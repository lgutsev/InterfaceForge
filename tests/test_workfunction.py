from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

_HAS_ASE = importlib.util.find_spec("ase") is not None


def _write_locpot(path: Path, *, cell: list[list[float]], value: float) -> None:
    """Write a minimal, realistic LOCPOT: raw grid values are the bare
    potential in eV (VASP does NOT multiply LOCPOT by the cell volume,
    unlike CHGCAR). A uniform 1x1x2 grid keeps the expected answer trivial.
    """

    lines = [
        "test cell",
        "1.0",
        f"{cell[0][0]:.6f} {cell[0][1]:.6f} {cell[0][2]:.6f}",
        f"{cell[1][0]:.6f} {cell[1][1]:.6f} {cell[1][2]:.6f}",
        f"{cell[2][0]:.6f} {cell[2][1]:.6f} {cell[2][2]:.6f}",
        "H",
        "1",
        "Direct",
        "0.0 0.0 0.0",
        "",
        "1 1 2",
        f"{value:.10E} {value:.10E}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@unittest.skipUnless(_HAS_ASE, "ASE is required for LOCPOT analysis")
class WorkFunctionTests(unittest.TestCase):
    def test_work_function_sign_is_independent_of_cell_handedness(self) -> None:
        from interfaceforge.workfunction import analyze_workfunction

        value = 4.5
        right_handed = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 8.0]]
        # Swapping two lattice vectors negates det(cell) (left-handed basis)
        # without changing any axis length, including the z-axis LOCPOT is
        # averaged along.
        left_handed = [[0.0, 4.0, 0.0], [4.0, 0.0, 0.0], [0.0, 0.0, 8.0]]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outcar = root / "OUTCAR"
            outcar.write_text("E-fermi :   0.0000     XC(G=0)\n", encoding="utf-8")

            results = {}
            for label, cell in (("right", right_handed), ("left", left_handed)):
                locpot = root / f"LOCPOT_{label}"
                _write_locpot(locpot, cell=cell, value=value)
                summary = analyze_workfunction(
                    locpot,
                    outcar,
                    data_output=root / f"locpot_{label}.dat",
                    summary_output=root / f"summary_{label}.json",
                )
                results[label] = summary["work_function_ev"]

            # A left-handed cell describes the exact same physical potential
            # landscape; the recovered work function must not flip sign.
            self.assertAlmostEqual(results["right"], value)
            self.assertAlmostEqual(results["left"], value)


if __name__ == "__main__":
    unittest.main()
