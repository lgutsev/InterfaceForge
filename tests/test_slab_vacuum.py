from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from interfaceforge.cli import main
from interfaceforge.geometry import extend_slab_vacuum, slab_vacuum

try:
    from ase import Atoms
    from ase.io import write as ase_write

    HAVE_ASE = True
except ImportError:  # pragma: no cover
    HAVE_ASE = False


def _slab(vac_low: float, vac_high: float, thickness: float = 6.0) -> Atoms:
    """A filled toy slab (2x2 in-plane, dense in c) with a known vacuum split."""

    c = vac_low + thickness + vac_high
    positions = [
        (x, y, z)
        for x in (0.0, 3.0)
        for y in (0.0, 3.0)
        for z in np.linspace(vac_low, vac_low + thickness, 5)
    ]
    return Atoms(f"Ni{len(positions)}", positions=positions, cell=[6.0, 6.0, c], pbc=True)


@unittest.skipUnless(HAVE_ASE, "ASE not installed")
class SlabVacuumTests(unittest.TestCase):
    def _write(self, atoms: Atoms, path: Path) -> Path:
        ase_write(str(path), atoms, format="vasp", direct=True, vasp5=True)
        return path

    def test_reports_asymmetric_vacuum_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            poscar = self._write(_slab(10.0, 3.0), Path(tmp) / "POSCAR")
            report = slab_vacuum(poscar)
            self.assertEqual(report["axis"], "c")
            self.assertAlmostEqual(report["vacuum_low_a"], 10.0, places=3)
            self.assertAlmostEqual(report["vacuum_high_a"], 3.0, places=3)
            self.assertAlmostEqual(report["vacuum_total_a"], 13.0, places=3)
            self.assertAlmostEqual(report["min_side_a"], 3.0, places=3)

    def test_auto_axis_picks_the_normal_on_a_real_slab(self) -> None:
        from ase.build import fcc111

        with tempfile.TemporaryDirectory() as tmp:
            slab = fcc111("Ni", size=(3, 3, 4), vacuum=9.0)
            poscar = self._write(slab, Path(tmp) / "POSCAR")
            report = slab_vacuum(poscar)
            self.assertEqual(report["axis"], "c")
            self.assertAlmostEqual(report["vacuum_total_a"], 18.0, delta=0.5)

    def test_wrapped_slab_is_recentred_for_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            atoms = _slab(3.0, 3.0, thickness=6.0)  # c = 12, total vacuum 6
            atoms.positions[:, 2] += 8.0  # push the slab across the top boundary
            poscar = self._write(atoms, Path(tmp) / "POSCAR")
            report = slab_vacuum(poscar)
            self.assertTrue(report["wrapped"])
            self.assertAlmostEqual(report["vacuum_total_a"], 6.0, places=3)

    def test_extend_recentres_to_target_per_side(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = self._write(_slab(10.0, 2.5), Path(tmp) / "POSCAR")
            out = Path(tmp) / "POSCAR.ext"
            summary = extend_slab_vacuum(src, out, vacuum_per_side=12.0)
            after = slab_vacuum(out)
            self.assertAlmostEqual(after["vacuum_low_a"], 12.0, places=2)
            self.assertAlmostEqual(after["vacuum_high_a"], 12.0, places=2)
            self.assertTrue(summary["recentred"])

    def test_extend_keep_position_only_adds_missing_vacuum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = self._write(_slab(14.0, 2.0), Path(tmp) / "POSCAR")
            out = Path(tmp) / "POSCAR.ext"
            extend_slab_vacuum(src, out, vacuum_per_side=12.0, keep_position=True)
            after = slab_vacuum(out)
            # low was already above target, so it is left alone; high is raised.
            self.assertAlmostEqual(after["vacuum_low_a"], 14.0, places=2)
            self.assertAlmostEqual(after["vacuum_high_a"], 12.0, places=2)

    def test_cli_audit_flags_thin_face_and_suggests_fix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = self._write(_slab(10.0, 3.0), Path(tmp) / "POSCAR")
            summary_path = Path(tmp) / "out.json"
            rc = main(
                ["vasp", "geom", "vacuum", str(src), "--min-vacuum", "12", "--summary", str(summary_path)]
            )
            self.assertEqual(rc, 0)
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "THIN")
            self.assertIn("--extend", payload["fix"])

    def test_cli_extend_writes_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = self._write(_slab(10.0, 3.0), Path(tmp) / "POSCAR")
            out = Path(tmp) / "POSCAR.big"
            rc = main(["vasp", "geom", "vacuum", str(src), "--extend", "15", "-o", str(out)])
            self.assertEqual(rc, 0)
            self.assertTrue(out.is_file())
            self.assertGreaterEqual(slab_vacuum(out)["min_side_a"], 14.9)


if __name__ == "__main__":
    unittest.main()
