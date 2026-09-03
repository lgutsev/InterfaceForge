from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from interfaceforge.cli import build_parser
from interfaceforge.slab_alignment import largest_periodic_gap, parse_poscar_lines
from interfaceforge.slab_publication import (
    _selected_side_plot_window,
    match_excess_atoms,
    plot_slab_publication,
    read_sumo_curve,
)

REFERENCE_POSCAR = """Reference
1.0
10 0 0
0 10 0
0 0 40
Pb I C N H
1 1 1 1 1
Direct
0.50 0.50 0.30
0.50 0.50 0.55
0.40 0.40 0.45
0.42 0.40 0.47
0.44 0.40 0.49
"""

PASSIVATED_POSCAR = """Passivated
1.0
10 0 0
0 10 0
0 0 40
Pb I C N H O Br
1 1 2 2 2 2 1
Direct
0.50 0.50 0.30
0.50 0.50 0.55
0.40 0.40 0.45
0.60 0.60 0.76
0.42 0.40 0.47
0.62 0.60 0.78
0.44 0.40 0.49
0.64 0.60 0.80
0.66 0.60 0.82
0.68 0.60 0.84
0.70 0.60 0.86
"""


def _write_sumo(folder: Path, passivated: bool) -> None:
    data = folder / "publication_dos_data"
    data.mkdir()
    energy = np.linspace(-2.0, 3.0, 101)
    total = np.exp(-((energy + 1.0) / 0.35) ** 2) + np.exp(-((energy - 1.0) / 0.35) ** 2)

    def write(stem: str, density: np.ndarray) -> None:
        np.savetxt(data / f"{stem}_dos.dat", np.column_stack((energy, density)))

    write("total", total)
    write("Pb", 0.6 * np.exp(-((energy - 1.0) / 0.35) ** 2))
    write("I", 0.7 * np.exp(-((energy + 1.0) / 0.35) ** 2))
    if passivated:
        for symbol, scale in (("C", 0.08), ("N", 0.04), ("H", 0.02), ("O", 0.12), ("Br", 0.06)):
            write(symbol, scale * np.exp(-((energy - 0.4) / 0.5) ** 2))


def _write_case(folder: Path, poscar: str, vacuum: float, vbm: float, cbm: float) -> None:
    folder.mkdir()
    structure = parse_poscar_lines(poscar.splitlines())
    gap_start, _gap_end, gap_width = largest_periodic_gap(structure.z_angstrom, structure.c_length)
    cut = (gap_start + 0.5 * gap_width) % structure.c_length
    shifted_atoms = np.mod(structure.z_angstrom - cut, structure.c_length)
    nz = 160
    z_grid = np.arange(nz) * structure.c_length / nz
    shifted = np.mod(z_grid - cut, structure.c_length)
    potential = np.where(
        shifted <= np.min(shifted_atoms) - 2.0,
        vacuum - 0.2,
        np.where(shifted >= np.max(shifted_atoms) + 2.0, vacuum, 2.0),
    )
    raw = [value for value in potential for _ in range(4)]
    (folder / "LOCPOT").write_text(
        poscar + "\n2 2 160\n" + " ".join(str(value) for value in raw) + "\n",
        encoding="utf-8",
    )
    (folder / "OUTCAR").write_text(" E-fermi : 0.000000 eV\n", encoding="utf-8")
    (folder / "vasprun.xml").write_text(
        f'<modeling><i name="efermi">0.0</i><eigenvalues><array><set><set>'
        f"<r>{vbm} 2.0</r><r>{cbm} 0.0</r>"
        "</set></set></array></eigenvalues></modeling>",
        encoding="utf-8",
    )


class SlabPublicationTests(unittest.TestCase):
    def test_cli_parser(self) -> None:
        args = build_parser().parse_args(
            ["vasp", "slab-publish", ".", "--config", "publish.json", "--run-sumo"]
        )
        self.assertEqual(args.config, "publish.json")
        self.assertTrue(args.run_sumo)

    def test_added_atoms_are_species_local_and_exclude_ma(self) -> None:
        reference = parse_poscar_lines(REFERENCE_POSCAR.splitlines())
        passivated = parse_poscar_lines(PASSIVATED_POSCAR.splitlines())
        excess = match_excess_atoms(reference, passivated)
        self.assertEqual(excess["C"], [2])
        self.assertEqual(excess["N"], [2])
        self.assertEqual(excess["H"], [2])
        self.assertEqual(excess["O"], [1, 2])
        self.assertEqual(excess["Br"], [1])
        self.assertNotIn("Pb", excess)
        self.assertNotIn("I", excess)

    def test_added_atoms_allow_different_vacuum_length(self) -> None:
        reference = parse_poscar_lines(REFERENCE_POSCAR.splitlines())
        taller_cell = PASSIVATED_POSCAR.replace("0 0 40", "0 0 44", 1)
        passivated = parse_poscar_lines(taller_cell.splitlines())
        # Preserve Cartesian slab geometry while adding 2 A of vacuum to each side.
        passivated.fractional[:, 2] = 0.5 + (passivated.fractional[:, 2] - 0.5) * 40.0 / 44.0
        excess = match_excess_atoms(reference, passivated)
        self.assertEqual(excess["C"], [2])
        self.assertEqual(excess["O"], [1, 2])

    def test_added_atoms_reject_in_plane_cell_change(self) -> None:
        reference = parse_poscar_lines(REFERENCE_POSCAR.splitlines())
        changed_surface = PASSIVATED_POSCAR.replace("10 0 0", "10.5 0 0", 1)
        passivated = parse_poscar_lines(changed_surface.splitlines())
        with self.assertRaisesRegex(Exception, "in-plane cells differ"):
            match_excess_atoms(reference, passivated)

    def test_sumo_curve_combines_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "Pb_dos.dat"
            path.write_text("# E s p\n0 1 -2\n1 3 -4\n", encoding="utf-8")
            energy, density = read_sumo_curve(path)
        self.assertTrue(np.allclose(energy, [0, 1]))
        self.assertTrue(np.allclose(density, [3, 7]))

    def test_publication_crop_keeps_only_selected_surface_side(self) -> None:
        high = SimpleNamespace(
            name="high",
            selected=SimpleNamespace(side="high-z", window_start_A=41.0, window_end_A=48.0),
            profile=SimpleNamespace(c_length_A=52.0),
        )
        low = SimpleNamespace(
            name="low",
            selected=SimpleNamespace(side="low-z", window_start_A=2.0, window_end_A=9.0),
            profile=SimpleNamespace(c_length_A=52.0),
        )
        self.assertEqual(_selected_side_plot_window(high, 2.0), (39.0, 48.0))
        self.assertEqual(_selected_side_plot_window(low, 2.0), (2.0, 11.0))

    def test_end_to_end_publication_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label in ("MAI", "PbI2"):
                reference = root / f"MAPI_{label}_Surf"
                passivated = root / f"MAPI_{label}_Surf_BPDCA"
                _write_case(reference, REFERENCE_POSCAR, 5.4, -1.0, 1.0)
                _write_case(passivated, PASSIVATED_POSCAR, 5.2, -0.9, 1.1)
                _write_sumo(reference, False)
                _write_sumo(passivated, True)
            config = root / "slab_publication.json"
            config.write_text(
                json.dumps(
                    {
                        "side": "high-z",
                        "pairs": [
                            {
                                "label": "MAI-rich",
                                "reference": "MAPI_MAI_Surf",
                                "passivated": "MAPI_MAI_Surf_BPDCA",
                            },
                            {
                                "label": "PbI2-rich",
                                "reference": "MAPI_PbI2_Surf",
                                "passivated": "MAPI_PbI2_Surf_BPDCA",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            payload = plot_slab_publication(root, config=config)
            self.assertEqual(payload["status"], "OK")
            destination = root / "publication_figures"
            for name in (
                "vacuum_validation.pdf",
                "vacuum_validation.png",
                "vacuum_validation.svg",
                "electronic_alignment.pdf",
                "electronic_alignment.png",
                "electronic_alignment.svg",
                "publication_band_edges.tsv",
                "publication_manifest.json",
            ):
                self.assertTrue((destination / name).is_file(), name)
            with (destination / "publication_band_edges.tsv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertAlmostEqual(float(rows[0]["delta_cbm_eV"]), 0.3)
            self.assertAlmostEqual(float(rows[0]["delta_vbm_eV"]), 0.3)
            manifest = json.loads((destination / "publication_manifest.json").read_text())
            ligand = manifest["passivant_species_local_indices"]["MAPI_MAI_Surf_BPDCA"]
            self.assertEqual(ligand["C"], [2])
            self.assertEqual(manifest["vacuum_figure_scope"]["mode"], "selected-side-only")
            self.assertEqual(manifest["vacuum_figure_scope"]["side"], "high-z")


if __name__ == "__main__":
    unittest.main()
