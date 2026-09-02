from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from interfaceforge.slab_alignment import (
    add_alignment_deltas,
    analyze_profile,
    analyze_slab_alignment,
    band_edges_from_vasprun,
    ionic_center_fraction,
    parse_poscar_lines,
    read_locpot,
    write_dipole_preview,
)

POSCAR = """Synthetic slab
1.0
10 0 0
0 10 0
0 0 40
Pb I H
1 1 1
Direct
0.5 0.5 0.30
0.5 0.5 0.60
0.5 0.5 0.65
"""


def _write_calculation(
    folder: Path,
    high_vacuum: float,
    vbm: float,
    cbm: float,
    *,
    high_slope: float = 0.0,
) -> None:
    folder.mkdir()
    nz = 80
    z_grid = np.arange(nz) * 40.0 / nz
    shifted = np.mod(z_grid - 39.0, 40.0)
    planar = np.where(
        shifted < 13.0,
        4.8,
        np.where(shifted > 27.0, high_vacuum + high_slope * (shifted - 27.0), 2.0),
    )
    raw = [value for value in planar for _ in range(4)]
    (folder / "LOCPOT").write_text(
        POSCAR + "\n2 2 80\n" + " ".join(str(value) for value in raw) + "\n",
        encoding="utf-8",
    )
    (folder / "OUTCAR").write_text(" E-fermi : 0.000000 eV\n", encoding="utf-8")
    (folder / "INCAR").write_text(
        "LDIPOL = .TRUE.\nIDIPOL = 3\nDIPOL = 0.5 0.5 0.5\n",
        encoding="utf-8",
    )
    (folder / "vasprun.xml").write_text(
        f'<modeling><i name="efermi">0.0</i><eigenvalues><array><set><set>'
        f"<r>{vbm} 2.0</r><r>{cbm} 0.0</r>"
        "</set></set></array></eigenvalues></modeling>",
        encoding="utf-8",
    )


class SlabAlignmentTests(unittest.TestCase):
    def test_poscar_and_fractional_dipole_center(self) -> None:
        structure = parse_poscar_lines(POSCAR.splitlines())
        self.assertAlmostEqual(structure.c_length, 40.0)
        center, compactness, missing = ionic_center_fraction(structure)
        self.assertGreater(center, 0.25)
        self.assertLess(center, 0.65)
        self.assertGreater(compactness, 0.0)
        self.assertEqual(missing, [])

    def test_raw_locpot_has_no_volume_scaling(self) -> None:
        values = " ".join(str(float(index)) for index in range(1, 17))
        locpot = POSCAR + "\n2 2 4\n" + values + "\n"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "LOCPOT"
            path.write_text(locpot, encoding="utf-8")
            structure, _z_grid, planar = read_locpot(path)
        self.assertTrue(np.allclose(planar, [2.5, 6.5, 10.5, 14.5]))
        self.assertAlmostEqual(structure.c_length, 40.0)

    def test_physical_sides_are_never_merged(self) -> None:
        structure = parse_poscar_lines(POSCAR.splitlines())
        z_grid = np.linspace(0, 40, 800, endpoint=False)
        potential = np.full_like(z_grid, 1.0)
        potential[z_grid < 10] = 4.0
        potential[z_grid > 28] = 5.0
        profile, _shifted_z, _shifted_potential = analyze_profile(
            structure,
            z_grid,
            potential,
            buffer_angstrom=1.0,
            minimum_window_angstrom=1.0,
        )
        self.assertAlmostEqual(abs(profile.high.plateau_eV - profile.low.plateau_eV), 1.0)

    def test_slope_is_detected(self) -> None:
        structure = parse_poscar_lines(POSCAR.splitlines())
        z_grid = np.linspace(0, 40, 800, endpoint=False)
        potential = 4.0 + 0.02 * z_grid
        profile, _shifted_z, _shifted_potential = analyze_profile(
            structure,
            z_grid,
            potential,
            buffer_angstrom=1.0,
            minimum_window_angstrom=1.0,
        )
        self.assertFalse(profile.high.correction_step_detected)
        self.assertAlmostEqual(abs(profile.high.slope_eV_per_A), 0.02, places=4)

    def test_high_side_dipole_step_is_excluded_from_flatness_fit(self) -> None:
        structure = parse_poscar_lines(POSCAR.splitlines())
        z_grid = np.linspace(0, 40, 800, endpoint=False)
        shifted = np.mod(z_grid - 39.0, 40.0)
        high_vacuum = 5.2 - 0.2 * np.clip((shifted - 34.6) / 0.8, 0.0, 1.0)
        potential = np.where(
            shifted < 13.0,
            4.8,
            np.where(shifted < 27.0, 2.0, high_vacuum),
        )
        profile, _shifted_z, _shifted_potential = analyze_profile(
            structure,
            z_grid,
            potential,
            buffer_angstrom=1.0,
            minimum_window_angstrom=1.0,
        )
        self.assertTrue(profile.high.correction_step_detected)
        self.assertAlmostEqual(profile.high.correction_step_A, 35.0, delta=0.1)
        self.assertAlmostEqual(profile.high.correction_step_width_A, 0.8, delta=0.15)
        self.assertLess(profile.high.window_end_A, profile.high.correction_step_A)
        self.assertAlmostEqual(profile.high.plateau_eV, 5.2, places=6)
        self.assertAlmostEqual(profile.high.swing_eV, 0.0, places=6)

    def test_low_side_dipole_step_keeps_surface_adjacent_plateau(self) -> None:
        structure = parse_poscar_lines(POSCAR.splitlines())
        z_grid = np.linspace(0, 40, 800, endpoint=False)
        shifted = np.mod(z_grid - 39.0, 40.0)
        potential = np.where(
            shifted < 6.0,
            4.5,
            np.where(shifted < 13.0, 4.8, np.where(shifted < 27.0, 2.0, 5.2)),
        )
        profile, _shifted_z, _shifted_potential = analyze_profile(
            structure,
            z_grid,
            potential,
            buffer_angstrom=1.0,
            minimum_window_angstrom=1.0,
        )
        self.assertTrue(profile.low.correction_step_detected)
        self.assertGreater(profile.low.window_start_A, profile.low.correction_step_A)
        self.assertAlmostEqual(profile.low.plateau_eV, 4.8, places=6)
        self.assertAlmostEqual(profile.low.swing_eV, 0.0, places=6)

    def test_wrapped_slab_still_has_two_sides(self) -> None:
        wrapped = """Wrapped slab
1.0
10 0 0
0 10 0
0 0 40
Pb I
1 1
Direct
0.5 0.5 0.90
0.5 0.5 0.10
"""
        structure = parse_poscar_lines(wrapped.splitlines())
        z_grid = np.linspace(0, 40, 800, endpoint=False)
        shifted = np.mod(z_grid - 20.0, 40.0)
        potential = np.where(
            shifted < 14,
            4.0,
            np.where(shifted > 26, 5.0, 1.0),
        )
        profile, _shifted_z, _shifted_potential = analyze_profile(structure, z_grid, potential, 1.0, 1.0)
        self.assertAlmostEqual(profile.low.plateau_eV, 4.0)
        self.assertAlmostEqual(profile.high.plateau_eV, 5.0)

    def test_vasprun_edges(self) -> None:
        xml = """<modeling>
<i name="efermi">0.2</i>
<eigenvalues><array><set><set>
<r>-1.0 2.0</r><r>1.5 0.0</r>
</set></set></array></eigenvalues>
</modeling>"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "vasprun.xml"
            path.write_text(xml, encoding="utf-8")
            result = band_edges_from_vasprun(path)
        self.assertEqual(result, (0.2, -1.0, 1.5, 2.5))

    def test_alignment_sign(self) -> None:
        rows = [
            {
                "folder": "MAPI_MAI_Surf",
                "reference": "MAPI_MAI_Surf",
                "status": "OK",
                "vbm_vac_eV": -5.7,
                "cbm_vac_eV": -4.1,
            },
            {
                "folder": "MAPI_MAI_Surf_BPDCA",
                "reference": "MAPI_MAI_Surf",
                "status": "OK",
                "vbm_vac_eV": -5.6,
                "cbm_vac_eV": -3.9,
            },
        ]
        add_alignment_deltas(rows)
        self.assertAlmostEqual(rows[1]["delta_vbm_eV"], 0.1)
        self.assertAlmostEqual(rows[1]["delta_cbm_eV"], 0.2)

    def test_dipole_fix_is_non_destructive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incar = root / "INCAR"
            original = "ENCUT = 520\nLDIPOL = .FALSE.\nDIPOL = 0.25 0.75 0.5\n"
            incar.write_text(original, encoding="utf-8")
            preview = write_dipole_preview(root, 0.625)
            self.assertEqual(incar.read_text(encoding="utf-8"), original)
            generated = preview.read_text(encoding="utf-8")
            self.assertIn("LDIPOL = .TRUE.", generated)
            self.assertIn("IDIPOL = 3", generated)
            self.assertIn("DIPOL  = 0.250000 0.750000 0.625000", generated)

    def test_two_folder_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_calculation(root / "MAPI_MAI_Surf", 5.4, -1.0, 1.0)
            _write_calculation(root / "MAPI_MAI_Surf_BPDCA", 5.2, -0.9, 1.1)
            config_path = root / "slab_alignment.json"
            config_path.write_text(
                json.dumps(
                    {
                        "side": "high-z",
                        "references": [
                            {
                                "prefix": "MAPI_MAI_Surf",
                                "reference": "MAPI_MAI_Surf",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            payload = analyze_slab_alignment(root, config=config_path)
            self.assertEqual(payload["status"], "OK")
            with (root / "band_edge_alignment.tsv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            case = next(row for row in rows if row["folder"].endswith("BPDCA"))
            self.assertAlmostEqual(float(case["delta_cbm_eV"]), 0.3)
            self.assertAlmostEqual(float(case["delta_vbm_eV"]), 0.3)
            self.assertTrue((root / "MAPI_MAI_Surf_BPDCA" / "vacuum_profile.png").is_file())
            self.assertTrue((root / "MAPI_MAI_Surf_BPDCA" / "Workfunction.png").is_file())
            self.assertTrue((root / "MAPI_MAI_Surf_BPDCA" / "LOCPOT_FLATNESS_OK").is_file())
            self.assertFalse((root / "MAPI_MAI_Surf_BPDCA" / "INCAR.dipole_fix").exists())
            self.assertTrue((root / "dipole_flatness_audit.tsv").is_file())
            self.assertTrue((root / "relaunch_review_queue.txt").is_file())

    def test_nonflat_case_is_flagged_and_gets_preview_automatically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = root / "MAPI_MAI_Surf_BPDCA"
            _write_calculation(case, 5.2, -0.9, 1.1, high_slope=0.03)
            config_path = root / "slab_alignment.json"
            config_path.write_text(
                json.dumps(
                    {
                        "side": "high-z",
                        "references": [
                            {"prefix": "MAPI_MAI_Surf", "reference": "MAPI_MAI_Surf"}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            payload = analyze_slab_alignment(root, config=config_path)
            row = payload["rows"][0]
            self.assertEqual(row["flatness_status"], "FAILED_FLATNESS")
            self.assertTrue(row["relaunch_review_required"])
            self.assertTrue((case / "RELAUNCH_REVIEW_REQUIRED").is_file())
            self.assertTrue((case / "INCAR.dipole_fix").is_file())
            self.assertFalse((case / "LOCPOT_FLATNESS_OK").exists())
            self.assertIn("MAPI_MAI_Surf_BPDCA", (root / "relaunch_review_queue.txt").read_text())

    def test_flatness_audit_survives_missing_band_edge_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = root / "MAPI_MAI_Surf"
            _write_calculation(case, 5.4, -1.0, 1.0)
            (case / "INCAR.dipole_fix").write_text("obsolete proposal\n", encoding="utf-8")
            (case / "vasprun.xml").unlink()
            config_path = root / "slab_alignment.json"
            config_path.write_text(
                json.dumps(
                    {
                        "side": "high-z",
                        "references": [
                            {"prefix": "MAPI_MAI_Surf", "reference": "MAPI_MAI_Surf"}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            payload = analyze_slab_alignment(root, config=config_path)
            row = payload["rows"][0]
            self.assertEqual(row["flatness_status"], "OK")
            self.assertEqual(row["band_edge_status"], "FAILED_BAND_EDGES")
            self.assertFalse(row["relaunch_review_required"])
            self.assertTrue((case / "LOCPOT_FLATNESS_OK").is_file())
            self.assertFalse((case / "INCAR.dipole_fix").exists())


if __name__ == "__main__":
    unittest.main()
