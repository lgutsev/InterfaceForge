from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from interfaceforge.adhesion import audit_adhesion, prepare_adhesion
from interfaceforge.errors import SafetyError

_POSCAR = """test interface
1.0
  5.00000000000000    0.00000000000000    0.00000000000000
  0.00000000000000    5.00000000000000    0.00000000000000
  0.00000000000000    0.00000000000000   20.00000000000000
Ti N
2 2
Direct
  0.0  0.0  0.10
  0.5  0.5  0.15
  0.0  0.0  0.60
  0.5  0.5  0.65
"""


def _write_reference(root: Path, *, with_ml_ff: bool = True) -> Path:
    source = root / "interface"
    source.mkdir()
    (source / "POSCAR").write_text(_POSCAR, encoding="utf-8")
    (source / "INCAR").write_text("ENCUT = 500\n", encoding="utf-8")
    (source / "KPOINTS").write_text("Automatic\n0\nGamma\n1 1 1\n", encoding="utf-8")
    (source / "POTCAR").write_text(
        "Ti block\nEnd of Dataset\nN block\nEnd of Dataset\n", encoding="utf-8"
    )
    if with_ml_ff:
        (source / "ML_FF").write_text("fake model bytes\n", encoding="utf-8")
    return source


class AdhesionTests(unittest.TestCase):
    def test_mlff_mode_hard_links_model_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_reference(root)

            manifest = prepare_adhesion(source, method="mlff", distances=[1, 2])

            self.assertEqual(manifest["interface_area_A2"], 25.0)
            self.assertEqual(manifest["split_plane_z_A"], 7.5)
            names = {record["directory"]: record for record in manifest["slabs"]}
            self.assertIn(str(Path("slabs") / "lower"), names)
            self.assertIn(str(Path("slabs") / "upper"), names)
            self.assertEqual(names[str(Path("slabs") / "lower")]["formula"], "Ti2")
            self.assertEqual(names[str(Path("slabs") / "upper")]["formula"], "N2")
            self.assertEqual(
                [record["separation_A"] for record in manifest["rigid_curve"]], [1.0, 2.0]
            )

            output = Path(manifest["output_directory"])
            lower_ff = output / "slabs" / "lower" / "ML_FF"
            self.assertTrue(lower_ff.is_file())
            self.assertEqual(lower_ff.stat().st_ino, (source / "ML_FF").stat().st_ino)
            self.assertTrue((output / "reference").is_symlink())
            self.assertTrue((output / "manifest.json").is_file())

            lower_incar = (output / "slabs" / "lower" / "INCAR").read_text(encoding="utf-8")
            self.assertIn("ENCUT = 500", lower_incar)
            self.assertIn("ML_LMLFF", lower_incar)
            self.assertIn("IBRION          = 2", lower_incar)

    def test_dft_mode_creates_no_ml_ff_and_strips_ml_tags(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_reference(root, with_ml_ff=False)
            (source / "INCAR").write_text(
                "ENCUT = 500\nML_LMLFF = .TRUE.\nML_MODE = run\n", encoding="utf-8"
            )

            manifest = prepare_adhesion(source, method="dft", distances=[1])

            output = Path(manifest["output_directory"])
            self.assertFalse((output / "slabs" / "lower" / "ML_FF").exists())
            incar = (output / "slabs" / "lower" / "INCAR").read_text(encoding="utf-8")
            self.assertNotIn("ML_LMLFF", incar)
            self.assertNotIn("ML_MODE", incar)
            self.assertIn("ENCUT = 500", incar)
            self.assertIn("EDIFF           = 1E-6", incar)

    def test_slab_mode_static_disables_ionic_relaxation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_reference(root, with_ml_ff=False)

            manifest = prepare_adhesion(
                source, method="dft", distances=[], slab_mode="static"
            )

            self.assertEqual(manifest["slab_mode"], "static")
            output = Path(manifest["output_directory"])
            for name in ("lower", "upper"):
                incar = (output / "slabs" / name / "INCAR").read_text(encoding="utf-8")
                self.assertIn("IBRION          = -1", incar)
                self.assertNotIn("IBRION          = 2", incar)

    def test_slab_mode_defaults_to_relax(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_reference(root, with_ml_ff=False)

            manifest = prepare_adhesion(source, method="dft", distances=[])

            self.assertEqual(manifest["slab_mode"], "relax")
            output = Path(manifest["output_directory"])
            incar = (output / "slabs" / "lower" / "INCAR").read_text(encoding="utf-8")
            self.assertIn("IBRION          = 2", incar)

    def test_slab_mode_static_works_for_mlff_too(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_reference(root)

            manifest = prepare_adhesion(
                source, method="mlff", distances=[], slab_mode="static"
            )

            output = Path(manifest["output_directory"])
            incar = (output / "slabs" / "lower" / "INCAR").read_text(encoding="utf-8")
            self.assertIn("IBRION          = -1", incar)
            self.assertIn("ML_LMLFF", incar)
            # A hard link to ML_FF is still expected in static mode.
            self.assertTrue((output / "slabs" / "lower" / "ML_FF").exists())

    def test_invalid_slab_mode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_reference(root, with_ml_ff=False)

            with self.assertRaises(ValueError):
                prepare_adhesion(source, method="dft", distances=[], slab_mode="bogus")

    def test_rigid_curve_shifts_upper_fragment_and_expands_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_reference(root, with_ml_ff=False)

            manifest = prepare_adhesion(source, method="dft", distances=[1])

            output = Path(manifest["output_directory"])
            poscar = (output / "rigid_curve" / "sep_001.00_A" / "POSCAR").read_text(
                encoding="utf-8"
            )
            lines = poscar.splitlines()
            self.assertIn("21.00000000000000", lines[4])  # c vector grew by the separation
            # Fractional z for the shifted upper atoms: (12+1)/21 and (13+1)/21.
            self.assertTrue(any("0.61904761904762" in line for line in lines))
            self.assertTrue(any("0.66666666666667" in line for line in lines))

    def test_guard_distance_violation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_reference(root, with_ml_ff=False)

            with self.assertRaisesRegex(ValueError, "too close|from the cut"):
                prepare_adhesion(source, method="dft", z_plane=2.05, guard=0.20)

    def test_existing_output_directory_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_reference(root, with_ml_ff=False)
            output_dir = root / "existing"
            output_dir.mkdir()

            with self.assertRaises(SafetyError):
                prepare_adhesion(source, method="dft", output_dir=str(output_dir))

    def test_explicit_z_plane_overrides_auto_detection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_reference(root, with_ml_ff=False)

            manifest = prepare_adhesion(source, method="dft", z_plane=5.0, distances=[])

            self.assertEqual(manifest["split_plane_z_A"], 5.0)
            self.assertEqual(manifest["plane_source"], "explicit")
            self.assertIsNone(manifest["detected_gap_A"])

    def test_launcher_is_propagated_to_every_generated_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_reference(root, with_ml_ff=False)
            launcher_text = "#!/bin/bash\nsrun vasp_gam\n"
            (source / "runvasp.sh").write_text(launcher_text, encoding="utf-8")

            manifest = prepare_adhesion(source, method="dft", distances=[1])

            self.assertEqual(manifest["launcher"], "runvasp.sh")
            output = Path(manifest["output_directory"])
            for relative in ("slabs/lower", "slabs/upper", "rigid_curve/sep_001.00_A"):
                copied = output / Path(relative) / "runvasp.sh"
                self.assertTrue(copied.is_file(), relative)
                self.assertEqual(copied.read_text(encoding="utf-8"), launcher_text)

    def test_launcher_prefers_runvasp_over_run_slurm(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_reference(root, with_ml_ff=False)
            (source / "run.slurm").write_text("slurm launcher\n", encoding="utf-8")
            (source / "runvasp.sh").write_text("runvasp launcher\n", encoding="utf-8")

            manifest = prepare_adhesion(source, method="dft", distances=[])

            self.assertEqual(manifest["launcher"], "runvasp.sh")

    def test_no_launcher_opts_out_even_when_one_is_present(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_reference(root, with_ml_ff=False)
            (source / "runvasp.sh").write_text("#!/bin/bash\n", encoding="utf-8")

            manifest = prepare_adhesion(
                source, method="dft", distances=[1], propagate_launcher=False
            )

            self.assertIsNone(manifest["launcher"])
            output = Path(manifest["output_directory"])
            self.assertFalse((output / "slabs" / "lower" / "runvasp.sh").exists())

    def test_missing_launcher_is_reported_as_none_not_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_reference(root, with_ml_ff=False)

            manifest = prepare_adhesion(source, method="dft", distances=[])

            self.assertIsNone(manifest["launcher"])


def _fake_outcar(energy: float) -> str:
    return (
        " FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)\n"
        "  ---------------------------------------------------\n"
        f"  free  energy   TOTEN  =      {energy:.6f} eV\n\n"
        f"  energy  without entropy=     {energy + 0.01:.6f}  "
        f"energy(sigma->0) =     {energy:.6f}\n\n"
        " General timing and accounting informations for this job\n"
    )


class AdhesionAuditTests(unittest.TestCase):
    def test_audit_computes_work_of_adhesion_and_curve_once_runs_finish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_reference(root, with_ml_ff=False)
            (source / "OUTCAR").write_text(_fake_outcar(-200.0), encoding="utf-8")

            manifest = prepare_adhesion(source, method="dft", distances=[1, 2])
            output = Path(manifest["output_directory"])
            (output / "slabs" / "lower" / "OUTCAR").write_text(
                _fake_outcar(-90.0), encoding="utf-8"
            )
            (output / "slabs" / "upper" / "OUTCAR").write_text(
                _fake_outcar(-95.0), encoding="utf-8"
            )
            (output / "rigid_curve" / "sep_001.00_A" / "OUTCAR").write_text(
                _fake_outcar(-184.0), encoding="utf-8"
            )
            (output / "rigid_curve" / "sep_002.00_A" / "OUTCAR").write_text(
                _fake_outcar(-186.0), encoding="utf-8"
            )

            result = audit_adhesion(output)

            row = result["work_of_adhesion"]["rows"][0]
            self.assertAlmostEqual(row["work_of_adhesion_ev_a2"], 0.6)
            self.assertAlmostEqual(row["work_of_adhesion_j_m2"], 0.6 * 16.02176634)
            self.assertEqual(result["rigid_curve_points_ready"], 2)
            self.assertEqual(result["rigid_curve_points_total"], 2)
            self.assertTrue(Path(result["audit_json"]).is_file())
            self.assertTrue(Path(result["audit_markdown"]).is_file())

    def test_audit_reports_none_before_anything_has_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_reference(root, with_ml_ff=False)

            manifest = prepare_adhesion(source, method="dft", distances=[1])
            output = Path(manifest["output_directory"])

            result = audit_adhesion(output)

            self.assertIsNone(result["work_of_adhesion"])
            self.assertIsNone(result["separation_curve"])
            self.assertEqual(result["rigid_curve_points_ready"], 0)
            self.assertFalse(result["reference"]["finished_normally"])

    def test_audit_computes_partial_curve_with_some_points_still_running(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_reference(root, with_ml_ff=False)
            (source / "OUTCAR").write_text(_fake_outcar(-200.0), encoding="utf-8")

            manifest = prepare_adhesion(source, method="dft", distances=[1, 2])
            output = Path(manifest["output_directory"])
            (output / "slabs" / "lower" / "OUTCAR").write_text(
                _fake_outcar(-90.0), encoding="utf-8"
            )
            (output / "slabs" / "upper" / "OUTCAR").write_text(
                _fake_outcar(-95.0), encoding="utf-8"
            )
            # Only one of the two rigid-curve points has finished.
            (output / "rigid_curve" / "sep_001.00_A" / "OUTCAR").write_text(
                _fake_outcar(-184.0), encoding="utf-8"
            )

            result = audit_adhesion(output)

            self.assertIsNotNone(result["work_of_adhesion"])
            self.assertIsNotNone(result["separation_curve"])
            self.assertEqual(result["rigid_curve_points_ready"], 1)
            self.assertEqual(result["rigid_curve_points_total"], 2)

    def test_audit_requires_an_existing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(SafetyError):
                audit_adhesion(Path(temporary) / "never_prepared")


if __name__ == "__main__":
    unittest.main()
