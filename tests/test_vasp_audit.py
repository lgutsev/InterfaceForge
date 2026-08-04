from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from interfaceforge.audit import audit_run, find_runs, run_audit
from interfaceforge.vasp import (
    apply_incar_preset,
    package_outputs,
    parse_incar,
    prepare_recovery,
    update_incar,
)


class VaspTests(unittest.TestCase):
    def test_incar_update_preserves_unrelated_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            incar = Path(temporary) / "INCAR"
            incar.write_text("ENCUT = 520 ! keep\nNSW = 10\n# note\n", encoding="utf-8")
            update_incar(incar, {"NSW": 200, "IBRION": 2})
            text = incar.read_text(encoding="utf-8")
            self.assertIn("ENCUT = 520 ! keep", text)
            self.assertIn("# note", text)
            self.assertEqual(parse_incar(incar)["NSW"], "200")

    def test_md_preset_leaves_convergence_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            incar = Path(temporary) / "INCAR"
            incar.write_text("ENCUT = 600\nEDIFF = 1E-7\n", encoding="utf-8")
            apply_incar_preset(
                incar, "md", temperature=450, nsw=1000, potim=0.5
            )
            parsed = parse_incar(incar)
            self.assertEqual(parsed["ENCUT"], "600")
            self.assertEqual(parsed["EDIFF"], "1E-7")
            self.assertEqual(parsed["TEBEG"], "450")
            self.assertEqual(parsed["POTIM"], "0.5")

    def test_portable_archive_excludes_potcar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "INCAR").write_text("NSW=0\n", encoding="utf-8")
            (root / "POTCAR").write_text("licensed\n", encoding="utf-8")
            output = root / "portable.zip"
            package_outputs(root, output)
            with zipfile.ZipFile(output) as archive:
                self.assertIn("INCAR", archive.namelist())
                self.assertNotIn("POTCAR", archive.namelist())

    def test_continue_recovery_does_not_self_copy_only_ml_ab(self) -> None:
        # When ML_ABN is absent/empty, _continue_source() falls back to
        # ML_AB itself; recovery must not shutil.copy2() that file onto
        # itself (shutil.SameFileError) when rebuilding it as the source.
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "INCAR").write_text("ML_MODE=train\nNSW=10\nTEBEG=300\n", encoding="utf-8")
            (run / "POTCAR").write_text("licensed\n", encoding="utf-8")
            (run / "KPOINTS").write_text("Automatic\n0\nGamma\n1 1 1\n", encoding="utf-8")
            (run / "CONTCAR").write_text("dummy contcar\n", encoding="utf-8")
            (run / "ML_AB").write_text("dummy ml_ab payload\n", encoding="utf-8")

            result = prepare_recovery(run, "continue")

            self.assertEqual(result["operation"], "continue")
            self.assertEqual((run / "ML_AB").read_text(encoding="utf-8"), "dummy ml_ab payload\n")

    def test_mode_aware_audit_recognizes_completed_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "INCAR").write_text(
                "ML_MODE=train\nNSW=2\nPOTIM=1\nTEBEG=300\nTEEND=300\n",
                encoding="utf-8",
            )
            (run / "OSZICAR").write_text(
                " 1 T= 300.0 E= -1\n 2 T= 301.0 E= -1\n", encoding="utf-8"
            )
            (run / "ML_LOGFILE").write_text(
                "STATUS accepted\nSTATUS accepted\nERR a b 0.02\n", encoding="utf-8"
            )
            (run / "OUTCAR").write_text(
                "General timing and accounting informations for this job\n",
                encoding="utf-8",
            )
            row = audit_run(run, run)
            self.assertEqual(row["ml_mode"], "train")
            self.assertEqual(row["progress_pct"], 100.0)
            self.assertEqual(row["health"], "ready to refit and test")

    def test_run_discovery_excludes_archives_unless_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "campaign"
            active = root / "FAPI_001_PBGCl"
            legacy_archive = root / "FAPI_001_PBGCl_old" / "expand_archive_20260728_125535"
            internal_archive = root / "FAPI_001_PBGCl" / ".interfaceforge" / "archive" / "continue_20260728"
            for run in (active, legacy_archive, internal_archive):
                run.mkdir(parents=True, exist_ok=True)
                (run / "INCAR").write_text("ML_MODE=train\n", encoding="utf-8")

            self.assertEqual(find_runs(root), [active])
            self.assertEqual(find_runs(root, include_archives=True), [active, legacy_archive])

    def test_archive_named_ancestor_outside_root_does_not_hide_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "archive_storage" / "campaign"
            active = root / "active_run"
            active.mkdir(parents=True)
            (active / "INCAR").write_text("ML_MODE=train\n", encoding="utf-8")

            self.assertEqual(find_runs(root), [active])

    def test_audit_writes_compact_summary_alongside_full_table(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "campaign"
            run = root / "active_run"
            run.mkdir(parents=True)
            (run / "INCAR").write_text("ML_MODE=train\nNSW=2\n", encoding="utf-8")
            (run / "OSZICAR").write_text(" 1 T= 300 E= -1\n", encoding="utf-8")
            (run / "ML_LOGFILE").write_text("STATUS accepted\n", encoding="utf-8")

            payload = run_audit(root)

            summary = Path(payload["outputs"]["summary_csv"])
            full = Path(payload["outputs"]["csv"])
            self.assertTrue(summary.is_file())
            self.assertTrue(full.is_file())
            self.assertLess(len(summary.read_text().split(",")), len(full.read_text().split(",")))
            self.assertIn("Next action", summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
