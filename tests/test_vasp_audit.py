from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from interfaceforge.audit import audit_run, find_runs, run_audit
from interfaceforge.vasp import (
    apply_incar_preset,
    mlff_accuracy_profile_tags,
    package_outputs,
    parse_incar,
    prepare_recovery,
    resolve_launcher,
    submit_run,
    update_incar,
)


class VaspTests(unittest.TestCase):
    def test_accurate_profile_matches_vasp_two_stage_recipe(self) -> None:
        train = mlff_accuracy_profile_tags("accurate", "train")
        refit = mlff_accuracy_profile_tags("accurate", "refit")

        self.assertEqual(train["ML_IALGO_LINREG"], "1")
        self.assertEqual(train["ML_SION1"], "0.3")
        self.assertEqual(train["ML_MRB2"], "12")
        self.assertEqual(refit["ML_IALGO_LINREG"], "4")
        self.assertEqual(refit["ML_SION1"], "0.5")
        self.assertEqual(refit["ML_MRB2"], "12")
        self.assertEqual(refit["ML_EPS_LOW"], "1E-11")

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

    def test_capacity_recovery_accepts_vasp_local_reference_stop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "INCAR").write_text(
                "ML_MODE=train\nIBRION=0\nNSW=2000\nPOTIM=0.5\n"
                "MDALGO=2\nSMASS=1.0\nTEBEG=300\nTEEND=300\nML_MB=6000\n",
                encoding="utf-8",
            )
            (run / "POTCAR").write_text("licensed\n", encoding="utf-8")
            (run / "KPOINTS").write_text("Automatic\n0\nGamma\n1 1 1\n", encoding="utf-8")
            (run / "CONTCAR").write_text("continued geometry\n", encoding="utf-8")
            (run / "ML_ABN").write_text("training database\n", encoding="utf-8")
            (run / "OUTCAR").write_text(
                "Not enough storage reserved for local reference configurations, "
                "please increase ML_MB.\n",
                encoding="utf-8",
            )

            result = prepare_recovery(run, "expand", ml_mb=12000)

            parsed = parse_incar(run / "INCAR")
            self.assertEqual(result["operation"], "expand")
            self.assertEqual(parsed["ML_MODE"], "train")
            self.assertEqual(parsed["ML_MB"], "12000")
            self.assertEqual(parsed["ML_LBASIS_DISCARD"], ".FALSE.")
            self.assertEqual(parsed["POTIM"], "0.5")
            self.assertEqual(
                (run / "ML_AB").read_text(encoding="utf-8"), "training database\n"
            )
            self.assertEqual(
                (run / "POSCAR").read_text(encoding="utf-8"), "continued geometry\n"
            )

            archived_outcar = list(
                (run / ".interfaceforge" / "archive").glob("expand_*/OUTCAR")
            )
            self.assertEqual(len(archived_outcar), 1)

            audit = audit_run(
                run / ".interfaceforge" / "archive" / archived_outcar[0].parent.name,
                run,
            )
            self.assertTrue(audit["ml_capacity_stop"])
            self.assertEqual(audit["health"], "stopped: ML local-reference capacity")

    def test_capacity_discard_recovery_keeps_memory_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "INCAR").write_text(
                "ML_MODE=train\nNSW=2000\nPOTIM=0.5\nTEBEG=300\nML_MB=6000\n",
                encoding="utf-8",
            )
            (run / "POTCAR").write_text("licensed\n", encoding="utf-8")
            (run / "KPOINTS").write_text("Automatic\n0\nGamma\n1 1 1\n", encoding="utf-8")
            (run / "CONTCAR").write_text("continued geometry\n", encoding="utf-8")
            (run / "ML_ABN").write_text("training database\n", encoding="utf-8")
            (run / "OUTCAR").write_text(
                "Not enough storage reserved for local reference configurations.\n",
                encoding="utf-8",
            )

            result = prepare_recovery(run, "discard")

            parsed = parse_incar(run / "INCAR")
            self.assertEqual(result["operation"], "discard")
            self.assertEqual(parsed["ML_MB"], "6000")
            self.assertEqual(parsed["ML_LBASIS_DISCARD"], ".TRUE.")
            self.assertEqual(parsed["ML_MODE"], "train")
            self.assertNotIn("ML_EPS_LOW", parsed)

    def test_capacity_recovery_can_apply_guarded_eps_low_increase(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "INCAR").write_text(
                "ML_MODE=train\nNSW=2000\nPOTIM=0.5\nTEBEG=300\nML_MB=6000\n",
                encoding="utf-8",
            )
            (run / "POTCAR").write_text("licensed\n", encoding="utf-8")
            (run / "KPOINTS").write_text("Automatic\n0\nGamma\n1 1 1\n", encoding="utf-8")
            (run / "CONTCAR").write_text("continued geometry\n", encoding="utf-8")
            (run / "ML_ABN").write_text("training database\n", encoding="utf-8")
            (run / "OUTCAR").write_text(
                "Not enough storage reserved for local reference configurations.\n",
                encoding="utf-8",
            )

            result = prepare_recovery(run, "discard", increase_eps_low=True)

            parsed = parse_incar(run / "INCAR")
            self.assertEqual(parsed["ML_EPS_LOW"], "1E-08")
            self.assertEqual(result["ml_eps_low"], "1E-08")

    def test_eps_low_increase_refuses_vasp_upper_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "INCAR").write_text(
                "ML_MODE=train\nNSW=2000\nML_EPS_LOW=1E-8\n",
                encoding="utf-8",
            )
            (run / "POTCAR").write_text("licensed\n", encoding="utf-8")
            (run / "KPOINTS").write_text("Automatic\n0\nGamma\n1 1 1\n", encoding="utf-8")
            (run / "CONTCAR").write_text("continued geometry\n", encoding="utf-8")
            (run / "ML_ABN").write_text("training database\n", encoding="utf-8")
            (run / "OUTCAR").write_text(
                "Not enough storage reserved for local reference configurations.\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(Exception, "strictly below 1E-7"):
                prepare_recovery(run, "discard", increase_eps_low=True)

    def test_submit_prefers_runvasp_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "run.slurm").write_text("#!/bin/bash\n", encoding="utf-8")
            (run / "runvasp.sh").write_text("#!/bin/bash\n", encoding="utf-8")

            self.assertEqual(resolve_launcher(run).name, "runvasp.sh")
            with patch("interfaceforge.vasp.subprocess.run") as mocked:
                mocked.return_value.stdout = "Submitted batch job 12345\n"
                self.assertEqual(submit_run(run), "12345")
                mocked.assert_called_once()
                self.assertEqual(mocked.call_args.args[0], ["sbatch", "runvasp.sh"])

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
            self.assertEqual(
                set(find_runs(root, include_archives=True)),
                {active, legacy_archive, internal_archive},
            )

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
