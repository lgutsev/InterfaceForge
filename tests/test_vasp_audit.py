from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from interfaceforge.audit import audit_run, find_runs, run_audit
from interfaceforge.cli import build_parser, main
from interfaceforge.errors import SafetyError
from interfaceforge.vasp import (
    apply_incar_preset,
    archive_mlff_models,
    assemble_potcar,
    ensure_run_potcar,
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

    def test_model_archive_preserves_ml_ab_and_checksum_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            accepted = root / "accepted"
            ignored = root / "ignored"
            accepted.mkdir()
            ignored.mkdir()
            (accepted / "ML_AB").write_text("accepted model\n", encoding="utf-8")
            (accepted / "INCAR").write_text("ML_MODE=train\n", encoding="utf-8")
            (accepted / "OSZICAR").write_text(" 1 T= 300 E= -1\n", encoding="utf-8")
            (accepted / "POTCAR").write_text("licensed\n", encoding="utf-8")
            (ignored / "INCAR").write_text("NSW=0\n", encoding="utf-8")
            output = root / "stored_models.zip"

            result = archive_mlff_models(root, output)

            self.assertEqual(result["runs"], 1)
            self.assertEqual(result["files"], 3)
            with zipfile.ZipFile(output) as archive:
                names = set(archive.namelist())
                self.assertIn("accepted/ML_AB", names)
                self.assertIn("accepted/INCAR", names)
                self.assertNotIn("accepted/POTCAR", names)
                self.assertNotIn("ignored/INCAR", names)
                manifest = json.loads(
                    archive.read("interfaceforge-model-archive.json")
                )
            artifacts = {
                artifact["path"]: artifact
                for artifact in manifest["runs"][0]["artifacts"]
            }
            self.assertEqual(
                artifacts["accepted/ML_AB"]["sha256"],
                hashlib.sha256(b"accepted model\n").hexdigest(),
            )
            self.assertTrue(manifest["potcar_excluded"])

    def test_model_archive_requires_nonempty_ml_ab(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "ML_AB").touch()
            with self.assertRaisesRegex(SafetyError, "No nonempty ML_AB"):
                archive_mlff_models(root, root / "models.zip")

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
            (run / "POTCAR").write_text("existing licensed data\n", encoding="utf-8")

            self.assertEqual(resolve_launcher(run).name, "runvasp.sh")
            with patch("interfaceforge.vasp.subprocess.run") as mocked:
                mocked.return_value.stdout = "Submitted batch job 12345\n"
                self.assertEqual(submit_run(run), "12345")
                mocked.assert_called_once()
                self.assertEqual(mocked.call_args.args[0], ["sbatch", "runvasp.sh"])

    def test_builtin_potcar_dictionary_matches_supplied_variants(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            poscar = root / "POSCAR"
            poscar.write_text(
                "test\n1.0\n1 0 0\n0 1 0\n0 0 1\nTc Po Am\n1 1 1\nDirect\n"
                "0 0 0\n0 0 0\n0 0 0\n",
                encoding="utf-8",
            )
            pp_root = root / "potpaw_PBE"
            expected = [("Tc_pv", b"tc\n"), ("Po_d", b"po\n"), ("Am", b"am\n")]
            for variant, content in expected:
                source = pp_root / variant / "POTCAR"
                source.parent.mkdir(parents=True)
                source.write_bytes(content)

            payload = assemble_potcar(
                poscar,
                root / "POTCAR",
                pseudopotential_root=pp_root,
            )

            self.assertEqual(payload["variants"], ["Tc_pv", "Po_d", "Am"])
            self.assertEqual((root / "POTCAR").read_bytes(), b"tc\npo\nam\n")

    def test_potcar_generator_supports_legacy_first_line_elements(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            poscar = root / "POSCAR"
            poscar.write_text(
                "Cs Pb I\n1.0\n1 0 0\n0 1 0\n0 0 1\n1 1 1\nDirect\n"
                "0 0 0\n0 0 0\n0 0 0\n",
                encoding="utf-8",
            )
            pp_root = root / "potpaw_PBE"
            for variant in ("Cs_sv", "Pb_d", "I"):
                source = pp_root / variant / "POTCAR"
                source.parent.mkdir(parents=True)
                source.write_text(variant + "\n", encoding="utf-8")

            payload = assemble_potcar(
                poscar,
                root / "POTCAR",
                pseudopotential_root=pp_root,
            )

            self.assertEqual(payload["variants"], ["Cs_sv", "Pb_d", "I"])

    def test_launcher_generates_missing_potcar_before_submission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            pp_root = Path(temporary) / "potpaw_PBE"
            run.mkdir()
            (run / "POSCAR").write_text(
                "test\n1.0\n1 0 0\n0 1 0\n0 0 1\nCs Pb I\n1 1 1\nDirect\n"
                "0 0 0\n0 0 0\n0 0 0\n",
                encoding="utf-8",
            )
            for variant in ("Cs_sv", "Pb_d", "I"):
                source = pp_root / variant / "POTCAR"
                source.parent.mkdir(parents=True)
                source.write_text(variant + "\n", encoding="utf-8")

            payload = ensure_run_potcar(run, pseudopotential_root=pp_root)

            self.assertEqual(payload["status"], "generated")
            self.assertEqual(payload["variants"], ["Cs_sv", "Pb_d", "I"])
            self.assertTrue((run / "POTCAR").stat().st_size)

    def test_submit_recover_continue_prepares_and_launches_in_one_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "INCAR").write_text(
                "ML_MODE=train\nNSW=2000\nPOTIM=0.5\nTEBEG=300\nTEEND=300\n",
                encoding="utf-8",
            )
            (run / "POTCAR").write_text("licensed\n", encoding="utf-8")
            (run / "KPOINTS").write_text("Automatic\n0\nGamma\n1 1 1\n", encoding="utf-8")
            (run / "CONTCAR").write_text("continued geometry\n", encoding="utf-8")
            (run / "ML_ABN").write_text("continued database\n", encoding="utf-8")
            (run / "OUTCAR").write_text("old output\n", encoding="utf-8")
            (run / "runvasp.sh").write_text("#!/bin/bash\n", encoding="utf-8")

            with patch("interfaceforge.vasp.subprocess.run") as mocked:
                mocked.return_value.stdout = "Submitted batch job 24680\n"
                result = main(
                    [
                        "vasp",
                        "submit",
                        str(run),
                        "--ml-continue",
                        "--temperature",
                        "450",
                        "--nsw",
                        "3000",
                    ]
                )

            self.assertEqual(result, 0)
            parsed = parse_incar(run / "INCAR")
            self.assertEqual(parsed["ML_MODE"], "train")
            self.assertEqual(parsed["TEBEG"], "450.0")
            self.assertEqual(parsed["TEEND"], "450.0")
            self.assertEqual(parsed["NSW"], "3000")
            self.assertEqual(parsed["POTIM"], "0.5")
            self.assertEqual((run / "POSCAR").read_text(), "continued geometry\n")
            self.assertEqual((run / "ML_AB").read_text(), "continued database\n")
            self.assertFalse((run / "OUTCAR").exists())
            self.assertEqual(len(list((run / ".interfaceforge/archive").glob("continue_*"))), 1)
            self.assertEqual(mocked.call_args.args[0], ["sbatch", "runvasp.sh"])

    def test_submit_recovery_flags_are_mutually_exclusive(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "vasp",
                    "submit",
                    "run",
                    "--ml-continue",
                    "--ml-capacity-recovery",
                ]
            )

    def test_ml_recovery_compatibility_aliases_remain_available(self) -> None:
        parser = build_parser()
        primary = parser.parse_args(["vasp", "ml-recover", "continue", "run"])
        legacy = parser.parse_args(["vasp", "recover", "continue", "run"])
        self.assertEqual(primary.operation, "continue")
        self.assertEqual(legacy.operation, "continue")
        submit_legacy = parser.parse_args(
            ["vasp", "submit", "run", "--recover-continue"]
        )
        self.assertTrue(submit_legacy.recover_continue)

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
                "STATUS accepted\nSTATUS accepted\nERR 2 0.001 0.02 0.3\n",
                encoding="utf-8",
            )
            (run / "OUTCAR").write_text(
                "General timing and accounting informations for this job\n",
                encoding="utf-8",
            )
            row = audit_run(run, run)
            self.assertEqual(row["ml_mode"], "train")
            self.assertEqual(row["progress_pct"], 100.0)
            self.assertEqual(row["health"], "ready to refit and test")
            self.assertEqual(row["training_rmse_records"], 1)
            self.assertEqual(row["training_energy_rmse_ev_atom_last"], 0.001)
            self.assertEqual(row["training_force_rmse_ev_a_last"], 0.02)
            self.assertEqual(row["training_stress_rmse_kbar_last"], 0.3)
            self.assertEqual(row["true_force_error_ev_a_last"], 0.02)

    def test_perovskite_profile_accepts_stationary_capacity_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "INCAR").write_text(
                "ML_MODE=train\nNSW=2000\nPOTIM=0.5\nTEBEG=300\nTEEND=300\n",
                encoding="utf-8",
            )
            (run / "OSZICAR").write_text(
                "".join(f" {step} T= 300.0 E= -1\n" for step in range(1, 251)),
                encoding="utf-8",
            )
            lines = []
            for step in range(1, 251):
                state = "learning" if step % 7 == 0 else "accurate"
                beef = 0.0195 + 0.0001 * (step % 10)
                lines.append(f"STATUS {step} {state}\n")
                lines.append(f"BEEF {step} 0.0 {beef:.6f} 0.0 0.020000\n")
            (run / "ML_LOGFILE").write_text("".join(lines), encoding="utf-8")
            (run / "OUTCAR").write_text(
                "Not enough storage reserved for local reference configurations.\n",
                encoding="utf-8",
            )

            general = audit_run(run, run)
            perovskite = audit_run(run, run, readiness_profile="perovskite")

            self.assertEqual(general["health"], "stopped: ML local-reference capacity")
            self.assertTrue(perovskite["perovskite_sampling_plateau"])
            self.assertLess(perovskite["recent_learning_event_rate_pct"], 20.0)
            self.assertEqual(perovskite["recent_critical_events"], 0)
            self.assertEqual(
                perovskite["health"], "perovskite sampling checkpoint reached"
            )
            self.assertIn("SELECT/REFIT", perovskite["next_action"])

    def test_perovskite_profile_rejects_recent_critical_events(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "INCAR").write_text(
                "ML_MODE=train\nNSW=2000\nPOTIM=0.5\n",
                encoding="utf-8",
            )
            (run / "OSZICAR").write_text(
                "".join(f" {step} T= 300.0 E= -1\n" for step in range(1, 251)),
                encoding="utf-8",
            )
            lines = []
            for step in range(1, 251):
                state = "critical" if step == 245 else "accurate"
                lines.append(f"STATUS {step} {state}\n")
                lines.append(f"BEEF {step} 0.0 0.020000 0.0 0.020000\n")
            (run / "ML_LOGFILE").write_text("".join(lines), encoding="utf-8")

            row = audit_run(run, run, readiness_profile="perovskite")

            self.assertFalse(row["perovskite_sampling_plateau"])
            self.assertEqual(row["health"], "perovskite sampling still critical")

    def test_cli_exposes_named_perovskite_readiness_profile(self) -> None:
        parser = build_parser()
        audit = parser.parse_args(
            ["audit", "runs", "--readiness-profile", "perovskite"]
        )
        status = parser.parse_args(
            ["status", "runs", "--readiness-profile", "perovskite"]
        )

        self.assertEqual(audit.readiness_profile, "perovskite")
        self.assertEqual(status.readiness_profile, "perovskite")

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
            (run / "ML_LOGFILE").write_text(
                "STATUS accepted\nERR 1 0.001 0.02 0.3\n", encoding="utf-8"
            )

            payload = run_audit(root)

            summary = Path(payload["outputs"]["summary_csv"])
            full = Path(payload["outputs"]["csv"])
            self.assertTrue(summary.is_file())
            self.assertTrue(full.is_file())
            self.assertLess(len(summary.read_text().split(",")), len(full.read_text().split(",")))
            self.assertIn("Training force RMSE (eV/A)", summary.read_text(encoding="utf-8"))
            self.assertIn("Next action", summary.read_text(encoding="utf-8"))
            markdown = Path(payload["outputs"]["markdown"]).read_text(encoding="utf-8")
            self.assertIn("Force RMSE (train)", markdown)
            self.assertIn("0.02 eV/A", markdown)


if __name__ == "__main__":
    unittest.main()
