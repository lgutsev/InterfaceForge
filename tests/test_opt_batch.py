from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from interfaceforge.cli import build_parser
from interfaceforge.errors import SafetyError
from interfaceforge.vasp import launch_opt_runs, prepare_opt_tree


def _poscar() -> str:
    return (
        "NiO hydroxylation fixture\n"
        "1.0\n"
        "12 0 0\n"
        "0 12 0\n"
        "0 0 30\n"
        "H Ni O\n"
        "1 2 1\n"
        "Selective dynamics\n"
        "Direct\n"
        "0.5 0.5 0.7 T T T\n"
        "0.0 0.0 0.3 F F F\n"
        "0.5 0.5 0.5 T T T\n"
        "0.1 0.1 0.3 F F F\n"
    )


def _incar() -> str:
    return (
        "SYSTEM = NiO OPT\n"
        "IBRION = 2\n"
        "ISIF = 2\n"
        "NSW = 500\n"
        "ISPIN = 2\n"
        "MAGMOM = 0 2 -2 0\n"
        "LDAU = .TRUE.\n"
        "LDAUL = -1 2 -1\n"
        "LDAUU = 0 4.6 0\n"
        "LDAUJ = 0 0 0\n"
        "LDIPOL = .TRUE.\n"
        "IDIPOL = 3\n"
        "DIPOL = 0.5 0.5 0.5\n"
    )


class OptBatchTests(unittest.TestCase):
    def _fixture(self, temporary: str) -> tuple[Path, Path, Path]:
        base = Path(temporary)
        root = base / "generated"
        runs = [
            root / "OH25/NiO_OH25_scattered_dissoc",
            root / "OH50/NiO_OH50_scattered_dissoc_Me4PACz_boundary",
        ]
        for run in runs:
            run.mkdir(parents=True)
            (run / "POSCAR").write_text(_poscar(), encoding="utf-8")
            (run / "INCAR").write_text(_incar(), encoding="utf-8")
            (run / "KPOINTS").write_text("Gamma\n0\nG\n1 1 1\n", encoding="utf-8")
        existing = "existing licensed POTCAR\n"
        (runs[0] / "POTCAR").write_text(existing, encoding="utf-8")
        manifest = root / "manifest_batch.csv"
        manifest.write_text(
            "case,path,ligand\n"
            f"case25,{runs[0].relative_to(root).as_posix()},\n"
            f"case50,{runs[1].relative_to(root).as_posix()},Me4PACz\n",
            encoding="utf-8",
        )
        launcher = base / "runvasp.sh"
        launcher.write_text(
            "#!/usr/bin/env bash\n"
            "module load vasp6/6.5.1-cpu\n"
            "srun -n128 vasp_gam\n",
            encoding="utf-8",
        )
        launcher.chmod(0o755)
        generator = base / "POTCAR_gen"
        generator.write_text(
            "#!/usr/bin/env bash\n"
            "printf ' VRHFIN =H: s1\\n VRHFIN =Ni: d8\\n VRHFIN =O: s2p4\\n' > POTCAR\n",
            encoding="utf-8",
        )
        generator.chmod(0o755)
        return root, manifest, launcher

    def test_prepare_runs_local_generator_only_for_missing_potcar_and_audits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, manifest, launcher = self._fixture(temporary)
            generator = Path(temporary) / "POTCAR_gen"
            existing = root / "OH25/NiO_OH25_scattered_dissoc/POTCAR"

            result = prepare_opt_tree(
                root,
                manifest=manifest,
                launcher_template=launcher,
                potcar_command=str(generator),
            )

            self.assertEqual(result["mode"], "prepared-and-audited")
            self.assertEqual(result["audit"]["status"], "PASS")
            self.assertEqual(existing.read_text(encoding="utf-8"), "existing licensed POTCAR\n")
            generated = root / "OH50/NiO_OH50_scattered_dissoc_Me4PACz_boundary/POTCAR"
            self.assertIn("VRHFIN =Ni", generated.read_text(encoding="utf-8"))
            audit = json.loads((root / "opt_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(len(audit["runs"]), 2)
            self.assertTrue(all(row["status"] == "PASS" for row in audit["runs"]))
            self.assertTrue((root / "opt_audit.tsv").is_file())
            self.assertTrue((root / "opt_audit.md").is_file())

    def test_dry_run_does_not_generate_potcar_or_copy_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, manifest, launcher = self._fixture(temporary)
            missing_run = root / "OH50/NiO_OH50_scattered_dissoc_Me4PACz_boundary"

            result = prepare_opt_tree(
                root,
                manifest=manifest,
                launcher_template=launcher,
                potcar_command=str(Path(temporary) / "POTCAR_gen"),
                dry_run=True,
            )

            self.assertEqual(result["mode"], "dry-run")
            self.assertFalse((missing_run / "POTCAR").exists())
            self.assertFalse((missing_run / "runvasp.sh").exists())
            self.assertFalse((root / "opt_audit.json").exists())

    def test_audit_rejects_wrong_magmom_length(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, manifest, launcher = self._fixture(temporary)
            generator = Path(temporary) / "POTCAR_gen"
            bad = root / "OH25/NiO_OH25_scattered_dissoc/INCAR"
            bad.write_text(_incar().replace("MAGMOM = 0 2 -2 0", "MAGMOM = 2 -2"), encoding="utf-8")

            with self.assertRaisesRegex(SafetyError, "OPT audit FAILED"):
                prepare_opt_tree(
                    root,
                    manifest=manifest,
                    launcher_template=launcher,
                    potcar_command=str(generator),
                )

            audit = json.loads((root / "opt_audit.json").read_text(encoding="utf-8"))
            failed = next(row for row in audit["runs"] if row["status"] == "FAIL")
            self.assertTrue(any("MAGMOM has 2 values" in issue for issue in failed["issues"]))

    def test_manifest_cannot_escape_selected_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, manifest, _launcher = self._fixture(temporary)
            manifest.write_text("path\n../outside\n", encoding="utf-8")
            with self.assertRaisesRegex(SafetyError, "escapes"):
                prepare_opt_tree(root, manifest=manifest, dry_run=True)

    def test_oh0_prefix_is_excluded_and_remembered_by_audit_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, manifest, launcher = self._fixture(temporary)
            source = root / "OH25/NiO_OH25_scattered_dissoc"
            oh0 = root / "OH0/NiO_pristine_Me4PACz"
            shutil.copytree(source, oh0)
            with manifest.open("a", encoding="utf-8") as handle:
                handle.write("pristine,OH0/NiO_pristine_Me4PACz,Me4PACz\n")

            prepared = prepare_opt_tree(
                root,
                manifest=manifest,
                launcher_template=launcher,
                potcar_command=str(Path(temporary) / "POTCAR_gen"),
                exclude_prefixes=["OH0"],
            )
            self.assertEqual(prepared["runs"], 2)
            self.assertEqual(prepared["excluded_prefixes"], ["OH0"])
            internal = json.loads((root / "opt_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(internal["excluded_prefixes"], ["OH0"])
            self.assertFalse(any(row["relative_path"].startswith("OH0/") for row in internal["runs"]))

            audited = prepare_opt_tree(root, audit_only=True)
            self.assertEqual(audited["mode"], "audited")
            self.assertEqual(audited["excluded_prefixes"], ["OH0"])
            self.assertEqual(audited["runs"], 2)

    def test_launch_is_dry_by_default_then_records_jobs_and_blocks_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, manifest, launcher = self._fixture(temporary)
            prepare_opt_tree(
                root,
                manifest=manifest,
                launcher_template=launcher,
                potcar_command=str(Path(temporary) / "POTCAR_gen"),
            )

            preview = launch_opt_runs([root])
            self.assertEqual(preview["mode"], "dry-run")
            self.assertEqual(preview["runs"], 2)

            responses = []
            for job_id in (8123, 8124):
                response = Mock()
                response.stdout = f"Submitted batch job {job_id}\n"
                responses.append(response)
            with patch("interfaceforge.vasp.subprocess.run", side_effect=responses) as mocked:
                submitted = launch_opt_runs([root], execute=True)

            self.assertEqual(submitted["submitted"], 2)
            self.assertEqual(mocked.call_count, 2)
            record = json.loads((root / "opt_launch.json").read_text(encoding="utf-8"))
            self.assertEqual([row["job_id"] for row in record["runs"]], ["8123", "8124"])
            with self.assertRaisesRegex(SafetyError, "duplicate launch"):
                launch_opt_runs([root])

    def test_launch_rechecks_audited_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, manifest, launcher = self._fixture(temporary)
            prepare_opt_tree(
                root,
                manifest=manifest,
                launcher_template=launcher,
                potcar_command=str(Path(temporary) / "POTCAR_gen"),
            )
            changed = root / "OH25/NiO_OH25_scattered_dissoc/INCAR"
            changed.write_text(changed.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")

            with self.assertRaisesRegex(SafetyError, "changed after the PASS audit"):
                launch_opt_runs([root])

    def test_launch_rejects_an_unaudited_launcher_override(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, manifest, launcher = self._fixture(temporary)
            prepare_opt_tree(
                root,
                manifest=manifest,
                launcher_template=launcher,
                potcar_command=str(Path(temporary) / "POTCAR_gen"),
            )
            for run in (
                root / "OH25/NiO_OH25_scattered_dissoc",
                root / "OH50/NiO_OH50_scattered_dissoc_Me4PACz_boundary",
            ):
                alternate = run / "alternate.sh"
                alternate.write_text("#!/usr/bin/env bash\nsrun vasp_gam\n", encoding="utf-8")
                alternate.chmod(0o755)

            with self.assertRaisesRegex(SafetyError, "was not part of the PASS audit"):
                launch_opt_runs([root], launcher="alternate.sh")

    def test_cli_exposes_safe_defaults(self) -> None:
        parser = build_parser()
        prepare = parser.parse_args(
            [
                "vasp",
                "opt-prepare",
                "generated",
                "--manifest",
                "generated/manifest_batch.csv",
                "--exclude-prefix",
                "OH0",
            ]
        )
        launch = parser.parse_args(["vasp", "opt-launch", "generated"])
        self.assertEqual(prepare.potcar_command, "POTCAR_gen")
        self.assertEqual(prepare.require_module, "vasp6/6.5.1-cpu")
        self.assertEqual(prepare.exclude_prefix, ["OH0"])
        self.assertFalse(launch.execute)


if __name__ == "__main__":
    unittest.main()
