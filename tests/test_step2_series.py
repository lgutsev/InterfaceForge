from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from interfaceforge.cli import build_parser, main
from interfaceforge.errors import SafetyError
from interfaceforge.vasp import launch_step2_runs, parse_incar, prepare_step2_series


def _poscar(elements: list[str]) -> str:
    counts = " ".join("1" for _ in elements)
    coordinates = "\n".join("0 0 0" for _ in elements)
    return (
        "step1 final geometry\n"
        "1.0\n"
        "1 0 0\n"
        "0 1 0\n"
        "0 0 1\n"
        f"{' '.join(elements)}\n"
        f"{counts}\n"
        "Direct\n"
        f"{coordinates}\n"
    )


class Step2SeriesTests(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        step1 = root / "Step1"
        step1.mkdir()
        (step1 / "KPOINTS").write_text(
            "Gamma\n0\nGamma\n1 1 1\n0 0 0\n", encoding="utf-8"
        )
        launcher = step1 / "runvasp.sh"
        launcher.write_text("#!/usr/bin/env bash\nsbatch payload\n", encoding="utf-8")
        launcher.chmod(0o755)

        cases = (
            (
                "Real/N_Term/x0.25",
                ["C", "H", "N", "Ni", "O", "P"],
                "-1 -1 -1 2 -1 -1",
                "0.0 0.0 0.0 4.6 0.0 0.0",
            ),
            (
                "Ideal/Ti_Term/x0.50",
                ["Si", "N", "Ti", "O"],
                "-1 -1 2 -1",
                "0.0 0.0 3.2 0.0",
            ),
        )
        for relative, elements, ldaul, ldauu in cases:
            run = step1 / relative
            run.mkdir(parents=True)
            (run / "INCAR").write_text(
                "ENCUT = 400\n"
                "LWAVE = .TRUE.\n"
                "TEBEG = 300\n"
                "LDAU      = .TRUE.\n"
                "LDAUTYPE  = 2\n"
                f"LDAUL     = {ldaul}\n"
                f"LDAUU     = {ldauu}\n"
                f"LDAUJ     = {' '.join('0.0' for _ in elements)}\n"
                "LDAUPRINT = 0\n"
                "LMAXMIX   = 4\n",
                encoding="utf-8",
            )
            (run / "CONTCAR").write_text(_poscar(elements), encoding="utf-8")
            (run / "POTCAR").write_text(
                f"licensed fixture for {' '.join(elements)}\n", encoding="utf-8"
            )
            (run / "OUTCAR").write_text("must not be inherited\n", encoding="utf-8")
        return step1

    def test_dry_run_validates_without_creating_temperature_trees(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            step1 = self._fixture(root)

            result = prepare_step2_series(step1, dry_run=True)

            self.assertEqual(result["mode"], "dry-run")
            self.assertEqual(result["source_runs"], 2)
            self.assertEqual(result["prepared_runs"], 6)
            self.assertFalse((root / "Step2_300K").exists())
            self.assertFalse((root / "Step2_450K").exists())
            self.assertFalse((root / "Step2_600K").exists())

    def test_temperature_series_uses_template_and_preserves_each_hubbard_array(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            step1 = self._fixture(root)

            result = prepare_step2_series(step1, temperatures=[300, 450, 600])

            self.assertEqual(result["mode"], "prepared-and-audited")
            self.assertEqual(result["audit"]["status"], "PASS")
            expected_u = {
                "Real/N_Term/x0.25": "0.0 0.0 0.0 4.6 0.0 0.0",
                "Ideal/Ti_Term/x0.50": "0.0 0.0 3.2 0.0",
            }
            for temperature in (300, 450, 600):
                output = root / f"Step2_{temperature}K"
                manifest = json.loads(
                    (output / "step2_manifest.json").read_text(encoding="utf-8")
                )
                audit = json.loads(
                    (output / "step2_audit.json").read_text(encoding="utf-8")
                )
                self.assertEqual(len(manifest["runs"]), 2)
                self.assertEqual(audit["status"], "PASS")
                self.assertTrue((output / "step2_audit.tsv").is_file())
                self.assertTrue((output / "step2_audit.md").is_file())
                for relative, ldauu in expected_u.items():
                    source = step1 / relative
                    run = output / relative
                    incar = parse_incar(run / "INCAR")
                    self.assertEqual(incar["SYSTEM"], f"Step2_DFT_MD_{temperature}K")
                    self.assertEqual(incar["TEBEG"], str(temperature))
                    self.assertEqual(incar["TEEND"], str(temperature))
                    self.assertEqual(incar["ENCUT"], "520")
                    self.assertEqual(incar["LWAVE"], ".FALSE.")
                    self.assertEqual(incar["LDAUU"], ldauu)
                    self.assertEqual(incar["LMAXMIX"], "4")
                    self.assertEqual(
                        (run / "POSCAR").read_text(encoding="utf-8"),
                        (source / "CONTCAR").read_text(encoding="utf-8"),
                    )
                    self.assertEqual(
                        (run / "POTCAR").read_text(encoding="utf-8"),
                        (source / "POTCAR").read_text(encoding="utf-8"),
                    )
                    self.assertTrue((run / "runvasp.sh").stat().st_mode & 0o100)
                    self.assertFalse((run / "OUTCAR").exists())

    def test_custom_template_cannot_override_source_hubbard_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            step1 = self._fixture(root)
            template = root / "INCAR_FINAL"
            template.write_text(
                "ENCUT = 610\nTEBEG = 1\nTEEND = 1\n"
                "LDAUU = 99 99\nLMAXMIX = 6\n",
                encoding="utf-8",
            )

            prepare_step2_series(step1, temperatures=[450], template=template)

            parsed = parse_incar(root / "Step2_450K/Real/N_Term/x0.25/INCAR")
            self.assertEqual(parsed["ENCUT"], "610")
            self.assertEqual(parsed["TEBEG"], "450")
            self.assertEqual(parsed["LDAUU"], "0.0 0.0 0.0 4.6 0.0 0.0")
            self.assertEqual(parsed["LMAXMIX"], "4")

    def test_mismatched_hubbard_array_fails_before_writing_anything(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            step1 = self._fixture(root)
            bad = step1 / "Real/N_Term/x0.25/INCAR"
            bad.write_text(
                bad.read_text(encoding="utf-8").replace(
                    "LDAUU     = 0.0 0.0 0.0 4.6 0.0 0.0",
                    "LDAUU     = 0.0 4.6",
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(SafetyError, "LDAUU has 2 entries"):
                prepare_step2_series(step1)

            self.assertFalse((root / "Step2_300K").exists())

    def test_existing_destination_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            step1 = self._fixture(root)
            existing = root / "Step2_300K"
            existing.mkdir()
            marker = existing / "user-data"
            marker.write_text("keep\n", encoding="utf-8")

            with self.assertRaisesRegex(SafetyError, "Refusing to overwrite"):
                prepare_step2_series(step1)

            self.assertEqual(marker.read_text(encoding="utf-8"), "keep\n")
            self.assertFalse((root / "Step2_450K").exists())

    def test_cli_exposes_default_temperature_series(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["vasp", "step2-series", "Step1", "--dry-run"])
        self.assertEqual(args.temperatures, [300.0, 450.0, 600.0])
        self.assertTrue(args.dry_run)

    def test_source_root_can_itself_be_one_calculation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tree = self._fixture(root)
            source = tree / "Real/N_Term/x0.25"
            (source / "KPOINTS").write_bytes((tree / "KPOINTS").read_bytes())
            (source / "runvasp.sh").write_bytes((tree / "runvasp.sh").read_bytes())
            (source / "runvasp.sh").chmod(0o755)

            result = prepare_step2_series(
                source,
                temperatures=[300],
                output_root=root / "single",
            )

            destination = root / "single/Step2_300K"
            self.assertEqual(result["source_runs"], 1)
            self.assertTrue((destination / "INCAR").is_file())
            self.assertTrue((destination / "POSCAR").is_file())
            self.assertTrue((destination / "step2_manifest.json").is_file())

    def test_audit_only_detects_a_post_preparation_incar_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            step1 = self._fixture(root)
            prepare_step2_series(step1, temperatures=[300])
            changed = root / "Step2_300K/Real/N_Term/x0.25/INCAR"
            changed.write_text(
                changed.read_text(encoding="utf-8").replace("ENCUT  = 520", "ENCUT  = 400"),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(SafetyError, "audit FAILED"):
                prepare_step2_series(step1, temperatures=[300], audit_only=True)

            audit = json.loads(
                (root / "Step2_300K/step2_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(audit["status"], "FAIL")
            failed = [row for row in audit["runs"] if row["status"] == "FAIL"]
            self.assertTrue(any("INCAR differs" in issue for issue in failed[0]["issues"]))

    def test_launch_is_dry_run_by_default_and_execute_submits_all_audited_leaves(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            step1 = self._fixture(root)
            prepare_step2_series(step1, temperatures=[300])
            step2 = root / "Step2_300K"

            preview = launch_step2_runs([step2])

            self.assertEqual(preview["mode"], "dry-run")
            self.assertEqual(preview["preflight"], "PASS")
            self.assertEqual(preview["runs"], 2)
            responses = []
            for job_id in (12345, 12346):
                response = Mock()
                response.stdout = f"Submitted batch job {job_id}\n"
                responses.append(response)
            with patch("interfaceforge.vasp.subprocess.run", side_effect=responses) as mocked:
                result = launch_step2_runs([step2], execute=True)

            self.assertEqual(result["mode"], "submitted")
            self.assertEqual(result["submitted"], 2)
            self.assertEqual(mocked.call_count, 2)
            launch_record = json.loads(
                (step2 / "step2_launch.json").read_text(encoding="utf-8")
            )
            self.assertEqual(launch_record["status"], "SUBMITTED")
            self.assertEqual(
                [row["job_id"] for row in launch_record["runs"]], ["12345", "12346"]
            )
            with self.assertRaisesRegex(SafetyError, "duplicate launch"):
                launch_step2_runs([step2])

    def test_launch_rechecks_hashes_after_pass_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            step1 = self._fixture(root)
            prepare_step2_series(step1, temperatures=[450])
            step2 = root / "Step2_450K"
            incar = step2 / "Ideal/Ti_Term/x0.50/INCAR"
            incar.write_text(incar.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")

            with self.assertRaisesRegex(SafetyError, "changed after preparation"):
                launch_step2_runs([step2])

    def test_cli_can_prepare_a_single_temperature(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            step1 = self._fixture(root)

            result = main(
                ["vasp", "step2-series", str(step1), "--temperatures", "450"]
            )

            self.assertEqual(result, 0)
            self.assertTrue((root / "Step2_450K/step2_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
