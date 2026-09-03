from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from interfaceforge.cli import main
from interfaceforge.errors import SafetyError
from interfaceforge.step1_launch import launch_step1_runs
from interfaceforge.vasp import _sha256_file

_POSCAR = "s\n1.0\n10 0 0\n0 10 0\n0 0 20\nNi O\n1 1\nDirect\n0.0 0.0 0.0\n0.5 0.5 0.5\n"
_INCAR = "SYSTEM = Step1\nIBRION = 0\nNSW = 400\nPOTIM = 1.0\nSMASS = -1\n"


def _run_dir(root: Path, name: str) -> Path:
    run = root / name
    run.mkdir(parents=True)
    (run / "INCAR").write_text(_INCAR, encoding="utf-8")
    (run / "POSCAR").write_text(_POSCAR, encoding="utf-8")
    (run / "KPOINTS").write_text("Gamma\n0\nGamma\n1 1 1\n0 0 0\n", encoding="utf-8")
    (run / "POTCAR").write_text("fixture POTCAR\n", encoding="utf-8")
    (run / "runvasp.sh").write_text("#!/bin/sh\nsbatch payload\n", encoding="utf-8")
    return run


def _tree(root: Path) -> Path:
    step1 = root / "Step1"
    step1.mkdir()
    fresh = _run_dir(step1, "fresh_run")
    repaired = _run_dir(step1, "repaired_run")
    done = _run_dir(step1, "done_run")
    (done / "OUTCAR").write_text("General timing and accounting\n", encoding="utf-8")

    (step1 / "step1_manifest.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "relative_path": "fresh_run",
                        "step1_incar_sha256": _sha256_file(fresh / "INCAR"),
                        "step1_poscar_sha256": _sha256_file(fresh / "POSCAR"),
                    },
                    {
                        "relative_path": "done_run",
                        "step1_incar_sha256": _sha256_file(done / "INCAR"),
                        "step1_poscar_sha256": _sha256_file(done / "POSCAR"),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (repaired / "step1_repair.json").write_text(
        json.dumps({"status": "PREPARED", "safe_prefix_steps": 40}), encoding="utf-8"
    )
    return step1


class Step1LaunchTests(unittest.TestCase):
    def test_dry_run_lists_prepared_and_repaired_skips_started(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            step1 = _tree(Path(tmp))
            plan = launch_step1_runs([step1])
            self.assertEqual(plan["mode"], "dry-run")
            self.assertEqual(plan["runs"], 2)
            kinds = {row["relative_path"]: row["kind"] for row in plan["planned"]}
            self.assertEqual(kinds, {"fresh_run": "prepared", "repaired_run": "repair-prepared"})
            reasons = {row["relative_path"]: row["reason"] for row in plan["skipped_runs"]}
            self.assertIn("already started", reasons["done_run"])

    def test_only_repaired_filters_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            step1 = _tree(Path(tmp))
            plan = launch_step1_runs([step1], only_repaired=True)
            self.assertEqual([row["relative_path"] for row in plan["planned"]], ["repaired_run"])

    def test_execute_submits_and_blocks_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            step1 = _tree(Path(tmp))
            responses = []
            for job_id in (5001, 5002):
                response = Mock()
                response.stdout = f"Submitted batch job {job_id}\n"
                responses.append(response)
            with patch("interfaceforge.vasp.subprocess.run", side_effect=responses) as mocked:
                result = launch_step1_runs([step1], execute=True)
            self.assertEqual(result["mode"], "submitted")
            self.assertEqual(result["submitted"], 2)
            self.assertEqual(mocked.call_count, 2)
            record = json.loads((step1 / "step1_launch.json").read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "SUBMITTED")
            self.assertEqual(sorted(r["job_id"] for r in record["runs"]), ["5001", "5002"])
            # a second launch sees both as already submitted -> nothing left
            with self.assertRaises(SafetyError):
                launch_step1_runs([step1])

    def test_hash_mismatch_after_prepare_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            step1 = _tree(Path(tmp))
            (step1 / "fresh_run" / "INCAR").write_text(_INCAR + "NELM = 200\n", encoding="utf-8")
            with self.assertRaisesRegex(SafetyError, "changed since step1-prepare"):
                launch_step1_runs([step1])

    def test_cli_is_dry_run_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            step1 = _tree(Path(tmp))
            self.assertEqual(main(["vasp", "step1-launch", str(step1)]), 0)
            self.assertFalse((step1 / "step1_launch.json").exists())


if __name__ == "__main__":
    unittest.main()
