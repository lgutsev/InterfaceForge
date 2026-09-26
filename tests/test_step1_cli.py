"""CLI tests for the Step1 recovery commands (``iface vasp step1-status/-repair/-resume/-launch/-recover``).

Exercised through both entry points: ``interfaceforge.cli.main`` and the
installed ``iface`` entry point ``interfaceforge.profile_cli.main`` (which
builds on ``cli.build_parser``).  Synthetic fixtures only; ``sbatch`` is
patched and ``squeue`` is never found (``--scheduler auto``) or disabled
(``--scheduler none``).
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any
from unittest.mock import patch

from interfaceforge import cli, profile_cli
from interfaceforge.step1_lineage import LAUNCH_LEDGER, RECOVER_JOURNAL, REPAIR_RECORD, RESUME_RECORD, read_json
from interfaceforge.step1_resume import prepare_step1_resume

_TESTS = str(Path(__file__).resolve().parent)
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)
from step1_fixtures import (  # noqa: E402
    fake_guard,
    mock_sbatch,
    tree_snapshot,
    write_manifest,
    write_step1_run,
)

REFERENCE_F = -10.0
ENTRY_POINTS: dict[str, Callable[[list[str]], int]] = {"cli": cli.main, "profile_cli": profile_cli.main}


def _runaway(first_bad: int) -> Callable[[int], float]:
    return lambda step: REFERENCE_F + (110.0 if step >= first_bad else 0.02 * (step % 3))


def _idle_run(root: Path, name: str, **kwargs: Any) -> Path:
    return write_step1_run(root, name, steps=0, contcar=None, xdatcar=False, outcar=None, age_hours=48.0, **kwargs)


def _run(main: Callable[[list[str]], int], argv: list[str]) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


class Step1CliTestCase(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        patcher = patch("interfaceforge.step1_scheduler.shutil.which", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()
        self.step1 = self.root / "Step1"

    def small_tree(self) -> dict[str, Path]:
        runs = {
            "done": write_step1_run(self.step1, "done", steps=400, outcar="finished"),
            "resume": write_step1_run(self.step1, "resume", steps=33),
            "repair": write_step1_run(self.step1, "repair", steps=30, energies=_runaway(25)),
            "launch": _idle_run(self.step1, "launch"),
        }
        write_manifest(self.step1, [runs["launch"]])
        return runs


class Step1RecoverCliTests(Step1CliTestCase):
    def test_recover_human_plan_and_json_through_both_entry_points(self) -> None:
        self.small_tree()
        before = tree_snapshot(self.root)
        for name, main in ENTRY_POINTS.items():
            with self.subTest(entry_point=name):
                code, out, err = _run(main, ["vasp", "step1-recover", str(self.step1), "--scheduler", "none"])
                self.assertEqual(code, 0, err)
                self.assertTrue(out.startswith(f"Step1 recovery plan: {self.step1}"))
                self.assertIn("scheduler: NOT verified - scheduler check disabled (--scheduler none)", out)
                self.assertIn("activity window: 6 h", out)
                for header in ("active (0)", "done (1)", "review (0)", "launch (1)", "resume (1)", "repair (1)"):
                    self.assertIn(header, out)
                self.assertIn("repair g1: rewind to segment step 16 (cumulative 16/400), NSW=384 @ 0.5 fs", out)
                self.assertIn(
                    "Dry run: nothing changed. Execute resume+repair+launch with: iface vasp step1-recover", out
                )

                code, out, err = _run(main, ["vasp", "step1-recover", str(self.step1), "--scheduler", "none", "--json"])
                self.assertEqual(code, 0, err)
                payload = json.loads(out)
                self.assertEqual(payload["format"], "interfaceforge-step1-recover-plan")
                self.assertEqual(payload["mode"], "dry-run")
                self.assertEqual(
                    payload["counts"], {"done": 1, "resume": 1, "repair": 1, "launch": 1, "review": 0, "active": 0}
                )
                self.assertEqual(payload["selected_categories"], ["resume", "repair", "launch"])
                self.assertTrue(payload["submit"])
        self.assertEqual(tree_snapshot(self.root), before)
        self.assertFalse((self.step1 / RECOVER_JOURNAL).exists())

    def test_recover_options_reach_the_plan(self) -> None:
        self.small_tree()
        argv = [
            "vasp", "step1-recover", str(self.step1), "--scheduler", "none", "--json", "--only", "repair",
            "--no-submit", "--potim", "0.25", "--algo", "All", "--no-precondition", "--no-ramp", "--langevin",
            "--langevin-gamma", "5", "--safety-steps", "4", "--contcar-tolerance", "2", "--fresh-start",
            "--energy-jump", "60", "--startup-grace-steps", "5", "--catastrophic-energy", "800", "--stale-hours", "3",
        ]  # fmt: skip
        code, out, err = _run(cli.main, argv)
        self.assertEqual(code, 0, err)
        payload = json.loads(out)
        self.assertEqual((payload["selected_categories"], payload["submit"]), (["repair"], False))
        repair = payload["settings"]["repair"]
        self.assertEqual(
            (repair["potim_fs"], repair["algo"], repair["precondition"], repair["ramp_from"]),
            (0.25, "All", False, None),
        )
        self.assertEqual((repair["langevin_gamma"], repair["safety_steps"]), (5.0, 4))
        self.assertEqual(payload["settings"]["resume"]["contcar_tolerance_angstrom"], 2.0)
        self.assertTrue(payload["settings"]["resume"]["fresh_start"])
        diagnostic = payload["settings"]["diagnostic"]
        self.assertEqual(
            (diagnostic["energy_jump_ev"], diagnostic["startup_grace_steps"], diagnostic["catastrophic_energy_ev"]),
            (60.0, 5, 800.0),
        )
        self.assertEqual(payload["settings"]["stale_hours"], 3.0)
        # --safety-steps 4: rewind to 25 - 1 - 4 = 20 -> frame 5.
        self.assertEqual(payload["categories"]["repair"][0]["action"]["safe_segment_steps"], 20)

        default = json.loads(
            _run(cli.main, ["vasp", "step1-recover", str(self.step1), "--scheduler", "none", "--json"])[1]
        )
        self.assertEqual(default["settings"]["repair"]["ramp_from"], 100.0)
        self.assertTrue(default["settings"]["repair"]["precondition"])

    def test_recover_rejects_contradictory_options(self) -> None:
        self.small_tree()
        code, _out, err = _run(
            cli.main,
            ["vasp", "step1-recover", str(self.step1), "--scheduler", "none", "--no-ramp", "--ramp-from", "50"],
        )
        self.assertEqual(code, 2)
        self.assertIn("--no-ramp and --ramp-from contradict", err)
        code, _out, err = _run(
            cli.main,
            ["vasp", "step1-recover", str(self.step1), "--scheduler", "none", "--resume-precondition", "--fresh-start"],
        )
        self.assertEqual(code, 2)
        self.assertIn("mutually exclusive", err)
        self.assertFalse((self.step1 / RECOVER_JOURNAL).exists())

    def test_recover_execute_prints_changed_and_submitted_runs_on_stderr(self) -> None:
        runs = self.small_tree()
        done_before = tree_snapshot(runs["done"])
        with mock_sbatch(5101) as calls:
            code, out, err = _run(
                profile_cli.main, ["vasp", "step1-recover", str(self.step1), "--execute", "--scheduler", "none"]
            )
        self.assertEqual(code, 0, err)
        self.assertEqual(
            [cwd for _command, cwd in calls], [str(runs["repair"]), str(runs["resume"]), str(runs["launch"])]
        )
        self.assertIn("CHANGED repair: prepared g1-repair-", err)
        self.assertIn("CHANGED resume: prepared g1-resume-", err)
        self.assertIn("SUBMITTED repair: job 5101", err)
        self.assertIn("SUBMITTED launch: job 5103", err)
        self.assertIn(
            "step1-recover: COMPLETED; changed 2 run(s): repair, resume; submitted 3 job(s): repair (job 5101), "
            "resume (job 5102), launch (job 5103)",
            err,
        )
        self.assertIn(": COMPLETED (selected resume, repair, launch; submit on)", out)
        self.assertIn(f"Journal: {self.step1 / RECOVER_JOURNAL}", out)
        self.assertEqual(read_json(self.step1 / RECOVER_JOURNAL)["executions"][0]["status"], "COMPLETED")
        self.assertEqual(tree_snapshot(runs["done"]), done_before)

    def test_recover_execute_json_and_failure_exit_code(self) -> None:
        self.small_tree()
        (self.step1 / RECOVER_JOURNAL).write_text("{broken", encoding="utf-8")
        with mock_sbatch() as calls:
            code, out, err = _run(
                cli.main, ["vasp", "step1-recover", str(self.step1), "--execute", "--scheduler", "none", "--json"]
            )
        self.assertEqual((code, calls, out), (2, [], ""))
        self.assertIn("ERROR: ", err)
        self.assertIn("not a readable step1-recover journal", err)

        (self.step1 / RECOVER_JOURNAL).unlink()
        with mock_sbatch():
            code, out, err = _run(
                cli.main,
                ["vasp", "step1-recover", str(self.step1), "--execute", "--scheduler", "none", "--json", "--no-submit"],
            )
        self.assertEqual(code, 0, err)
        payload = json.loads(out)
        self.assertEqual((payload["status"], payload["submitted"]), ("COMPLETED", []))
        self.assertEqual([item["relative_path"] for item in payload["changed"]], ["repair", "resume"])


class Step1ResumeRepairCliTests(Step1CliTestCase):
    def test_resume_dry_run_json(self) -> None:
        runs = self.small_tree()
        before = tree_snapshot(self.root)
        code, out, err = _run(cli.main, ["vasp", "step1-resume", str(self.step1), "--scheduler", "none"])
        self.assertEqual(code, 0, err)
        payload = json.loads(out)
        self.assertEqual(
            (payload["format"], payload["mode"], payload["resumable"]),
            ("interfaceforge-step1-resume-plan", "dry-run", 1),
        )
        self.assertEqual(payload["runs"][0]["run"], str(runs["resume"]))
        self.assertEqual(payload["settings"]["stale_hours"], 6.0)  # unverified scheduler
        self.assertFalse(payload["scheduler"]["verified"])
        self.assertEqual(tree_snapshot(self.root), before)

    def test_resume_and_repair_submit_need_execute(self) -> None:
        self.small_tree()
        before = tree_snapshot(self.root)
        for command in ("step1-repair", "step1-resume"):
            with self.subTest(command=command), mock_sbatch() as calls:
                code, out, err = _run(cli.main, ["vasp", command, str(self.step1), "--submit", "--scheduler", "none"])
                self.assertEqual(code, 2)
                self.assertEqual((out, calls), ("", []))
                self.assertIn(f"{command} --submit needs --execute", err)
        self.assertEqual(tree_snapshot(self.root), before)

    def test_resume_execute_submit_prepares_and_submits_exactly_the_prepared_runs(self) -> None:
        runs = self.small_tree()
        with mock_sbatch(5201) as calls:
            code, out, err = _run(
                cli.main, ["vasp", "step1-resume", str(self.step1), "--execute", "--submit", "--scheduler", "none"]
            )
        self.assertEqual(code, 0, err)
        self.assertEqual([cwd for _command, cwd in calls], [str(runs["resume"])])
        payload = json.loads(out)
        self.assertEqual(payload["mode"], "prepared")
        self.assertEqual(payload["launch"]["submitted"], 1)
        self.assertIn("step1-resume: CHANGED 1 run(s): resume (g1-resume-", err)
        self.assertIn("step1-resume: submitted 1 job(s): resume (job 5201)", err)
        self.assertEqual(read_json(runs["resume"] / RESUME_RECORD)["status"], "SUBMITTED")

    def test_repair_execute_submit_with_new_options(self) -> None:
        runs = self.small_tree()
        argv = [
            "vasp", "step1-repair", str(self.step1), "--execute", "--submit", "--scheduler", "none",
            "--precondition", "--ramp-from", "100", "--startup-grace-steps", "10", "--catastrophic-energy", "500",
        ]  # fmt: skip
        with mock_sbatch(5301) as calls:
            code, out, err = _run(cli.main, argv)
        self.assertEqual(code, 0, err)
        self.assertEqual([cwd for _command, cwd in calls], [str(runs["repair"])])
        payload = json.loads(out)
        self.assertEqual((payload["mode"], payload["repairable"]), ("prepared", 1))
        self.assertEqual(
            (payload["settings"]["startup_grace_steps"], payload["settings"]["catastrophic_energy_ev"]), (10, 500.0)
        )
        self.assertIn("step1-repair: CHANGED 1 run(s): repair (g1-repair-", err)
        self.assertIn("step1-repair: submitted 1 job(s): repair (job 5301)", err)
        self.assertEqual(read_json(runs["repair"] / REPAIR_RECORD)["status"], "SUBMITTED")
        self.assertEqual([row["job_id"] for row in read_json(self.step1 / LAUNCH_LEDGER)["runs"]], ["5301"])

    def test_repair_execute_submit_honours_launcher(self) -> None:
        runs = self.small_tree()
        repair = runs["repair"]
        (repair / "job.sh").write_bytes((repair / "runvasp.sh").read_bytes())
        (repair / "runvasp.sh").unlink()  # only a non-default launcher
        argv = ["vasp", "step1-repair", str(self.step1), "--execute", "--submit", "--scheduler", "none"]
        with mock_sbatch(5401) as calls:
            code, out, err = _run(cli.main, [*argv, "--launcher", "job.sh"])
        self.assertEqual(code, 0, err)
        self.assertEqual([(command[-1], cwd) for command, cwd in calls], [("job.sh", str(repair))])
        self.assertEqual(read_json(repair / REPAIR_RECORD)["status"], "SUBMITTED")

    def test_launcher_that_bypasses_precondition_is_refused_before_anything_changes(self) -> None:
        self.small_tree()
        before = tree_snapshot(self.root)
        for command in ("step1-repair", "step1-resume"):
            argv = ["vasp", command, str(self.step1), "--execute", "--submit", "--scheduler", "none"]
            with self.subTest(command=command), mock_sbatch() as calls:
                code, out, err = _run(cli.main, [*argv, "--precondition", "--launcher", "job.sh"])
                self.assertEqual((code, out, calls), (2, "", []))
                self.assertIn("would bypass the preconditioning", err)
        self.assertEqual(tree_snapshot(self.root), before)

    def test_repair_stale_hours_default_resolves_from_the_scheduler(self) -> None:
        self.small_tree()
        code, out, err = _run(cli.main, ["vasp", "step1-repair", str(self.step1), "--scheduler", "none"])
        self.assertEqual(code, 0, err)
        settings = json.loads(out)["settings"]
        self.assertEqual((settings["stale_hours"], settings["stale_hours_requested"]), (6.0, None))
        with patch("interfaceforge.cli.SchedulerGuard", side_effect=lambda mode: fake_guard()):
            code, out, err = _run(cli.main, ["vasp", "step1-repair", str(self.step1), "--scheduler", "slurm"])
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["settings"]["stale_hours"], 0.1)  # Slurm verified: settle window
        code, out, err = _run(cli.main, ["vasp", "step1-repair", str(self.step1), "--stale-hours", "2"])
        self.assertEqual(json.loads(out)["settings"]["stale_hours"], 2.0)


class Step1StatusLaunchCliTests(Step1CliTestCase):
    def test_status_scheduler_none_json_and_human(self) -> None:
        self.small_tree()
        code, out, err = _run(cli.main, ["vasp", "step1-status", str(self.step1), "--scheduler", "none", "--json"])
        self.assertEqual(code, 0, err)
        payload = json.loads(out)
        self.assertEqual((payload["scheduler"]["requested"], payload["scheduler"]["verified"]), ("none", False))
        self.assertEqual(payload["stale_hours"], 6.0)
        self.assertEqual(payload["action_tally"], {"done": 1, "resume": 1, "repair": 1, "launch": 1})
        code, out, err = _run(profile_cli.main, ["vasp", "step1-status", str(self.step1), "--scheduler", "none"])
        self.assertEqual(code, 0, err)
        self.assertIn("scheduler: NOT verified", out)
        self.assertIn("actions  (done: 1  launch: 1  resume: 1  repair: 1)", out)

    def test_status_human_output_survives_a_cp1252_stdout(self) -> None:
        self.small_tree()
        buffer = io.BytesIO()
        stream = io.TextIOWrapper(buffer, encoding="cp1252")  # e.g. a redirected stdout on Windows
        with patch("sys.stdout", stream):
            self.assertEqual(cli.main(["vasp", "step1-status", str(self.step1), "--scheduler", "none"]), 0)
            stream.flush()
            text = buffer.getvalue().decode("cp1252")
        stream.detach()
        self.assertIn("] done  -> done: complete, stable and thermally ready for Step2", text)

    def test_launch_only_resumed(self) -> None:
        runs = self.small_tree()
        prepare_step1_resume(self.step1, execute=True, scheduler=fake_guard())
        code, out, err = _run(
            cli.main, ["vasp", "step1-launch", str(self.step1), "--only-resumed", "--scheduler", "none"]
        )
        self.assertEqual(code, 0, err)
        payload = json.loads(out)
        self.assertEqual(
            [(row["relative_path"], row["kind"]) for row in payload["planned"]], [("resume", "resume-prepared")]
        )
        self.assertIn(
            "not a resumed run (--only-resumed)",
            {row["relative_path"]: row["reason"] for row in payload["skipped_runs"]}["launch"],
        )
        self.assertFalse((self.step1 / LAUNCH_LEDGER).exists())

        with mock_sbatch(5401) as calls:
            code, out, err = _run(
                profile_cli.main,
                ["vasp", "step1-launch", str(self.step1), "--only-resumed", "--scheduler", "none", "--execute"],
            )
        self.assertEqual(code, 0, err)
        self.assertEqual([cwd for _command, cwd in calls], [str(runs["resume"])])
        self.assertIn("step1-launch: submitted 1 job(s): resume (job 5401)", err)

    def test_old_invocations_still_work(self) -> None:
        runs = self.small_tree()
        before = tree_snapshot(self.root)
        # Exactly the argv the existing tests and launch scripts use.
        for argv in (
            ["vasp", "step1-status", str(self.step1), "--json"],
            ["vasp", "step1-status", str(self.step1)],
            ["vasp", "step1-status", str(self.step1), "--stale-hours", "6"],
            ["vasp", "step1-repair", str(runs["repair"])],
            ["vasp", "step1-repair", str(self.step1), "--potim", "0.5", "--algo", "Normal", "--precondition",
             "--ramp-from", "100", "--stale-hours", "6", "--langevin", "--langevin-gamma", "10"],
            ["vasp", "step1-launch", str(self.step1)],
            ["vasp", "step1-launch", str(self.step1), "--only-repaired"],
        ):  # fmt: skip
            with self.subTest(argv=argv):
                code, out, err = _run(cli.main, argv)
                if argv[1] == "step1-launch" and "--only-repaired" in argv:
                    # Nothing repaired yet: step1-launch refuses with its usual message.
                    self.assertEqual(code, 2)
                    self.assertIn("No launchable Step1 runs", err)
                else:
                    self.assertEqual(code, 0, err)
        self.assertEqual(tree_snapshot(self.root), before)

        # The old two-step rescue (repair --execute, then launch --only-repaired --execute) still works.
        with mock_sbatch(5501) as calls:
            code, _out, err = _run(
                cli.main,
                ["vasp", "step1-repair", str(runs["repair"]), "--precondition", "--ramp-from", "100", "--execute"],
            )
            self.assertEqual(code, 0, err)
            self.assertEqual(calls, [])
            code, out, err = _run(
                cli.main, ["vasp", "step1-launch", str(runs["repair"]), "--only-repaired", "--execute"]
            )
        self.assertEqual(code, 0, err)
        self.assertEqual([cwd for _command, cwd in calls], [str(runs["repair"])])
        self.assertEqual(json.loads(out)["submitted"], 1)


if __name__ == "__main__":
    unittest.main()
