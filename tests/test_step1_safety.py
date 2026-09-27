"""Cross-command safety: per-run lock, post-archive re-check, no write-through, shared activity rule."""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import interfaceforge.step1_repair as repair_module
from interfaceforge.errors import SafetyError
from interfaceforge.step1_launch import launch_step1_runs
from interfaceforge.step1_lineage import (
    ARCHIVE_MANIFEST,
    interrupted_archive,
    read_json,
    run_lock,
    run_lock_path,
)
from interfaceforge.step1_repair import prepare_step1_repair
from interfaceforge.step1_resume import prepare_step1_resume
from interfaceforge.step1_scheduler import SchedulerGuard, SchedulerSnapshot
from interfaceforge.step1_status import step1_status

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
from test_step1_launch import _tree  # noqa: E402
from test_step1_repair import _runaway  # noqa: E402


def _switchable_guard(state: dict[str, object]) -> SchedulerGuard:
    """A verified guard whose snapshot lists ``state['active']`` (a run dir or None) as RUNNING."""

    def factory() -> SchedulerSnapshot:
        active = state.get("active")
        jobs = (
            [{"job_id": "9900", "state": "RUNNING", "workdir": str(active), "workdir_real": os.path.realpath(active)}]
            if active
            else []
        )
        return SchedulerSnapshot(
            requested="slurm",
            mode="slurm",
            verified=True,
            reason="switchable fixture",
            taken_at="2026-09-25T00:00:00+00:00",
            jobs=jobs,
            monotonic=time.monotonic(),
        )

    return SchedulerGuard("slurm", recheck_seconds=0.0, snapshot_factory=factory)


class RunLockTests(unittest.TestCase):
    def test_lock_is_exclusive_and_leaves_no_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            with run_lock(run, "repair"):
                with self.assertRaisesRegex(SafetyError, "holds .*step1.lock"):
                    with run_lock(run, "resume"):
                        pass
            self.assertFalse((run / ".interfaceforge").exists())

    def test_held_lock_blocks_launch_without_sbatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            step1 = _tree(Path(tmp))
            with run_lock(step1 / "fresh_run", "resume"), mock_sbatch() as calls:
                with self.assertRaisesRegex(SafetyError, "holds"):
                    launch_step1_runs([step1], execute=True, scheduler=fake_guard())
            self.assertEqual(calls, [])

    def test_concurrent_launches_submit_each_run_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            step1 = _tree(Path(tmp))
            results: list[object] = []
            with mock_sbatch() as calls:
                from interfaceforge import vasp

                real = vasp.subprocess.run

                def slow(*args, **kwargs):
                    time.sleep(0.3)  # sbatch latency: the window in which both used to pass every check
                    return real(*args, **kwargs)

                with patch("interfaceforge.vasp.subprocess.run", side_effect=slow):

                    def worker() -> None:
                        try:
                            result = launch_step1_runs([step1], execute=True, scheduler=fake_guard())
                            results.append(result["submitted"])
                        except SafetyError as exc:
                            results.append(exc)

                    threads = [threading.Thread(target=worker) for _ in range(2)]
                    for thread in threads:
                        thread.start()
                    for thread in threads:
                        thread.join()
            directories = [cwd for _command, cwd in calls]
            self.assertEqual(len(directories), len(set(directories)), directories)

    def test_held_lock_blocks_resume_and_repair_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            step1 = Path(tmp).resolve() / "Step1"
            healthy = write_step1_run(step1, "healthy")
            unstable = write_step1_run(step1, "unstable", steps=30, energies=_runaway(25))
            write_manifest(step1, [healthy, unstable])
            before = tree_snapshot(step1)
            with run_lock(healthy, "repair"), run_lock(unstable, "resume"):
                with self.assertRaisesRegex(SafetyError, "holds"):
                    prepare_step1_resume(healthy, execute=True, scheduler=fake_guard())
                with self.assertRaisesRegex(SafetyError, "holds"):
                    prepare_step1_repair(unstable, execute=True, scheduler=fake_guard())
            self.assertEqual(tree_snapshot(step1), before)
            self.assertIsNone(interrupted_archive(healthy))
            self.assertIsNone(interrupted_archive(unstable))


class PostArchiveRecheckTests(unittest.TestCase):
    def _assert_abandoned(self, run: Path, inputs_before: dict[str, bytes]) -> None:
        self.assertIsNone(interrupted_archive(run))
        manifests = list((run / ".interfaceforge" / "archive").glob(f"*/{ARCHIVE_MANIFEST}"))
        self.assertEqual([read_json(path)["status"] for path in manifests], ["ABANDONED"])
        for name, data in inputs_before.items():
            self.assertEqual((run / name).read_bytes(), data, name)
        self.assertFalse(run_lock_path(run).exists())

    def test_job_starting_during_resume_archive_leaves_run_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            step1 = Path(tmp).resolve() / "Step1"
            run = write_step1_run(step1, "healthy")
            write_manifest(step1, [run])
            inputs = {name: (run / name).read_bytes() for name in ("POSCAR", "INCAR", "OSZICAR", "CONTCAR")}
            state: dict[str, object] = {"active": None}
            import interfaceforge.step1_resume as resume_module

            original = resume_module.archive_step1_state

            def archive_then_start_job(folder, operation):
                archive = original(folder, operation)
                state["active"] = run  # sbatch'ed by someone else while GB were being copied
                return archive

            with patch.object(resume_module, "archive_step1_state", archive_then_start_job):
                with self.assertRaisesRegex(SafetyError, "active"):
                    prepare_step1_resume(run, execute=True, scheduler=_switchable_guard(state))
            self._assert_abandoned(run, inputs)

    def test_job_starting_during_repair_archive_leaves_run_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            step1 = Path(tmp).resolve() / "Step1"
            run = write_step1_run(step1, "unstable", steps=30, energies=_runaway(25))
            write_manifest(step1, [run])
            inputs = {name: (run / name).read_bytes() for name in ("POSCAR", "INCAR", "OSZICAR", "XDATCAR")}
            state: dict[str, object] = {"active": None}
            original = repair_module.archive_step1_state

            def archive_then_start_job(folder, operation):
                archive = original(folder, operation)
                state["active"] = run
                return archive

            with patch.object(repair_module, "archive_step1_state", archive_then_start_job):
                with self.assertRaisesRegex(SafetyError, "active"):
                    prepare_step1_repair(run, execute=True, scheduler=_switchable_guard(state))
            self._assert_abandoned(run, inputs)


class NoWriteThroughTests(unittest.TestCase):
    def _link_outside(self, run: Path, name: str, tmp: Path) -> Path:
        shared = tmp / f"shared_{name}"
        shared.write_bytes((run / name).read_bytes())
        (run / name).unlink()
        os.link(shared, run / name)
        return shared

    def test_resume_does_not_modify_hard_linked_template_poscar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp).resolve()
            step1 = tmp_path / "Step1"
            run = write_step1_run(step1, "healthy")
            shared = self._link_outside(run, "POSCAR", tmp_path)
            write_manifest(step1, [run])
            before = shared.read_bytes()
            result = prepare_step1_resume(run, execute=True, scheduler=fake_guard())
            self.assertEqual(result["runs"][0]["status"], "PREPARED")
            self.assertEqual(shared.read_bytes(), before)
            self.assertNotEqual((run / "POSCAR").read_bytes(), before)

    def test_repair_precondition_does_not_modify_shared_poscar_or_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp).resolve()
            step1 = tmp_path / "Step1"
            run = write_step1_run(step1, "unstable", steps=30, energies=_runaway(25))
            poscar = self._link_outside(run, "POSCAR", tmp_path)
            launcher = self._link_outside(run, "runvasp.sh", tmp_path)
            write_manifest(step1, [run])
            poscar_before, launcher_before = poscar.read_bytes(), launcher.read_bytes()
            result = prepare_step1_repair(run, execute=True, precondition=True, scheduler=fake_guard())
            self.assertEqual(result["runs"][0]["status"], "PREPARED")
            self.assertEqual(poscar.read_bytes(), poscar_before)
            self.assertEqual(launcher.read_bytes(), launcher_before)
            self.assertNotEqual((run / "runvasp.sh").read_bytes(), launcher_before)


class SharedActivityRuleTests(unittest.TestCase):
    def test_repair_refuses_run_with_old_oszicar_but_fresh_outcar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            step1 = Path(tmp).resolve() / "Step1"
            run = write_step1_run(step1, "unstable", steps=30, energies=_runaway(25), age_hours=10)
            write_manifest(step1, [run])
            os.utime(run / "OUTCAR", None)  # VASP still inside a long SCF step
            status = step1_status(step1, scheduler="none")["runs"][0]
            self.assertEqual(status["recovery"]["category"], "active")
            plan = prepare_step1_repair(step1, scheduler="none")["runs"][0]
            self.assertEqual(plan["status"], "ACTIVE_OR_RECENT")
            self.assertIn("OUTCAR", plan["skip_reason"])
            before = tree_snapshot(step1)
            with self.assertRaises(SafetyError):
                prepare_step1_repair(step1, execute=True, scheduler="none")
            self.assertEqual(tree_snapshot(step1), before)


if __name__ == "__main__":
    unittest.main()
