"""Tests for the Step1 Slurm activity guard (``interfaceforge.step1_scheduler``).

squeue and ``shutil.which`` are always patched: nothing here talks to a real
scheduler, so the tests run the same on Windows and on Linux CI.
"""

from __future__ import annotations

import codecs
import locale
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

from interfaceforge.errors import SafetyError
from interfaceforge.step1_scheduler import (
    DEFAULT_RECHECK_SECONDS,
    DEFAULT_STALE_HOURS,
    SCHEDULER_MODES,
    SLURM_VERIFIED_SETTLE_HOURS,
    SchedulerGuard,
    SchedulerSnapshot,
    as_guard,
    resolve_stale_hours,
    take_snapshot,
)

_TESTS = str(Path(__file__).resolve().parent)
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)
from step1_fixtures import fake_guard, sequence_guard  # noqa: E402

WHICH = "interfaceforge.step1_scheduler.shutil.which"
RUN = "interfaceforge.step1_scheduler.subprocess.run"
GETUSER = "interfaceforge.step1_scheduler.getpass.getuser"
REALPATH = "interfaceforge.step1_scheduler.os.path.realpath"
# The module's ``time`` is the stdlib module, so this also drives time.monotonic() in _CountingFactory.
MONOTONIC = "interfaceforge.step1_scheduler.time.monotonic"
SQUEUE_PATH = "/usr/bin/squeue"
SQUEUE_KWARGS = {"check": True, "capture_output": True, "text": True, "errors": "surrogateescape", "timeout": 60}


def _completed(stdout: str | None) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["squeue"], returncode=0, stdout=stdout, stderr="")


def _squeue_user(run: Mock) -> str:
    command = run.call_args.args[0]
    return command[command.index("-u") + 1]


_SLURM_SELF_VARIABLES = ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID")


@contextmanager
def _env(
    *,
    user: str | None = "alice",
    job_id: str | None = None,
    array_job_id: str | None = None,
    array_task_id: str | None = None,
) -> Iterator[None]:
    """os.environ with a controlled USER / SLURM_JOB_ID / SLURM_ARRAY_* (restored afterwards)."""

    with patch.dict(os.environ, {}, clear=False):
        for name in (*_SLURM_SELF_VARIABLES, "USER"):
            os.environ.pop(name, None)
        values = {
            "USER": user,
            "SLURM_JOB_ID": job_id,
            "SLURM_ARRAY_JOB_ID": array_job_id,
            "SLURM_ARRAY_TASK_ID": array_task_id,
        }
        for name, value in values.items():
            if value is not None:
                os.environ[name] = value
        yield


class _FakeClock:
    """Deterministic stand-in for time.monotonic(): independent of machine uptime."""

    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _snapshot(
    jobs: list[dict[str, Any]], *, verified: bool = True, monotonic: float | None = None
) -> SchedulerSnapshot:
    return SchedulerSnapshot(
        requested="slurm",
        mode="slurm",
        verified=verified,
        reason="test snapshot",
        taken_at="2026-09-23T00:00:00+00:00",
        jobs=jobs,
        monotonic=time.monotonic() if monotonic is None else monotonic,
    )


def _job(job_id: str, state: str, workdir: str | Path) -> dict[str, Any]:
    return {"job_id": job_id, "state": state, "workdir": str(workdir), "workdir_real": os.path.realpath(str(workdir))}


def _make_directory_link(target: Path, link: Path) -> str | None:
    """Create a directory link: ``"symlink"``, else a Windows ``"junction"``, else None.

    Windows without Developer Mode refuses unprivileged symlinks; a junction
    still exercises the realpath resolution the guard relies on.
    """

    try:
        os.symlink(target, link, target_is_directory=True)
        return "symlink"
    except (OSError, NotImplementedError):
        pass
    if os.name == "nt":
        try:
            import _winapi

            _winapi.CreateJunction(str(target), str(link))
            return "junction"
        except (ImportError, AttributeError, OSError):
            pass
    return None


class _CountingFactory:
    """snapshot_factory that counts calls; ``age_seconds`` backdates each snapshot."""

    def __init__(self, jobs: list[dict[str, Any]] | None = None, *, age_seconds: float = 0.0, stamp: bool = True):
        self.jobs = list(jobs or [])
        self.age_seconds = age_seconds
        self.stamp = stamp
        self.calls = 0

    def __call__(self) -> SchedulerSnapshot:
        self.calls += 1
        monotonic = time.monotonic() - self.age_seconds if self.stamp else 0.0
        return _snapshot([dict(job) for job in self.jobs], monotonic=monotonic)


class _FlakyFactory:
    """snapshot_factory that raises SafetyError while ``failing`` is True; counts calls."""

    MESSAGE = "could not query Slurm (TimeoutExpired: squeue timed out); refusing rather than guessing job state"

    def __init__(self, *, failing: bool = True) -> None:
        self.failing = failing
        self.calls = 0

    def __call__(self) -> SchedulerSnapshot:
        self.calls += 1
        if self.failing:
            raise SafetyError(self.MESSAGE)
        return _snapshot([])


class TakeSnapshotTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_constants_match_spec(self) -> None:
        self.assertEqual(SCHEDULER_MODES, ("auto", "slurm", "none"))
        self.assertEqual(DEFAULT_RECHECK_SECONDS, 15.0)
        self.assertEqual(DEFAULT_STALE_HOURS, 6.0)
        self.assertEqual(SLURM_VERIFIED_SETTLE_HOURS, 0.1)

    def test_squeue_output_parsing_skips_malformed_and_missing_workdirs(self) -> None:
        run_a = self.root / "a"
        run_b = self.root / "b"
        stdout = "\n".join(
            [
                f"101|RUNNING|{run_a}",
                "garbage line without separators",
                "102|PENDING",
                "103|PENDING|N/A",
                "104|RUNNING|(null)",
                "105|RUNNING|",
                "106|RUNNING|   ",
                "",
                f"  107 | PENDING | {run_b}  ",
            ]
        )
        with _env(), patch(WHICH, return_value=SQUEUE_PATH), patch(RUN, return_value=_completed(stdout)):
            snapshot = take_snapshot("auto")
        self.assertEqual(snapshot.requested, "auto")
        self.assertEqual(snapshot.mode, "slurm")
        self.assertTrue(snapshot.verified)
        self.assertEqual([job["job_id"] for job in snapshot.jobs], ["101", "107"])
        self.assertEqual([job["state"] for job in snapshot.jobs], ["RUNNING", "PENDING"])
        self.assertEqual(snapshot.jobs[0]["workdir"], str(run_a))
        self.assertEqual(snapshot.jobs[0]["workdir_real"], os.path.realpath(str(run_a)))
        self.assertEqual(snapshot.jobs[1]["workdir"], str(run_b))
        self.assertEqual(snapshot.reason, "squeue -u alice: 2 active job(s)")
        self.assertRegex(snapshot.taken_at, r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00$")
        self.assertGreater(snapshot.monotonic, 0.0)

    def test_pipe_inside_workdir_is_kept(self) -> None:
        # split("|", 2): only the first two separators delimit fields.
        with _env(), patch(WHICH, return_value=SQUEUE_PATH), patch(RUN, return_value=_completed("7|RUNNING|/x/a|b\n")):
            snapshot = take_snapshot("auto")
        self.assertEqual(len(snapshot.jobs), 1)
        self.assertEqual(snapshot.jobs[0]["workdir"], "/x/a|b")

    def test_own_slurm_job_is_excluded(self) -> None:
        stdout = f"555|RUNNING|{self.root}\n556|RUNNING|{self.root / 'leaf'}\n"
        with _env(job_id="555"), patch(WHICH, return_value=SQUEUE_PATH), patch(RUN, return_value=_completed(stdout)):
            snapshot = take_snapshot("auto")
        self.assertEqual([job["job_id"] for job in snapshot.jobs], ["556"])
        self.assertIn("1 active job(s)", snapshot.reason)
        self.assertIn("555", snapshot.reason)

    def test_other_jobs_kept_without_slurm_job_id(self) -> None:
        stdout = f"555|RUNNING|{self.root}\n"
        with _env(), patch(WHICH, return_value=SQUEUE_PATH), patch(RUN, return_value=_completed(stdout)):
            snapshot = take_snapshot("slurm")
        self.assertEqual([job["job_id"] for job in snapshot.jobs], ["555"])

    def test_own_array_task_is_excluded(self) -> None:
        # squeue lists an array task as <SLURM_ARRAY_JOB_ID>_<SLURM_ARRAY_TASK_ID>, never by the
        # task's own SLURM_JOB_ID; a recovery array task run from the leaf must not block itself.
        stdout = f"1234567_4|RUNNING|{self.root}\n1234567_5|RUNNING|{self.root / 'leaf'}\n"
        with (
            _env(job_id="1234571", array_job_id="1234567", array_task_id="4"),
            patch(RUN, return_value=_completed(stdout)),
        ):
            snapshot = take_snapshot("slurm")
        self.assertEqual([job["job_id"] for job in snapshot.jobs], ["1234567_5"])
        self.assertEqual(snapshot.active_jobs_for(self.root / "other"), [])
        self.assertEqual(snapshot.reason, "squeue -u alice: 1 active job(s) (ignoring this job 1234567_4)")

    def test_array_self_id_needs_both_array_variables(self) -> None:
        stdout = f"1234567_4|RUNNING|{self.root}\n"
        for partial in ({"array_job_id": "1234567"}, {"array_task_id": "4"}):
            with self.subTest(**partial):
                with _env(**partial), patch(RUN, return_value=_completed(stdout)):
                    snapshot = take_snapshot("slurm")
                self.assertEqual([job["job_id"] for job in snapshot.jobs], ["1234567_4"])
                self.assertNotIn("ignoring", snapshot.reason)

    def test_auto_without_squeue_is_unverified_none(self) -> None:
        runner = Mock()
        with _env(), patch(WHICH, return_value=None) as which, patch(RUN, runner):
            snapshot = take_snapshot("auto")
        which.assert_called_once_with("squeue")
        runner.assert_not_called()
        self.assertEqual(snapshot.requested, "auto")
        self.assertEqual(snapshot.mode, "none")
        self.assertFalse(snapshot.verified)
        self.assertEqual(snapshot.reason, "no squeue on PATH; relying on file-age guard")
        self.assertEqual(snapshot.jobs, [])

    def test_mode_none_never_queries(self) -> None:
        which = Mock()
        runner = Mock()
        with patch(WHICH, which), patch(RUN, runner):
            snapshot = take_snapshot("none")
        which.assert_not_called()
        runner.assert_not_called()
        self.assertEqual((snapshot.requested, snapshot.mode, snapshot.verified), ("none", "none", False))
        self.assertEqual(snapshot.reason, "scheduler check disabled (--scheduler none)")
        self.assertEqual(snapshot.jobs, [])

    def test_auto_with_squeue_calls_squeue_with_expected_arguments(self) -> None:
        # -a: include hidden partitions; errors="surrogateescape": decoding can never fail.
        with _env(user="alice"), patch(WHICH, return_value=SQUEUE_PATH), patch(RUN, return_value=_completed("")) as run:
            snapshot = take_snapshot("auto")
        run.assert_called_once_with(["squeue", "-h", "-a", "-u", "alice", "-o", "%i|%T|%Z"], **SQUEUE_KWARGS)
        self.assertTrue(snapshot.verified)
        self.assertEqual(snapshot.jobs, [])
        self.assertEqual(snapshot.reason, "squeue -u alice: 0 active job(s)")

    def test_explicit_user_overrides_environment(self) -> None:
        with _env(user="alice"), patch(WHICH, return_value=SQUEUE_PATH), patch(RUN, return_value=_completed("")) as run:
            take_snapshot("auto", user="bob")
        self.assertEqual(_squeue_user(run), "bob")

    def test_user_falls_back_to_getpass(self) -> None:
        with (
            _env(user=None),
            patch(GETUSER, return_value="carol"),
            patch(WHICH, return_value=SQUEUE_PATH),
            patch(RUN, return_value=_completed("")) as run,
        ):
            take_snapshot("slurm")
        self.assertEqual(_squeue_user(run), "carol")

    def test_unknown_user_refuses_in_slurm_mode(self) -> None:
        runner = Mock()
        with _env(user=None), patch(GETUSER, side_effect=OSError("no user")), patch(RUN, runner):
            with self.assertRaisesRegex(SafetyError, "could not query Slurm"):
                take_snapshot("slurm")
        runner.assert_not_called()

    def test_slurm_mode_does_not_consult_path(self) -> None:
        # --scheduler slurm asks squeue directly; a missing binary is a failure, not a fallback.
        which = Mock(return_value=None)
        with _env(), patch(WHICH, which), patch(RUN, return_value=_completed("")):
            snapshot = take_snapshot("slurm")
        which.assert_not_called()
        self.assertEqual((snapshot.requested, snapshot.mode, snapshot.verified), ("slurm", "slurm", True))

    def test_slurm_mode_failures_raise_safety_error(self) -> None:
        command = ["squeue", "-h", "-a", "-u", "alice", "-o", "%i|%T|%Z"]
        failures = [
            subprocess.CalledProcessError(1, command, output="", stderr="slurm_load_jobs error: timeout"),
            FileNotFoundError(2, "No such file or directory", "squeue"),
            subprocess.TimeoutExpired(command, 60),
            OSError("permission denied"),
            UnicodeDecodeError("utf-8", b"\x81", 0, 1, "invalid start byte"),
            ValueError("I/O operation on closed file"),
        ]
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                with _env(), patch(RUN, side_effect=failure):
                    with self.assertRaises(SafetyError) as caught:
                        take_snapshot("slurm")
                message = str(caught.exception)
                self.assertIn("could not query Slurm (", message)
                self.assertIn("refusing rather than guessing job state", message)
                self.assertIn(type(failure).__name__, message)
                self.assertIs(caught.exception.__cause__, failure)

    def test_called_process_error_stderr_is_reported(self) -> None:
        failure = subprocess.CalledProcessError(1, ["squeue"], output="", stderr="slurm_load_jobs error: timeout\n")
        with _env(), patch(RUN, side_effect=failure):
            with self.assertRaisesRegex(SafetyError, "slurm_load_jobs error: timeout"):
                take_snapshot("slurm")

    def test_undecodable_stderr_gives_printable_message(self) -> None:
        # surrogateescape decoding leaves lone surrogates in stderr; the message must still encode as UTF-8.
        failure = subprocess.CalledProcessError(1, ["squeue"], output="", stderr="bad \udc81 byte\n")
        with _env(), patch(RUN, side_effect=failure):
            with self.assertRaises(SafetyError) as caught:
                take_snapshot("slurm")
        message = str(caught.exception)
        message.encode("utf-8")  # must not raise UnicodeEncodeError
        self.assertIn("stderr: bad \\udc81 byte", message)

    def test_undecodable_squeue_output_never_loses_the_job(self) -> None:
        # Regression: without errors="surrogateescape" a non-UTF-8 WorkDir made the Windows reader
        # thread drop stdout (a VERIFIED snapshot with 0 jobs) and raised a bare ValueError on POSIX.
        # A real child process (not squeue) prints the bytes so the actual decoding path runs.
        real_run = subprocess.run
        code = "import sys; sys.stdout.buffer.write(b'1|RUNNING|/scratch/\\x81\\xff/run\\n')"

        def fake_squeue(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            self.assertEqual(command[0], "squeue")
            return real_run([sys.executable, "-c", code], **kwargs)

        with _env(), patch(RUN, side_effect=fake_squeue):
            snapshot = take_snapshot("slurm")
        self.assertTrue(snapshot.verified)
        self.assertEqual([job["job_id"] for job in snapshot.jobs], ["1"])
        self.assertEqual(snapshot.reason, "squeue -u alice: 1 active job(s)")
        workdir = snapshot.jobs[0]["workdir"]
        self.assertTrue(workdir.startswith("/scratch/") and workdir.endswith("/run"), ascii(workdir))
        same_codec = (
            codecs.lookup(locale.getpreferredencoding(False)).name == codecs.lookup(sys.getfilesystemencoding()).name
        )
        if os.name == "posix" and same_codec:
            # The escaped bytes map back to the real directory name.
            self.assertEqual(os.fsencode(workdir), b"/scratch/\x81\xff/run")

    def test_lost_stdout_refuses_instead_of_reporting_no_jobs(self) -> None:
        # capture_output always yields a str; None means the output was lost (e.g. a failed reader thread).
        with _env(), patch(RUN, return_value=_completed(None)):
            with self.assertRaisesRegex(SafetyError, r"could not query Slurm \(ValueError: squeue output could not"):
                take_snapshot("slurm")

    def test_unusable_workdir_refuses(self) -> None:
        # e.g. a WorkDir with an embedded NUL on POSIX: realpath raises ValueError -> SafetyError, not a crash.
        with (
            _env(),
            patch(RUN, return_value=_completed("1|RUNNING|/x/a\n")),
            patch(REALPATH, side_effect=ValueError("embedded null byte")),
        ):
            with self.assertRaisesRegex(SafetyError, r"could not query Slurm \(ValueError: embedded null byte\)"):
                take_snapshot("slurm")

    def test_auto_with_squeue_present_but_failing_refuses(self) -> None:
        # squeue exists but errors: auto behaves as slurm and refuses (no silent fallback).
        failure = subprocess.CalledProcessError(1, ["squeue"], stderr="down")
        with _env(), patch(WHICH, return_value=SQUEUE_PATH), patch(RUN, side_effect=failure):
            with self.assertRaises(SafetyError):
                take_snapshot("auto")

    def test_unknown_mode_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            take_snapshot("pbs")
        with self.assertRaises(ValueError):
            SchedulerGuard("pbs")
        with self.assertRaises(ValueError):
            as_guard("pbs")

    def test_to_dict_has_exact_keys_and_counts_jobs(self) -> None:
        snapshot = _snapshot([_job("1", "RUNNING", self.root / "a"), _job("2", "PENDING", self.root / "b")])
        self.assertEqual(
            snapshot.to_dict(),
            {
                "requested": "slurm",
                "mode": "slurm",
                "verified": True,
                "reason": "test snapshot",
                "taken_at": "2026-09-23T00:00:00+00:00",
                "active_jobs": 2,
            },
        )


class ActiveJobsMatchingTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.series = self.root / "step1"
        self.run = self.series / "OH50" / "leaf1"
        self.run.mkdir(parents=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_exact_workdir_is_active(self) -> None:
        snapshot = _snapshot([_job("11", "RUNNING", self.run)])
        self.assertEqual([job["job_id"] for job in snapshot.active_jobs_for(self.run)], ["11"])
        self.assertEqual(len(snapshot.active_jobs_for(str(self.run))), 1)

    def test_descendant_workdir_is_active(self) -> None:
        snapshot = _snapshot([_job("12", "RUNNING", self.run / "precondition")])
        self.assertEqual([job["job_id"] for job in snapshot.active_jobs_for(self.run)], ["12"])

    def test_ancestor_workdir_is_not_active(self) -> None:
        # A recovery job submitted from the Step1 root must not block every leaf.
        snapshot = _snapshot([_job("13", "RUNNING", self.series), _job("14", "RUNNING", self.series / "OH50")])
        self.assertEqual(snapshot.active_jobs_for(self.run), [])
        # ...but the root itself (and its parent group) do see them.
        self.assertEqual(len(snapshot.active_jobs_for(self.series)), 2)

    def test_sibling_with_common_prefix_is_not_active(self) -> None:
        snapshot = _snapshot([_job("15", "RUNNING", self.series / "OH50" / "leaf10")])
        self.assertEqual(snapshot.active_jobs_for(self.run), [])

    def test_trailing_separator_and_dot_segments_are_normalised(self) -> None:
        snapshot = _snapshot([_job("16", "RUNNING", str(self.run) + os.sep)])
        self.assertEqual(len(snapshot.active_jobs_for(self.run)), 1)
        self.assertEqual(len(snapshot.active_jobs_for(self.run / ".." / "leaf1")), 1)

    def test_job_rows_without_workdir_real_are_resolved(self) -> None:
        snapshot = _snapshot(
            [
                {"job_id": "17", "state": "RUNNING", "workdir": str(self.run)},
                {"job_id": "18", "state": "RUNNING", "workdir": ""},
                {"job_id": "19", "state": "RUNNING", "workdir": "N/A"},
            ]
        )
        self.assertEqual([job["job_id"] for job in snapshot.active_jobs_for(self.run)], ["17"])

    def test_empty_workdir_never_matches_cwd(self) -> None:
        # realpath("") would be the cwd; such a row must be ignored rather than match it.
        snapshot = _snapshot([{"job_id": "20", "state": "RUNNING", "workdir": ""}])
        self.assertEqual(snapshot.active_jobs_for(os.getcwd()), [])

    def test_symlinked_paths_match_by_realpath(self) -> None:
        link = self.root / "link"
        kind = _make_directory_link(self.series, link)
        if kind is None:
            self.skipTest("neither directory symlinks nor junctions can be created here")
        try:
            via_link = link / "OH50" / "leaf1"
            self.assertEqual(os.path.realpath(str(via_link)), os.path.realpath(str(self.run)))
            # Job WorkDir reported through the link, run addressed by its real path.
            self.assertEqual(len(_snapshot([_job("21", "RUNNING", via_link)]).active_jobs_for(self.run)), 1)
            # Job WorkDir by real path (descendant), run addressed through the link.
            self.assertEqual(
                len(_snapshot([_job("22", "RUNNING", self.run / "precondition")]).active_jobs_for(via_link)), 1
            )
            # An ancestor reached through the link is still an ancestor.
            self.assertEqual(_snapshot([_job("23", "RUNNING", link)]).active_jobs_for(self.run), [])
            # squeue output through the link resolves workdir_real to the real directory.
            stdout = f"24|RUNNING|{via_link}\n"
            with _env(), patch(WHICH, return_value=SQUEUE_PATH), patch(RUN, return_value=_completed(stdout)):
                snapshot = take_snapshot("auto")
            self.assertEqual(snapshot.jobs[0]["workdir_real"], os.path.realpath(str(self.run)))
            self.assertEqual(len(snapshot.active_jobs_for(self.run)), 1)
        finally:
            if kind == "junction":
                os.rmdir(link)  # removes the junction only, never its target

    @unittest.skipUnless(os.name == "nt", "case-insensitive paths are a Windows property")
    def test_windows_match_is_case_insensitive(self) -> None:
        # realpath returns the on-disk case for EXISTING paths, which would make both sides equal
        # before normcase runs.  The tail here does not exist, so realpath keeps it as given and
        # only normcase can make the two spellings match.
        tail = self.run / "precondition"
        self.assertFalse(tail.exists())
        upper, lower = str(tail).upper(), str(tail).lower()
        self.assertNotEqual(os.path.realpath(upper), os.path.realpath(lower))
        self.assertEqual(len(_snapshot([_job("25", "RUNNING", upper)]).active_jobs_for(lower)), 1)
        self.assertEqual(len(_snapshot([_job("26", "RUNNING", str(tail / "SUB").upper())]).active_jobs_for(lower)), 1)


class SchedulerGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.run = self.root / "leaf"
        self.other = self.root / "other"
        self.run.mkdir()
        self.other.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_assert_inactive_raises_with_job_id_and_state(self) -> None:
        guard = SchedulerGuard.fixed([{"job_id": 4242, "state": "RUNNING", "workdir": self.run / "precondition"}])
        with self.assertRaises(SafetyError) as caught:
            guard.assert_inactive(self.run)
        message = str(caught.exception)
        self.assertTrue(message.startswith(f"Refusing to mutate {self.run}: active Slurm job(s) "), message)
        self.assertIn("4242 (RUNNING)", message)
        self.assertTrue(message.endswith("use it as WorkDir"), message)

    def test_assert_inactive_lists_every_active_job(self) -> None:
        guard = SchedulerGuard.fixed(
            [
                {"job_id": "1", "state": "RUNNING", "workdir": str(self.run)},
                {"job_id": "2", "state": "PENDING", "workdir": str(self.run / "precondition")},
            ]
        )
        with self.assertRaisesRegex(SafetyError, re.escape("1 (RUNNING), 2 (PENDING)")):
            guard.assert_inactive(self.run)

    def test_assert_inactive_returns_snapshot_for_idle_run(self) -> None:
        guard = SchedulerGuard.fixed([{"job_id": "1", "state": "RUNNING", "workdir": str(self.run)}])
        snapshot = guard.assert_inactive(self.other)
        self.assertIsInstance(snapshot, SchedulerSnapshot)
        self.assertIs(snapshot, guard.snapshot)
        self.assertEqual(snapshot.to_dict()["active_jobs"], 1)

    def test_fixed_computes_workdir_real_and_validates_jobs(self) -> None:
        guard = SchedulerGuard.fixed([{"job_id": 7, "state": "PENDING", "workdir": self.run, "workdir_real": "/bogus"}])
        job = guard.snapshot.jobs[0]
        self.assertEqual(job["job_id"], "7")
        self.assertEqual(job["workdir"], str(self.run))
        self.assertEqual(job["workdir_real"], os.path.realpath(str(self.run)))
        self.assertTrue(guard.snapshot.verified)
        self.assertEqual(guard.snapshot.mode, "slurm")
        with self.assertRaises(ValueError):
            SchedulerGuard.fixed([{"job_id": "1", "state": "RUNNING"}])

    def test_fixed_skips_jobs_without_workdir(self) -> None:
        # Same rule as squeue parsing; realpath("") would otherwise make the job block the cwd.
        no_workdir = [
            {"job_id": str(index), "state": "RUNNING", "workdir": workdir}
            for index, workdir in enumerate(["", "N/A", "(null)", "   ", None], start=1)
        ]
        guard = SchedulerGuard.fixed([*no_workdir, {"job_id": "6", "state": "RUNNING", "workdir": self.run}])
        self.assertEqual([job["job_id"] for job in guard.snapshot.jobs], ["6"])
        self.assertEqual(guard.snapshot.reason, "fixed scheduler snapshot: 1 active job(s)")
        only_missing = SchedulerGuard.fixed(no_workdir)
        self.assertEqual(only_missing.snapshot.jobs, [])
        self.assertEqual(only_missing.active_jobs_for(os.getcwd()), [])
        only_missing.assert_inactive(os.getcwd())

    def test_fixed_unverified(self) -> None:
        guard = SchedulerGuard.fixed(verified=False)
        self.assertFalse(guard.snapshot.verified)
        self.assertEqual(guard.snapshot.mode, "none")
        self.assertEqual(guard.snapshot.jobs, [])
        guard.assert_inactive(self.run)

    def test_fixed_snapshots_are_independent_copies(self) -> None:
        guard = SchedulerGuard.fixed([{"job_id": "1", "state": "RUNNING", "workdir": str(self.run)}])
        guard.snapshot.jobs.clear()
        self.assertEqual(len(guard.refresh().jobs), 1)

    def test_snapshot_is_taken_lazily_once(self) -> None:
        factory = _CountingFactory()
        guard = SchedulerGuard("slurm", snapshot_factory=factory)
        self.assertEqual(factory.calls, 0)
        first = guard.snapshot
        self.assertIs(guard.snapshot, first)
        guard.active_jobs_for(self.run)
        self.assertEqual(factory.calls, 1)
        self.assertIsNot(guard.refresh(), first)
        self.assertEqual(factory.calls, 2)

    def test_recheck_zero_refreshes_on_every_assert(self) -> None:
        factory = _CountingFactory()
        guard = SchedulerGuard("slurm", recheck_seconds=0.0, snapshot_factory=factory)
        _ = guard.snapshot  # planning snapshot
        for _ in range(3):
            guard.assert_inactive(self.run)
        self.assertEqual(factory.calls, 4)

    def test_negative_recheck_refreshes_on_every_assert(self) -> None:
        factory = _CountingFactory()
        guard = SchedulerGuard("slurm", recheck_seconds=-1.0, snapshot_factory=factory)
        guard.assert_inactive(self.run)
        guard.assert_inactive(self.run)
        self.assertEqual(factory.calls, 2)

    def test_large_recheck_reuses_snapshot(self) -> None:
        factory = _CountingFactory()
        guard = SchedulerGuard("slurm", recheck_seconds=3600.0, snapshot_factory=factory)
        planning = guard.snapshot
        for _ in range(3):
            self.assertIs(guard.assert_inactive(self.run), planning)
        self.assertEqual(factory.calls, 1)

    def test_first_assert_without_snapshot_takes_one(self) -> None:
        factory = _CountingFactory()
        guard = SchedulerGuard("slurm", recheck_seconds=3600.0, snapshot_factory=factory)
        guard.assert_inactive(self.run)
        guard.assert_inactive(self.run)
        self.assertEqual(factory.calls, 1)

    def test_snapshot_older_than_recheck_is_refreshed(self) -> None:
        factory = _CountingFactory(age_seconds=DEFAULT_RECHECK_SECONDS * 10)
        with patch(MONOTONIC, new=_FakeClock(1000.0)):
            guard = SchedulerGuard("slurm", snapshot_factory=factory)
            _ = guard.snapshot  # planning snapshot
            guard.assert_inactive(self.run)
            guard.assert_inactive(self.run)
        self.assertEqual(factory.calls, 3)

    def test_backdated_snapshot_shortly_after_boot_is_still_refreshed(self) -> None:
        # Regression: monotonic() counts from boot, so 100 s after boot a snapshot taken 150 s
        # earlier is stamped -50.  It used to be mistaken for "unstamped", re-stamped to now and
        # never re-checked.
        factory = _CountingFactory(age_seconds=150.0)
        with patch(MONOTONIC, new=_FakeClock(100.0)):
            guard = SchedulerGuard("slurm", snapshot_factory=factory)
            self.assertEqual(guard.snapshot.monotonic, -50.0)
            guard.assert_inactive(self.run)
            guard.assert_inactive(self.run)
        self.assertEqual(factory.calls, 3)

    def test_snapshot_reused_until_recheck_window_elapses(self) -> None:
        clock = _FakeClock(1000.0)
        factory = _CountingFactory()
        with patch(MONOTONIC, new=clock):
            guard = SchedulerGuard("slurm", snapshot_factory=factory)
            guard.assert_inactive(self.run)
            clock.now = 1000.0 + DEFAULT_RECHECK_SECONDS - 0.1
            guard.assert_inactive(self.run)
            self.assertEqual(factory.calls, 1)
            clock.now = 1000.0 + DEFAULT_RECHECK_SECONDS  # age == recheck_seconds: re-check
            guard.assert_inactive(self.run)
            self.assertEqual(factory.calls, 2)
            clock.now += DEFAULT_RECHECK_SECONDS - 0.1
            guard.assert_inactive(self.run)
        self.assertEqual(factory.calls, 2)

    def test_clock_running_backwards_forces_refresh(self) -> None:
        clock = _FakeClock(1000.0)
        factory = _CountingFactory()
        with patch(MONOTONIC, new=clock):
            guard = SchedulerGuard("slurm", recheck_seconds=3600.0, snapshot_factory=factory)
            guard.assert_inactive(self.run)
            clock.now = 990.0
            guard.assert_inactive(self.run)
        self.assertEqual(factory.calls, 2)

    def test_nan_snapshot_stamp_forces_refresh(self) -> None:
        calls = []

        def factory() -> SchedulerSnapshot:
            calls.append(1)
            return _snapshot([], monotonic=math.nan)

        guard = SchedulerGuard("slurm", recheck_seconds=3600.0, snapshot_factory=factory)
        for _ in range(3):
            guard.assert_inactive(self.run)
        self.assertEqual(len(calls), 3)

    def test_non_finite_recheck_seconds_is_rejected(self) -> None:
        # NaN or +inf would silently switch off the pre-mutation re-check (spec invariant 2).
        for bad in (math.nan, math.inf, "nan", "inf"):
            with self.subTest(value=bad):
                with self.assertRaisesRegex(ValueError, "recheck_seconds"):
                    SchedulerGuard("slurm", recheck_seconds=bad)  # type: ignore[arg-type]
                guard = SchedulerGuard("slurm")
                with self.assertRaises(ValueError):
                    guard.recheck_seconds = bad  # type: ignore[assignment]
                self.assertEqual(guard.recheck_seconds, DEFAULT_RECHECK_SECONDS)

    def test_minus_infinity_recheck_always_refreshes(self) -> None:
        factory = _CountingFactory()
        guard = SchedulerGuard("slurm", recheck_seconds=-math.inf, snapshot_factory=factory)
        guard.assert_inactive(self.run)
        guard.assert_inactive(self.run)
        self.assertEqual(factory.calls, 2)

    def test_unstamped_factory_snapshots_are_stamped_by_guard(self) -> None:
        factory = _CountingFactory(stamp=False)
        guard = SchedulerGuard("slurm", recheck_seconds=3600.0, snapshot_factory=factory)
        guard.assert_inactive(self.run)
        guard.assert_inactive(self.run)
        self.assertEqual(factory.calls, 1)
        self.assertGreater(guard.snapshot.monotonic, 0.0)

    def test_throttled_guard_catches_new_job_once_recheck_elapses(self) -> None:
        factory = _CountingFactory()
        guard = SchedulerGuard("slurm", recheck_seconds=3600.0, snapshot_factory=factory)
        guard.assert_inactive(self.run)
        factory.jobs = [_job("31", "PENDING", self.run)]
        guard.assert_inactive(self.run)  # within the recheck window: planning snapshot reused
        guard.recheck_seconds = 0.0
        with self.assertRaisesRegex(SafetyError, re.escape("31 (PENDING)")):
            guard.assert_inactive(self.run)

    def test_active_jobs_for_does_not_refresh(self) -> None:
        factory = _CountingFactory()
        guard = SchedulerGuard("slurm", recheck_seconds=0.0, snapshot_factory=factory)
        _ = guard.snapshot  # planning snapshot
        factory.jobs = [_job("32", "RUNNING", self.run)]
        self.assertEqual(guard.active_jobs_for(self.run), [])
        self.assertEqual(factory.calls, 1)

    def test_default_guard_uses_take_snapshot_with_its_mode(self) -> None:
        with _env(), patch(WHICH, return_value=None):
            guard = SchedulerGuard()
            self.assertEqual(guard.mode, "auto")
            self.assertEqual(guard.recheck_seconds, DEFAULT_RECHECK_SECONDS)
            snapshot = guard.assert_inactive(self.run)
        self.assertEqual((snapshot.requested, snapshot.mode, snapshot.verified), ("auto", "none", False))

    def test_slurm_guard_propagates_squeue_failure(self) -> None:
        guard = SchedulerGuard("slurm")
        with _env(), patch(RUN, side_effect=subprocess.TimeoutExpired(["squeue"], 60)):
            with self.assertRaisesRegex(SafetyError, "could not query Slurm"):
                guard.assert_inactive(self.run)

    def test_try_snapshot_returns_current_snapshot_without_refreshing(self) -> None:
        factory = _CountingFactory()
        guard = SchedulerGuard("slurm", recheck_seconds=0.0, snapshot_factory=factory)
        snapshot, error = guard.try_snapshot()
        self.assertIsNone(error)
        self.assertIsInstance(snapshot, SchedulerSnapshot)
        self.assertIs(guard.try_snapshot()[0], snapshot)
        self.assertIs(guard.snapshot, snapshot)
        self.assertEqual(factory.calls, 1)

    def test_try_snapshot_remembers_failure_so_hung_squeue_runs_once(self) -> None:
        # A read-only status pass over many rows must not re-run a hung squeue (60 s timeout) per row.
        guard = SchedulerGuard("slurm")
        timeout = subprocess.TimeoutExpired(["squeue"], 60)
        with _env(), patch(RUN, side_effect=timeout) as run:
            results = [guard.try_snapshot() for _ in range(5)]
        self.assertEqual(run.call_count, 1)
        for snapshot, error in results:
            self.assertIsNone(snapshot)
            self.assertIsInstance(error, str)
            self.assertIn("could not query Slurm (TimeoutExpired", str(error))
            self.assertIn("refusing rather than guessing job state", str(error))

    def test_mutation_still_requeries_after_remembered_failure(self) -> None:
        factory = _FlakyFactory(failing=True)
        guard = SchedulerGuard("slurm", recheck_seconds=3600.0, snapshot_factory=factory)
        self.assertEqual(guard.try_snapshot(), (None, _FlakyFactory.MESSAGE))
        # assert_inactive fails closed: it queries again instead of trusting (or skipping) anything.
        with self.assertRaisesRegex(SafetyError, "could not query Slurm"):
            guard.assert_inactive(self.run)
        self.assertEqual(factory.calls, 2)
        self.assertEqual(guard.try_snapshot(), (None, _FlakyFactory.MESSAGE))
        self.assertEqual(factory.calls, 2)
        # Once squeue answers again the remembered failure is cleared.
        factory.failing = False
        snapshot = guard.assert_inactive(self.run)
        self.assertEqual(guard.try_snapshot(), (snapshot, None))
        self.assertEqual(factory.calls, 3)

    def test_failed_refresh_drops_previous_snapshot(self) -> None:
        factory = _FlakyFactory(failing=False)
        guard = SchedulerGuard("slurm", recheck_seconds=3600.0, snapshot_factory=factory)
        planning = guard.snapshot
        factory.failing = True
        with self.assertRaises(SafetyError):
            guard.refresh()
        # The old answer is not reused for display or for a mutation check.
        self.assertEqual(guard.try_snapshot(), (None, _FlakyFactory.MESSAGE))
        with self.assertRaises(SafetyError):
            guard.assert_inactive(self.run)
        factory.failing = False
        refreshed = guard.refresh()
        self.assertIsNot(refreshed, planning)
        self.assertEqual(guard.try_snapshot(), (refreshed, None))
        self.assertEqual(factory.calls, 4)

    def test_as_guard(self) -> None:
        guard = SchedulerGuard.fixed()
        self.assertIs(as_guard(guard), guard)
        for mode in SCHEDULER_MODES:
            with self.subTest(mode=mode):
                built = as_guard(mode)
                self.assertIsInstance(built, SchedulerGuard)
                self.assertEqual(built.mode, mode)
        with self.assertRaises(TypeError):
            as_guard(None)  # type: ignore[arg-type]


class ResolveStaleHoursTest(unittest.TestCase):
    def test_explicit_value_wins(self) -> None:
        verified = _snapshot([], verified=True)
        unverified = _snapshot([], verified=False)
        for snapshot in (verified, unverified):
            hours, reason = resolve_stale_hours(2.5, snapshot)
            self.assertEqual(hours, 2.5)
            self.assertIn("explicit", reason)
        self.assertEqual(resolve_stale_hours(0, verified)[0], 0.0)

    def test_verified_slurm_uses_settle_window(self) -> None:
        hours, reason = resolve_stale_hours(None, _snapshot([], verified=True))
        self.assertEqual(hours, SLURM_VERIFIED_SETTLE_HOURS)
        self.assertIn("verified", reason)
        self.assertIn("6 min", reason)

    def test_unverified_uses_default_file_age_guard(self) -> None:
        with _env(), patch(WHICH, return_value=None):
            snapshot = take_snapshot("auto")
        hours, reason = resolve_stale_hours(None, snapshot)
        self.assertEqual(hours, DEFAULT_STALE_HOURS)
        self.assertIn("not verified", reason)
        self.assertIn("no squeue on PATH", reason)

    def test_negative_or_nan_is_rejected(self) -> None:
        for bad in (-1.0, float("nan")):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                resolve_stale_hours(bad, _snapshot([]))


class SharedFixtureGuardsTest(unittest.TestCase):
    """tests/step1_fixtures.py builds guards through the public API; keep them working."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.run = self.root / "OH50" / "leaf"
        self.idle = self.root / "OH25" / "leaf"
        self.run.mkdir(parents=True)
        self.idle.mkdir(parents=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_fake_guard(self) -> None:
        guard = fake_guard({self.run: "RUNNING", str(self.root / "OH75"): "PENDING"})
        self.assertTrue(guard.snapshot.verified)
        self.assertEqual(guard.snapshot.to_dict()["active_jobs"], 2)
        with self.assertRaisesRegex(SafetyError, re.escape("9000 (RUNNING)")):
            guard.assert_inactive(self.run)
        self.assertIsInstance(guard.assert_inactive(self.idle), SchedulerSnapshot)
        # The Step1 root is an ancestor of the active leaf: it is not itself active...
        self.assertEqual(guard.active_jobs_for(self.root / "OH75" / "leaf"), [])
        # ...while the active WorkDir's own ancestors can still see it.
        self.assertEqual([job["job_id"] for job in guard.active_jobs_for(self.root)], ["9000", "9001"])

    def test_fake_guard_idle_and_unverified(self) -> None:
        self.assertEqual(fake_guard().assert_inactive(self.run).jobs, [])
        unverified = fake_guard(verified=False)
        self.assertFalse(unverified.snapshot.verified)
        self.assertEqual(resolve_stale_hours(None, unverified.snapshot)[0], DEFAULT_STALE_HOURS)

    def test_sequence_guard_catches_job_started_after_planning(self) -> None:
        guard = sequence_guard([{}, {self.run: "PENDING"}])
        planning = guard.snapshot
        self.assertEqual(planning.reason, "fixture snapshot 0")
        self.assertEqual(planning.active_jobs_for(self.run), [])
        with self.assertRaises(SafetyError) as caught:
            guard.assert_inactive(self.run)
        self.assertIn("9100 (PENDING)", str(caught.exception))
        self.assertIn(str(self.run), str(caught.exception))
        # The last snapshot repeats.
        with self.assertRaises(SafetyError):
            guard.assert_inactive(self.run)
        self.assertEqual(guard.snapshot.reason, "fixture snapshot 1")
        self.assertIsInstance(guard.assert_inactive(self.idle), SchedulerSnapshot)


if __name__ == "__main__":
    unittest.main()
