from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from interfaceforge.cli import main
from interfaceforge.errors import SafetyError
from interfaceforge.step1_launch import launch_step1_runs
from interfaceforge.step1_lineage import (
    REPAIR_RECORD,
    RESUME_RECORD,
    archive_step1_state,
    atomic_write_json,
    build_segment_record,
    current_generation,
    finalize_archive,
    incar_schedule,
    mark_record_submitted,
    new_generation_id,
    read_json,
    utc_now_iso,
    utc_stamp,
)
from interfaceforge.step1_scheduler import SchedulerGuard, SchedulerSnapshot
from interfaceforge.vasp import _sha256_file, parse_incar

_TESTS = str(Path(__file__).resolve().parent)
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

from step1_fixtures import (  # noqa: E402
    fake_guard,
    mock_sbatch,
    sequence_guard,
    tree_snapshot,
    write_legacy_launch_ledger,
    write_legacy_repair_record,
    write_manifest,
    write_step1_run,
)

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


# --------------------------------------------------------------------------- #
# Synthetic generation helpers (fabricated fixtures, never real NiO data)
# --------------------------------------------------------------------------- #


def _idle_run(root: Path, name: str) -> Path:
    """A launchable-looking run: inputs + launcher, no runtime outputs."""

    return write_step1_run(root, name, steps=0, contcar=None, xdatcar=False, outcar=None, age_hours=48.0)


def _write_segment_record(run: Path, kind: str, *, accepted: int = 40, stamp: str | None = None) -> str:
    """Write a new-format (generation-aware) repair/resume record exactly as the planners would; return its id."""

    parent = current_generation(run)
    generation = parent.generation + 1
    stamp = stamp or utc_stamp()
    generation_id = new_generation_id(kind, generation, stamp)
    original = parent.original_nsw or 400
    prefix = parent.accepted_prefix_steps + accepted
    potim = parent.segment_potim_fs or 1.0
    record = build_segment_record(
        kind,
        run=run,
        generation=generation,
        generation_id=generation_id,
        parent=parent,
        prepared_at=utc_now_iso(),
        original_nsw=original,
        accepted_prefix_steps=prefix,
        accepted_segments=[
            *parent.ledger,
            {
                "generation": parent.generation,
                "generation_id": parent.generation_id,
                "kind": parent.kind,
                "steps": accepted,
                "potim_fs": potim,
            },
        ],
        ledger_exact=True,
        segment_nsw=original - prefix,
        segment_potim_fs=0.5 if kind == "repair" else potim,
        segment_schedule=incar_schedule(parse_incar(run / "INCAR")),
        archive=str(run / ".interfaceforge" / "archive" / f"step1_{kind}_g{generation}_{stamp}"),
        extra={},
    )
    atomic_write_json(run / (REPAIR_RECORD if kind == "repair" else RESUME_RECORD), record)
    return generation_id


def _legacy_leaf_row(leaf: Path, job_id: str = "111") -> dict[str, str]:
    """The schema-1 row an old ``step1-launch <leaf>`` wrote (leaf launched as its own root)."""

    return {
        "status": "SUBMITTED",
        "job_id": job_id,
        "kind": "prepared",
        "root": str(leaf),
        "relative_path": ".",
        "directory": str(leaf),
        "launcher": "runvasp.sh",
        "notes": "",
        "detail": "",
    }


def _stamp_hours_ago(hours: float) -> str:
    return datetime.fromtimestamp(time.time() - hours * 3600.0, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _reasons(plan: dict) -> dict[str, str]:
    return {row["relative_path"]: row["reason"] for row in plan["skipped_runs"]}


def _planned(plan: dict) -> list[str]:
    return [row["relative_path"] for row in plan["planned"]]


def _reset_record_to_prepared(run: Path, name: str = REPAIR_RECORD) -> None:
    """Simulate a crash between the ledger append and the record update: only the ledger knows."""

    record = read_json(run / name)
    record["status"] = "PREPARED"
    record["submissions"] = []
    atomic_write_json(run / name, record)


class _HermeticSchedulerMixin(unittest.TestCase):
    """``--scheduler auto`` never finds a real squeue, even on a Slurm login node."""

    def setUp(self) -> None:
        super().setUp()
        patcher = patch("interfaceforge.step1_scheduler.shutil.which", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)


class Step1LaunchTests(_HermeticSchedulerMixin):
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

    def test_cli_execute_still_works_with_the_old_keyword_arguments(self) -> None:
        # cli.py still calls launch_step1_runs(roots, execute=, launcher=, only_repaired=, progress=).
        with tempfile.TemporaryDirectory() as tmp:
            step1 = _tree(Path(tmp).resolve())
            stdout, stderr = io.StringIO(), io.StringIO()
            with mock_sbatch(5951) as calls, redirect_stdout(stdout), redirect_stderr(stderr):
                self.assertEqual(main(["vasp", "step1-launch", str(step1), "--only-repaired", "--execute"]), 0)
            self.assertEqual([cwd for _command, cwd in calls], [str(step1 / "repaired_run")])
            payload = json.loads(stdout.getvalue())
            self.assertEqual((payload["mode"], payload["submitted"]), ("submitted", 1))
            self.assertTrue(payload["batch_id"].startswith("b-"))
            ledger = read_json(step1 / "step1_launch.json")
            self.assertEqual(ledger["schema_version"], 2)
            self.assertEqual(
                [(row["relative_path"], row["job_id"], row["generation"]) for row in ledger["runs"]],
                [("repaired_run", "5951", 1)],
            )
            self.assertEqual(read_json(step1 / "repaired_run" / REPAIR_RECORD)["status"], "SUBMITTED")


class GenerationAwareLaunchTests(_HermeticSchedulerMixin):
    def setUp(self) -> None:
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.step1 = self.root / "Step1"
        self.step1.mkdir()

    # ------------------------------------------------------------------ #
    # (1) + (3): the real bug and the provenance it must keep
    # ------------------------------------------------------------------ #

    def _leaf_launched_as_own_root_then_repaired(self) -> tuple[Path, str]:
        leaf = _idle_run(self.step1, "leaf")
        write_legacy_launch_ledger(leaf, [_legacy_leaf_row(leaf)], age_hours=30.0)
        generation_id = _write_segment_record(leaf, "repair")
        return leaf, generation_id

    def test_new_repair_generation_is_launchable_despite_old_leaf_ledger(self) -> None:
        leaf, generation_id = self._leaf_launched_as_own_root_then_repaired()
        names_before = {path.name for path in leaf.iterdir()}

        plan = launch_step1_runs([leaf], only_repaired=True, scheduler="none")
        self.assertEqual(
            [(row["relative_path"], row["kind"], row["generation_id"]) for row in plan["planned"]],
            [(".", "repair-prepared", generation_id)],
        )
        self.assertEqual(plan["planned"][0]["historical_submissions"], 1)
        self.assertNotIn("_fingerprint", plan["planned"][0])
        self.assertIsNone(plan["batch_id"])
        self.assertEqual(plan["scheduler"]["mode"], "none")

        with mock_sbatch(5001) as calls:
            result = launch_step1_runs([leaf], only_repaired=True, execute=True, scheduler="none")
        self.assertEqual(result["mode"], "submitted")
        self.assertEqual([cwd for _command, cwd in calls], [str(leaf)])
        self.assertEqual(result["jobs"][0]["job_id"], "5001")
        self.assertTrue(result["batch_id"].startswith("b-"))

        # Nobody renamed or removed anything; only the TSV mirror is new.
        names_after = {path.name for path in leaf.iterdir()}
        self.assertLessEqual(names_before, names_after)
        self.assertEqual(names_after - names_before, {"step1_launch.tsv"})

    def test_historical_provenance_is_retained_and_current_generation_is_guarded(self) -> None:
        leaf, generation_id = self._leaf_launched_as_own_root_then_repaired()
        with mock_sbatch(5001):
            result = launch_step1_runs([leaf], only_repaired=True, execute=True, scheduler="none")
        batch_id = result["batch_id"]

        ledger = read_json(leaf / "step1_launch.json")
        self.assertEqual(ledger["schema_version"], 2)
        self.assertEqual(ledger["latest_batch_id"], batch_id)
        self.assertEqual(ledger["status"], "SUBMITTED")
        old = [row for row in ledger["runs"] if row.get("job_id") == "111"]
        self.assertEqual(len(old), 1)
        self.assertTrue(old[0]["legacy"])
        self.assertEqual(old[0]["status"], "SUBMITTED")
        self.assertTrue(old[0]["legacy_recorded_at"])
        new = [row for row in ledger["runs"] if row.get("generation_id") == generation_id]
        self.assertEqual(len(new), 1)
        self.assertEqual(
            (new[0]["status"], new[0]["job_id"], new[0]["generation"], new[0]["batch_id"], new[0]["kind"]),
            ("SUBMITTED", "5001", 1, batch_id, "repair-prepared"),
        )
        self.assertTrue(new[0]["submitted_at"])
        self.assertEqual(
            [(batch["batch_id"], batch["status"], batch["submitted"], batch["failed"]) for batch in ledger["batches"]],
            [(batch_id, "SUBMITTED", 1, 0)],
        )
        self.assertEqual(result["reports"], [str(leaf / "step1_launch.json"), str(leaf / "step1_launch.tsv")])

        tsv = (leaf / "step1_launch.tsv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(tsv), 3)  # header + old row + new row
        self.assertIn("111", tsv[1])
        self.assertIn("5001", tsv[2])
        self.assertIn(generation_id, tsv[2])

        record = read_json(leaf / REPAIR_RECORD)
        self.assertEqual(record["status"], "SUBMITTED")
        self.assertEqual(len(record["submissions"]), 1)
        submission = record["submissions"][0]
        self.assertEqual((submission["job_id"], submission["batch_id"]), ("5001", batch_id))
        self.assertEqual(submission["ledger"], str(leaf / "step1_launch.json"))
        self.assertEqual(submission["generation_id"], generation_id)

        pattern = rf"current generation {re.escape(generation_id)} already submitted \(job 5001\)"
        with self.assertRaisesRegex(SafetyError, pattern):
            launch_step1_runs([leaf], only_repaired=True, scheduler="none")
        # The ledger row alone (record rewritten as if the process died before marking it) still guards.
        _reset_record_to_prepared(leaf)
        with self.assertRaisesRegex(SafetyError, pattern):
            launch_step1_runs([leaf], only_repaired=True, scheduler="none")

    def test_several_leaves_as_roots_share_one_batch(self) -> None:
        # rescue_step1_conservative.sh passes every repaired leaf as its own root.
        first, first_id = self._leaf_launched_as_own_root_then_repaired()
        second = _idle_run(self.step1, "second")
        second_id = _write_segment_record(second, "repair")
        with mock_sbatch(6001) as calls:
            result = launch_step1_runs([first, second], only_repaired=True, execute=True, scheduler="none")
        self.assertEqual(len(calls), 2)
        for leaf, generation_id in ((first, first_id), (second, second_id)):
            ledger = read_json(leaf / "step1_launch.json")
            self.assertEqual(ledger["latest_batch_id"], result["batch_id"])
            self.assertEqual([row["generation_id"] for row in ledger["runs"] if not row.get("legacy")], [generation_id])
            # Each leaf ledger counts only the run it records, not the whole batch.
            self.assertEqual([(batch["planned"], batch["submitted"]) for batch in ledger["batches"]], [(1, 1)])

    # ------------------------------------------------------------------ #
    # (2): legacy repair record vs legacy launch ledger
    # ------------------------------------------------------------------ #

    def test_legacy_repair_prepared_after_legacy_launch_is_launchable(self) -> None:
        leaf = _idle_run(self.step1, "leaf")
        write_legacy_launch_ledger(leaf, [_legacy_leaf_row(leaf)], age_hours=48.0)
        stamp = utc_stamp()  # the repair archive is newer than the launch ledger
        write_legacy_repair_record(leaf, archive=str(leaf / ".interfaceforge" / "archive" / f"step1_repair_{stamp}"))
        generation_id = f"legacy-repair-g1-{stamp}"

        plan = launch_step1_runs([leaf], only_repaired=True, scheduler="none")
        self.assertEqual(
            [(row["kind"], row["generation_id"]) for row in plan["planned"]], [("repair-prepared", generation_id)]
        )

        with mock_sbatch(5101) as calls:
            launch_step1_runs([leaf], only_repaired=True, execute=True, scheduler="none")
        self.assertEqual(len(calls), 1)
        record = read_json(leaf / REPAIR_RECORD)
        self.assertEqual(record["status"], "SUBMITTED")
        self.assertEqual(record["legacy_generation_id"], generation_id)  # schema-1 record keeps no generation_id
        self.assertNotIn("generation_id", record)
        with self.assertRaisesRegex(SafetyError, rf"current generation {re.escape(generation_id)} already submitted"):
            launch_step1_runs([leaf], only_repaired=True, scheduler="none")

    def test_legacy_repair_prepared_before_legacy_launch_stays_blocked(self) -> None:
        # The legacy row is newer than the legacy repair: it IS the launch of that repair.
        leaf = _idle_run(self.step1, "leaf")
        stamp = _stamp_hours_ago(72.0)
        write_legacy_repair_record(leaf, archive=str(leaf / ".interfaceforge" / "archive" / f"step1_repair_{stamp}"))
        write_legacy_launch_ledger(leaf, [_legacy_leaf_row(leaf)], age_hours=1.0)
        before = tree_snapshot(self.root)
        pattern = rf"current generation legacy-repair-g1-{stamp} already submitted \(job 111\)"
        with self.assertRaisesRegex(SafetyError, pattern):
            launch_step1_runs([leaf], only_repaired=True, scheduler="none")
        self.assertEqual(tree_snapshot(self.root), before)

    # ------------------------------------------------------------------ #
    # (4): cross-root ledger matching
    # ------------------------------------------------------------------ #

    def _tree_with_repaired_leaf(self) -> tuple[Path, Path, str]:
        leaf = _idle_run(self.step1, "leaf")
        other = _idle_run(self.step1, "other")
        write_manifest(self.step1, [other])  # "other" is a gen-0 prepared run
        generation_id = _write_segment_record(leaf, "repair")
        return leaf, other, generation_id

    def test_launch_from_root_then_leaf_directly_is_refused(self) -> None:
        leaf, _other, generation_id = self._tree_with_repaired_leaf()
        with mock_sbatch(5201) as calls:
            launch_step1_runs([self.step1], only_repaired=True, execute=True, scheduler="none")
        self.assertEqual([cwd for _command, cwd in calls], [str(leaf)])
        self.assertFalse((leaf / "step1_launch.json").exists())
        self.assertEqual(read_json(self.step1 / "step1_launch.json")["runs"][0]["relative_path"], "leaf")

        pattern = rf"current generation {re.escape(generation_id)} already submitted \(job 5201\)"
        with self.assertRaisesRegex(SafetyError, pattern):
            launch_step1_runs([leaf], scheduler="none")
        _reset_record_to_prepared(leaf)  # only the Step1-root ledger row remains as evidence
        with self.assertRaisesRegex(SafetyError, pattern):
            launch_step1_runs([leaf], scheduler="none")

    def test_launch_leaf_directly_then_from_root_is_refused(self) -> None:
        leaf, _other, generation_id = self._tree_with_repaired_leaf()
        with mock_sbatch(5301):
            launch_step1_runs([leaf], execute=True, scheduler="none")
        self.assertFalse((self.step1 / "step1_launch.json").exists())

        for reset in (False, True):
            if reset:
                _reset_record_to_prepared(leaf)  # only the leaf ledger row remains as evidence
            plan = launch_step1_runs([self.step1], scheduler="none")
            self.assertEqual(_planned(plan), ["other"])
            self.assertEqual(_reasons(plan)["leaf"], f"current generation {generation_id} already submitted (job 5301)")

    def test_overlapping_roots_preflight_each_run_once(self) -> None:
        leaf, other, _generation_id = self._tree_with_repaired_leaf()
        plan = launch_step1_runs([self.step1, leaf, self.step1], scheduler="none")
        self.assertEqual(sorted(row["directory"] for row in plan["planned"]), sorted([str(leaf), str(other)]))
        self.assertEqual(len(plan["roots"]), 2)
        self.assertEqual(
            [row["reason"] for row in plan["skipped_runs"]], [f"already preflighted under root {self.step1}"]
        )

    # ------------------------------------------------------------------ #
    # (5): the scheduler guard
    # ------------------------------------------------------------------ #

    def _two_prepared_runs(self) -> tuple[Path, Path]:
        first = _idle_run(self.step1, "a_first")
        second = _idle_run(self.step1, "b_second")
        write_manifest(self.step1, [first, second])
        return first, second

    def test_active_slurm_run_is_skipped_and_never_submitted(self) -> None:
        busy, idle = self._two_prepared_runs()
        plan = launch_step1_runs([self.step1], scheduler=fake_guard({busy: "PENDING"}))
        self.assertEqual(_planned(plan), ["b_second"])
        self.assertEqual(_reasons(plan)["a_first"], "active in Slurm (job 9000 PENDING)")
        self.assertTrue(plan["scheduler"]["verified"])

        with mock_sbatch(5401) as calls:
            result = launch_step1_runs([self.step1], execute=True, scheduler=fake_guard({busy: "RUNNING"}))
        self.assertEqual([cwd for _command, cwd in calls], [str(idle)])
        self.assertEqual(_reasons(result)["a_first"], "active in Slurm (job 9000 RUNNING)")
        self.assertEqual([row["directory"] for row in read_json(self.step1 / "step1_launch.json")["runs"]], [str(idle)])

    def test_job_appearing_between_planning_and_sbatch_blocks_that_run(self) -> None:
        first, second = self._two_prepared_runs()
        guard = sequence_guard([{}, {second: "RUNNING"}])  # inactive at planning, active at submit time
        with mock_sbatch(5501) as calls, self.assertRaisesRegex(SafetyError, "active Slurm job") as caught:
            launch_step1_runs([self.step1], execute=True, scheduler=guard)
        self.assertEqual([cwd for _command, cwd in calls], [str(first)])  # never sbatch'ed the busy run
        self.assertIn(str(self.step1 / "step1_launch.json"), str(caught.exception))
        rows = read_json(self.step1 / "step1_launch.json")["runs"]
        self.assertEqual(
            [(row["relative_path"], row["status"], row["job_id"]) for row in rows],
            [("a_first", "SUBMITTED", "5501"), ("b_second", "FAILED", "")],
        )
        self.assertIn("Refusing to mutate", rows[1]["detail"])

    def test_leaf_root_that_became_active_is_not_written_to(self) -> None:
        leaf = _idle_run(self.step1, "leaf")
        _write_segment_record(leaf, "repair")
        before = tree_snapshot(leaf)
        # Already active at planning time: skipped by the dry run and by execute alike.
        for execute in (False, True):
            with mock_sbatch() as calls, self.assertRaisesRegex(SafetyError, r"active in Slurm \(job 9000 RUNNING\)"):
                launch_step1_runs([leaf], execute=execute, scheduler=fake_guard({leaf: "RUNNING"}))
            self.assertEqual(calls, [])
        self.assertEqual(tree_snapshot(leaf), before)
        # Inactive at planning, active at submit time.
        with mock_sbatch() as calls, self.assertRaisesRegex(SafetyError, "active Slurm job"):
            launch_step1_runs([leaf], execute=True, scheduler=sequence_guard([{}, {leaf: "PENDING"}]))
        self.assertEqual(calls, [])
        self.assertEqual(tree_snapshot(leaf), before)  # no ledger written into the active WorkDir

    def test_run_changed_between_planning_and_sbatch_is_refused(self) -> None:
        first, _second = self._two_prepared_runs()
        calls_seen = {"count": 0}

        def factory() -> SchedulerSnapshot:
            calls_seen["count"] += 1
            if calls_seen["count"] == 2:  # the pre-sbatch re-check: someone edits the INCAR meanwhile
                with (first / "INCAR").open("a", encoding="utf-8") as handle:
                    handle.write("NELM = 200\n")
            return SchedulerSnapshot("slurm", "slurm", True, "fixture", utc_now_iso(), [], time.monotonic())

        guard = SchedulerGuard("slurm", recheck_seconds=0.0, snapshot_factory=factory)
        with mock_sbatch() as calls, self.assertRaisesRegex(SafetyError, "changed since planning"):
            launch_step1_runs([self.step1], execute=True, scheduler=guard)
        self.assertEqual(calls, [])
        rows = read_json(self.step1 / "step1_launch.json")["runs"]
        self.assertEqual([(row["relative_path"], row["status"]) for row in rows], [("a_first", "FAILED")])

    def test_unqueryable_slurm_refuses_before_anything(self) -> None:
        self._two_prepared_runs()
        before = tree_snapshot(self.root)
        with patch("interfaceforge.step1_scheduler.subprocess.run", side_effect=FileNotFoundError("squeue")):
            with self.assertRaisesRegex(SafetyError, "could not query Slurm"):
                launch_step1_runs([self.step1], scheduler="slurm")
        self.assertEqual(tree_snapshot(self.root), before)

    # ------------------------------------------------------------------ #
    # (6): mid-batch failure
    # ------------------------------------------------------------------ #

    def test_mid_batch_sbatch_failure_is_recorded_incrementally(self) -> None:
        first = _idle_run(self.step1, "a_first")
        second = _idle_run(self.step1, "b_second")
        first_id = _write_segment_record(first, "repair")
        _write_segment_record(second, "repair")
        second_record = (second / REPAIR_RECORD).read_bytes()
        ledger_path = self.step1 / "step1_launch.json"
        seen_before_second: list[list[tuple[str, str]]] = []

        def fake_sbatch(command: list[str], cwd: object = None, **_kwargs: object) -> Mock:
            if Path(str(cwd)) == first:
                result = Mock()
                result.stdout = "Submitted batch job 7001\n"
                return result
            payload = read_json(ledger_path)  # the first job must already be on record
            seen_before_second.append([(row["status"], row["job_id"]) for row in payload.get("runs", [])])
            raise subprocess.CalledProcessError(1, command, stderr="sbatch: error: QOSMaxSubmitJobPerUserLimit")

        with patch("interfaceforge.vasp.subprocess.run", side_effect=fake_sbatch):
            with self.assertRaises(SafetyError) as caught:
                launch_step1_runs([self.step1], only_repaired=True, execute=True, scheduler="none")
        message = str(caught.exception)
        self.assertIn(str(ledger_path), message)
        self.assertIn("a_first (job 7001)", message)
        self.assertEqual(seen_before_second, [[("SUBMITTED", "7001")]])

        ledger = read_json(ledger_path)
        self.assertEqual(
            [(row["relative_path"], row["status"], row["job_id"]) for row in ledger["runs"]],
            [("a_first", "SUBMITTED", "7001"), ("b_second", "FAILED", "")],
        )
        self.assertIn("non-zero exit status", ledger["runs"][1]["detail"])
        self.assertEqual(ledger["status"], "FAILED")
        self.assertEqual(
            [(batch["status"], batch["submitted"], batch["failed"], batch["planned"]) for batch in ledger["batches"]],
            [("FAILED", 1, 1, 2)],
        )
        first_record = read_json(first / REPAIR_RECORD)
        self.assertEqual(first_record["status"], "SUBMITTED")
        self.assertEqual(first_record["submissions"][0]["job_id"], "7001")
        self.assertEqual(first_record["generation_id"], first_id)
        self.assertEqual((second / REPAIR_RECORD).read_bytes(), second_record)  # untouched

        # The failed run is still launchable (its generation was never submitted); the first is not.
        plan = launch_step1_runs([self.step1], only_repaired=True, scheduler="none")
        self.assertEqual(_planned(plan), ["b_second"])
        self.assertIn("already submitted (job 7001)", _reasons(plan)["a_first"])

        # Relaunching appends a second batch (even within the same second); the history stays cumulative.
        with mock_sbatch(7101) as calls:
            retry = launch_step1_runs([self.step1], only_repaired=True, execute=True, scheduler="none")
        self.assertEqual([cwd for _command, cwd in calls], [str(second)])
        ledger = read_json(ledger_path)
        self.assertEqual(
            [(row["relative_path"], row["status"], row["job_id"]) for row in ledger["runs"]],
            [("a_first", "SUBMITTED", "7001"), ("b_second", "FAILED", ""), ("b_second", "SUBMITTED", "7101")],
        )
        self.assertEqual(
            [(batch["status"], batch["submitted"], batch["failed"]) for batch in ledger["batches"]],
            [("FAILED", 1, 1), ("SUBMITTED", 1, 0)],
        )
        self.assertNotEqual(ledger["batches"][0]["batch_id"], retry["batch_id"])
        self.assertEqual((ledger["latest_batch_id"], ledger["status"]), (retry["batch_id"], "SUBMITTED"))
        self.assertEqual(len((self.step1 / "step1_launch.tsv").read_text(encoding="utf-8").splitlines()), 4)

    # ------------------------------------------------------------------ #
    # (7): dry run performs zero mutation
    # ------------------------------------------------------------------ #

    def test_dry_run_is_zero_mutation_and_never_upgrades_schema1_ledgers(self) -> None:
        fresh = _idle_run(self.step1, "a_fresh")
        busy = _idle_run(self.step1, "b_busy")
        leaf = _idle_run(self.step1, "c_leaf")
        resumed = _idle_run(self.step1, "d_resumed")
        write_manifest(self.step1, [fresh, busy])
        old_row = dict(_legacy_leaf_row(self.step1 / "z_old", job_id="42"), relative_path="z_old", root=str(self.step1))
        write_legacy_launch_ledger(self.step1, [old_row], age_hours=72.0)
        write_legacy_launch_ledger(leaf, [_legacy_leaf_row(leaf)], age_hours=30.0)
        _write_segment_record(leaf, "repair")
        _write_segment_record(resumed, "resume")
        before = tree_snapshot(self.root)

        plan = launch_step1_runs([self.step1], scheduler=fake_guard({busy: "PENDING"}))
        self.assertEqual(_planned(plan), ["a_fresh", "c_leaf", "d_resumed"])
        self.assertEqual(launch_step1_runs([leaf], only_repaired=True, scheduler="none")["runs"], 1)
        with self.assertRaises(SafetyError):  # nothing launchable is also read-only
            launch_step1_runs([self.step1], runs=[busy], scheduler=fake_guard({busy: "PENDING"}))

        self.assertEqual(tree_snapshot(self.root), before)
        self.assertEqual(read_json(self.step1 / "step1_launch.json")["schema_version"], 1)
        self.assertEqual(read_json(leaf / "step1_launch.json")["schema_version"], 1)

    # ------------------------------------------------------------------ #
    # (8): resume-prepared records
    # ------------------------------------------------------------------ #

    def test_resume_prepared_records_and_kind_filters(self) -> None:
        fresh = _idle_run(self.step1, "a_fresh")
        repaired = _idle_run(self.step1, "b_repaired")
        resumed = _idle_run(self.step1, "c_resumed")
        write_manifest(self.step1, [fresh])
        _write_segment_record(repaired, "repair")
        resume_id = _write_segment_record(resumed, "resume")

        plan = launch_step1_runs([self.step1], scheduler="none")
        self.assertEqual(
            {row["relative_path"]: row["kind"] for row in plan["planned"]},
            {"a_fresh": "prepared", "b_repaired": "repair-prepared", "c_resumed": "resume-prepared"},
        )
        only_repaired = launch_step1_runs([self.step1], only_repaired=True, scheduler="none")
        self.assertEqual(_planned(only_repaired), ["b_repaired"])
        self.assertEqual(_reasons(only_repaired)["c_resumed"], "not a repaired run (--only-repaired)")
        only_resumed = launch_step1_runs([self.step1], only_resumed=True, scheduler="none")
        self.assertEqual(_planned(only_resumed), ["c_resumed"])
        self.assertEqual(_reasons(only_resumed)["b_repaired"], "not a resumed run (--only-resumed)")
        self.assertEqual(_reasons(only_resumed)["a_fresh"], "not a resumed run (--only-resumed)")
        both = launch_step1_runs([self.step1], only_repaired=True, only_resumed=True, scheduler="none")
        self.assertEqual(_planned(both), ["b_repaired", "c_resumed"])

        with mock_sbatch(5601) as calls:
            result = launch_step1_runs([self.step1], only_resumed=True, execute=True, scheduler="none")
        self.assertEqual([cwd for _command, cwd in calls], [str(resumed)])
        self.assertEqual(result["jobs"][0]["generation_id"], resume_id)
        record = read_json(resumed / RESUME_RECORD)
        self.assertEqual(record["status"], "SUBMITTED")
        self.assertEqual(record["submissions"][0]["job_id"], "5601")

    # ------------------------------------------------------------------ #
    # Record states, fail-closed ledgers, runs= restriction
    # ------------------------------------------------------------------ #

    def test_unreadable_and_conflicting_records_are_never_launched(self) -> None:
        unreadable = _idle_run(self.step1, "a_unreadable")
        (unreadable / REPAIR_RECORD).write_text("{not json", encoding="utf-8")
        conflict = _idle_run(self.step1, "b_conflict")
        write_legacy_repair_record(conflict)
        _write_segment_record(conflict, "resume")
        good = _idle_run(self.step1, "c_good")
        _write_segment_record(good, "repair")

        plan = launch_step1_runs([self.step1], scheduler="none")
        self.assertEqual(_planned(plan), ["c_good"])
        reasons = _reasons(plan)
        self.assertEqual(reasons["a_unreadable"], "step1_repair.json status is 'UNREADABLE', not PREPARED")
        self.assertRegex(reasons["b_conflict"], r"status is 'CONFLICT', not PREPARED \(both ")

    def test_record_marked_submitted_without_ledger_row_is_guarded(self) -> None:
        leaf = _idle_run(self.step1, "leaf")
        generation_id = _write_segment_record(leaf, "repair")
        mark_record_submitted(leaf, generation_id, {"job_id": "777", "batch_id": "b-x", "launcher": "runvasp.sh"})
        pattern = rf"current generation {re.escape(generation_id)} already submitted \(job 777\)"
        with self.assertRaisesRegex(SafetyError, pattern):
            launch_step1_runs([leaf], scheduler="none")

    def test_unreadable_root_ledger_fails_closed(self) -> None:
        leaf = _idle_run(self.step1, "leaf")
        _write_segment_record(leaf, "repair")
        (self.step1 / "step1_launch.json").write_text('{"runs": [{"status": "SUBMI', encoding="utf-8")
        before = tree_snapshot(self.root)
        with self.assertRaisesRegex(SafetyError, r"already submitted \(job unknown\): launch ledger .* is unreadable"):
            launch_step1_runs([self.step1], scheduler="none")
        self.assertEqual(tree_snapshot(self.root), before)

    def test_runs_restriction(self) -> None:
        first, second = self._two_prepared_runs()
        plan = launch_step1_runs([self.step1], runs=[second], scheduler="none")
        self.assertEqual(_planned(plan), ["b_second"])
        self.assertEqual(plan["skipped_runs"], [])

        outside = self.root / "elsewhere"
        outside.mkdir()
        with self.assertRaisesRegex(SafetyError, "not under any of the given Step1 roots"):
            launch_step1_runs([self.step1], runs=[outside], scheduler="none")
        (self.step1 / "notes").mkdir()
        with self.assertRaisesRegex(SafetyError, "Not a Step1 run directory"):
            launch_step1_runs([self.step1], runs=[self.step1 / "notes"], scheduler="none")
        with self.assertRaisesRegex(SafetyError, "runs= is empty"):
            launch_step1_runs([self.step1], runs=[], scheduler="none")

        with mock_sbatch(5701) as calls:
            result = launch_step1_runs([self.step1], runs=[str(second)], execute=True, scheduler="none")
        self.assertEqual([cwd for _command, cwd in calls], [str(second)])
        self.assertEqual(result["submitted"], 1)
        self.assertEqual(launch_step1_runs([self.step1], scheduler="none")["planned"][0]["directory"], str(first))

    # ------------------------------------------------------------------ #
    # Backward compatibility, batch identity, interrupted mutations
    # ------------------------------------------------------------------ #

    def test_schema1_root_ledger_still_blocks_its_runs_and_is_upgraded_on_execute(self) -> None:
        launched, fresh = self._two_prepared_runs()
        old_row = dict(_legacy_leaf_row(launched, job_id="42"), relative_path="a_first", root=str(self.step1))
        write_legacy_launch_ledger(self.step1, [old_row], age_hours=5.0)

        plan = launch_step1_runs([self.step1], scheduler="none")
        self.assertEqual(_planned(plan), ["b_second"])
        self.assertEqual(_reasons(plan)["a_first"], "current generation g0-prepare already submitted (job 42)")
        self.assertEqual(read_json(self.step1 / "step1_launch.json")["schema_version"], 1)

        with mock_sbatch(5801) as calls:
            result = launch_step1_runs([self.step1], execute=True, scheduler="none")
        self.assertEqual([cwd for _command, cwd in calls], [str(fresh)])
        ledger = read_json(self.step1 / "step1_launch.json")
        self.assertEqual((ledger["schema_version"], ledger["legacy_import"]["schema_version"]), (2, 1))
        self.assertEqual(
            [
                (row["relative_path"], row["job_id"], row.get("legacy", False), row.get("generation_id"))
                for row in ledger["runs"]
            ],
            [("a_first", "42", True, None), ("b_second", "5801", False, "g0-prepare")],
        )
        old, new = ledger["runs"]
        self.assertEqual((new["generation"], new["batch_id"]), (0, result["batch_id"]))
        # The legacy row keeps its original (pre-upgrade) time, about 5 h before the new submission.
        recorded = datetime.fromisoformat(old["legacy_recorded_at"])
        self.assertLess(recorded, datetime.fromisoformat(new["submitted_at"]) - timedelta(hours=4))
        self.assertEqual(len((self.step1 / "step1_launch.tsv").read_text(encoding="utf-8").splitlines()), 3)
        with self.assertRaisesRegex(SafetyError, "No launchable Step1 runs"):
            launch_step1_runs([self.step1], scheduler="none")

    def test_launches_within_one_second_get_distinct_batches(self) -> None:
        first, second = self._two_prepared_runs()
        with patch("interfaceforge.step1_launch.utc_stamp", return_value="20260923T101500Z"), mock_sbatch(5901):
            one = launch_step1_runs([self.step1], runs=[first], execute=True, scheduler="none")
            two = launch_step1_runs([self.step1], runs=[second], execute=True, scheduler="none")
        self.assertEqual((one["batch_id"], two["batch_id"]), ("b-20260923T101500Z", "b-20260923T101500Z-2"))
        ledger = read_json(self.step1 / "step1_launch.json")
        self.assertEqual(
            [(batch["batch_id"], batch["planned"], batch["submitted"], batch["status"]) for batch in ledger["batches"]],
            [("b-20260923T101500Z", 1, 1, "SUBMITTED"), ("b-20260923T101500Z-2", 1, 1, "SUBMITTED")],
        )
        self.assertEqual([row["batch_id"] for row in ledger["runs"]], [one["batch_id"], two["batch_id"]])

    def test_interrupted_recovery_mutation_is_never_launched(self) -> None:
        leaf = _idle_run(self.step1, "leaf")
        generation_id = _write_segment_record(leaf, "repair")
        archive = archive_step1_state(leaf, "step1_repair_g2")  # manifest stays IN_PROGRESS: the repair "died"
        before = tree_snapshot(self.root)
        with self.assertRaisesRegex(SafetyError, r"leaf: interrupted recovery mutation \(archive .+ is IN_PROGRESS\)"):
            launch_step1_runs([self.step1], scheduler="none")
        self.assertEqual(tree_snapshot(self.root), before)

        finalize_archive(archive, generation_id=generation_id)
        self.assertEqual(_planned(launch_step1_runs([self.step1], scheduler="none")), ["leaf"])

        # A recovery mutation that starts between planning and sbatch is caught by the re-check.
        refreshes = {"count": 0}

        def factory() -> SchedulerSnapshot:
            refreshes["count"] += 1
            if refreshes["count"] == 2:  # the pre-sbatch re-check
                archive_step1_state(leaf, "step1_resume_g2")
            return SchedulerSnapshot("slurm", "slurm", True, "fixture", utc_now_iso(), [], time.monotonic())

        guard = SchedulerGuard("slurm", recheck_seconds=0.0, snapshot_factory=factory)
        with mock_sbatch() as calls, self.assertRaisesRegex(SafetyError, "interrupted recovery mutation"):
            launch_step1_runs([self.step1], execute=True, scheduler=guard)
        self.assertEqual(calls, [])
        self.assertEqual(read_json(leaf / REPAIR_RECORD)["status"], "PREPARED")
        rows = read_json(self.step1 / "step1_launch.json")["runs"]
        self.assertEqual([(row["relative_path"], row["status"]) for row in rows], [("leaf", "FAILED")])


if __name__ == "__main__":
    unittest.main()
