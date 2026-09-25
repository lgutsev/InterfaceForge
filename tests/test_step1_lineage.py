"""Tests for Step1 generations, launch ledgers and archives (``interfaceforge.step1_lineage``).

Every run, record and ledger here is synthetic (``tests/step1_fixtures.py``);
nothing calls Slurm, sbatch or VASP.  The legacy repair-of-repair case
reproduces the real NiO/OH50 accounting: 16 accepted steps at 1.0 fs, then
52 more at 0.5 fs -> cumulative prefix 68, remaining 332.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

from interfaceforge import step1_lineage
from interfaceforge.errors import SafetyError
from interfaceforge.step1_lineage import (
    ARCHIVE_MANIFEST,
    FINGERPRINT_FILES,
    GEN0_ID,
    LAUNCH_LEDGER,
    LAUNCH_TSV,
    LAUNCH_TSV_COLUMNS,
    LEGACY_REPAIR_KEYS,
    REPAIR_RECORD,
    RESUME_KEYS,
    RESUME_RECORD,
    SCHEDULE_KEYS,
    Generation,
    accepted_ps,
    append_launch_rows,
    archive_step1_state,
    atomic_write_json,
    atomic_write_text,
    build_segment_record,
    current_generation,
    finalize_archive,
    format_temperature,
    incar_schedule,
    interrupted_archive,
    ledger_paths_for,
    load_ledger_rows,
    mark_record_submitted,
    new_generation_id,
    read_json,
    retire_current_records,
    rows_for_run,
    run_fingerprint,
    schedule_temperature,
    seal_launch_rows,
    submission_state,
    unreadable_ledgers,
    utc_now_iso,
    utc_stamp,
)
from interfaceforge.vasp import parse_incar

_TESTS = str(Path(__file__).resolve().parent)
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)
from step1_fixtures import (  # noqa: E402
    incar_text,
    tree_snapshot,
    write_legacy_launch_ledger,
    write_legacy_repair_record,
    write_manifest,
    write_step1_run,
)

FIRST_ARCHIVE = "step1_repair_20260910T080000Z"
SECOND_ARCHIVE = "step1_repair_20260920T101500Z"


def _stamp(hours_ago: float) -> str:
    moment = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return moment.strftime("%Y%m%dT%H%M%SZ")


def _epoch(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def _incar(folder: Path, **kwargs: Any) -> dict[str, str]:
    path = folder / "INCAR.test"
    path.write_text(incar_text(**kwargs), encoding="utf-8")
    return parse_incar(path)


def _launch_row(run: Path, root: Path, *, job_id: str, **fields: Any) -> dict[str, Any]:
    relative = run.relative_to(root).as_posix() if run != root else "."
    row = {
        "status": "SUBMITTED",
        "job_id": job_id,
        "kind": "repair-prepared",
        "root": str(root),
        "relative_path": relative,
        "directory": str(run),
        "launcher": "runvasp.sh",
        "notes": "",
        "detail": "",
    }
    row.update(fields)
    return row


def _legacy_record(run: Path, **fields: Any) -> dict[str, Any]:
    write_legacy_repair_record(run, **fields)
    return json.loads((run / REPAIR_RECORD).read_text(encoding="utf-8"))


def _real_case_run(
    root: Path,
    name: str = "OH50_run",
    *,
    foreign_archive_paths: bool = False,
    first_archive: str = FIRST_ARCHIVE,
    second_archive: str = SECOND_ARCHIVE,
) -> Path:
    """Legacy repair-of-repair exactly as the pre-generation code left it on disk.

    Repair 1 kept 16 steps of the original 1.0 fs / 300 K segment; repair 2
    kept 52 steps of repair 1's 0.5 fs 100->300 K segment.  Repair 2's archive
    holds repair 1's record (archive_run copies step1_repair.json).
    """

    run = write_step1_run(root, name, nsw=332, potim=0.5, tebeg=100.0, teend=300.0, steps=0, contcar=None)
    archives = run / ".interfaceforge" / "archive"
    first = archives / first_archive
    second = archives / second_archive
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "INCAR").write_text(incar_text(nsw=400, potim=1.0, tebeg=300.0, teend=300.0), encoding="utf-8")
    (second / "INCAR").write_text(incar_text(nsw=384, potim=0.5, tebeg=100.0, teend=300.0), encoding="utf-8")

    def archive_value(path: Path) -> str:
        # A tree copied off the cluster keeps the cluster's absolute paths.
        if foreign_archive_paths:
            return f"/cluster/scratch/Step1/{name}/.interfaceforge/archive/{path.name}"
        return str(path)

    write_legacy_repair_record(
        run,
        safe_prefix_steps=16,
        safe_segment_steps=16,
        previous_safe_prefix_steps=0,
        rewind_frame=4,
        repair_nsw=384,
        original_potim_fs=1.0,
        repair_potim_fs=0.5,
        archive=archive_value(first),
    )
    shutil.copy2(run / REPAIR_RECORD, second / REPAIR_RECORD)
    write_legacy_repair_record(
        run,
        safe_prefix_steps=68,
        safe_segment_steps=52,
        previous_safe_prefix_steps=16,
        rewind_frame=13,
        repair_nsw=332,
        original_potim_fs=0.5,
        repair_potim_fs=0.5,
        archive=archive_value(second),
    )
    return run


class LineageTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()


class ScheduleTests(LineageTestCase):
    def test_constant_temperature_is_flat(self) -> None:
        for step in (0, 1, 200, 400, 1000):
            self.assertEqual(schedule_temperature(300.0, 300.0, 400, step), 300.0)
        self.assertEqual(schedule_temperature(300.0, None, 400, 250), 300.0)

    def test_ramp_is_linear_in_ionic_step(self) -> None:
        self.assertEqual(schedule_temperature(100.0, 300.0, 400, 0), 100.0)
        self.assertEqual(schedule_temperature(100.0, 300.0, 400, 100), 150.0)
        self.assertEqual(schedule_temperature(100.0, 300.0, 400, 200), 200.0)
        self.assertEqual(schedule_temperature(100.0, 300.0, 400, 400), 300.0)
        self.assertAlmostEqual(schedule_temperature(100.0, 300.0, 388, 52), 126.804, places=3)

    def test_ramp_clamps_outside_the_segment(self) -> None:
        self.assertEqual(schedule_temperature(100.0, 300.0, 400, -5), 100.0)
        self.assertEqual(schedule_temperature(100.0, 300.0, 400, 401), 300.0)
        self.assertEqual(schedule_temperature(100.0, 300.0, 400, 10_000), 300.0)

    def test_falsy_nsw_returns_the_endpoint(self) -> None:
        self.assertEqual(schedule_temperature(100.0, 300.0, 0, 5), 300.0)
        self.assertEqual(schedule_temperature(100.0, 300.0, None, 5), 300.0)

    def test_format_temperature(self) -> None:
        self.assertEqual(format_temperature(300.0), "300")
        self.assertEqual(format_temperature(116.5), "116.5")
        self.assertEqual(format_temperature(126.80412), "126.8")
        self.assertEqual(format_temperature(127.08333), "127.08")

    def test_incar_schedule_velocity_rescale_defaults(self) -> None:
        schedule = incar_schedule(_incar(self.root))
        self.assertEqual(
            schedule,
            {
                "tebeg_k": 300.0,
                "teend_k": 300.0,
                "teend_explicit": True,
                "nsw": 400,
                "potim_fs": 1.0,
                "nblock": 4,
                "thermostat": "velocity-rescale (SMASS=-1)",
                "ramp": False,
            },
        )

    def test_incar_schedule_thermostat_detection(self) -> None:
        langevin = incar_schedule(_incar(self.root, smass=None, extra={"MDALGO": 3, "LANGEVIN_GAMMA": "10 10"}))
        self.assertEqual(langevin["thermostat"], "langevin (MDALGO=3)")
        # MDALGO=3 wins even when a stale SMASS=-1 is still present.
        both = incar_schedule(_incar(self.root, smass=-1, extra={"MDALGO": 3}))
        self.assertEqual(both["thermostat"], "langevin (MDALGO=3)")
        self.assertEqual(incar_schedule(_incar(self.root, smass=0))["thermostat"], "nose (SMASS=0)")
        self.assertEqual(
            incar_schedule(_incar(self.root, smass=None, extra={"SMASS": "0.5"}))["thermostat"], "nose (SMASS=0.5)"
        )
        self.assertEqual(incar_schedule(_incar(self.root, smass=None))["thermostat"], "unknown")
        self.assertEqual(incar_schedule(_incar(self.root, smass=None, extra={"SMASS": -3}))["thermostat"], "unknown")

    def test_incar_schedule_ramp_and_missing_teend(self) -> None:
        ramp = incar_schedule(_incar(self.root, tebeg=100.0, teend=300.0))
        self.assertTrue(ramp["ramp"])
        self.assertEqual((ramp["tebeg_k"], ramp["teend_k"]), (100.0, 300.0))
        constant = incar_schedule(_incar(self.root, tebeg=100.0, teend=None))
        self.assertEqual(constant["teend_k"], 100.0)
        self.assertFalse(constant["teend_explicit"])
        self.assertFalse(constant["ramp"])
        self.assertEqual(incar_schedule({"NSW": "10"})["nblock"], 1)
        self.assertIsNone(incar_schedule({})["potim_fs"])


class GenerationTests(LineageTestCase):
    def test_generation_zero_defaults(self) -> None:
        run = write_step1_run(self.root, "run")
        generation = current_generation(run)
        self.assertEqual(generation.generation, 0)
        self.assertEqual(generation.generation_id, GEN0_ID)
        self.assertEqual(generation.kind, "original")
        self.assertTrue(generation.legacy)
        self.assertIsNone(generation.record_path)
        self.assertEqual(generation.record, {})
        self.assertIsNone(generation.prepared_at)
        self.assertIsNone(generation.prepared_epoch)
        self.assertEqual(generation.original_nsw, 400)
        self.assertEqual(generation.accepted_prefix_steps, 0)
        self.assertEqual(generation.segment_nsw, 400)
        self.assertEqual(generation.segment_potim_fs, 1.0)
        self.assertEqual(generation.ledger, [])
        self.assertTrue(generation.ledger_exact)
        self.assertIsNone(generation.status)
        self.assertEqual(generation.submissions, [])

    def test_new_generation_id_and_accepted_ps(self) -> None:
        self.assertEqual(new_generation_id("repair", 2, "20260920T101500Z"), "g2-repair-20260920T101500Z")
        self.assertRegex(new_generation_id("resume", 1), r"^g1-resume-\d{8}T\d{6}Z$")
        self.assertEqual(accepted_ps([]), 0.0)
        self.assertAlmostEqual(accepted_ps([{"steps": 16, "potim_fs": 1.0}, {"steps": 52, "potim_fs": 0.5}]), 0.042)
        self.assertIsNone(accepted_ps([{"steps": 16, "potim_fs": None}, {"steps": 52, "potim_fs": 0.5}]))

    def test_new_repair_record_round_trip(self) -> None:
        run = write_step1_run(self.root, "run", tebeg=100.0, teend=300.0)
        parent = current_generation(run)
        gid = new_generation_id("repair", 1, "20260923T101500Z")
        segments = [
            {
                "generation": 0,
                "generation_id": GEN0_ID,
                "kind": "original",
                "steps": 16,
                "potim_fs": 1.0,
                "tebeg_k": 100.0,
                "teend_k": 108.0,
                "restart_source": "XDATCAR frame 4",
            }
        ]
        # A legacy-shaped plan dict as extra: its status/archive must not leak.
        extra = {
            "status": "READY",
            "archive": None,
            "safe_segment_steps": 16,
            "rewind_frame": 4,
            "original_potim_fs": 1.0,
            "repair_algo": "Normal",
            "repair_electronic": {"EDIFF": "1E-5", "NELM": "120", "NELMIN": "6"},
            "repair_ramp_from_k": 100.0,
            "repair_precondition": True,
            "source": "XDATCAR",
            "diagnostic": {"unstable": True},
            "age_hours": 10.0,
        }
        record = _build_repair(run, parent, gid, segments, extra)
        self.assertEqual(record["format"], "interfaceforge-step1-repair")
        self.assertEqual(record["schema_version"], 2)
        self.assertEqual(record["status"], "PREPARED")
        self.assertEqual(record["archive"], "archive-dir")
        self.assertEqual(record["parent_generation_id"], GEN0_ID)
        for key in LEGACY_REPAIR_KEYS:
            self.assertIn(key, record)
        self.assertEqual(record["safe_prefix_steps"], 16)
        self.assertEqual(record["previous_safe_prefix_steps"], 0)
        self.assertEqual(record["repair_nsw"], 384)
        self.assertEqual(record["repair_potim_fs"], 0.5)
        self.assertIsNone(record["repair_langevin_gamma"])
        self.assertAlmostEqual(record["accepted_ps"], 0.016)
        self.assertAlmostEqual(record["accepted_segments"][0]["ps"], 0.016)
        self.assertEqual(sorted(record["segment_schedule"]), sorted(SCHEDULE_KEYS))
        self.assertEqual(record["submissions"], [])

        atomic_write_json(run / REPAIR_RECORD, record)
        generation = current_generation(run)
        self.assertEqual(generation.generation, 1)
        self.assertEqual(generation.generation_id, gid)
        self.assertEqual(generation.kind, "repair")
        self.assertFalse(generation.legacy)
        self.assertEqual(generation.record_path, run / REPAIR_RECORD)
        self.assertEqual(generation.prepared_at, "2026-09-23T10:15:00+00:00")
        self.assertEqual(generation.prepared_epoch, _epoch("2026-09-23T10:15:00+00:00"))
        self.assertEqual(generation.original_nsw, 400)
        self.assertEqual(generation.accepted_prefix_steps, 16)
        self.assertEqual(generation.segment_nsw, 384)
        self.assertEqual(generation.segment_potim_fs, 0.5)
        self.assertEqual(generation.ledger, record["accepted_segments"])
        self.assertTrue(generation.ledger_exact)
        self.assertEqual(generation.status, "PREPARED")
        self.assertEqual(generation.submissions, [])

    def test_resume_record_keys_and_newer_generation_wins(self) -> None:
        run = write_step1_run(self.root, "run", tebeg=100.0, teend=300.0)
        repair = _build_repair(run, current_generation(run), "g1-repair-20260923T101500Z", [], {})
        repair["accepted_segments"] = [{"generation": 0, "kind": "original", "steps": 16, "potim_fs": 1.0}]
        atomic_write_json(run / REPAIR_RECORD, repair)
        parent = current_generation(run)
        resume = _build_resume(run, parent, "g2-resume-20260923T111500Z")
        self.assertEqual(resume["format"], "interfaceforge-step1-resume")
        self.assertEqual(resume["schema_version"], 1)
        self.assertEqual(resume["parent_generation_id"], "g1-repair-20260923T101500Z")
        self.assertEqual(resume["resume_nsw"], 351)
        self.assertEqual(resume["segment_accepted_steps"], 33)
        for key in RESUME_KEYS:
            self.assertIn(key, resume)
        # Hand-edit scenario: both records present -> larger generation wins.
        atomic_write_json(run / RESUME_RECORD, resume)
        generation = current_generation(run)
        self.assertEqual(generation.generation, 2)
        self.assertEqual(generation.kind, "resume")
        self.assertEqual(generation.segment_nsw, 351)
        self.assertEqual(generation.accepted_prefix_steps, 49)
        # Consistent pair (larger generation prepared later): no conflict flagged.
        self.assertIsNone(generation.conflict)
        self.assertEqual(generation.status, "PREPARED")

    def test_equal_generations_newer_prepared_at_wins(self) -> None:
        run = write_step1_run(self.root, "run")
        parent = current_generation(run)
        repair = _build_repair(run, parent, "g1-repair-20260923T101500Z", [], {})
        resume = _build_resume(run, parent, "g1-resume-20260923T111500Z")  # prepared one hour later
        atomic_write_json(run / REPAIR_RECORD, repair)
        atomic_write_json(run / RESUME_RECORD, resume)
        old = datetime.now(timezone.utc).timestamp() - 86400.0
        os.utime(run / RESUME_RECORD, (old, old))  # older mtime must not beat a newer prepared_at
        generation = current_generation(run)
        self.assertEqual(generation.generation, 1)
        self.assertEqual(generation.generation_id, "g1-resume-20260923T111500Z")
        self.assertEqual(generation.kind, "resume")

        # Same generation and prepared_at: the newer mtime decides.
        resume["prepared_at"] = repair["prepared_at"]
        atomic_write_json(run / RESUME_RECORD, resume)
        os.utime(run / RESUME_RECORD, (old, old))
        self.assertEqual(current_generation(run).generation_id, "g1-repair-20260923T101500Z")

    def test_build_segment_record_rejects_unknown_kind(self) -> None:
        run = write_step1_run(self.root, "run")
        with self.assertRaises(ValueError):
            build_segment_record(
                "original",
                run=run,
                generation=1,
                generation_id="g1-original-x",
                parent=current_generation(run),
                prepared_at=utc_now_iso(),
                original_nsw=400,
                accepted_prefix_steps=0,
                accepted_segments=[],
                ledger_exact=True,
                segment_nsw=400,
                segment_potim_fs=1.0,
                segment_schedule={},
                archive="a",
                extra={},
            )

    def test_unreadable_record_is_not_generation_zero(self) -> None:
        run = write_step1_run(self.root, "run")
        (run / REPAIR_RECORD).write_text("{not json", encoding="utf-8")
        generation = current_generation(run)
        self.assertEqual(generation.status, "UNREADABLE")
        self.assertGreaterEqual(generation.generation, 1)
        self.assertEqual(generation.record_path, run / REPAIR_RECORD)
        self.assertFalse(generation.ledger_exact)

    def test_repair_record_defaults_follow_a_parent_with_an_accepted_prefix(self) -> None:
        # Schema-1 meaning of the defaulted keys on a repair of a repair (16 -> +52 -> 68 / 332).
        run = write_step1_run(self.root, "run", potim=1.0)
        first = _build_repair(run, current_generation(run), "g1-repair-20260910T080000Z", [], {})
        self.assertEqual(first["previous_safe_prefix_steps"], 0)
        self.assertEqual(first["safe_segment_steps"], 16)
        self.assertEqual(first["original_potim_fs"], 1.0)  # POTIM of the rewound generation-0 segment
        atomic_write_json(run / REPAIR_RECORD, first)
        parent = current_generation(run)
        self.assertEqual((parent.accepted_prefix_steps, parent.segment_potim_fs), (16, 0.5))
        second = build_segment_record(
            "repair",
            run=run,
            generation=2,
            generation_id="g2-repair-20260920T101500Z",
            parent=parent,
            prepared_at="2026-09-20T10:15:00+00:00",
            original_nsw=400,
            accepted_prefix_steps=68,
            accepted_segments=[],
            ledger_exact=True,
            segment_nsw=332,
            segment_potim_fs=0.5,
            segment_schedule={},
            archive="archive-2",
            extra={},
        )
        self.assertEqual(second["previous_safe_prefix_steps"], 16)
        self.assertEqual(second["safe_segment_steps"], 52)
        self.assertEqual(second["original_potim_fs"], 0.5)
        self.assertEqual(second["safe_prefix_steps"], 68)
        self.assertEqual((second["repair_nsw"], second["repair_potim_fs"]), (332, 0.5))
        self.assertEqual(second["parent_generation_id"], "g1-repair-20260910T080000Z")

    def test_non_finite_numbers_read_as_unknown(self) -> None:
        # json.loads accepts NaN/Infinity; a hand-edited record must never crash a read-only path.
        run = write_step1_run(self.root, "run", nsw=388, potim=0.5)
        (run / REPAIR_RECORD).write_text(
            '{"safe_prefix_steps": NaN, "previous_safe_prefix_steps": Infinity, "repair_nsw": -Infinity,'
            ' "original_nsw": 1e400, "repair_potim_fs": NaN, "status": "PREPARED", "archive": "x"}',
            encoding="utf-8",
        )
        generation = current_generation(run)
        self.assertEqual(generation.accepted_prefix_steps, 0)
        self.assertEqual((generation.segment_nsw, generation.segment_potim_fs), (388, 0.5))  # INCAR fallback
        self.assertEqual(generation.original_nsw, 388)
        self.assertEqual(generation.status, "PREPARED")

        (run / REPAIR_RECORD).write_text(
            '{"generation_id": "g1-repair-x", "generation": NaN, "accepted_prefix_steps": Infinity,'
            ' "accepted_segments": [{"steps": NaN, "potim_fs": 0.5}], "ledger_exact": true}',
            encoding="utf-8",
        )
        generation = current_generation(run)
        self.assertEqual((generation.generation, generation.accepted_prefix_steps), (1, 0))
        self.assertIsNone(accepted_ps(generation.ledger))
        schedule = incar_schedule({"NSW": "inf", "POTIM": "nan", "TEBEG": "nan", "NBLOCK": "inf"})
        self.assertEqual((schedule["nsw"], schedule["potim_fs"], schedule["tebeg_k"]), (None, None, 300.0))
        self.assertEqual(schedule["nblock"], 1)
        ledger = self.root / LAUNCH_LEDGER
        ledger.write_text('{"schema_version": NaN, "runs": [{"status": "SUBMITTED", "job_id": "1"}]}', "utf-8")
        self.assertTrue(load_ledger_rows(ledger)[0]["legacy"])

    def test_contradictory_records_are_flagged_as_a_conflict(self) -> None:
        # Downgrade: an older step1-repair wrote a schema-1 record next to a stale gen-2 resume record,
        # so "larger generation wins" would pick the stale one.
        run = write_step1_run(self.root, "run", nsw=395, potim=0.5)
        _legacy_record(
            run,
            safe_prefix_steps=5,
            safe_segment_steps=5,
            previous_safe_prefix_steps=0,
            repair_nsw=395,
            archive=str(run / ".interfaceforge" / "archive" / "step1_repair_20260924T000000Z"),
        )
        resume_gid = "g2-resume-20260915T000000Z"
        resume = {
            "generation": 2,
            "generation_id": resume_gid,
            "prepared_at": "2026-09-15T00:00:00+00:00",
            "status": "SUBMITTED",
            "accepted_prefix_steps": 49,
            "segment_nsw": 351,
            "submissions": [{"job_id": "900", "generation_id": resume_gid}],
        }
        atomic_write_json(run / RESUME_RECORD, resume)
        generation = current_generation(run)
        self.assertEqual(generation.generation_id, resume_gid)  # the spec tie-break is still applied...
        self.assertEqual(generation.status, "CONFLICT")  # ...but nothing may launch or resume from it
        self.assertIn("generation-aware", generation.conflict)
        self.assertEqual(generation.record["status"], "SUBMITTED")
        state = submission_state(run, generation, [])
        self.assertTrue(state["current_submitted"])  # the record's own submission stays visible
        self.assertEqual(state["current_submission"]["job_id"], "900")

        # Both generation-aware, but generation 2 was prepared before generation 1.
        generation_zero = current_generation(write_step1_run(self.root, "fresh"))
        atomic_write_json(run / REPAIR_RECORD, _build_repair(run, generation_zero, "g1-repair-late", [], {}))
        resume["status"] = "PREPARED"
        atomic_write_json(run / RESUME_RECORD, resume)
        generation = current_generation(run)
        self.assertEqual(generation.status, "CONFLICT")
        self.assertIn("prepared before generation 1", generation.conflict)

        # An unreadable loser is a conflict too.
        (run / REPAIR_RECORD).write_text("{truncated", encoding="utf-8")
        generation = current_generation(run)
        self.assertEqual(generation.generation_id, resume_gid)
        self.assertEqual(generation.status, "CONFLICT")
        self.assertIn("unreadable", generation.conflict)


def _build_repair(
    run: Path, parent: Generation, gid: str, segments: list[dict[str, Any]], extra: dict[str, Any]
) -> dict[str, Any]:
    return build_segment_record(
        "repair",
        run=run,
        generation=parent.generation + 1,
        generation_id=gid,
        parent=parent,
        prepared_at="2026-09-23T10:15:00+00:00",
        original_nsw=400,
        accepted_prefix_steps=16,
        accepted_segments=segments,
        ledger_exact=True,
        segment_nsw=384,
        segment_potim_fs=0.5,
        segment_schedule={**incar_schedule(parse_incar(run / "INCAR")), "nsw": 384},
        archive="archive-dir",
        extra=extra,
    )


def _build_resume(run: Path, parent: Generation, gid: str) -> dict[str, Any]:
    return build_segment_record(
        "resume",
        run=run,
        generation=parent.generation + 1,
        generation_id=gid,
        parent=parent,
        prepared_at="2026-09-23T11:15:00+00:00",
        original_nsw=400,
        accepted_prefix_steps=parent.accepted_prefix_steps + 33,
        accepted_segments=[*parent.ledger, {"generation": 1, "kind": "repair", "steps": 33, "potim_fs": 0.5}],
        ledger_exact=True,
        segment_nsw=351,
        segment_potim_fs=0.5,
        segment_schedule={"tebeg_k": 117.19, "teend_k": 300.0, "nsw": 351, "thermostat": "x", "ramp": True},
        archive="archive-dir-2",
        extra={"restart_source": "CONTCAR", "restart_frame": None},
    )


def _generation_aware(run: Path) -> Generation:
    """A generation-aware (non-legacy) generation 1 of ``run``, built in memory: nothing is written."""

    return Generation(
        generation=1,
        generation_id="g1-repair-20260923T101500Z",
        kind="repair",
        legacy=False,
        record_path=run / REPAIR_RECORD,
        record={"status": "PREPARED"},
        prepared_at="2026-09-23T10:15:00+00:00",
        prepared_epoch=_epoch("2026-09-23T10:15:00+00:00"),
        original_nsw=400,
        accepted_prefix_steps=16,
        segment_nsw=384,
        segment_potim_fs=0.5,
        ledger=[],
        ledger_exact=True,
        status="PREPARED",
        submissions=[],
    )


class LegacyGenerationTests(LineageTestCase):
    def test_single_legacy_repair_is_generation_one_with_exact_ledger(self) -> None:
        run = write_step1_run(self.root, "run", nsw=388, potim=0.5, tebeg=100.0, teend=300.0)
        _legacy_record(run)
        generation = current_generation(run)
        self.assertEqual(generation.generation, 1)
        self.assertEqual(generation.kind, "repair")
        self.assertTrue(generation.legacy)
        self.assertEqual(generation.generation_id, "legacy-repair-g1-20260901T000000Z")
        self.assertEqual(generation.prepared_at, "2026-09-01T00:00:00+00:00")
        self.assertEqual(generation.prepared_epoch, _epoch("2026-09-01T00:00:00+00:00"))
        self.assertEqual(generation.accepted_prefix_steps, 12)
        self.assertEqual(generation.original_nsw, 400)
        self.assertEqual(generation.segment_nsw, 388)
        self.assertEqual(generation.segment_potim_fs, 0.5)
        self.assertEqual(generation.status, "PREPARED")
        self.assertTrue(generation.ledger_exact)
        self.assertEqual(
            generation.ledger,
            [
                {
                    "generation": 0,
                    "generation_id": GEN0_ID,
                    "kind": "original",
                    "steps": 12,
                    "potim_fs": 1.0,
                    "ps": 0.012,
                    "restart_source": "XDATCAR frame 3",
                    "legacy": True,
                }
            ],
        )
        self.assertAlmostEqual(accepted_ps(generation.ledger), 0.012)

    def test_legacy_repair_of_repair_reconstructs_cumulative_prefix(self) -> None:
        run = _real_case_run(self.root)
        generation = current_generation(run)
        self.assertEqual(generation.generation, 2)
        self.assertEqual(generation.kind, "repair")
        self.assertTrue(generation.legacy)
        self.assertEqual(generation.generation_id, "legacy-repair-g2-20260920T101500Z")
        self.assertEqual(generation.prepared_at, "2026-09-20T10:15:00+00:00")
        self.assertEqual(generation.accepted_prefix_steps, 68)
        self.assertEqual(generation.original_nsw, 400)
        self.assertEqual(generation.original_nsw - generation.accepted_prefix_steps, 332)
        self.assertEqual(generation.segment_nsw, 332)
        self.assertEqual(generation.segment_potim_fs, 0.5)
        self.assertTrue(generation.ledger_exact)
        self.assertEqual(
            generation.ledger,
            [
                {
                    "generation": 0,
                    "generation_id": GEN0_ID,
                    "kind": "original",
                    "steps": 16,
                    "potim_fs": 1.0,
                    "ps": 0.016,
                    "restart_source": "XDATCAR frame 4",
                    "legacy": True,
                    "tebeg_k": 300.0,
                    "teend_k": 300.0,
                },
                {
                    "generation": 1,
                    "generation_id": "legacy-repair-g1-20260910T080000Z",
                    "kind": "repair",
                    "steps": 52,
                    "potim_fs": 0.5,
                    "ps": 0.026,
                    "restart_source": "XDATCAR frame 13",
                    "legacy": True,
                    "tebeg_k": 100.0,
                    "teend_k": 127.08,  # 100 + 200 * 52 / 384
                },
            ],
        )
        self.assertEqual(accepted_ps(generation.ledger), 0.042)
        self.assertEqual(sum(row["steps"] for row in generation.ledger), generation.accepted_prefix_steps)

    def test_three_legacy_repairs_reconstruct_generation_three(self) -> None:
        run = _real_case_run(self.root)
        third = run / ".interfaceforge" / "archive" / "step1_repair_20260922T120000Z"
        third.mkdir(parents=True)
        (third / "INCAR").write_text(incar_text(nsw=332, potim=0.5, tebeg=100.0, teend=300.0), encoding="utf-8")
        shutil.copy2(run / REPAIR_RECORD, third / REPAIR_RECORD)
        write_legacy_repair_record(
            run,
            safe_prefix_steps=108,
            safe_segment_steps=40,
            previous_safe_prefix_steps=68,
            rewind_frame=10,
            repair_nsw=292,
            original_potim_fs=0.5,
            repair_potim_fs=0.5,
            archive=str(third),
        )
        generation = current_generation(run)
        self.assertEqual(generation.generation, 3)
        self.assertEqual(generation.generation_id, "legacy-repair-g3-20260922T120000Z")
        self.assertTrue(generation.ledger_exact)
        self.assertEqual(generation.accepted_prefix_steps, 108)
        self.assertEqual(generation.original_nsw - generation.accepted_prefix_steps, 292)
        self.assertEqual([row["steps"] for row in generation.ledger], [16, 52, 40])
        self.assertEqual([row["generation"] for row in generation.ledger], [0, 1, 2])
        self.assertEqual(
            [row["generation_id"] for row in generation.ledger],
            [GEN0_ID, "legacy-repair-g1-20260910T080000Z", "legacy-repair-g2-20260920T101500Z"],
        )
        self.assertEqual(generation.ledger[2]["teend_k"], round(100.0 + 200.0 * 40 / 332, 2))
        self.assertEqual(accepted_ps(generation.ledger), 0.062)

    def test_archive_chain_survives_a_moved_tree(self) -> None:
        run = _real_case_run(self.root, foreign_archive_paths=True)
        generation = current_generation(run)
        self.assertEqual(generation.generation, 2)
        self.assertTrue(generation.ledger_exact)
        self.assertEqual([row["steps"] for row in generation.ledger], [16, 52])
        self.assertEqual(generation.ledger[1]["tebeg_k"], 100.0)

    def test_legacy_generation_id_is_deterministic(self) -> None:
        run = _real_case_run(self.root)
        first = current_generation(run)
        (run / REPAIR_RECORD).touch()
        second = current_generation(run)
        self.assertEqual(first.generation_id, second.generation_id)
        self.assertEqual(first.prepared_epoch, second.prepared_epoch)
        self.assertEqual(first.ledger, second.ledger)

        other = write_step1_run(self.root, "nostamp")
        _legacy_record(other, archive=str(other / ".interfaceforge" / "archive" / "handmade"))
        ids = {current_generation(other).generation_id for _ in range(3)}
        self.assertEqual(len(ids), 1)
        gid = ids.pop()
        self.assertRegex(gid, r"^legacy-repair-g1-\d{8}T\d{6}Z$")

    def test_marking_a_stampless_legacy_record_pins_its_identity(self) -> None:
        # Without an archive stamp the id comes from the record mtime; marking the
        # record submitted rewrites it, so the derived identity must be pinned.
        run = write_step1_run(self.root, "nostamp", nsw=388, potim=0.5)
        _legacy_record(run, archive=str(run / ".interfaceforge" / "archive" / "handmade"))
        record_path = run / REPAIR_RECORD
        old = datetime.now(timezone.utc).timestamp() - 7200.0
        os.utime(record_path, (old, old))
        before = current_generation(run)
        self.assertAlmostEqual(before.prepared_epoch, old, delta=1.0)
        mark_record_submitted(run, before.generation_id, {"job_id": "8001", "batch_id": "b-1"})
        self.assertGreater(record_path.stat().st_mtime, old + 3600.0)  # the rewrite moved the mtime
        after = current_generation(run)
        self.assertEqual(after.generation_id, before.generation_id)
        self.assertEqual(after.prepared_at, before.prepared_at)
        self.assertEqual(after.prepared_epoch, before.prepared_epoch)
        self.assertEqual(after.status, "SUBMITTED")
        self.assertEqual(read_json(record_path)["legacy_generation_id"], before.generation_id)
        self.assertEqual(after.submissions[0]["job_id"], "8001")

    def test_broken_chain_marks_the_ledger_inexact(self) -> None:
        run = write_step1_run(self.root, "run", nsw=332, potim=0.5)
        _legacy_record(
            run,
            safe_prefix_steps=68,
            safe_segment_steps=52,
            previous_safe_prefix_steps=16,
            rewind_frame=13,
            repair_nsw=332,
            original_potim_fs=0.5,
            archive=str(run / ".interfaceforge" / "archive" / SECOND_ARCHIVE),  # never created
        )
        generation = current_generation(run)
        self.assertEqual(generation.generation, 2)
        self.assertEqual(generation.generation_id, "legacy-repair-g2-20260920T101500Z")
        self.assertFalse(generation.ledger_exact)
        self.assertEqual(generation.accepted_prefix_steps, 68)
        unknown = dict(generation.ledger[0])
        self.assertIn("not found", unknown.pop("note"))
        self.assertEqual(
            unknown,
            {
                "generation": None,
                "generation_id": None,
                "kind": "unknown",
                "steps": 16,
                "potim_fs": None,
                "ps": None,
                "legacy": True,
            },
        )
        self.assertEqual(len(generation.ledger), 2)
        self.assertEqual(generation.ledger[1]["steps"], 52)
        self.assertEqual(generation.ledger[1]["potim_fs"], 0.5)
        self.assertEqual(generation.ledger[1]["kind"], "repair")
        self.assertIsNone(accepted_ps(generation.ledger))

    def test_archive_chain_cycle_terminates_as_broken(self) -> None:
        run = write_step1_run(self.root, "run", nsw=332, potim=0.5)
        archive = run / ".interfaceforge" / "archive" / SECOND_ARCHIVE
        archive.mkdir(parents=True)
        looping = {"safe_prefix_steps": 16, "previous_safe_prefix_steps": 8, "archive": str(archive)}
        (archive / REPAIR_RECORD).write_text(json.dumps(looping), encoding="utf-8")
        _legacy_record(
            run, safe_prefix_steps=68, safe_segment_steps=52, previous_safe_prefix_steps=16, archive=str(archive)
        )
        generation = current_generation(run)
        self.assertFalse(generation.ledger_exact)
        self.assertEqual(generation.accepted_prefix_steps, 68)
        self.assertEqual(generation.ledger[0]["kind"], "unknown")
        # Stopped by the cycle guard (two walked records), not by the depth cap.
        self.assertIn("cycles", generation.ledger[0]["note"])
        self.assertEqual(generation.generation, 3)

    def test_repair_after_a_first_repair_that_rewound_to_step_zero(self) -> None:
        # Repair 1 rewound to step 0 (early first bad step / SCF-only failure), so repair 2 recorded
        # previous_safe_prefix_steps == 0; repair 2's archive still holds repair 1's record.
        run = write_step1_run(self.root, "run", nsw=372, potim=0.5, tebeg=100.0, teend=300.0, steps=0, contcar=None)
        archives = run / ".interfaceforge" / "archive"
        first, second = archives / FIRST_ARCHIVE, archives / SECOND_ARCHIVE
        first.mkdir(parents=True)
        second.mkdir(parents=True)
        (first / "INCAR").write_text(incar_text(nsw=400, potim=1.0), encoding="utf-8")
        (second / "INCAR").write_text(incar_text(nsw=400, potim=0.5, tebeg=100.0, teend=300.0), encoding="utf-8")
        write_legacy_repair_record(
            run,
            safe_prefix_steps=0,
            safe_segment_steps=0,
            previous_safe_prefix_steps=0,
            rewind_frame=None,
            source="POSCAR",
            repair_nsw=400,
            original_potim_fs=1.0,
            archive=str(first),
        )
        shutil.copy2(run / REPAIR_RECORD, second / REPAIR_RECORD)
        write_legacy_repair_record(
            run,
            safe_prefix_steps=28,
            safe_segment_steps=28,
            previous_safe_prefix_steps=0,
            rewind_frame=7,
            repair_nsw=372,
            original_potim_fs=0.5,
            archive=str(second),
        )
        generation = current_generation(run)
        self.assertEqual(generation.generation, 2)
        self.assertEqual(generation.generation_id, "legacy-repair-g2-20260920T101500Z")
        self.assertTrue(generation.ledger_exact)
        self.assertEqual(generation.accepted_prefix_steps, 28)
        self.assertEqual(
            [
                (row["generation"], row["generation_id"], row["kind"], row["steps"], row["potim_fs"])
                for row in generation.ledger
            ],
            [(0, GEN0_ID, "original", 0, 1.0), (1, "legacy-repair-g1-20260910T080000Z", "repair", 28, 0.5)],
        )
        self.assertEqual((generation.ledger[1]["tebeg_k"], generation.ledger[1]["teend_k"]), (100.0, 114.0))

        # The same record whose archived predecessor cannot be read: a broken chain, never generation 1.
        (second / REPAIR_RECORD).write_text("{truncated", encoding="utf-8")
        broken = current_generation(run)
        self.assertEqual(broken.generation, 2)
        self.assertFalse(broken.ledger_exact)
        self.assertEqual(broken.ledger[0]["kind"], "unknown")
        self.assertIn("unreadable", broken.ledger[0]["note"])
        self.assertEqual(broken.accepted_prefix_steps, 28)

    def test_legacy_record_on_top_of_a_generation_aware_record(self) -> None:
        # Downgrade scenario: an older InterfaceForge repaired a schema-2 generation.
        run = write_step1_run(self.root, "run", nsw=332, potim=0.5, tebeg=100.0, teend=300.0)
        base_gid = "g1-repair-20260915T000000Z"
        base = _build_repair(
            run, current_generation(run), base_gid, [{"generation": 0, "steps": 16, "potim_fs": 1.0}], {}
        )
        archive = run / ".interfaceforge" / "archive" / SECOND_ARCHIVE
        archive.mkdir(parents=True)
        atomic_write_json(archive / REPAIR_RECORD, base)
        _legacy_record(
            run, safe_prefix_steps=68, safe_segment_steps=52, previous_safe_prefix_steps=16, archive=str(archive)
        )
        generation = current_generation(run)
        self.assertEqual(generation.generation, 2)
        self.assertTrue(generation.legacy)
        self.assertTrue(generation.ledger_exact)
        self.assertEqual([row["steps"] for row in generation.ledger], [16, 52])
        self.assertEqual(generation.ledger[1]["generation_id"], base_gid)
        self.assertEqual(generation.ledger[1]["generation"], 1)

    def test_pre_cumulative_fix_record_is_generation_one(self) -> None:
        run = write_step1_run(self.root, "run", nsw=388, potim=0.5)
        archive = run / ".interfaceforge" / "archive" / "step1_repair_20260801T000000Z"
        archive.mkdir(parents=True)  # no step1_repair.json inside: nothing earlier to follow
        record = _legacy_record(run, archive=str(archive))
        del record["previous_safe_prefix_steps"]
        del record["safe_segment_steps"]
        (run / REPAIR_RECORD).write_text(json.dumps(record), encoding="utf-8")
        generation = current_generation(run)
        self.assertEqual(generation.generation, 1)
        self.assertTrue(generation.ledger_exact)
        self.assertEqual(generation.accepted_prefix_steps, 12)
        self.assertEqual(len(generation.ledger), 1)
        self.assertEqual(generation.ledger[0]["steps"], 12)
        self.assertEqual(generation.ledger[0]["kind"], "original")
        self.assertEqual(generation.generation_id, "legacy-repair-g1-20260801T000000Z")

    def test_pre_cumulative_fix_repair_of_repair_is_rebuilt_from_the_chain(self) -> None:
        # Both repairs written before the cumulative-prefix fix: R2's safe_prefix_steps is
        # segment-only and its original_nsw is R1's repair NSW; R2's archive holds R1.
        run = write_step1_run(self.root, "run", nsw=316, potim=0.5)
        archives = run / ".interfaceforge" / "archive"
        first, second = archives / "step1_repair_20260801T000000Z", archives / "step1_repair_20260802T000000Z"
        first.mkdir(parents=True)
        second.mkdir(parents=True)
        (first / "INCAR").write_text("NSW = 400\nPOTIM = 1.0\nTEBEG = 300\n", encoding="utf-8")
        (second / "INCAR").write_text("NSW = 368\nPOTIM = 0.5\nTEBEG = 300\n", encoding="utf-8")
        pre_fix = ("previous_safe_prefix_steps", "safe_segment_steps")
        r1 = _legacy_record(run, archive=str(first), safe_prefix_steps=32, original_nsw=400, repair_nsw=368,
                            original_potim_fs=1.0)
        r2 = _legacy_record(run, archive=str(second), safe_prefix_steps=52, original_nsw=368, repair_nsw=316,
                            original_potim_fs=0.5)
        for record in (r1, r2):
            for key in pre_fix:
                del record[key]
        (second / REPAIR_RECORD).write_text(json.dumps(r1), encoding="utf-8")
        (run / REPAIR_RECORD).write_text(json.dumps(r2), encoding="utf-8")

        generation = current_generation(run)
        self.assertEqual(generation.generation, 2)
        self.assertEqual(generation.accepted_prefix_steps, 84)
        self.assertEqual(generation.original_nsw, 400)
        self.assertEqual(generation.segment_nsw, 316)
        self.assertTrue(generation.ledger_exact)
        self.assertEqual(
            [(row["generation"], row["kind"], row["steps"]) for row in generation.ledger],
            [(0, "original", 32), (1, "repair", 52)],
        )
        self.assertEqual(generation.generation_id, "legacy-repair-g2-20260802T000000Z")


class RecordMutationTests(LineageTestCase):
    def test_mark_record_submitted_on_new_record(self) -> None:
        run = write_step1_run(self.root, "run")
        gid = "g1-repair-20260923T101500Z"
        atomic_write_json(run / REPAIR_RECORD, _build_repair(run, current_generation(run), gid, [], {}))
        before = (run / REPAIR_RECORD).read_bytes()
        mark_record_submitted(run, "g9-repair-other", {"job_id": "1"})
        self.assertEqual((run / REPAIR_RECORD).read_bytes(), before)

        mark_record_submitted(run, gid, {"job_id": "5001", "batch_id": "b-1", "launcher": "runvasp.sh"})
        record = read_json(run / REPAIR_RECORD)
        self.assertEqual(record["status"], "SUBMITTED")
        self.assertEqual(record["submissions"][0]["job_id"], "5001")
        self.assertEqual(record["submissions"][0]["generation_id"], gid)
        self.assertIn("submitted_at", record["submissions"][0])
        state = submission_state(run, current_generation(run), [])
        self.assertTrue(state["current_submitted"])
        self.assertEqual(state["current_submission"]["job_id"], "5001")

    def test_mark_record_submitted_is_noop_for_generation_zero(self) -> None:
        run = write_step1_run(self.root, "run")
        mark_record_submitted(run, GEN0_ID, {"job_id": "1"})
        self.assertFalse((run / REPAIR_RECORD).exists())
        self.assertFalse((run / RESUME_RECORD).exists())

    def test_mark_legacy_record_keeps_its_identity(self) -> None:
        run = _real_case_run(self.root)
        gid = current_generation(run).generation_id
        mark_record_submitted(run, gid, {"job_id": "7001"})
        record = read_json(run / REPAIR_RECORD)
        self.assertNotIn("generation_id", record)
        self.assertEqual(record["legacy_generation_id"], gid)
        self.assertEqual(record["status"], "SUBMITTED")
        generation = current_generation(run)
        self.assertEqual(generation.generation_id, gid)
        self.assertEqual(generation.generation, 2)
        self.assertEqual(generation.accepted_prefix_steps, 68)
        self.assertTrue(submission_state(run, generation, [])["current_submitted"])

    def test_retire_refuses_without_an_archived_copy(self) -> None:
        run = write_step1_run(self.root, "run")
        _legacy_record(run)
        empty = self.root / "not-an-archive"
        empty.mkdir()
        with self.assertRaises(SafetyError):
            retire_current_records(run, empty)
        self.assertTrue((run / REPAIR_RECORD).is_file())

        archive = archive_step1_state(run, "step1_repair_g2")
        (run / REPAIR_RECORD).write_text("{}\n", encoding="utf-8")  # changed after archiving
        with self.assertRaises(SafetyError):
            retire_current_records(run, archive)
        self.assertTrue((run / REPAIR_RECORD).is_file())

    def test_retire_checks_every_record_before_deleting_any(self) -> None:
        run = write_step1_run(self.root, "run")
        _legacy_record(run)
        atomic_write_json(run / RESUME_RECORD, {"generation_id": "g2-resume-x", "generation": 2})
        archive = archive_step1_state(run, "step1_repair_g3")
        (archive / RESUME_RECORD).unlink()  # the second record has no archived copy
        with self.assertRaises(SafetyError):
            retire_current_records(run, archive)
        self.assertTrue((run / REPAIR_RECORD).is_file())  # the first (archived) record was not deleted either
        self.assertTrue((run / RESUME_RECORD).is_file())

    def test_marking_a_hand_written_resume_record_keeps_its_identity(self) -> None:
        # No generation_id and no prepared_at: the id derives from the mtime, which the rewrite moves.
        run = write_step1_run(self.root, "run", nsw=367, potim=0.5)
        path = run / RESUME_RECORD
        path.write_text(json.dumps({"status": "PREPARED", "accepted_prefix_steps": 33}), encoding="utf-8")
        old = datetime.now(timezone.utc).timestamp() - 7200.5
        os.utime(path, (old, old))
        before = current_generation(run)
        self.assertTrue(before.legacy)
        self.assertEqual(before.prepared_epoch, float(int(old)))
        row = {**_launch_row(run, run, job_id="8101"), "generation": 1, "generation_id": before.generation_id}
        append_launch_rows(run, [row], batch={"batch_id": "b-1"})
        mark_record_submitted(run, before.generation_id, {"job_id": "8101", "batch_id": "b-1"})
        self.assertGreater(path.stat().st_mtime, old + 3600.0)
        after = current_generation(run)
        self.assertEqual(after.generation_id, before.generation_id)
        self.assertEqual((after.prepared_at, after.prepared_epoch), (before.prepared_at, before.prepared_epoch))
        self.assertEqual(after.status, "SUBMITTED")
        state = submission_state(run, after, rows_for_run(run, ledger_paths_for(run)))
        self.assertTrue(state["current_submitted"])
        self.assertEqual(state["current_submission"]["job_id"], "8101")
        self.assertEqual(state["historical_submissions"], [])

    def test_retire_removes_archived_records(self) -> None:
        run = write_step1_run(self.root, "run")
        _legacy_record(run)
        atomic_write_json(run / RESUME_RECORD, {"generation_id": "g2-resume-x", "generation": 2})
        archive = archive_step1_state(run, "step1_resume_g3")
        removed = retire_current_records(run, archive)
        self.assertEqual(removed, [REPAIR_RECORD, RESUME_RECORD])
        self.assertFalse((run / REPAIR_RECORD).exists())
        self.assertFalse((run / RESUME_RECORD).exists())
        self.assertTrue((archive / REPAIR_RECORD).is_file())
        self.assertTrue((archive / RESUME_RECORD).is_file())
        self.assertEqual(retire_current_records(run, archive), [])


class ArchiveTests(LineageTestCase):
    def test_archive_copies_extras_and_tracks_progress(self) -> None:
        run = write_step1_run(self.root, "run", launcher="precondition", wavecar=True)
        _legacy_record(run)
        (run / RESUME_RECORD).write_text("{}\n", encoding="utf-8")
        (run / LAUNCH_LEDGER).write_text("{}\n", encoding="utf-8")
        (run / "vasp_md.dat").write_text("md\n", encoding="utf-8")
        (run / "CHGCAR").write_bytes(b"chg")
        precondition = run / "precondition"
        precondition.mkdir()
        (precondition / "INCAR").write_text("NSW = 0\n", encoding="utf-8")
        (precondition / "OSZICAR").write_text("1 F= -1.0\n", encoding="utf-8")
        (precondition / "WAVECAR").write_bytes(b"w" * 64)

        archive = archive_step1_state(run, "step1_resume_g2")
        self.assertEqual(archive.parent, run / ".interfaceforge" / "archive")
        for name in (
            "INCAR",
            "POSCAR",
            "OSZICAR",
            "runvasp.sh",
            REPAIR_RECORD,
            RESUME_RECORD,
            LAUNCH_LEDGER,
            "INCAR.precondition",
            "vasp_md.dat",
            "precondition/INCAR",
            "precondition/OSZICAR",
        ):
            self.assertTrue((archive / name).is_file(), name)
        for name in ("WAVECAR", "CHGCAR", "precondition/WAVECAR"):
            self.assertFalse((archive / name).exists(), name)

        manifest = read_json(archive / ARCHIVE_MANIFEST)
        self.assertEqual(manifest["format"], "interfaceforge-step1-archive")
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["operation"], "step1_resume_g2")
        self.assertEqual(manifest["status"], "IN_PROGRESS")
        self.assertIsNone(manifest["generation_id"])
        self.assertEqual(manifest["not_archived"], ["WAVECAR", "CHGCAR"])
        names = {item["name"]: item["bytes"] for item in manifest["files"]}
        self.assertEqual(names["precondition/INCAR"], (precondition / "INCAR").stat().st_size)
        self.assertIn(REPAIR_RECORD, names)
        self.assertNotIn(ARCHIVE_MANIFEST, names)

        self.assertEqual(interrupted_archive(run), archive)
        finalize_archive(archive, generation_id="g2-resume-20260923T101500Z")
        manifest = read_json(archive / ARCHIVE_MANIFEST)
        self.assertEqual(manifest["status"], "COMPLETE")
        self.assertEqual(manifest["generation_id"], "g2-resume-20260923T101500Z")
        self.assertIsNone(interrupted_archive(run))

    def test_interrupted_archive_is_none_without_archives(self) -> None:
        run = write_step1_run(self.root, "run")
        self.assertIsNone(interrupted_archive(run))

    def test_interrupted_archive_returns_the_newest_in_progress(self) -> None:
        run = write_step1_run(self.root, "run")
        base = run / ".interfaceforge" / "archive"
        for name, created, status in (
            ("step1_repair_g1_20260901T000000Z", "2026-09-01T00:00:00+00:00", "IN_PROGRESS"),
            ("step1_resume_g2_20260910T000000Z", "2026-09-10T00:00:00+00:00", "IN_PROGRESS"),
            ("step1_repair_g3_20260920T000000Z", "2026-09-20T00:00:00+00:00", "COMPLETE"),
        ):
            (base / name).mkdir(parents=True)
            atomic_write_json(base / name / ARCHIVE_MANIFEST, {"status": status, "created_at": created})
        (base / "no_manifest_20260930T000000Z").mkdir()
        self.assertEqual(interrupted_archive(run), base / "step1_resume_g2_20260910T000000Z")
        finalize_archive(base / "step1_resume_g2_20260910T000000Z", generation_id="g2-resume-x")
        self.assertEqual(interrupted_archive(run), base / "step1_repair_g1_20260901T000000Z")

    def test_finalize_refuses_an_archive_without_manifest(self) -> None:
        folder = self.root / "archive"
        folder.mkdir()
        with self.assertRaises(SafetyError):
            finalize_archive(folder, generation_id="g1-repair-x")
        self.assertFalse((folder / ARCHIVE_MANIFEST).exists())

    def test_read_only_helpers_perform_zero_mutation(self) -> None:
        step1 = self.root / "Step1"
        run = _real_case_run(step1)
        fresh = write_step1_run(step1, "fresh")
        write_manifest(step1, [fresh])
        write_legacy_launch_ledger(run, [_launch_row(run, run, job_id="1")], age_hours=72.0)
        write_legacy_launch_ledger(step1, [_launch_row(fresh, step1, job_id="2")], age_hours=5.0)
        before = tree_snapshot(self.root)
        for folder in (run, fresh):
            generation = current_generation(folder)
            paths = ledger_paths_for(folder, [step1])
            for path in paths:
                load_ledger_rows(path)
            submission_state(folder, generation, rows_for_run(folder, paths))
            run_fingerprint(folder)
            interrupted_archive(folder)
            incar_schedule(parse_incar(folder / "INCAR"))
        self.assertEqual(tree_snapshot(self.root), before)
        self.assertFalse((fresh / ".interfaceforge").exists())

    def test_run_fingerprint_changes_when_oszicar_changes(self) -> None:
        run = write_step1_run(self.root, "run")
        before = run_fingerprint(run)
        self.assertEqual(set(before), set(FINGERPRINT_FILES))
        self.assertIsNone(before[RESUME_RECORD])
        self.assertEqual(before["OSZICAR"][0], (run / "OSZICAR").stat().st_size)
        with (run / "OSZICAR").open("a", encoding="utf-8") as handle:
            handle.write("   34 T=   300.0 E= -9.0 F= -10.0\n")
        after = run_fingerprint(run)
        self.assertNotEqual(before, after)
        self.assertNotEqual(before["OSZICAR"], after["OSZICAR"])
        self.assertEqual(before["INCAR"], after["INCAR"])
        self.assertEqual(json.loads(json.dumps(after)), after)


class JsonIoTests(LineageTestCase):
    def test_atomic_write_json_leaves_no_tmp(self) -> None:
        path = self.root / "record.json"
        atomic_write_json(path, {"b": 1, "a": [1, 2]})
        text = path.read_text(encoding="utf-8")
        self.assertTrue(text.endswith("\n"))
        self.assertLess(text.index('"a"'), text.index('"b"'))
        self.assertEqual(json.loads(text), {"a": [1, 2], "b": 1})
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_atomic_write_text_keeps_newlines_and_leaves_no_tmp(self) -> None:
        path = self.root / "table.tsv"
        atomic_write_text(path, "a\tb\n1\t2\n")
        atomic_write_text(path, "a\tb\n3\t4\n")
        self.assertEqual(path.read_bytes(), b"a\tb\n3\t4\n")
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_atomic_write_failure_keeps_the_old_file(self) -> None:
        path = self.root / "record.json"
        atomic_write_json(path, {"old": True})
        with patch("interfaceforge.step1_lineage.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                atomic_write_json(path, {"new": True})
        self.assertEqual(read_json(path), {"old": True})
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_atomic_writes_use_unique_temporary_names(self) -> None:
        path = self.root / "record.json"
        sources: list[str] = []
        real_replace = os.replace

        def spy(source: Any, target: Any) -> None:
            sources.append(Path(source).name)
            real_replace(source, target)

        with patch("interfaceforge.step1_lineage.os.replace", side_effect=spy):
            atomic_write_json(path, {"n": 1})
            atomic_write_json(path, {"n": 2})
        self.assertEqual(len(set(sources)), 2)  # concurrent writers never share (and truncate) one temp file
        for name in sources:
            self.assertTrue(name.startswith("record.json.") and name.endswith(".tmp"), name)
            self.assertNotEqual(name, "record.json.tmp")
        self.assertEqual(read_json(path), {"n": 2})
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_read_json_tolerates_bad_input(self) -> None:
        self.assertEqual(read_json(self.root / "missing.json"), {})
        (self.root / "bad.json").write_text("{nope", encoding="utf-8")
        self.assertEqual(read_json(self.root / "bad.json"), {})
        (self.root / "list.json").write_text("[1, 2]", encoding="utf-8")
        self.assertEqual(read_json(self.root / "list.json"), {})

    def test_time_helpers(self) -> None:
        self.assertRegex(utc_now_iso(), r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00$")
        self.assertRegex(utc_stamp(), r"^\d{8}T\d{6}Z$")


class SubmissionStateTests(LineageTestCase):
    def _legacy_repaired_run(self, *, prepared_hours_ago: float) -> Path:
        step1 = self.root / "Step1"
        run = write_step1_run(step1, "runA", nsw=388, potim=0.5)
        archive = run / ".interfaceforge" / "archive" / f"step1_repair_{_stamp(prepared_hours_ago)}"
        _legacy_record(run, archive=str(archive))
        return run

    def _state(self, run: Path, roots: tuple[Path, ...] = ()) -> dict[str, Any]:
        return submission_state(run, current_generation(run), rows_for_run(run, ledger_paths_for(run, roots)))

    def test_historical_leaf_row_before_legacy_repair_is_not_current(self) -> None:
        run = self._legacy_repaired_run(prepared_hours_ago=1.0)
        write_legacy_launch_ledger(run, [_launch_row(run, run, job_id="111")], age_hours=48.0)
        state = self._state(run)
        self.assertFalse(state["current_submitted"])
        self.assertIsNone(state["current_submission"])
        self.assertEqual([row["job_id"] for row in state["historical_submissions"]], ["111"])
        self.assertIn("1 historical", state["match_rule"])

    def test_legacy_row_after_legacy_repair_is_current(self) -> None:
        run = self._legacy_repaired_run(prepared_hours_ago=1.0)
        write_legacy_launch_ledger(run, [_launch_row(run, run, job_id="222")])
        state = self._state(run)
        self.assertTrue(state["current_submitted"])
        self.assertEqual(state["current_submission"]["job_id"], "222")
        self.assertEqual(state["historical_submissions"], [])
        self.assertIn(">=", state["match_rule"])

    def test_legacy_gen0_row_with_bumped_ledger_mtime_is_not_a_repair_submission(self) -> None:
        # The old launcher wrote kind "prepared" only for generation 0; a copy or edit that
        # refreshes the ledger mtime must not turn that row into the repair's submission.
        run = self._legacy_repaired_run(prepared_hours_ago=1.0)
        write_legacy_launch_ledger(run, [_launch_row(run, run, job_id="444", kind="prepared")])  # mtime: now
        state = self._state(run)
        self.assertFalse(state["current_submitted"])
        self.assertEqual([row["job_id"] for row in state["historical_submissions"]], ["444"])

    def test_legacy_row_counts_for_generation_zero(self) -> None:
        run = write_step1_run(self.root, "run")
        write_legacy_launch_ledger(run, [_launch_row(run, run, job_id="333")], age_hours=48.0)
        state = self._state(run)
        self.assertTrue(state["current_submitted"])
        self.assertIn("generation 0", state["match_rule"])

    def test_generation_ids_decide_for_new_records(self) -> None:
        step1 = self.root / "Step1"
        run = write_step1_run(step1, "runA")
        gid = "g1-repair-20260923T101500Z"
        atomic_write_json(run / REPAIR_RECORD, _build_repair(run, current_generation(run), gid, [], {}))
        generation = current_generation(run)
        other = {**_launch_row(run, step1, job_id="444"), "generation": 1, "generation_id": "g1-repair-OTHER"}
        legacy = {**_launch_row(run, step1, job_id="445"), "legacy": True}
        state = submission_state(run, generation, [other, legacy])
        self.assertFalse(state["current_submitted"])
        self.assertEqual({row["job_id"] for row in state["historical_submissions"]}, {"444", "445"})

        matching = {**_launch_row(run, step1, job_id="446"), "generation": 1, "generation_id": gid}
        failed = {**matching, "status": "FAILED", "job_id": ""}
        state = submission_state(run, generation, [other, failed, matching])
        self.assertTrue(state["current_submitted"])
        self.assertEqual(state["current_submission"]["job_id"], "446")
        self.assertEqual([row["job_id"] for row in state["historical_submissions"]], ["444"])

    def test_rows_match_by_directory_across_invocation_roots(self) -> None:
        step1 = self.root / "Step1"
        run = write_step1_run(step1, "runA")
        sibling = write_step1_run(step1, "runB")
        write_manifest(step1, [run, sibling])
        rows = [
            # Written on the cluster: stale absolute directory, relative path still valid.
            {**_launch_row(run, step1, job_id="555"), "directory": "/cluster/scratch/Step1/runA"},
            _launch_row(sibling, step1, job_id="556"),
        ]
        write_legacy_launch_ledger(step1, rows, age_hours=2.0)
        paths = ledger_paths_for(run)
        self.assertEqual(paths, [(step1 / LAUNCH_LEDGER).resolve()])
        matched = rows_for_run(run, paths)
        self.assertEqual([row["job_id"] for row in matched], ["555"])
        self.assertTrue(matched[0]["legacy"])
        self.assertEqual(matched[0]["_ledger"], str(paths[0]))
        state = submission_state(run, current_generation(run), matched)
        self.assertTrue(state["current_submitted"])
        self.assertEqual(state["current_submission"]["job_id"], "555")
        # The sibling's row never leaks into runA's state, even if passed in.
        state = submission_state(run, current_generation(run), load_ledger_rows(paths[0]))
        self.assertEqual(state["current_submission"]["job_id"], "555")
        self.assertEqual(state["historical_submissions"], [])

    def test_real_failure_old_leaf_ledger_does_not_block_second_legacy_repair(self) -> None:
        # Regression for the manual-rename incident: the leaf was once launched as
        # its own root (leaf ledger, relative_path "."), repair 1 was launched from
        # the Step1 root, and a second legacy repair was then prepared.
        step1 = self.root / "Step1"
        run = _real_case_run(
            step1,
            first_archive=f"step1_repair_{_stamp(48.0)}",
            second_archive=f"step1_repair_{_stamp(1.0)}",
        )
        write_manifest(step1, [run])
        write_legacy_launch_ledger(run, [_launch_row(run, run, job_id="111")], age_hours=72.0)
        write_legacy_launch_ledger(step1, [_launch_row(run, step1, job_id="222")], age_hours=47.0)
        generation = current_generation(run)
        self.assertEqual((generation.generation, generation.accepted_prefix_steps), (2, 68))
        paths = ledger_paths_for(run, [step1])
        self.assertEqual(paths, [(run / LAUNCH_LEDGER).resolve(), (step1 / LAUNCH_LEDGER).resolve()])
        rows = rows_for_run(run, paths)
        self.assertEqual(sorted(row["job_id"] for row in rows), ["111", "222"])
        state = submission_state(run, generation, rows)
        self.assertFalse(state["current_submitted"])
        self.assertEqual(sorted(row["job_id"] for row in state["historical_submissions"]), ["111", "222"])

        # Launching generation 2 records a generation-aware row: now it is current.
        row = {
            **_launch_row(run, step1, job_id="333"),
            "generation": generation.generation,
            "generation_id": generation.generation_id,
            "submitted_at": utc_now_iso(),
        }
        append_launch_rows(step1, [row], batch={"batch_id": "b-now"})
        state = submission_state(run, generation, rows_for_run(run, ledger_paths_for(run, [step1])))
        self.assertTrue(state["current_submitted"])
        self.assertEqual(state["current_submission"]["job_id"], "333")
        self.assertEqual(sorted(row["job_id"] for row in state["historical_submissions"]), ["111", "222"])

    def test_different_generation_id_never_counts_even_for_generation_zero(self) -> None:
        run = write_step1_run(self.root, "run")
        generation = current_generation(run)
        now = utc_now_iso()
        foreign = {**_launch_row(run, self.root, job_id="10"), "generation_id": "g1-repair-X", "submitted_at": now}
        state = submission_state(run, generation, [foreign])
        self.assertFalse(state["current_submitted"])
        self.assertEqual([row["job_id"] for row in state["historical_submissions"]], ["10"])
        own = {**_launch_row(run, self.root, job_id="11"), "generation_id": GEN0_ID, "submitted_at": now}
        state = submission_state(run, generation, [foreign, own])
        self.assertTrue(state["current_submitted"])
        self.assertEqual(state["current_submission"]["job_id"], "11")

    def test_legacy_row_without_any_time_is_conservatively_current(self) -> None:
        run = self._legacy_repaired_run(prepared_hours_ago=1.0)
        row = _launch_row(run, run, job_id="12")  # no submitted_at / legacy_recorded_at
        state = submission_state(run, current_generation(run), [row])
        self.assertTrue(state["current_submitted"])
        self.assertIn("conservatively", state["match_rule"])

    def test_leaf_and_root_ledgers_both_match_the_run(self) -> None:
        # (f) the same run launched once from the Step1 root and once as its own root.
        step1 = self.root / "Step1"
        run = write_step1_run(step1, "runA")
        gid = "g1-repair-20260923T101500Z"
        atomic_write_json(run / REPAIR_RECORD, _build_repair(run, current_generation(run), gid, [], {}))
        write_manifest(step1, [run])
        write_legacy_launch_ledger(run, [_launch_row(run, run, job_id="20")], age_hours=30.0)
        root_row = {**_launch_row(run, step1, job_id="21"), "generation": 1, "generation_id": gid}
        append_launch_rows(step1, [root_row], batch={"batch_id": "b-1"})
        paths = ledger_paths_for(run)  # found without passing the root: nearest manifest ancestor
        self.assertEqual(paths, [(run / LAUNCH_LEDGER).resolve(), (step1 / LAUNCH_LEDGER).resolve()])
        rows = rows_for_run(run, [*paths, *paths])  # duplicates never double the rows
        self.assertEqual([row["job_id"] for row in rows], ["20", "21"])
        self.assertEqual({row["_directory_real"] for row in rows}, {os.path.realpath(run)})
        state = submission_state(run, current_generation(run), rows)
        self.assertTrue(state["current_submitted"])
        self.assertEqual(state["current_submission"]["job_id"], "21")
        self.assertEqual([row["job_id"] for row in state["historical_submissions"]], ["20"])
        self.assertIn(gid, state["match_rule"])

    def test_unreadable_ledger_fails_closed(self) -> None:
        # A truncated ledger (the pre-generation step1-launch wrote it non-atomically) may hold the
        # current generation's SUBMITTED row: the duplicate check must not read it as "never submitted".
        step1 = self.root / "Step1"
        run = write_step1_run(step1, "runA", steps=0, contcar=None)
        write_manifest(step1, [run])
        ledger = write_legacy_launch_ledger(step1, [_launch_row(run, step1, job_id="777")])
        self.assertTrue(self._state(run)["current_submitted"])
        text = ledger.read_text(encoding="utf-8")
        for broken in (text[:-40], "", "[1, 2]", '{"format": "interfaceforge-step1-launch"}'):
            ledger.write_text(broken, encoding="utf-8")
            before = tree_snapshot(self.root)
            paths = ledger_paths_for(run)
            self.assertEqual(paths, [ledger.resolve()])
            self.assertEqual(unreadable_ledgers([*paths, *paths]), [ledger.resolve()])
            self.assertEqual(load_ledger_rows(ledger), [])
            rows = rows_for_run(run, paths)
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["unreadable_ledger"])
            self.assertNotEqual(rows[0]["status"], "SUBMITTED")
            for generation in (current_generation(run), _generation_aware(run)):
                state = submission_state(run, generation, rows)
                self.assertTrue(state["current_submitted"], broken)
                self.assertEqual(state["current_submission"]["job_id"], "unknown")
                self.assertEqual(state["unreadable_ledgers"], [str(ledger.resolve())])
                self.assertIn("unreadable", state["match_rule"])
            self.assertEqual(tree_snapshot(self.root), before)  # read-only
        ledger.write_text('{"runs": []}', encoding="utf-8")  # empty but valid: nothing submitted
        self.assertEqual(unreadable_ledgers([ledger]), [])
        state = self._state(run)
        self.assertFalse(state["current_submitted"])
        self.assertEqual(state["unreadable_ledgers"], [])

    def test_unreadable_ledger_is_reported_beside_a_real_submission(self) -> None:
        step1 = self.root / "Step1"
        run = write_step1_run(step1, "runA")
        write_manifest(step1, [run])
        (run / LAUNCH_LEDGER).write_text("{broken", encoding="utf-8")
        row = {**_launch_row(run, step1, job_id="81"), "generation": 0, "generation_id": GEN0_ID}
        append_launch_rows(step1, [row], batch={"batch_id": "b-1"})
        state = self._state(run)
        self.assertTrue(state["current_submitted"])
        self.assertEqual(state["current_submission"]["job_id"], "81")
        self.assertEqual(state["unreadable_ledgers"], [str((run / LAUNCH_LEDGER).resolve())])

    def test_legacy_row_time_boundary_and_utc_z_suffix(self) -> None:
        run = write_step1_run(self.root, "run", nsw=388, potim=0.5)
        _legacy_record(run, archive=str(run / ".interfaceforge" / "archive" / SECOND_ARCHIVE))
        generation = current_generation(run)
        self.assertEqual(generation.prepared_at, "2026-09-20T10:15:00+00:00")
        cases = {
            "2026-09-20T10:15:00+00:00": True,  # recorded in the same second the repair was prepared
            "2026-09-20T10:15:00Z": True,  # Python 3.10 fromisoformat rejects "Z"
            "2026-09-20T10:14:59+00:00": False,
            "2026-09-20T10:14:59Z": False,
        }
        for stamp, expected in cases.items():
            row = {**_launch_row(run, run, job_id="5"), "submitted_at": stamp, "legacy": True}
            state = submission_state(run, generation, [row])
            self.assertEqual(state["current_submitted"], expected, stamp)
            self.assertEqual(len(state["historical_submissions"]), 0 if expected else 1, stamp)

        # The same boundary through a schema-1 ledger's mtime.
        ledger = write_legacy_launch_ledger(run, [_launch_row(run, run, job_id="6")])
        for delta, expected in ((0.0, True), (-1.0, False)):
            moment = generation.prepared_epoch + delta
            os.utime(ledger, (moment, moment))
            self.assertEqual(self._state(run)["current_submitted"], expected, delta)

    def test_every_current_submission_is_listed(self) -> None:
        step1 = self.root / "Step1"
        run = write_step1_run(step1, "runA")
        generation = _generation_aware(run)
        gid = generation.generation_id
        first = {**_launch_row(run, step1, job_id="91"), "generation_id": gid, "submitted_at": "2026-09-23T11:00:00Z"}
        second = {**_launch_row(run, step1, job_id="92"), "generation_id": gid, "submitted_at": "2026-09-23T12:00:00Z"}
        copy = {**first, "_ledger": str(run / LAUNCH_LEDGER)}  # the same job recorded in a second ledger
        state = submission_state(run, generation, [second, first, copy])
        self.assertEqual(state["current_submission"]["job_id"], "92")
        self.assertEqual([row["job_id"] for row in state["current_submissions"]], ["91", "92"])
        self.assertIn("2 submissions", state["match_rule"])
        single = submission_state(run, generation, [first])
        self.assertEqual([row["job_id"] for row in single["current_submissions"]], ["91"])
        self.assertNotIn("submissions of this generation", single["match_rule"])
        self.assertEqual(submission_state(run, generation, [])["current_submissions"], [])

    def test_ledger_ancestors_are_limited_to_eight_levels(self) -> None:
        top = self.root / "top"
        parent = top
        for level in range(1, 9):
            parent = parent / f"d{level}"
        run = write_step1_run(parent, "run")  # ancestors: d8 (1) ... d1 (8), top (9)
        write_legacy_launch_ledger(top, [])
        self.assertEqual(ledger_paths_for(run), [])
        write_legacy_launch_ledger(top / "d1", [])
        self.assertEqual(ledger_paths_for(run), [(top / "d1" / LAUNCH_LEDGER).resolve()])
        self.assertIn((top / LAUNCH_LEDGER).resolve(), ledger_paths_for(run, [top]))  # an explicit root still counts

    def test_ledger_paths_are_deduplicated_and_ordered(self) -> None:
        step1 = self.root / "Step1"
        run = write_step1_run(step1, "runA")
        write_manifest(step1, [run])
        write_legacy_launch_ledger(step1, [])
        write_legacy_launch_ledger(run, [])
        write_legacy_launch_ledger(self.root, [])  # above the manifest root: not an ancestor candidate
        paths = ledger_paths_for(run, [step1, step1])
        self.assertEqual(paths, [(run / LAUNCH_LEDGER).resolve(), (step1 / LAUNCH_LEDGER).resolve()])
        self.assertIn((self.root / LAUNCH_LEDGER).resolve(), ledger_paths_for(run, [self.root]))


class LaunchLedgerTests(LineageTestCase):
    def test_append_upgrades_schema_one_and_keeps_history(self) -> None:
        step1 = self.root / "Step1"
        run = write_step1_run(step1, "runA", nsw=388, potim=0.5)
        archive = run / ".interfaceforge" / "archive" / f"step1_repair_{_stamp(1.0)}"
        _legacy_record(run, archive=str(archive))
        write_legacy_launch_ledger(step1, [_launch_row(run, step1, job_id="111")], age_hours=5.0)
        generation = current_generation(run)

        new_row = {
            **_launch_row(run, step1, job_id="222"),
            "generation": generation.generation,
            "generation_id": generation.generation_id,
            "submitted_at": utc_now_iso(),
            "batch_id": "b-20260923T101500Z",
            "_ledger": "must not be persisted",
        }
        batch = {"batch_id": "b-20260923T101500Z", "started_at": utc_now_iso()}
        path = append_launch_rows(step1, [new_row], batch=batch)
        self.assertEqual(path, (step1 / LAUNCH_LEDGER).resolve())
        payload = read_json(path)
        self.assertEqual(payload["format"], "interfaceforge-step1-launch")
        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["status"], "SUBMITTED")
        self.assertEqual(payload["latest_batch_id"], "b-20260923T101500Z")
        self.assertEqual(payload["legacy_import"]["rows"], 1)
        self.assertEqual([row["job_id"] for row in payload["runs"]], ["111", "222"])
        legacy, current = payload["runs"]
        self.assertTrue(legacy["legacy"])
        recorded = _epoch(legacy["legacy_recorded_at"])
        expected = datetime.now(timezone.utc).timestamp() - 5.0 * 3600.0
        self.assertLess(abs(recorded - expected), 120.0)
        self.assertEqual(current["generation_id"], generation.generation_id)
        self.assertNotIn("_ledger", current)
        for key in ("notes", "detail", "root", "relative_path", "directory", "launcher", "kind"):
            self.assertIn(key, current)
        self.assertEqual(len(payload["batches"]), 1)
        self.assertEqual(payload["batches"][0]["submitted"], 1)
        self.assertEqual(payload["batches"][0]["failed"], 0)

        # The imported legacy row keeps its pre-upgrade time: still historical.
        state = submission_state(run, generation, rows_for_run(run, ledger_paths_for(run)))
        self.assertTrue(state["current_submitted"])
        self.assertEqual(state["current_submission"]["job_id"], "222")
        self.assertEqual([row["job_id"] for row in state["historical_submissions"]], ["111"])

        with (step1 / LAUNCH_TSV).open(encoding="utf-8", newline="") as handle:
            table = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(list(table[0]), list(LAUNCH_TSV_COLUMNS))
        self.assertEqual([row["job_id"] for row in table], ["111", "222"])
        self.assertEqual(table[1]["generation_id"], generation.generation_id)

        # Exact duplicates are skipped; a later failing batch is appended.
        append_launch_rows(step1, [new_row], batch=batch)
        failed = _launch_row(run, step1, job_id="", status="FAILED", detail="sbatch: error")
        append_launch_rows(step1, [failed], batch={"batch_id": "b-2"})
        payload = read_json(path)
        self.assertEqual([row["status"] for row in payload["runs"]], ["SUBMITTED", "SUBMITTED", "FAILED"])
        self.assertEqual(payload["status"], "FAILED")
        self.assertEqual(payload["latest_batch_id"], "b-2")
        self.assertEqual([item["batch_id"] for item in payload["batches"]], ["b-20260923T101500Z", "b-2"])
        self.assertEqual(payload["runs"][2]["batch_id"], "b-2")
        tsv = (step1 / LAUNCH_TSV).read_text(encoding="utf-8")
        self.assertIn("111", tsv)
        self.assertIn("sbatch: error", tsv)
        self.assertEqual(list(step1.glob("*.tmp")), [])

    def test_append_pins_the_time_of_schema_two_rows_without_generation_id(self) -> None:
        step1 = self.root / "Step1"
        run = write_step1_run(step1, "runA")
        sibling = write_step1_run(step1, "runB")
        payload = {
            "format": "interfaceforge-step1-launch",
            "schema_version": 2,
            "root": str(step1),
            "status": "SUBMITTED",
            "latest_batch_id": "b-old",
            "batches": [{"batch_id": "b-old", "status": "SUBMITTED", "submitted": 1, "failed": 0}],
            "runs": [_launch_row(run, step1, job_id="30")],  # hand-edited: no generation_id, no time
        }
        ledger = step1 / LAUNCH_LEDGER
        atomic_write_json(ledger, payload)
        old = datetime.now(timezone.utc).timestamp() - 10 * 3600.0
        os.utime(ledger, (old, old))
        old_iso = datetime.fromtimestamp(old, tz=timezone.utc).replace(microsecond=0).isoformat()
        self.assertEqual(load_ledger_rows(ledger)[0]["legacy_recorded_at"], old_iso)

        new = {**_launch_row(sibling, step1, job_id="31"), "generation": 0, "generation_id": GEN0_ID}
        append_launch_rows(step1, [new], batch={"batch_id": "b-new"})
        written = read_json(ledger)
        self.assertEqual(written["schema_version"], 2)
        self.assertNotIn("legacy_import", written)
        self.assertEqual([item["batch_id"] for item in written["batches"]], ["b-old", "b-new"])
        first, second = load_ledger_rows(ledger)
        self.assertTrue(first["legacy"])
        self.assertEqual(first["legacy_recorded_at"], old_iso)  # survived the rewrite
        self.assertNotIn("legacy", second)
        self.assertNotIn("legacy_recorded_at", second)

    def test_unreadable_ledger_is_preserved_before_rewrite(self) -> None:
        step1 = self.root / "Step1"
        run = write_step1_run(step1, "runA")
        (step1 / LAUNCH_LEDGER).write_text("{broken", encoding="utf-8")
        append_launch_rows(step1, [_launch_row(run, step1, job_id="9")], batch={"batch_id": "b-1"})
        backups = list(step1.glob(f"{LAUNCH_LEDGER}.unreadable-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(encoding="utf-8"), "{broken")
        self.assertEqual([row["job_id"] for row in read_json(step1 / LAUNCH_LEDGER)["runs"]], ["9"])

    def test_each_failed_batch_is_recorded(self) -> None:
        # FAILED rows carry no job id: a second failure of the same run/generation is new provenance.
        step1 = self.root / "Step1"
        run = write_step1_run(step1, "runA")
        failed = {
            **_launch_row(run, step1, job_id="", status="FAILED"),
            "generation": 1,
            "generation_id": "g1-repair-X",
        }
        append_launch_rows(step1, [{**failed, "detail": "QOS limit"}], batch={"batch_id": "b-1"})
        append_launch_rows(step1, [{**failed, "detail": "QOS limit"}], batch={"batch_id": "b-1"})  # verbatim retry
        path = append_launch_rows(step1, [{**failed, "detail": "invalid account"}], batch={"batch_id": "b-2"})
        payload = read_json(path)
        self.assertEqual([row["detail"] for row in payload["runs"]], ["QOS limit", "invalid account"])
        self.assertEqual([row["batch_id"] for row in payload["runs"]], ["b-1", "b-2"])
        batches = {item["batch_id"]: item for item in payload["batches"]}
        self.assertEqual((batches["b-1"]["failed"], batches["b-1"]["status"]), (1, "FAILED"))
        self.assertEqual((batches["b-2"]["failed"], batches["b-2"]["submitted"]), (1, 0))
        self.assertEqual(batches["b-2"]["status"], "FAILED")
        self.assertEqual((payload["status"], payload["latest_batch_id"]), ("FAILED", "b-2"))
        tsv = (step1 / LAUNCH_TSV).read_text(encoding="utf-8")
        self.assertIn("invalid account", tsv)

    def test_batch_status_follows_what_the_call_recorded(self) -> None:
        step1 = self.root / "Step1"
        run = write_step1_run(step1, "runA")
        row = {**_launch_row(run, step1, job_id="61"), "generation": 0, "generation_id": GEN0_ID}
        append_launch_rows(step1, [row], batch={"batch_id": "b-1"})
        # The same job re-recorded under a new batch is skipped, yet that batch did not fail.
        path = append_launch_rows(step1, [row], batch={"batch_id": "b-2"})
        payload = read_json(path)
        self.assertEqual(len(payload["runs"]), 1)
        batches = {item["batch_id"]: item for item in payload["batches"]}
        self.assertEqual((batches["b-2"]["submitted"], batches["b-2"]["status"]), (0, "SUBMITTED"))
        path = append_launch_rows(step1, [], batch={"batch_id": "b-3"})
        payload = read_json(path)
        self.assertEqual(payload["status"], "EMPTY")  # nothing submitted is never reported as SUBMITTED

    def test_concurrent_appends_keep_every_row(self) -> None:
        step1 = self.root / "Step1"
        step1.mkdir()
        errors: list[BaseException] = []

        def worker(index: int) -> None:
            for step in range(6):
                row = {
                    "status": "SUBMITTED",
                    "job_id": f"{index}-{step}",
                    "directory": str(step1 / f"run{index}"),
                    "relative_path": f"run{index}",
                    "generation": 0,
                    "generation_id": GEN0_ID,
                }
                try:
                    append_launch_rows(step1, [row], batch={"batch_id": f"b-{index}"})
                except BaseException as exc:  # noqa: BLE001 - reported by the assertion below
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        self.assertEqual(errors, [])
        payload = read_json(step1 / LAUNCH_LEDGER)
        self.assertEqual(len(payload["runs"]), 24)
        self.assertEqual(len({row["job_id"] for row in payload["runs"]}), 24)
        self.assertEqual(sorted(path.name for path in step1.iterdir()), [LAUNCH_LEDGER, LAUNCH_TSV])

    def test_a_held_lock_blocks_writers_until_timeout(self) -> None:
        step1 = self.root / "Step1"
        run = write_step1_run(step1, "runA")
        ledger = write_legacy_launch_ledger(step1, [_launch_row(run, step1, job_id="1")])
        before = ledger.read_bytes()
        lock = step1 / f"{LAUNCH_LEDGER}.lock"
        lock.write_text("pid 1\n", encoding="utf-8")  # a live writer holds it
        with patch.object(step1_lineage, "_LOCK_TIMEOUT_SECONDS", 0.3):
            with self.assertRaises(SafetyError) as caught:
                append_launch_rows(step1, [_launch_row(run, step1, job_id="2")], batch={"batch_id": "b-1"})
            self.assertIn(str(lock), str(caught.exception))
            with self.assertRaises(SafetyError):
                seal_launch_rows(run, [ledger], retired_generation_id=GEN0_ID, new_generation_id="g1-repair-x")
        self.assertEqual(ledger.read_bytes(), before)
        self.assertTrue(lock.is_file())  # another process's lock is never removed while fresh

    def test_a_stale_lock_is_broken(self) -> None:
        step1 = self.root / "Step1"
        run = write_step1_run(step1, "runA")
        lock = step1 / f"{LAUNCH_LEDGER}.lock"
        lock.write_text("pid 1\n", encoding="utf-8")
        old = datetime.now(timezone.utc).timestamp() - 3600.0
        os.utime(lock, (old, old))  # left behind by a process that died an hour ago
        path = append_launch_rows(step1, [_launch_row(run, step1, job_id="3")], batch={"batch_id": "b-1"})
        self.assertEqual([row["job_id"] for row in read_json(path)["runs"]], ["3"])
        self.assertFalse(lock.exists())

    def test_seal_leaves_an_unreadable_ledger_untouched(self) -> None:
        step1 = self.root / "Step1"
        run = write_step1_run(step1, "runA")
        ledger = step1 / LAUNCH_LEDGER
        ledger.write_text('{"runs": [{"status": "SUBM', encoding="utf-8")
        self.assertEqual(
            seal_launch_rows(run, [ledger], retired_generation_id=GEN0_ID, new_generation_id="g1-repair-x"), []
        )
        self.assertEqual(ledger.read_text(encoding="utf-8"), '{"runs": [{"status": "SUBM')
        self.assertEqual(list(step1.glob("*.lock")), [])

    def test_seal_annotates_only_rows_of_that_run(self) -> None:
        step1 = self.root / "Step1"
        run = write_step1_run(step1, "runA")
        sibling = write_step1_run(step1, "runB")
        write_legacy_launch_ledger(step1, [_launch_row(run, step1, job_id="100")], age_hours=3.0)
        rows = [
            {**_launch_row(run, step1, job_id="101"), "generation": 1, "generation_id": "g1-repair-A"},
            {**_launch_row(run, step1, job_id="102"), "generation": 1, "generation_id": "g1-repair-KEEP"},
            {**_launch_row(sibling, step1, job_id="103"), "generation": 1, "generation_id": "g1-repair-A"},
        ]
        root_ledger = append_launch_rows(step1, rows, batch={"batch_id": "b-1"})
        leaf_ledger = write_legacy_launch_ledger(sibling, [_launch_row(sibling, sibling, job_id="104")])
        leaf_before = leaf_ledger.read_bytes()

        modified = seal_launch_rows(
            run, [root_ledger, leaf_ledger], retired_generation_id="g1-repair-A", new_generation_id="g2-resume-B"
        )
        self.assertEqual(modified, [str(root_ledger)])
        self.assertEqual(leaf_ledger.read_bytes(), leaf_before)
        sealed = {row["job_id"]: row for row in read_json(root_ledger)["runs"]}
        self.assertEqual(sealed["100"]["superseded_by_generation_id"], "g2-resume-B")
        self.assertEqual(sealed["101"]["superseded_by_generation_id"], "g2-resume-B")
        self.assertIn("superseded_at", sealed["101"])
        self.assertNotIn("superseded_by_generation_id", sealed["102"])
        self.assertNotIn("superseded_by_generation_id", sealed["103"])
        self.assertEqual(
            seal_launch_rows(run, [root_ledger], retired_generation_id="g1-repair-A", new_generation_id="g3-x"), []
        )

    def test_sealing_a_schema_one_ledger_pins_legacy_times(self) -> None:
        step1 = self.root / "Step1"
        run = write_step1_run(step1, "runA")
        sibling = write_step1_run(step1, "runB")
        ledger = write_legacy_launch_ledger(
            step1, [_launch_row(run, step1, job_id="1"), _launch_row(sibling, step1, job_id="2")], age_hours=24.0
        )
        old_time = datetime.fromtimestamp(ledger.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat()
        seal_launch_rows(run, [ledger], retired_generation_id=GEN0_ID, new_generation_id="g1-repair-new")
        payload = read_json(ledger)
        self.assertEqual(payload["schema_version"], 1)
        rows = {row["job_id"]: row for row in payload["runs"]}
        self.assertEqual(rows["1"]["superseded_by_generation_id"], "g1-repair-new")
        self.assertNotIn("superseded_by_generation_id", rows["2"])
        # The sibling's row keeps its original (pre-rewrite) time.
        self.assertEqual(rows["2"]["legacy_recorded_at"], old_time)
        self.assertEqual(load_ledger_rows(ledger)[1]["legacy_recorded_at"], old_time)


if __name__ == "__main__":
    unittest.main()
