from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

from interfaceforge.cli import main
from interfaceforge.errors import SafetyError
from interfaceforge.step1_lineage import (
    GEN0_ID,
    Generation,
    archive_step1_state,
    atomic_write_json,
    build_segment_record,
    mark_record_submitted,
)
from interfaceforge.step1_scheduler import SchedulerGuard
from interfaceforge.step1_status import main as status_main
from interfaceforge.step1_status import recovery_category, render, step1_status

_TESTS = str(Path(__file__).resolve().parent)
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

from step1_fixtures import (  # noqa: E402
    fake_guard,
    linear_ramp,
    tree_snapshot,
    write_legacy_launch_ledger,
    write_legacy_repair_record,
    write_manifest,
    write_step1_run,
)

_INCAR = (
    "ISTART=1\nENCUT=400\nPREC=Normal\nEDIFF=1E-4\nALGO=Fast\nLREAL=Auto\n"
    "ISMEAR=0\nSIGMA=0.05\nISPIN=2\nLDAU=.TRUE.\nLDAUU=4.6 0.0\nLMAXMIX=4\n"
    "IBRION=0\nNSW=400\nPOTIM=1.0\nSMASS=-1\nNBLOCK=4\nTEBEG=100\nTEEND=100\n"
)


def _oszicar(steps: int, temp: int = 100) -> str:
    return "".join(
        f"   {i} F= -.1E1 E0= -.1E1  d E =0  T= {temp} \n" for i in range(1, steps + 1)
    )


def _tree(root: Path) -> Path:
    s1 = root / "Step1"
    running = s1 / "runA"
    running.mkdir(parents=True)
    (running / "INCAR").write_text(_INCAR, encoding="utf-8")
    (running / "POTCAR").write_text("ENMAX  =  400.000; ENMIN\nENMAX  =  250.000; ENMIN\n", encoding="utf-8")
    (running / "OSZICAR").write_text(_oszicar(137), encoding="utf-8")
    (running / "OUTCAR").write_text("running output, no timing block yet\n", encoding="utf-8")
    (running / "XDATCAR").write_text(
        "".join(f"Direct configuration=  {i}\n0 0 0\n" for i in range(1, 138)), encoding="utf-8"
    )

    done = s1 / "runB"
    done.mkdir(parents=True)
    (done / "INCAR").write_text(_INCAR, encoding="utf-8")
    (done / "OSZICAR").write_text(_oszicar(400), encoding="utf-8")
    (done / "OUTCAR").write_text(
        "...\n General timing and accounting informations for this job\n", encoding="utf-8"
    )

    fresh = s1 / "runC"
    fresh.mkdir(parents=True)
    (fresh / "INCAR").write_text(_INCAR.replace("ISTART=1", "ISTART=0"), encoding="utf-8")
    return s1


class _HermeticSchedulerMixin(unittest.TestCase):
    """``--scheduler auto`` never finds a real squeue, even on a Slurm login node."""

    def setUp(self) -> None:
        super().setUp()
        patcher = patch("interfaceforge.step1_scheduler.shutil.which", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)


class Step1StatusTests(_HermeticSchedulerMixin):
    def test_states_frames_and_incar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            s1 = _tree(Path(tmp))
            payload = step1_status(s1)
            rows = {row["run"]: row for row in payload["runs"]}

            self.assertEqual(payload["state_tally"], {"running": 1, "done": 1, "not-started": 1})

            a = rows["runA"]
            self.assertEqual(a["state"], "running")
            self.assertEqual(a["frames_oszicar"], 137)
            self.assertEqual(a["frames_xdatcar"], 137)
            self.assertEqual(a["nsw_target"], 400)
            self.assertEqual(a["percent_complete"], 34.2)
            self.assertAlmostEqual(a["produced_ps"], 0.137)
            self.assertEqual(a["incar"]["encut_ev"], 400.0)
            self.assertEqual(a["incar"]["ldauu"], "4.6 0.0")
            self.assertEqual(a["incar"]["encut_over_enmax"], 1.0)

            self.assertEqual(rows["runB"]["state"], "done")
            self.assertEqual(rows["runC"]["state"], "not-started")
            self.assertEqual(rows["runC"]["incar"]["istart"], 0)

    def test_done_early_when_timing_block_but_short(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "Step1" / "run"
            run.mkdir(parents=True)
            (run / "INCAR").write_text(_INCAR, encoding="utf-8")
            (run / "OSZICAR").write_text(_oszicar(210), encoding="utf-8")
            (run / "OUTCAR").write_text(
                "General timing and accounting informations for this job\n", encoding="utf-8"
            )
            payload = step1_status(run)
            self.assertEqual(payload["runs"][0]["state"], "done-early")

    def test_stalled_when_oszicar_is_old(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "Step1" / "run"
            run.mkdir(parents=True)
            (run / "INCAR").write_text(_INCAR, encoding="utf-8")
            (run / "OSZICAR").write_text(_oszicar(50), encoding="utf-8")
            (run / "OUTCAR").write_text("no timing block\n", encoding="utf-8")
            old = time.time() - 10 * 3600
            for name in ("OSZICAR", "OUTCAR"):
                os.utime(run / name, (old, old))
            payload = step1_status(run, stale_hours=6.0)
            self.assertEqual(payload["runs"][0]["state"], "stalled?")
            self.assertTrue(payload["runs"][0]["stale"])

    def test_tail_temperature_marks_ramp_as_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "Step1" / "run"
            run.mkdir(parents=True)
            incar = _INCAR.replace("TEEND=100", "TEEND=300")
            (run / "INCAR").write_text(incar, encoding="utf-8")
            temps = [100 + round(200 * (i - 1) / 399) for i in range(1, 401)]
            (run / "OSZICAR").write_text(
                "".join(
                    f"   {i} F= -.1E1 E0= -.1E1  d E =0  T= {temp} \n"
                    for i, temp in enumerate(temps, start=1)
                ),
                encoding="utf-8",
            )
            (run / "OUTCAR").write_text(
                "General timing and accounting informations for this job\n", encoding="utf-8"
            )
            row = step1_status(run)["runs"][0]
            self.assertLess(row["temperature_mean_k"], 250)
            self.assertGreater(row["thermal_tail_mean_k"], 250)
            self.assertEqual(row["thermal_tail_window_steps"], 50)
            self.assertAlmostEqual(row["thermal_ready_threshold_k"], 250.0)
            self.assertTrue(row["thermal_ready"])
            rendered = render(step1_status(run))
            self.assertIn("Tmean=", rendered)
            self.assertIn("Ttail50=", rendered)
            self.assertIn("ready", rendered)

    def test_finished_but_unstable_is_not_reported_done(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "Step1" / "run"
            run.mkdir(parents=True)
            (run / "INCAR").write_text(_INCAR, encoding="utf-8")
            text = _oszicar(399) + " 400 F= 0.100E+03 E0= 0.100E+03 d E=0 T= 300\n"
            (run / "OSZICAR").write_text(text, encoding="utf-8")
            (run / "OUTCAR").write_text(
                "General timing and accounting informations for this job\n", encoding="utf-8"
            )
            row = step1_status(run)["runs"][0]
            self.assertEqual(row["state"], "unstable")
            self.assertFalse(row["thermal_ready"])

    def test_error_marker_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "Step1" / "run"
            run.mkdir(parents=True)
            (run / "INCAR").write_text(_INCAR, encoding="utf-8")
            (run / "OSZICAR").write_text(_oszicar(12), encoding="utf-8")
            (run / "OUTCAR").write_text("ZBRENT: fatal error in bracketing\n", encoding="utf-8")
            payload = step1_status(run)
            self.assertEqual(payload["runs"][0]["state"], "error")

    def test_render_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            s1 = _tree(Path(tmp))
            text = render(step1_status(s1))
            self.assertIn("Step1 status:", text)
            self.assertIn("runA", text)
            self.assertIn("frames 137/400", text)
            self.assertEqual(main(["vasp", "step1-status", str(s1), "--json"]), 0)


# --------------------------------------------------------------------------- #
# State-aware status (spec section 10).  Every trajectory below is synthetic.
# --------------------------------------------------------------------------- #

_OLD_ROW_KEYS = {
    "run", "path", "state", "stale", "age_hours", "frames_oszicar", "frames_oszicar_segment",
    "accepted_prefix_steps", "frames_xdatcar", "last_step", "nsw_target", "nsw_segment_target",
    "percent_complete", "produced_ps", "target_ps", "temperature_mean_k", "temperature_std_k",
    "temperature_last_k", "thermal_target_k", "thermal_tail_window_steps", "thermal_tail_mean_k",
    "thermal_tail_min_k", "thermal_tail_max_k", "thermal_ready_threshold_k", "thermal_ready",
    "wavecar_present", "contcar_present", "potcar_present", "potcar_enmax_ev", "updated", "incar",
    "stability", "repair",
}  # fmt: skip
_OLD_PAYLOAD_KEYS = {
    "schema_version", "root", "stale_hours", "protocol", "manifest_temperature_k", "manifest_nsw",
    "state_tally", "runs",
}  # fmt: skip
_LINEAGE_KEYS = {
    "generation", "generation_id", "segment_kind", "legacy_record", "record_status", "original_nsw",
    "accepted_prefix_steps", "current_segment_steps", "cumulative_steps", "remaining_steps", "segment_nsw",
    "segment_potim_fs", "accepted_prefix_ps", "current_segment_ps", "accepted_total_ps", "segments",
    "ledger_exact", "temperature", "submitted", "submission", "historical_submissions", "interrupted_mutation",
}  # fmt: skip

_G1_ID = "g1-repair-20260920T101500Z"
_G2_ID = "g2-repair-20260922T101500Z"


def _only_row(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload["runs"]
    if len(rows) != 1:
        raise AssertionError(f"expected one run, got {[row['relative_path'] for row in rows]}")
    return rows[0]


def _rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["relative_path"]: row for row in payload["runs"]}


def _row_block(text: str, run_name: str) -> str:
    """The rendered lines of one run (its ``[state] name`` line up to the next run or the footer)."""

    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("  [") and line.split("] ", 1)[1].split("  ", 1)[0] == run_name:
            block = [line]
            for follow in lines[index + 1 :]:
                if follow.startswith("  [") or not follow.strip():
                    break
                block.append(follow)
            return "\n".join(block)
    raise AssertionError(f"{run_name} not rendered:\n{text}")


def _write_ledger(directory: Path, rows: list[dict[str, Any]]) -> Path:
    """A schema-2 step1_launch.json written the way step1-launch leaves it (cumulative ``runs``)."""

    payload = {
        "format": "interfaceforge-step1-launch",
        "schema_version": 2,
        "root": str(directory),
        "status": "SUBMITTED",
        "preflight": "PASS",
        "latest_batch_id": rows[-1]["batch_id"],
        "batches": [
            {
                "batch_id": row["batch_id"],
                "started_at": row["submitted_at"],
                "finished_at": row["submitted_at"],
                "status": "SUBMITTED",
                "submitted": 1,
                "failed": 0,
            }
            for row in rows
        ],
        "runs": rows,
    }
    path = directory / "step1_launch.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _launch_row(
    run: Path, root: Path, *, job_id: str, generation: int, generation_id: str, when: str
) -> dict[str, Any]:
    return {
        "status": "SUBMITTED",
        "job_id": job_id,
        "kind": "repair-prepared",
        "root": str(root),
        "relative_path": run.relative_to(root).as_posix(),
        "directory": str(run),
        "launcher": "runvasp.sh",
        "notes": "",
        "detail": f"Submitted batch job {job_id}",
        "generation": generation,
        "generation_id": generation_id,
        "submitted_at": when,
        "batch_id": f"b-{when[:10].replace('-', '')}",
    }


def _write_repair_generation_two(step1: Path) -> Path:
    """A run on repair generation 2: accepted 16 @ 1.0 fs + 52 @ 0.5 fs, segment NSW 332 @ 0.5 fs (100->300 K).

    The current segment has 120 completed steps following the ramp, and the
    record was written with ``build_segment_record`` exactly as step1-repair
    writes it, then marked submitted as step1-launch does.
    """

    run = write_step1_run(
        step1,
        "repaired",
        nsw=332,
        potim=0.5,
        tebeg=100.0,
        teend=300.0,
        steps=120,
        temperatures=linear_ramp(100.0, 300.0, 332),
    )
    parent = Generation(
        generation=1,
        generation_id=_G1_ID,
        kind="repair",
        legacy=False,
        record_path=None,
        record={},
        prepared_at="2026-09-20T10:15:00+00:00",
        prepared_epoch=datetime(2026, 9, 20, 10, 15, tzinfo=timezone.utc).timestamp(),
        original_nsw=400,
        accepted_prefix_steps=16,
        segment_nsw=384,
        segment_potim_fs=0.5,
        ledger=[{"generation": 0, "generation_id": GEN0_ID, "kind": "original", "steps": 16, "potim_fs": 1.0}],
        ledger_exact=True,
        status="SUBMITTED",
    )
    record = build_segment_record(
        "repair",
        run=run,
        generation=2,
        generation_id=_G2_ID,
        parent=parent,
        prepared_at="2026-09-22T10:15:00+00:00",
        original_nsw=400,
        accepted_prefix_steps=68,
        accepted_segments=[
            {"generation": 0, "generation_id": GEN0_ID, "kind": "original", "steps": 16, "potim_fs": 1.0,
             "tebeg_k": 300.0, "teend_k": 300.0, "restart_source": None},
            {"generation": 1, "generation_id": _G1_ID, "kind": "repair", "steps": 52, "potim_fs": 0.5,
             "tebeg_k": 100.0, "teend_k": 127.08, "restart_source": "XDATCAR frame 13"},
        ],  # fmt: skip
        ledger_exact=True,
        segment_nsw=332,
        segment_potim_fs=0.5,
        segment_schedule={
            "tebeg_k": 100.0, "teend_k": 300.0, "nsw": 332, "thermostat": "velocity-rescale (SMASS=-1)", "ramp": True
        },  # fmt: skip
        archive=str(run / ".interfaceforge" / "archive" / "step1_repair_g2_20260922T101500Z"),
        extra={"safe_segment_steps": 52, "rewind_frame": 13, "repair_precondition": True, "source": "XDATCAR"},
    )
    atomic_write_json(run / "step1_repair.json", record)
    mark_record_submitted(
        run, _G2_ID, {"job_id": "456", "submitted_at": "2026-09-22T11:00:00+00:00", "batch_id": "b-20260922"}
    )
    _write_ledger(
        step1,
        [
            _launch_row(run, step1, job_id="123", generation=1, generation_id=_G1_ID, when="2026-09-20T11:00:00+00:00"),
            _launch_row(run, step1, job_id="456", generation=2, generation_id=_G2_ID, when="2026-09-22T11:00:00+00:00"),
        ],
    )
    return run


class Step1StatusReadinessTests(_HermeticSchedulerMixin):
    """Thermal state is reported separately from completeness, stability and Step2 readiness."""

    def test_incomplete_warm_trajectory_is_thermal_ok_but_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_step1_run(Path(tmp) / "Step1", "warm", steps=33)
            payload = step1_status(run, scheduler="none")
            row = _only_row(payload)
            self.assertEqual(row["state"], "stalled?")
            self.assertTrue(row["thermal_tail_ok"])
            self.assertTrue(row["trajectory_stable"])
            self.assertTrue(row["thermal_ready"])  # legacy flag: tail ok AND stable
            self.assertFalse(row["complete"])
            self.assertFalse(row["ready_for_step2"])
            self.assertEqual(row["severity"], "ok")
            self.assertFalse(row["review_required"])
            text = render(payload)
            self.assertIn("Ttail33=300 K thermal-ok; incomplete", text)
            self.assertNotIn("ready for Step2", text)
            self.assertNotIn("(<250 K)", text)

    def test_completed_ramp_is_judged_on_its_tail_and_ready_for_step2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_step1_run(
                Path(tmp) / "Step1",
                "ramp",
                tebeg=100.0,
                teend=300.0,
                steps=400,
                temperatures=linear_ramp(100.0, 300.0, 400),
                outcar="finished",
            )
            payload = step1_status(run, scheduler="none")
            row = _only_row(payload)
            # The whole-run mean of a 100->300 K ramp sits near 200 K; the late window is the diagnostic.
            self.assertGreaterEqual(row["temperature_mean_k"], 190.0)
            self.assertLessEqual(row["temperature_mean_k"], 210.0)
            self.assertGreaterEqual(row["thermal_tail_mean_k"], 280.0)
            self.assertLessEqual(row["thermal_tail_mean_k"], 292.0)
            self.assertAlmostEqual(row["thermal_ready_threshold_k"], 250.0)
            self.assertEqual(row["state"], "done")
            self.assertTrue(row["thermal_tail_ok"])
            self.assertTrue(row["complete"])
            self.assertTrue(row["ready_for_step2"])
            self.assertEqual(row["recovery"]["category"], "done")
            text = render(payload)
            self.assertIn("thermal-ok; ready for Step2", text)
            self.assertNotIn("(review:", text)

    def test_unstable_run_with_warm_tail_prints_thermal_ok_and_unstable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_step1_run(
                Path(tmp) / "Step1",
                "runaway",
                steps=400,
                energies=lambda step: -10.0 if step <= 396 else 150.0,
                outcar="finished",
            )
            payload = step1_status(run, scheduler="none")
            row = _only_row(payload)
            self.assertEqual(row["state"], "unstable")
            self.assertGreater(row["thermal_tail_mean_k"], 250.0)
            self.assertTrue(row["thermal_tail_ok"])
            self.assertFalse(row["trajectory_stable"])
            self.assertFalse(row["thermal_ready"])
            self.assertFalse(row["ready_for_step2"])
            self.assertEqual(row["severity"], "unstable")
            self.assertEqual(row["recovery"]["category"], "repair")
            text = render(payload)
            self.assertIn("Ttail50=300 K thermal-ok; UNSTABLE", text)
            self.assertNotIn("(<250 K)", text)
            self.assertIn("stability: UNSTABLE — first unsafe ionic step 397", text)

    def test_cold_tail_still_prints_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_step1_run(Path(tmp) / "Step1", "cold", steps=400, temperatures=100.0, outcar="finished")
            payload = step1_status(run, scheduler="none")
            row = _only_row(payload)
            self.assertFalse(row["thermal_tail_ok"])
            self.assertTrue(row["complete"])
            self.assertFalse(row["ready_for_step2"])
            text = render(payload)
            self.assertIn("Ttail50=100 K (<250 K); complete; not ready", text)
            self.assertNotIn("thermal-ok", text)

    def test_benign_startup_transient_is_done_with_a_warning_not_unstable(self) -> None:
        # Magnetic DFT+U startup relaxation: ionic step 1 sits 77 eV above the
        # post-grace reference, then the run settles and finishes 400/400 near 302 K.
        with tempfile.TemporaryDirectory() as tmp:
            run = write_step1_run(
                Path(tmp) / "Step1",
                "transient",
                steps=400,
                temperatures=302.0,
                energies=lambda step: -10.0 + 77.0 if step == 1 else -10.0,
                outcar="finished",
            )
            payload = step1_status(run, scheduler="none")
            row = _only_row(payload)
            self.assertEqual(row["state"], "done")
            self.assertEqual(row["severity"], "warning")
            self.assertTrue(row["trajectory_stable"])
            self.assertTrue(row["stability"]["benign_warnings_only"])
            self.assertTrue(row["complete"])
            self.assertTrue(row["ready_for_step2"])
            self.assertTrue(row["review_required"])
            self.assertEqual(row["recovery"]["category"], "review")
            self.assertIn("confirm before Step2", row["recovery"]["reason"])
            self.assertIn("benign startup transient", row["recovery"]["reason"])
            text = render(payload)
            self.assertIn("thermal-ok; ready for Step2 (review: startup energy excursion", text)
            self.assertIn("stability: WARNING", text)
            self.assertIn("(benign startup transient)", text)
            self.assertNotIn("UNSTABLE", text)

    def test_complete_run_with_error_marker_and_no_footer_is_reviewed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_step1_run(Path(tmp) / "Step1", "crashed", steps=400, outcar="error")
            row = _only_row(step1_status(run, scheduler="none"))
            self.assertEqual(row["state"], "error")
            self.assertTrue(row["ready_for_step2"])
            self.assertEqual(row["recovery"]["category"], "review")
            self.assertIn("VASP error marker", row["recovery"]["reason"])

    def test_every_existing_key_is_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_step1_run(Path(tmp) / "Step1", "warm", steps=33)
            payload = step1_status(run, scheduler="none")
            self.assertLessEqual(_OLD_PAYLOAD_KEYS | {"scheduler", "action_tally"}, set(payload))
            row = _only_row(payload)
            new_keys = {
                "thermal_tail_ok", "trajectory_stable", "complete", "ready_for_step2", "review_required", "severity",
                "scheduler", "lineage", "recovery", "relative_path", "started", "activity", "launch",
            }  # fmt: skip
            self.assertLessEqual(_OLD_ROW_KEYS | new_keys, set(row))
            self.assertLessEqual(_LINEAGE_KEYS, set(row["lineage"]))
            self.assertEqual(
                set(row["lineage"]["temperature"]),
                {"segment_tebeg_k", "segment_teend_k", "segment_nsw", "ramp", "current_target_k", "thermostat"},
            )
            self.assertEqual(set(row["scheduler"]) & {"verified", "active", "jobs"}, {"verified", "active", "jobs"})
            self.assertEqual(payload["action_tally"], {"resume": 1})
            json.dumps(payload)  # the payload is plain JSON


class Step1StatusLineageTests(_HermeticSchedulerMixin):
    def test_repair_generation_two_lineage_and_submissions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            step1 = Path(tmp) / "Step1"
            run = _write_repair_generation_two(step1)
            payload = step1_status(step1, scheduler="none")
            row = _only_row(payload)
            lineage = row["lineage"]

            self.assertEqual(lineage["generation"], 2)
            self.assertEqual(lineage["generation_id"], _G2_ID)
            self.assertEqual(lineage["segment_kind"], "repair")
            self.assertFalse(lineage["legacy_record"])
            self.assertEqual(lineage["record_status"], "SUBMITTED")
            self.assertEqual(lineage["original_nsw"], 400)
            self.assertEqual(lineage["accepted_prefix_steps"], 68)
            self.assertEqual(lineage["current_segment_steps"], 120)
            self.assertEqual(lineage["cumulative_steps"], 188)
            self.assertEqual(lineage["remaining_steps"], 212)
            self.assertEqual(lineage["segment_nsw"], 332)
            self.assertEqual(lineage["segment_potim_fs"], 0.5)
            self.assertAlmostEqual(lineage["accepted_prefix_ps"], 0.016 + 0.026)
            self.assertAlmostEqual(lineage["current_segment_ps"], 0.060)
            self.assertAlmostEqual(lineage["accepted_total_ps"], 0.042 + 0.060)
            self.assertEqual([segment["steps"] for segment in lineage["segments"]], [16, 52])
            self.assertTrue(lineage["ledger_exact"])
            temperature = lineage["temperature"]
            self.assertEqual((temperature["segment_tebeg_k"], temperature["segment_teend_k"]), (100.0, 300.0))
            self.assertEqual(temperature["segment_nsw"], 332)
            self.assertTrue(temperature["ramp"])
            self.assertAlmostEqual(temperature["current_target_k"], 100.0 + 200.0 * 120 / 332)
            self.assertIsNone(lineage["interrupted_mutation"])
            self.assertEqual(lineage["warnings"], [])

            # One SUBMITTED row of the current generation, one of the older generation.
            self.assertTrue(lineage["submitted"])
            self.assertEqual(lineage["submission"]["job_id"], "456")
            self.assertEqual([item["job_id"] for item in lineage["historical_submissions"]], ["123"])
            self.assertFalse(any(key.startswith("_") for key in lineage["submission"]))

            # Row-level totals keep their meaning, now priced from the ledger.
            self.assertEqual(row["accepted_prefix_steps"], 68)
            self.assertEqual(row["frames_oszicar"], 188)
            self.assertEqual(row["nsw_target"], 400)
            self.assertEqual(row["nsw_segment_target"], 332)
            self.assertEqual(row["percent_complete"], 47.0)
            self.assertAlmostEqual(row["produced_ps"], 0.102)
            self.assertAlmostEqual(row["target_ps"], 0.042 + 332 * 0.5 / 1000.0)
            self.assertEqual(row["repair"]["safe_prefix_steps"], 68)
            self.assertEqual(row["recovery"]["category"], "resume")

            block = _row_block(render(payload), run.name)
            lineage_line = next(line for line in block.splitlines() if "lineage:" in line)
            for fragment in (
                "repair g2",
                _G2_ID,
                "target 400",
                "accepted 68 + segment 120 = 188",
                "segment NSW 332 @ POTIM 0.5 fs",
                "accepted 0.102 ps",
                "T 100→300 K ramp (now 172 K)",
                "current generation submitted (job 456)",
                "1 older submission",
            ):
                self.assertIn(fragment, lineage_line)
            self.assertIn("(accepted prefix 68; repair XDATCAR 30)", block)

    def test_legacy_schema1_record_and_ledger_are_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            step1 = Path(tmp) / "Step1"
            run = write_step1_run(step1, "legacy", nsw=388, potim=0.5, tebeg=100.0, steps=0, outcar=None)
            write_legacy_repair_record(run)
            # A schema-1 ledger whose SUBMITTED row predates the legacy repair (2026-09-01):
            # it is history and must not block the prepared repair.
            write_legacy_launch_ledger(
                step1,
                [{"status": "SUBMITTED", "job_id": "321", "relative_path": "legacy", "directory": str(run)}],
                age_hours=24.0 * 40,
            )
            payload = step1_status(step1, scheduler="none")
            row = _only_row(payload)
            lineage = row["lineage"]
            self.assertEqual(row["state"], "repair-prepared")
            self.assertEqual(row["accepted_prefix_steps"], 12)
            self.assertEqual(row["nsw_target"], 400)
            self.assertEqual(row["frames_oszicar"], 12)
            self.assertEqual(lineage["generation"], 1)
            self.assertEqual(lineage["generation_id"], "legacy-repair-g1-20260901T000000Z")
            self.assertTrue(lineage["legacy_record"])
            self.assertEqual(lineage["record_status"], "PREPARED")
            self.assertAlmostEqual(lineage["accepted_prefix_ps"], 0.012)
            self.assertFalse(lineage["submitted"])
            self.assertEqual([item["job_id"] for item in lineage["historical_submissions"]], ["321"])
            self.assertEqual(row["launch"]["kind"], "repair-prepared")
            self.assertEqual(row["recovery"]["category"], "launch")
            text = render(payload)
            self.assertIn("lineage: repair g1 (legacy-repair-g1-20260901T000000Z; legacy record", text)
            self.assertIn("current generation not submitted", text)

    def test_generation_zero_submission_is_shown_and_duplicates_warned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            step1 = Path(tmp) / "Step1"
            run = write_step1_run(step1, "fresh", steps=0, outcar=None)
            write_manifest(step1, [run])
            write_legacy_launch_ledger(
                step1,
                [
                    {"status": "SUBMITTED", "job_id": "700", "relative_path": "fresh", "directory": str(run)},
                    {"status": "SUBMITTED", "job_id": "701", "relative_path": "fresh", "directory": str(run)},
                ],
            )
            payload = step1_status(step1, scheduler="none")
            row = _only_row(payload)
            self.assertTrue(row["lineage"]["submitted"])
            self.assertEqual(row["lineage"]["submission_count"], 2)
            self.assertTrue(any("2 times" in warning for warning in row["lineage"]["warnings"]))
            text = render(payload)
            self.assertIn("lineage: original g0 (g0-prepare)", text)
            self.assertIn("lineage warning: current generation recorded as submitted 2 times", text)


class Step1StatusSchedulerTests(unittest.TestCase):
    """Scheduler state beats file age; a failing squeue is reported, never raised."""

    def test_scheduler_state_overrides_file_age(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_step1_run(Path(tmp) / "Step1", "old", steps=33, age_hours=10.0)

            running = _only_row(step1_status(run, scheduler=fake_guard({run: "RUNNING"})))
            self.assertEqual(running["state"], "running")
            self.assertEqual(running["scheduler"]["active"], True)
            self.assertTrue(running["scheduler"]["verified"])
            self.assertEqual([job["state"] for job in running["scheduler"]["jobs"]], ["RUNNING"])
            self.assertEqual(running["recovery"], {"category": "active", "reason": "Slurm job 9000 RUNNING"})
            self.assertFalse(running["complete"])

            queued = _only_row(step1_status(run, scheduler=fake_guard({run: "PENDING"})))
            self.assertEqual(queued["state"], "queued")
            self.assertEqual(queued["recovery"]["category"], "active")

            gone = _only_row(step1_status(run, scheduler=fake_guard()))
            self.assertEqual(gone["state"], "interrupted")
            self.assertIs(gone["scheduler"]["active"], False)
            self.assertEqual(gone["recovery"]["category"], "resume")

            unverified = _only_row(step1_status(run, scheduler="none"))
            self.assertEqual(unverified["state"], "stalled?")
            self.assertIsNone(unverified["scheduler"]["active"])
            self.assertFalse(unverified["scheduler"]["verified"])
            self.assertEqual(unverified["recovery"]["category"], "resume")

    def test_job_inside_run_counts_but_ancestor_workdir_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            step1 = Path(tmp) / "Step1"
            run = write_step1_run(step1, "leaf", steps=33)
            inside = _only_row(step1_status(run, scheduler=fake_guard({run / "precondition": "RUNNING"})))
            self.assertEqual(inside["recovery"]["category"], "active")
            ancestor = _only_row(step1_status(run, scheduler=fake_guard({step1: "RUNNING"})))
            self.assertEqual(ancestor["state"], "interrupted")
            self.assertEqual(ancestor["recovery"]["category"], "resume")

    def test_failing_squeue_is_reported_once_and_never_raised(self) -> None:
        calls = {"count": 0}

        def broken() -> Any:
            calls["count"] += 1
            raise SafetyError("could not query Slurm (TimeoutExpired: squeue); refusing rather than guessing job state")

        with tempfile.TemporaryDirectory() as tmp:
            step1 = Path(tmp) / "Step1"
            write_step1_run(step1, "a", steps=33)
            write_step1_run(step1, "b", steps=12)
            write_step1_run(step1, "c", steps=400, outcar="finished")
            payload = step1_status(step1, scheduler=SchedulerGuard("slurm", snapshot_factory=broken))
            self.assertEqual(calls["count"], 1)
            self.assertFalse(payload["scheduler"]["verified"])
            self.assertIn("could not query Slurm", payload["scheduler"]["reason"])
            for row in payload["runs"]:
                self.assertFalse(row["scheduler"]["verified"])
                self.assertIsNone(row["scheduler"]["active"])
            self.assertEqual(_rows(payload)["a"]["state"], "stalled?")
            self.assertEqual(payload["stale_hours"], 6.0)
            text = render(payload)
            self.assertIn("scheduler: NOT verified — could not query Slurm", text)

    def test_real_squeue_failure_path_is_non_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_step1_run(Path(tmp) / "Step1", "a", steps=33)
            missing = FileNotFoundError("squeue")
            with patch("interfaceforge.step1_scheduler.subprocess.run", side_effect=missing) as mocked:
                payload = step1_status(run, scheduler="slurm")
            self.assertEqual(mocked.call_count, 1)
            self.assertFalse(payload["scheduler"]["verified"])
            self.assertEqual(_only_row(payload)["state"], "stalled?")

    def test_stale_hours_none_resolves_from_the_scheduler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_step1_run(Path(tmp) / "Step1", "recent", steps=33, age_hours=1.0)

            unverified = step1_status(run, stale_hours=None, scheduler="none")
            self.assertEqual(unverified["stale_hours"], 6.0)
            row = _only_row(unverified)
            self.assertEqual(row["state"], "running")
            self.assertEqual(row["recovery"]["category"], "active")
            self.assertTrue(row["recovery"]["reason"].startswith("updated 60 min ago"))

            verified = step1_status(run, stale_hours=None, scheduler=fake_guard())
            self.assertAlmostEqual(verified["stale_hours"], 0.1)
            self.assertIn("Slurm verified", verified["stale_hours_reason"])
            row = _only_row(verified)
            self.assertEqual(row["state"], "interrupted")
            self.assertEqual(row["recovery"]["category"], "resume")

            # An explicit 6 h keeps the file-age window even when Slurm is verified.
            explicit = _only_row(step1_status(run, stale_hours=6.0, scheduler=fake_guard()))
            self.assertEqual(explicit["state"], "interrupted")
            self.assertEqual(explicit["recovery"]["category"], "active")

            # The Python default and the module CLI resolve like iface vasp step1-status (and recover).
            default = step1_status(run, scheduler=fake_guard())
            self.assertAlmostEqual(default["stale_hours"], 0.1)
            self.assertEqual(_only_row(default)["recovery"]["category"], "resume")
            from interfaceforge import step1_status as status_module

            with patch.object(status_module, "as_guard", return_value=fake_guard()):
                stream = io.StringIO()
                with redirect_stdout(stream):
                    status_module.main([str(run), "--json"])
            payload = json.loads(stream.getvalue())
            self.assertAlmostEqual(payload["stale_hours"], 0.1)
            self.assertNotIn("explicit", payload["stale_hours_reason"])

    def test_state_stale_flag_and_category_use_one_activity_moment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_step1_run(Path(tmp) / "Step1", "a", steps=40, age_hours=10.0)
            moment = time.time() - 1800.0
            os.utime(run / "OUTCAR", (moment, moment))  # only OUTCAR moved (a long SCF step)
            row = _only_row(step1_status(run, scheduler="none"))
            self.assertEqual(row["recovery"]["category"], "active")
            self.assertEqual(row["state"], "running")
            self.assertFalse(row["stale"])
            self.assertAlmostEqual(row["age_hours"], 0.5, delta=0.05)
            self.assertEqual(row["updated"], row["activity"]["updated"])

    def test_shared_guard_snapshot_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_step1_run(Path(tmp) / "Step1", "a", steps=33)
            guard = fake_guard({run: "RUNNING"})
            first = guard.snapshot
            payload = step1_status(run, scheduler=guard)
            self.assertIs(guard.snapshot, first)
            self.assertEqual(_only_row(payload)["state"], "running")


class Step1StatusRecoveryCategoryTests(_HermeticSchedulerMixin):
    def test_categories_across_a_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            step1 = Path(tmp) / "Step1"
            write_step1_run(step1, "done", steps=400, outcar="finished")
            write_step1_run(step1, "resume", steps=33)
            write_step1_run(
                step1, "transient", steps=33, energies=lambda step: -10.0 + 77.0 if step == 1 else -10.0
            )
            write_step1_run(step1, "repair", steps=33, energies=lambda step: -10.0 if step < 25 else 100.0)
            fresh = write_step1_run(step1, "launch_gen0", steps=0, outcar=None)
            prepared = write_step1_run(step1, "launch_record", nsw=388, potim=0.5, steps=0, outcar=None)
            write_legacy_repair_record(prepared)
            unlisted = write_step1_run(step1, "unlisted", steps=0, outcar=None)
            no_kpoints = write_step1_run(step1, "no_kpoints", steps=0, outcar=None)
            (no_kpoints / "KPOINTS").unlink()
            no_launcher = write_step1_run(step1, "no_launcher", steps=0, outcar=None, launcher=None)
            write_step1_run(step1, "cold", steps=400, temperatures=100.0, outcar="finished")
            write_step1_run(step1, "spike", steps=33, energies=lambda step: 90.0 if step == 25 else -10.0)
            write_step1_run(step1, "no_step", steps=0, partial_scf_tail=5)
            write_step1_run(step1, "error", steps=33, outcar="error")
            interrupted = write_step1_run(step1, "interrupted", steps=33)
            archive = archive_step1_state(interrupted, "step1_resume_g1")  # left IN_PROGRESS on purpose
            unreadable = write_step1_run(step1, "unreadable", steps=33)
            (unreadable / "step1_repair.json").write_text("{truncated", encoding="utf-8")
            write_step1_run(step1, "recent", steps=33, age_hours=0.0)
            write_manifest(step1, [fresh, no_kpoints, no_launcher])
            del unlisted

            rows = _rows(step1_status(step1, scheduler="none"))
            expected = {
                "done": "done",
                "resume": "resume",
                "transient": "resume",
                "repair": "repair",
                "launch_gen0": "launch",
                "launch_record": "launch",
                "unlisted": "review",
                "no_kpoints": "review",
                "no_launcher": "review",
                "cold": "review",
                "spike": "review",
                "no_step": "review",
                "error": "review",
                "interrupted": "review",
                "unreadable": "review",
                "recent": "active",
            }
            self.assertEqual({name: row["recovery"]["category"] for name, row in rows.items()}, expected)

            def reason(name: str) -> str:
                return rows[name]["recovery"]["reason"]

            self.assertIn("benign startup transient", reason("transient"))
            self.assertIn("incomplete (33/400)", reason("resume"))
            self.assertIn("first unsafe step 25", reason("repair"))
            self.assertIn("prepared, never submitted", reason("launch_gen0"))
            self.assertIn("repair-prepared, never submitted", reason("launch_record"))
            self.assertEqual(rows["launch_record"]["state"], "repair-prepared")
            self.assertIn("not started and not launchable: not listed in", reason("unlisted"))
            self.assertIn("missing KPOINTS", reason("no_kpoints"))
            self.assertIn("no VASP launcher", reason("no_launcher"))
            self.assertIn("complete but Ttail below threshold", reason("cold"))
            self.assertIn("isolated energy spike", reason("spike"))
            self.assertIn("no completed ionic step", reason("no_step"))
            self.assertIn("VASP error marker", reason("error"))
            self.assertEqual(reason("interrupted"), f"interrupted recovery mutation; inspect {archive}")
            self.assertEqual(rows["interrupted"]["lineage"]["interrupted_mutation"], str(archive))
            self.assertIn("segment record UNREADABLE", reason("unreadable"))
            self.assertTrue(reason("recent").startswith("updated 0 min ago"))
            # The archive copy of the INCAR is never discovered as a run.
            self.assertNotIn("interrupted/.interfaceforge", " ".join(rows))

    def test_not_started_but_submitted_depends_on_scheduler_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            step1 = Path(tmp) / "Step1"
            run = write_step1_run(step1, "fresh", steps=0, outcar=None)
            write_manifest(step1, [run])
            write_legacy_launch_ledger(
                step1, [{"status": "SUBMITTED", "job_id": "777", "relative_path": "fresh", "directory": str(run)}]
            )
            unverified = _only_row(step1_status(step1, scheduler="none"))
            self.assertEqual(
                unverified["recovery"], {"category": "active", "reason": "submitted (job 777); scheduler not verified"}
            )
            verified = _only_row(step1_status(step1, scheduler=fake_guard()))
            self.assertEqual(verified["recovery"]["category"], "review")
            self.assertIn("submitted (job 777) but not queued and no output", verified["recovery"]["reason"])
            queued = _only_row(step1_status(step1, scheduler=fake_guard({run: "PENDING"})))
            self.assertEqual(queued["state"], "queued")
            self.assertEqual(queued["recovery"]["category"], "active")

    def test_unreadable_ledger_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            step1 = Path(tmp) / "Step1"
            run = write_step1_run(step1, "fresh", steps=0, outcar=None)
            write_manifest(step1, [run])
            (step1 / "step1_launch.json").write_text('{"runs": [', encoding="utf-8")
            row = _only_row(step1_status(step1, scheduler="none"))
            self.assertTrue(row["lineage"]["submitted"])
            self.assertEqual(row["recovery"]["category"], "review")
            self.assertIn("is unreadable", row["recovery"]["reason"])
            self.assertTrue(row["lineage"]["unreadable_ledgers"])

    def test_classifier_rules_on_synthetic_rows(self) -> None:
        base: dict[str, Any] = {
            "state": "stalled?",
            "started": True,
            "scheduler": {"verified": False, "active": None, "jobs": []},
            "lineage": {"current_segment_steps": 10, "cumulative_steps": 10, "original_nsw": 400},
            "stability": {"unstable": False, "severity": "ok"},
            "severity": "ok",
            "trajectory_stable": True,
            "complete": False,
            "activity": {"recent": False},
        }
        self.assertEqual(recovery_category(base)["category"], "resume")
        conflict = dict(base, lineage=dict(base["lineage"], record_status="CONFLICT", conflict="both records exist"))
        self.assertEqual(recovery_category(conflict)["category"], "review")
        self.assertIn("both records exist", recovery_category(conflict)["reason"])
        active = dict(base, scheduler={"verified": True, "active": True, "jobs": [{"job_id": "1", "state": "RUNNING"}]})
        self.assertEqual(recovery_category(dict(active, lineage=conflict["lineage"]))["category"], "active")
        self.assertEqual(recovery_category(dict(base, state="no-incar"))["category"], "review")
        warned = dict(base, severity="warning", stability={"severity": "warning", "warnings": ["scf elevated"]})
        self.assertEqual(
            recovery_category(warned), {"category": "review", "reason": "scf elevated; review before resuming"}
        )
        unknown_tail = dict(base, complete=True, ready_for_step2=False, thermal_tail_ok=None)
        self.assertEqual(recovery_category(unknown_tail)["category"], "review")


class Step1StatusReadOnlyTests(_HermeticSchedulerMixin):
    def test_status_render_and_cli_never_mutate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            step1 = Path(tmp) / "Step1"
            fresh = write_step1_run(step1, "fresh", steps=0, outcar=None)
            write_step1_run(step1, "stalled", steps=33)
            legacy = write_step1_run(step1, "legacy", nsw=388, potim=0.5, steps=20)
            # A legacy record whose archive chain is broken (previous prefix known only as a count).
            write_legacy_repair_record(
                legacy, safe_prefix_steps=30, safe_segment_steps=18, previous_safe_prefix_steps=12
            )
            leaf = write_step1_run(step1, "leaf_launched", steps=0, outcar=None)
            write_legacy_launch_ledger(
                leaf, [{"status": "SUBMITTED", "job_id": "11", "relative_path": ".", "directory": str(leaf)}]
            )
            write_legacy_launch_ledger(
                step1, [{"status": "SUBMITTED", "job_id": "12", "relative_path": "stalled"}], age_hours=30.0
            )
            half = write_step1_run(step1, "half_mutated", steps=33)
            archive_step1_state(half, "step1_repair_g1")
            _write_repair_generation_two(step1 / "gen2")
            write_manifest(step1, [fresh, leaf])

            before = tree_snapshot(Path(tmp))
            for scheduler in ("none", fake_guard(), fake_guard({fresh: "PENDING"})):
                render(step1_status(step1, scheduler=scheduler))
                render(step1_status(legacy, scheduler=scheduler))
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["vasp", "step1-status", str(step1), "--json"]), 0)
                self.assertEqual(main(["vasp", "step1-status", str(step1)]), 0)
                self.assertEqual(status_main([str(step1), "--scheduler", "none", "--json"]), 0)
            after = tree_snapshot(Path(tmp))
            self.assertEqual(before, after)
            leftovers = [name for name in after if name.endswith((".tmp", ".lock")) or ".unreadable-" in name]
            self.assertEqual(leftovers, [])

            legacy_row = _rows(step1_status(step1, scheduler="none"))["legacy"]
            self.assertFalse(legacy_row["lineage"]["ledger_exact"])
            self.assertIn("ledger inexact", _row_block(render(step1_status(step1, scheduler="none")), "legacy"))

    def test_module_main_survives_a_stdout_that_cannot_encode_arrows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_step1_run(Path(tmp) / "Step1", "done", steps=400, outcar="finished")
            buffer = io.BytesIO()
            stream = io.TextIOWrapper(buffer, encoding="cp1252")  # e.g. a redirected stdout on Windows
            with patch("sys.stdout", stream):
                self.assertEqual(status_main([str(run), "--scheduler", "none"]), 0)
                stream.flush()
                text = buffer.getvalue().decode("cp1252")
            stream.detach()
            self.assertIn("] done  -> done: complete, stable and thermally ready for Step2", text)

    def test_render_first_line_and_footer_tallies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            step1 = Path(tmp) / "Step1"
            write_step1_run(step1, "done", steps=400, outcar="finished")
            write_step1_run(step1, "resume", steps=33)
            text = render(step1_status(step1, scheduler="none"))
            self.assertIn("  [done       ] done  → done: complete, stable and thermally ready for Step2", text)
            self.assertIn("] resume  → resume: incomplete (33/400), trajectory healthy", text)
            self.assertIn("2 runs  (done: 1  stalled?: 1)", text)
            self.assertIn("actions  (done: 1  resume: 1)", text)
            self.assertIn("scheduler: NOT verified — scheduler check disabled (--scheduler none)", text)


if __name__ == "__main__":
    unittest.main()
