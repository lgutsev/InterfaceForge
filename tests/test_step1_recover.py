"""Tests for ``step1-recover``: whole-tree classification, planning, journalled execution, lifecycle.

Every trajectory is synthetic (tests/step1_fixtures: a two-ion H/O cell with
fabricated OSZICAR/XDATCAR/CONTCAR files); nothing here is real NiO data and
no real Slurm, sbatch or VASP is ever called (fake/sequence scheduler guards,
patched ``sbatch``).
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Callable
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

from interfaceforge.errors import SafetyError
from interfaceforge.step1_lineage import (
    GEN0_ID,
    LAUNCH_LEDGER,
    RECOVER_JOURNAL,
    REPAIR_RECORD,
    RESUME_RECORD,
    atomic_write_json,
    build_segment_record,
    current_generation,
    format_temperature,
    read_json,
    utc_stamp,
)
from interfaceforge.step1_recover import (
    AUTO_CATEGORIES,
    RECOVER_CATEGORIES,
    execute_command,
    execute_step1_recovery,
    plan_step1_recovery,
    render_recovery_execution,
    render_recovery_plan,
)
from interfaceforge.vasp import _PRECONDITION_MARKER, parse_incar

_TESTS = str(Path(__file__).resolve().parent)
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)
from step1_fixtures import (  # noqa: E402
    FINISHED_OUTCAR,
    H_STEP,
    O_POSITION,
    RUNNING_OUTCAR,
    contcar_text,
    fake_guard,
    h_position,
    linear_ramp,
    mock_sbatch,
    oszicar_text,
    sequence_guard,
    set_age,
    tree_snapshot,
    write_legacy_repair_record,
    write_manifest,
    write_step1_run,
    xdatcar_text,
)

REFERENCE_F = -10.0
RUNAWAY_EV = 110.0  # sustained departure, well below the 500 eV catastrophic limit
SEGMENT_DRIFT = 0.0003  # fractional x per step in a repair/resume segment (the original drifts H_STEP)
REPAIR_G1_ID = "g1-repair-20260920T101500Z"
TWO_VASP_LINES = "#!/bin/bash\n#SBATCH -N 1\nmodule load vasp\nsrun -n4 vasp_std\nsrun -n4 vasp_std\n"


def _runaway(first_bad: int) -> Callable[[int], float]:
    """Quiet energies near F_ref, then a sustained +110 eV departure from ``first_bad`` onward."""

    return lambda step: REFERENCE_F + (RUNAWAY_EV if step >= first_bad else 0.02 * (step % 3))


def _startup_transient(step: int) -> float:
    """The real false positive: ionic step 1 at F_ref + 77 eV, a quiet trajectory afterwards."""

    return REFERENCE_F + 77.0 if step == 1 else REFERENCE_F


def _isolated_spike(step: int) -> float:
    """One post-grace step 100 eV off with both neighbours in band: a review-level warning."""

    return REFERENCE_F + (100.0 if step == 25 else 0.0)


def _poscar_x(run: Path) -> float:
    """Fractional x of the H ion in the run's current POSCAR (segment start)."""

    lines = (run / "POSCAR").read_text(encoding="utf-8").splitlines()
    start = next(index for index, line in enumerate(lines) if line.strip().lower().startswith("direct")) + 1
    return float(lines[start].split()[0])


def _write_segment(
    run: Path,
    *,
    steps: int,
    start_x: float,
    drift: float = SEGMENT_DRIFT,
    energies: Any = None,
    temperatures: Any = None,
    outcar: str = RUNNING_OUTCAR,
    nblock: int = 4,
) -> None:
    """Let the prepared segment 'run' on the cluster: fresh outputs continuing from the POSCAR at ``start_x``.

    Segment step ``j`` puts the H ion at ``start_x + drift * j``; the CONTCAR
    holds the next position (as VASP writes it) with a velocity block.
    """

    frames = []
    for index, step in enumerate(range(nblock, steps + 1, nblock), start=1):
        frames.append(
            f"Direct configuration= {index:6d}\n"
            f"  {start_x + drift * step:.8f}  0.10000000  0.50000000\n"
            f"  {O_POSITION[0]:.8f}  {O_POSITION[1]:.8f}  {O_POSITION[2]:.8f}\n"
        )
    (run / "XDATCAR").write_text(xdatcar_text(0) + "".join(frames), encoding="utf-8")  # xdatcar_text(0) = header
    (run / "OSZICAR").write_text(oszicar_text(steps, energies=energies, temperatures=temperatures), encoding="utf-8")
    (run / "CONTCAR").write_text(contcar_text(0, h_override=start_x + drift * (steps + 1)), encoding="utf-8")
    (run / "OUTCAR").write_text(outcar, encoding="utf-8")
    set_age(run, 10.0)


def _write_repair_g1_record(run: Path, *, status: str = "SUBMITTED") -> None:
    """A schema-2 step1_repair.json for generation 1: 12 accepted @ 1.0 fs, segment NSW 388 @ 0.5 fs (100->300 K)."""

    record = build_segment_record(
        "repair",
        run=run,
        generation=1,
        generation_id=REPAIR_G1_ID,
        parent=current_generation(run),
        prepared_at="2026-09-20T10:15:00+00:00",
        original_nsw=400,
        accepted_prefix_steps=12,
        accepted_segments=[
            {"generation": 0, "generation_id": GEN0_ID, "kind": "original", "steps": 12, "potim_fs": 1.0}
        ],
        ledger_exact=True,
        segment_nsw=388,
        segment_potim_fs=0.5,
        segment_schedule={
            "tebeg_k": 100.0,
            "teend_k": 300.0,
            "nsw": 388,
            "thermostat": "velocity-rescale (SMASS=-1)",
            "ramp": True,
        },
        archive=str(run / ".interfaceforge" / "archive" / "step1_repair_g1_20260920T101500Z"),
        extra={"safe_segment_steps": 12, "original_potim_fs": 1.0, "rewind_frame": 3, "source": "XDATCAR"},
    )
    record["status"] = status
    atomic_write_json(run / REPAIR_RECORD, record)
    set_age(run, 10.0)


def _write_prepared_repair_record(run: Path) -> str:
    """A never-submitted (PREPARED) generation-aware repair record on an idle run; returns its generation id."""

    parent = current_generation(run)
    generation_id = f"g1-repair-{utc_stamp()}"
    record = build_segment_record(
        "repair",
        run=run,
        generation=1,
        generation_id=generation_id,
        parent=parent,
        prepared_at="2026-09-21T08:00:00+00:00",
        original_nsw=400,
        accepted_prefix_steps=12,
        accepted_segments=[
            {"generation": 0, "generation_id": GEN0_ID, "kind": "original", "steps": 12, "potim_fs": 1.0}
        ],
        ledger_exact=True,
        segment_nsw=388,
        segment_potim_fs=0.5,
        segment_schedule={"tebeg_k": 100.0, "teend_k": 300.0, "nsw": 388, "thermostat": "x", "ramp": True},
        archive=str(run / ".interfaceforge" / "archive" / "step1_repair_g1_20260921T080000Z"),
        extra={"safe_segment_steps": 12, "original_potim_fs": 1.0},
    )
    atomic_write_json(run / REPAIR_RECORD, record)
    set_age(run, 48.0)
    return generation_id


def _idle_run(root: Path, name: str, **kwargs: Any) -> Path:
    """Inputs + launcher, no runtime outputs (as step1-prepare leaves a run)."""

    return write_step1_run(root, name, steps=0, contcar=None, xdatcar=False, outcar=None, age_hours=48.0, **kwargs)


def _names(plan: dict[str, Any], category: str) -> list[str]:
    return [entry["relative_path"] for entry in plan["categories"][category]]


def _entry(plan: dict[str, Any], name: str) -> dict[str, Any]:
    for entries in plan["categories"].values():
        for entry in entries:
            if entry["relative_path"] == name:
                return entry
    raise AssertionError(f"{name} not in plan")


def _journal(step1: Path) -> dict[str, Any]:
    return read_json(step1 / RECOVER_JOURNAL)


def _ledger_rows(step1: Path) -> list[dict[str, Any]]:
    return read_json(step1 / LAUNCH_LEDGER).get("runs") or []


class RecoverTestCase(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        # --scheduler auto never finds a real squeue, even on a Slurm login node.
        patcher = patch("interfaceforge.step1_scheduler.shutil.which", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()
        self.step1 = self.root / "Step1"

    def mixed_tree(self) -> dict[str, Path]:
        """One run per category (and both kinds of resume and launch)."""

        step1 = self.step1
        runs: dict[str, Path] = {}
        runs["done_300K"] = write_step1_run(step1, "done_300K", steps=400, outcar="finished")
        # Healthy original stalled on wall time: 33/400 @ 1.0 fs, 300 -> 300 K.
        runs["resume_original"] = write_step1_run(step1, "resume_original", steps=33)
        # Interrupted conservative ramp on repair generation 1 (NSW 388 @ 0.5 fs, 100 -> 300 K, 150 steps).
        runs["resume_repair_ramp"] = write_step1_run(
            step1,
            "resume_repair_ramp",
            nsw=388,
            potim=0.5,
            tebeg=100.0,
            teend=300.0,
            algo="Normal",
            steps=150,
            temperatures=linear_ramp(100.0, 300.0, 388),
        )
        _write_repair_g1_record(runs["resume_repair_ramp"])
        runs["repair_runaway"] = write_step1_run(step1, "repair_runaway", steps=30, energies=_runaway(25))
        # Complete 400/400, ~302 K, clean tail, but ionic step 1 is +77 eV (startup transient).
        runs["review_transient"] = write_step1_run(
            step1, "review_transient", steps=400, temperatures=302.0, energies=_startup_transient, outcar="finished"
        )
        runs["review_spike"] = write_step1_run(step1, "review_spike", steps=33, energies=_isolated_spike)
        # 10 h old files, but squeue lists it: the scheduler beats file age.
        runs["active_running"] = write_step1_run(step1, "active_running", steps=33, age_hours=10.0)
        runs["launch_gen0"] = _idle_run(step1, "launch_gen0")
        runs["launch_repair_prepared"] = _idle_run(step1, "launch_repair_prepared", nsw=388, potim=0.5)
        self.prepared_gid = _write_prepared_repair_record(runs["launch_repair_prepared"])
        write_manifest(step1, [runs["launch_gen0"]])
        return runs

    def mixed_guard(self, runs: dict[str, Path]) -> Any:
        return fake_guard({runs["active_running"]: "RUNNING"})


# --------------------------------------------------------------------------- #
# (1) classification + render, (2) dry-run zero mutation
# --------------------------------------------------------------------------- #


class RecoverPlanTests(RecoverTestCase):
    def test_mixed_tree_classification_counts_reasons_and_attached_plans(self) -> None:
        runs = self.mixed_tree()
        plan = plan_step1_recovery(self.step1, scheduler=self.mixed_guard(runs))

        self.assertEqual(
            (plan["format"], plan["schema_version"], plan["mode"]), ("interfaceforge-step1-recover-plan", 1, "dry-run")
        )
        self.assertEqual(plan["auto_categories"], list(AUTO_CATEGORIES))
        self.assertEqual(set(plan["categories"]), set(RECOVER_CATEGORIES))
        self.assertEqual(plan["counts"], {"done": 1, "resume": 2, "repair": 1, "launch": 2, "review": 2, "active": 1})
        self.assertEqual(_names(plan, "done"), ["done_300K"])
        self.assertEqual(_names(plan, "resume"), ["resume_original", "resume_repair_ramp"])
        self.assertEqual(_names(plan, "repair"), ["repair_runaway"])
        self.assertEqual(_names(plan, "review"), ["review_spike", "review_transient"])
        self.assertEqual(_names(plan, "active"), ["active_running"])
        self.assertEqual(_names(plan, "launch"), ["launch_gen0", "launch_repair_prepared"])
        self.assertTrue(plan["scheduler"]["verified"])
        self.assertEqual(plan["settings"]["stale_hours"], 0.1)  # Slurm verified: 6 min settle window
        self.assertEqual(plan["settings"]["repair"]["potim_fs"], 0.5)
        self.assertTrue(plan["settings"]["repair"]["precondition"])
        self.assertEqual(plan["settings"]["repair"]["ramp_from"], 100.0)

        # Reasons.
        self.assertEqual(_entry(plan, "active_running")["reason"], "Slurm job 9000 RUNNING")
        transient = _entry(plan, "review_transient")["reason"]
        self.assertIn("complete and Step2-ready by hard criteria", transient)
        self.assertIn("benign startup transient", transient)
        self.assertIn("confirm before Step2", transient)
        self.assertIn("isolated energy spike", _entry(plan, "review_spike")["reason"])
        self.assertIn("first unsafe step 25", _entry(plan, "repair_runaway")["reason"])
        self.assertIn("incomplete (33/400), trajectory healthy", _entry(plan, "resume_original")["reason"])
        self.assertIn("prepared, never submitted", _entry(plan, "launch_gen0")["reason"])
        self.assertIn("repair-prepared, never submitted", _entry(plan, "launch_repair_prepared")["reason"])
        self.assertIn("complete, stable and thermally ready", _entry(plan, "done_300K")["reason"])

        # Progress and lineage per entry.
        self.assertEqual(_entry(plan, "resume_original")["progress"], "33/400")
        self.assertEqual(_entry(plan, "resume_repair_ramp")["progress"], "162/400")
        self.assertEqual(_entry(plan, "done_300K")["progress"], "400/400")
        self.assertEqual(_entry(plan, "launch_gen0")["generation_id"], GEN0_ID)
        self.assertEqual(_entry(plan, "resume_repair_ramp")["generation_id"], REPAIR_G1_ID)

        # Attached dry plans / preflights (never executed on a dry run).
        original = _entry(plan, "resume_original")["action"]
        self.assertEqual(original["status"], "READY")
        self.assertEqual((original["restart_source"], original["resume_nsw"]), ("CONTCAR", 367))
        ramp = _entry(plan, "resume_repair_ramp")["action"]
        self.assertEqual((ramp["generation"], ramp["accepted_prefix_steps"], ramp["resume_nsw"]), (2, 162, 238))
        self.assertEqual(ramp["incar_changes"]["TEBEG"], "177.32")  # the 100 -> 300 K ramp continues
        repair = _entry(plan, "repair_runaway")["action"]
        self.assertEqual(repair["status"], "READY")
        self.assertEqual(
            (repair["safe_segment_steps"], repair["safe_prefix_steps"], repair["repair_nsw"]), (16, 16, 384)
        )
        self.assertTrue(repair["repair_precondition"])
        self.assertEqual(repair["repair_ramp_from_k"], 100.0)
        gen0 = _entry(plan, "launch_gen0")["action"]
        self.assertEqual((gen0["kind"], gen0["generation_id"], gen0["launcher"]), ("prepared", GEN0_ID, "runvasp.sh"))
        prepared = _entry(plan, "launch_repair_prepared")["action"]
        self.assertEqual((prepared["kind"], prepared["generation_id"]), ("repair-prepared", self.prepared_gid))
        for name in ("done_300K", "review_spike", "review_transient", "active_running"):
            self.assertIsNone(_entry(plan, name)["action"], name)
        for entries in plan["categories"].values():
            for entry in entries:
                self.assertIn("fingerprint", entry)
                self.assertIn("state", entry)
        json.dumps(plan, default=str)  # the CLI prints it

    def test_render_shows_every_category_header_per_run_lines_and_the_dry_run_footer(self) -> None:
        runs = self.mixed_tree()
        plan = plan_step1_recovery(self.step1, scheduler=self.mixed_guard(runs))
        text = render_recovery_plan(plan)

        self.assertTrue(text.startswith(f"Step1 recovery plan: {self.step1}"))
        self.assertIn("scheduler: verified", text)
        self.assertIn("activity window: 0.1 h", text)
        headers = [
            "active (1) -- left untouched",
            "done (1) -- left untouched",
            "review (2) -- left untouched",
            "launch (2) -- acted on by --execute",
            "resume (2) -- acted on by --execute",
            "repair (1) -- acted on by --execute",
        ]
        positions = [text.index(header) for header in headers]
        self.assertEqual(positions, sorted(positions))  # active, done, review, launch, resume, repair
        for name, entry_line in (
            ("active_running", "33/400  Slurm job 9000 RUNNING"),
            ("done_300K", "400/400  complete, stable and thermally ready for Step2"),
            ("resume_original", "33/400  incomplete (33/400), trajectory healthy"),
            ("repair_runaway", "30/400  hard-unstable"),
        ):
            line = next(line for line in text.splitlines() if line.strip().startswith(name))
            self.assertIn(entry_line, line)
        self.assertIn("resume g1 from CONTCAR (+33 accepted -> 33/400), NSW=367", text)
        self.assertIn(
            "resume g2 from CONTCAR (+150 accepted -> 162/400), NSW=238 @ 0.5 fs, TEBEG 100 -> 177.32 K", text
        )
        self.assertIn(
            "repair g1: rewind to segment step 16 (cumulative 16/400), NSW=384 @ 0.5 fs, ALGO=Normal, precondition, "
            "ramp 100->300 K",
            text,
        )
        self.assertIn("launch prepared g0-prepare with runvasp.sh", text)
        footer = text.splitlines()[-1]
        self.assertEqual(
            footer, f"Dry run: nothing changed. Execute resume+repair+launch with: {execute_command(self.step1)}"
        )
        self.assertTrue(footer.endswith(" --execute"))
        text.encode("ascii")  # the typographic characters are spelled in ASCII

        # --only / --no-submit tailor the footer command.
        tailored = render_recovery_plan(dict(plan, selected_categories=["repair"], submit=False))
        self.assertTrue(tailored.splitlines()[-1].endswith("--execute --only repair --no-submit"))
        self.assertIn("Execute repair with:", tailored)

    def test_dry_run_performs_zero_mutation_and_writes_no_journal(self) -> None:
        runs = self.mixed_tree()
        before = tree_snapshot(self.root)
        for scheduler in (self.mixed_guard(runs), "none", fake_guard()):
            plan = plan_step1_recovery(self.step1, scheduler=scheduler)
            render_recovery_plan(plan)
        plan_step1_recovery(runs["resume_original"], scheduler=fake_guard())  # a single run as root
        after = tree_snapshot(self.root)
        self.assertEqual(before, after)
        self.assertFalse((self.step1 / RECOVER_JOURNAL).exists())
        self.assertEqual([name for name in after if name.endswith((".tmp", ".lock"))], [])

    def test_option_validation(self) -> None:
        self.mixed_tree()
        with self.assertRaises(ValueError):
            plan_step1_recovery(self.step1, scheduler="none", repair_options={"potim": 0.5})
        with self.assertRaises(ValueError):
            plan_step1_recovery(
                self.step1, scheduler="none", resume_options={"precondition": True, "fresh_start": True}
            )
        with self.assertRaises(ValueError):
            plan_step1_recovery(self.step1, scheduler="none", diagnostic_options={"energy_jump_ev": -1.0})
        with self.assertRaises(ValueError):
            execute_step1_recovery(self.step1, scheduler="none", only=("review",))
        with self.assertRaises(SafetyError):
            plan_step1_recovery(self.step1, scheduler="none", repair_options={"potim_fs": 0.0})
        # Validation happens before anything is written.
        self.assertFalse((self.step1 / RECOVER_JOURNAL).exists())

    def test_single_run_root(self) -> None:
        run = write_step1_run(self.step1, "only_run", steps=33)
        plan = plan_step1_recovery(run, scheduler=fake_guard())
        self.assertEqual(_names(plan, "resume"), ["."])


# --------------------------------------------------------------------------- #
# (3) execute, (4) mid-execution failure, (5) race, (6) --only / --no-submit
# --------------------------------------------------------------------------- #


class RecoverExecuteTests(RecoverTestCase):
    def test_execute_prepares_and_submits_and_journals_every_step(self) -> None:
        runs = self.mixed_tree()
        untouched = ("done_300K", "review_spike", "review_transient", "active_running")
        before = {name: tree_snapshot(runs[name]) for name in untouched}
        messages: list[str] = []
        with mock_sbatch(5001) as calls:
            result = execute_step1_recovery(self.step1, scheduler=self.mixed_guard(runs), progress=messages.append)

        order = ["repair_runaway", "resume_original", "resume_repair_ramp", "launch_gen0", "launch_repair_prepared"]
        self.assertEqual([cwd for _command, cwd in calls], [str(runs[name]) for name in order])
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual([item["relative_path"] for item in result["changed"]], order[:3])
        self.assertEqual(
            [(item["relative_path"], item["job_id"]) for item in result["submitted"]],
            list(zip(order, ["5001", "5002", "5003", "5004", "5005"], strict=True)),
        )

        # Journal: one COMPLETED execution with per-run outcomes.
        journal = _journal(self.step1)
        self.assertEqual((journal["format"], journal["schema_version"]), ("interfaceforge-step1-recover", 1))
        self.assertEqual(len(journal["executions"]), 1)
        execution = journal["executions"][0]
        self.assertEqual(execution["status"], "COMPLETED")
        self.assertEqual(execution["selected_categories"], ["resume", "repair", "launch"])
        self.assertTrue(execution["submit"])
        self.assertTrue(execution["scheduler"]["verified"])
        for key in ("execution_id", "started_at", "finished_at"):
            self.assertTrue(execution[key], key)
        rows = {row["relative_path"]: row for row in execution["runs"]}
        self.assertEqual([row["relative_path"] for row in execution["runs"]], order)
        for name in order:
            row = rows[name]
            for key in ("directory", "category", "action", "prepared", "archive", "generation_id", "submitted",
                        "job_id", "error", "finished_at"):  # fmt: skip
                self.assertIn(key, row)
            self.assertTrue(row["submitted"], name)
            self.assertEqual(row["outcome"], "submitted")
            self.assertIsNone(row["error"])
            self.assertEqual(row["directory"], str(runs[name]))
        for name in order[:3]:
            self.assertTrue(rows[name]["prepared"])
            self.assertTrue(Path(rows[name]["archive"]).is_dir(), name)
        for name in order[3:]:
            self.assertFalse(rows[name]["prepared"])
            self.assertIsNone(rows[name]["archive"])
        self.assertEqual(rows["repair_runaway"]["category"], "repair")
        self.assertTrue(rows["repair_runaway"]["generation_id"].startswith("g1-repair-"))
        self.assertTrue(rows["resume_original"]["generation_id"].startswith("g1-resume-"))
        self.assertTrue(rows["resume_repair_ramp"]["generation_id"].startswith("g2-resume-"))
        self.assertEqual(rows["launch_gen0"]["generation_id"], GEN0_ID)
        self.assertEqual(rows["launch_repair_prepared"]["generation_id"], self.prepared_gid)

        # Root ledger: every submission with its generation id; run records marked SUBMITTED.
        ledger = [
            (row["relative_path"], row["status"], row["job_id"], row["generation_id"])
            for row in _ledger_rows(self.step1)
        ]
        self.assertEqual(
            ledger, [(name, "SUBMITTED", rows[name]["job_id"], rows[name]["generation_id"]) for name in order]
        )
        self.assertEqual(read_json(runs["repair_runaway"] / REPAIR_RECORD)["status"], "SUBMITTED")
        self.assertEqual(read_json(runs["resume_original"] / RESUME_RECORD)["status"], "SUBMITTED")
        self.assertEqual(read_json(runs["resume_repair_ramp"] / RESUME_RECORD)["status"], "SUBMITTED")
        self.assertFalse((runs["resume_repair_ramp"] / REPAIR_RECORD).exists())  # retired into the archive
        self.assertEqual(read_json(runs["launch_repair_prepared"] / REPAIR_RECORD)["status"], "SUBMITTED")

        # Recover's conservative repair defaults were applied.
        incar = parse_incar(runs["repair_runaway"] / "INCAR")
        self.assertEqual((incar["POTIM"], incar["ALGO"], incar["TEBEG"], incar["NSW"]), ("0.5", "Normal", "100", "384"))
        self.assertIn(_PRECONDITION_MARKER, (runs["repair_runaway"] / "runvasp.sh").read_text(encoding="utf-8"))
        self.assertTrue((runs["repair_runaway"] / "INCAR.precondition").is_file())

        # done / review / active runs are byte-identical.
        for name in untouched:
            self.assertEqual(tree_snapshot(runs[name]), before[name], name)
        text = "\n".join(messages)
        self.assertIn("CHANGED repair_runaway", text)
        self.assertIn("SUBMITTED launch_gen0: job 5004", text)
        self.assertIn("COMPLETED; changed 3 run(s)", text)
        self.assertEqual([name for name in tree_snapshot(self.root) if name.endswith((".tmp", ".lock"))], [])

        rendered = render_recovery_execution(result)
        self.assertIn(f"Execution {execution['execution_id']}: COMPLETED", rendered)
        self.assertIn("[submitted] launch_gen0", rendered)
        self.assertIn(f"Journal: {self.step1 / RECOVER_JOURNAL}", rendered)

        # Everything recoverable was handled: a fresh plan has nothing left to do automatically.
        again = plan_step1_recovery(self.step1, scheduler=self.mixed_guard(runs))
        self.assertEqual(sum(again["counts"][name] for name in AUTO_CATEGORIES), 0)

    def test_second_sbatch_failure_stops_with_a_precise_journal(self) -> None:
        a = write_step1_run(self.step1, "a_resume", steps=33)
        b = write_step1_run(self.step1, "b_repair", steps=30, energies=_runaway(25))
        c = _idle_run(self.step1, "c_launch")
        write_manifest(self.step1, [c])
        c_before = tree_snapshot(c)
        ok = Mock()
        ok.stdout = "Submitted batch job 7001\n"
        failure = subprocess.CalledProcessError(1, ["sbatch", "runvasp.sh"], output="", stderr="QOSMaxSubmitJobPerUser")
        with patch("interfaceforge.vasp.subprocess.run", side_effect=[ok, failure]) as sbatch:
            with self.assertRaises(SafetyError) as caught:
                execute_step1_recovery(self.step1, scheduler=fake_guard())
        self.assertEqual(sbatch.call_count, 2)
        message = str(caught.exception)
        self.assertIn("step1-recover stopped at b_repair (repair)", message)
        self.assertIn("it WAS prepared (changed) but not submitted", message)
        self.assertIn("Changed (prepared): a_resume, b_repair", message)
        self.assertIn("Submitted: a_resume (job 7001)", message)
        self.assertIn("Not attempted: c_launch", message)
        self.assertIn(f"Journal: {self.step1 / RECOVER_JOURNAL}", message)

        execution = _journal(self.step1)["executions"][0]
        self.assertEqual(execution["status"], "FAILED")
        self.assertTrue(execution["error"])
        rows = {row["relative_path"]: row for row in execution["runs"]}
        self.assertEqual(
            (rows["a_resume"]["prepared"], rows["a_resume"]["submitted"], rows["a_resume"]["job_id"]),
            (True, True, "7001"),
        )
        self.assertEqual((rows["b_repair"]["prepared"], rows["b_repair"]["submitted"]), (True, False))
        self.assertEqual(rows["b_repair"]["outcome"], "failed")
        self.assertIn("sbatch", rows["b_repair"]["error"])
        self.assertTrue(rows["b_repair"]["generation_id"].startswith("g1-repair-"))
        self.assertEqual(rows["c_launch"]["outcome"], "not attempted")
        self.assertEqual((rows["c_launch"]["prepared"], rows["c_launch"]["submitted"]), (False, False))
        self.assertEqual(tree_snapshot(c), c_before)
        # The launch ledger records the submitted job and the FAILED attempt.
        ledger = [(row["relative_path"], row["status"]) for row in _ledger_rows(self.step1)]
        self.assertEqual(ledger, [("a_resume", "SUBMITTED"), ("b_repair", "FAILED")])
        self.assertEqual(read_json(b / REPAIR_RECORD)["status"], "PREPARED")
        self.assertEqual(read_json(a / RESUME_RECORD)["status"], "SUBMITTED")

        # A fresh plan lists the prepared-but-unsubmitted run under launch.
        plan = plan_step1_recovery(self.step1, scheduler=fake_guard())
        self.assertEqual(_names(plan, "launch"), ["b_repair", "c_launch"])
        self.assertIn("repair-prepared, never submitted", _entry(plan, "b_repair")["reason"])
        self.assertIn("submitted (job 7001) but not queued", _entry(plan, "a_resume")["reason"])

    def test_run_that_becomes_active_between_planning_and_mutation_is_refused(self) -> None:
        a = write_step1_run(self.step1, "a_resume", steps=33)
        b = write_step1_run(self.step1, "b_resume", steps=40)
        c = _idle_run(self.step1, "c_launch")
        write_manifest(self.step1, [c])
        b_before, c_before = tree_snapshot(b), tree_snapshot(c)
        # Snapshot 0 (planning) is empty; from the first re-check on, squeue lists b.
        guard = sequence_guard([{}, {b: "RUNNING"}])
        with mock_sbatch(8001) as calls, self.assertRaises(SafetyError) as caught:
            execute_step1_recovery(self.step1, scheduler=guard)
        self.assertEqual([cwd for _command, cwd in calls], [str(a)])
        self.assertIn("active Slurm job(s) 9100 (RUNNING)", str(caught.exception))
        self.assertIn("it was not modified", str(caught.exception))
        self.assertEqual(tree_snapshot(b), b_before)
        self.assertEqual(tree_snapshot(c), c_before)
        execution = _journal(self.step1)["executions"][0]
        self.assertEqual(execution["status"], "FAILED")
        rows = {row["relative_path"]: row for row in execution["runs"]}
        self.assertEqual((rows["a_resume"]["prepared"], rows["a_resume"]["submitted"]), (True, True))
        self.assertIn("Refusing to mutate", rows["b_resume"]["error"])
        self.assertEqual((rows["b_resume"]["prepared"], rows["b_resume"]["submitted"]), (False, False))
        self.assertEqual(rows["c_launch"]["outcome"], "not attempted")

    def test_first_entry_refused_changes_nothing(self) -> None:
        a = write_step1_run(self.step1, "a_repair", steps=30, energies=_runaway(25))
        b = write_step1_run(self.step1, "b_resume", steps=33)
        before = {name: tree_snapshot(run) for name, run in (("a", a), ("b", b))}
        with mock_sbatch() as calls, self.assertRaises(SafetyError):
            execute_step1_recovery(self.step1, scheduler=sequence_guard([{}, {a: "PENDING"}]))
        self.assertEqual(calls, [])
        self.assertEqual({"a": tree_snapshot(a), "b": tree_snapshot(b)}, before)
        rows = _journal(self.step1)["executions"][0]["runs"]
        self.assertEqual([row["outcome"] for row in rows], ["failed", "not attempted"])

    def test_only_repair_touches_only_repair_entries(self) -> None:
        resume = write_step1_run(self.step1, "a_resume", steps=33)
        repair = write_step1_run(self.step1, "b_repair", steps=30, energies=_runaway(25))
        launch = _idle_run(self.step1, "c_launch")
        write_manifest(self.step1, [launch])
        before = {"resume": tree_snapshot(resume), "launch": tree_snapshot(launch)}
        with mock_sbatch(6001) as calls:
            result = execute_step1_recovery(self.step1, only=["repair"], scheduler=fake_guard())
        self.assertEqual([cwd for _command, cwd in calls], [str(repair)])
        self.assertEqual({"resume": tree_snapshot(resume), "launch": tree_snapshot(launch)}, before)
        execution = _journal(self.step1)["executions"][0]
        self.assertEqual(execution["selected_categories"], ["repair"])
        self.assertEqual([row["relative_path"] for row in execution["runs"]], ["b_repair"])
        self.assertEqual(result["status"], "COMPLETED")

    def test_no_submit_prepares_without_sbatch_and_the_next_plan_launches_them(self) -> None:
        resume = write_step1_run(self.step1, "a_resume", steps=33)
        repair = write_step1_run(self.step1, "b_repair", steps=30, energies=_runaway(25))
        launch = _idle_run(self.step1, "c_launch")
        write_manifest(self.step1, [launch])
        launch_before = tree_snapshot(launch)
        messages: list[str] = []
        with mock_sbatch() as calls:
            result = execute_step1_recovery(self.step1, submit=False, scheduler=fake_guard(), progress=messages.append)
        self.assertEqual(calls, [])
        self.assertEqual(result["submitted"], [])
        self.assertEqual([item["relative_path"] for item in result["changed"]], ["a_resume", "b_repair"])
        self.assertEqual(result["execution"]["skipped_launch_entries"], ["c_launch"])
        self.assertIn("--no-submit: leaving 1 launch entr(y/ies) alone: c_launch", messages[0])
        self.assertEqual(tree_snapshot(launch), launch_before)
        self.assertEqual(read_json(resume / RESUME_RECORD)["status"], "PREPARED")
        self.assertEqual(read_json(repair / REPAIR_RECORD)["status"], "PREPARED")
        self.assertFalse((self.step1 / LAUNCH_LEDGER).exists())
        rows = _journal(self.step1)["executions"][0]["runs"]
        self.assertEqual(
            [(row["prepared"], row["submitted"], row["outcome"]) for row in rows], [(True, False, "prepared")] * 2
        )

        plan = plan_step1_recovery(self.step1, scheduler=fake_guard())
        self.assertEqual(_names(plan, "launch"), ["a_resume", "b_repair", "c_launch"])
        self.assertIn("resume-prepared", _entry(plan, "a_resume")["reason"])
        self.assertIn("repair-prepared", _entry(plan, "b_repair")["reason"])

    def test_crash_inside_a_repair_leaves_an_in_progress_archive_named_in_the_journal(self) -> None:
        repair = write_step1_run(self.step1, "a_repair", steps=30, energies=_runaway(25))
        resume = write_step1_run(self.step1, "b_resume", steps=33)
        resume_before = tree_snapshot(resume)
        with (
            mock_sbatch() as calls,
            patch("interfaceforge.step1_repair.update_incar", side_effect=OSError("disk full")),
            self.assertRaises(SafetyError) as caught,
        ):
            execute_step1_recovery(self.step1, scheduler=fake_guard())
        self.assertEqual(calls, [])
        self.assertIn("disk full", str(caught.exception))
        self.assertIn("interrupted mutation is archived at", str(caught.exception))
        self.assertIn("Changed (prepared): none. Submitted: none. Not attempted: b_resume", str(caught.exception))
        row = _journal(self.step1)["executions"][0]["runs"][0]
        self.assertEqual((row["relative_path"], row["prepared"], row["outcome"]), ("a_repair", False, "failed"))
        archive = Path(row["interrupted_archive"])
        self.assertEqual(read_json(archive / "ARCHIVE_MANIFEST.json")["status"], "IN_PROGRESS")
        self.assertEqual(archive.parent.parent.parent, repair)
        self.assertEqual(tree_snapshot(resume), resume_before)
        # The half-done run is never touched again automatically.
        plan = plan_step1_recovery(self.step1, scheduler=fake_guard())
        self.assertIn("interrupted recovery mutation", _entry(plan, "a_repair")["reason"])
        self.assertEqual(_names(plan, "review"), ["a_repair"])

    def test_keyboard_interrupt_is_journalled_and_re_raised(self) -> None:
        write_step1_run(self.step1, "a_resume", steps=33)
        write_step1_run(self.step1, "b_resume", steps=33)
        with (
            patch("interfaceforge.step1_recover.execute_resume_plan", side_effect=KeyboardInterrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            execute_step1_recovery(self.step1, scheduler=fake_guard())
        execution = _journal(self.step1)["executions"][0]
        self.assertEqual((execution["status"], execution["interrupted"]), ("FAILED", True))
        self.assertEqual([row["outcome"] for row in execution["runs"]], ["failed", "not attempted"])

    def test_nothing_selected_writes_nothing(self) -> None:
        write_step1_run(self.step1, "done", steps=400, outcar="finished")
        before = tree_snapshot(self.root)
        result = execute_step1_recovery(self.step1, scheduler=fake_guard())
        self.assertEqual(result["status"], "NOTHING_TO_DO")
        self.assertIsNone(result["journal"])
        self.assertEqual(tree_snapshot(self.root), before)

    def test_unreadable_existing_journal_refuses_before_any_mutation(self) -> None:
        run = write_step1_run(self.step1, "a_resume", steps=33)
        (self.step1 / RECOVER_JOURNAL).write_text("{not json", encoding="utf-8")
        before = tree_snapshot(self.root)
        with mock_sbatch() as calls, self.assertRaisesRegex(SafetyError, "not a readable step1-recover journal"):
            execute_step1_recovery(self.step1, scheduler=fake_guard())
        self.assertEqual(calls, [])
        self.assertEqual(tree_snapshot(self.root), before)
        self.assertFalse((run / RESUME_RECORD).exists())

    def test_journal_appends_one_execution_per_execute(self) -> None:
        write_step1_run(self.step1, "a_resume", steps=33)
        write_step1_run(self.step1, "b_repair", steps=30, energies=_runaway(25))
        with mock_sbatch():
            execute_step1_recovery(self.step1, only=["resume"], scheduler=fake_guard())
            execute_step1_recovery(self.step1, only=["repair"], scheduler=fake_guard())
        executions = _journal(self.step1)["executions"]
        self.assertEqual([item["selected_categories"] for item in executions], [["resume"], ["repair"]])
        self.assertEqual(len({item["execution_id"] for item in executions}), 2)
        self.assertTrue(all(item["status"] == "COMPLETED" for item in executions))


# --------------------------------------------------------------------------- #
# (7) end-to-end lifecycle on one synthetic run
# --------------------------------------------------------------------------- #


class RecoverLifecycleTests(RecoverTestCase):
    def recover(self, job_id: int, expected: str) -> dict[str, Any]:
        """Plan (asserting the single entry's category), then execute with a mocked sbatch."""

        plan = plan_step1_recovery(self.step1, scheduler=fake_guard())
        self.assertEqual(plan["categories"][expected][0]["relative_path"], "OH50_run", plan["counts"])
        self.assertEqual(sum(plan["counts"].values()), 1)
        with mock_sbatch(job_id) as calls:
            result = execute_step1_recovery(self.step1, scheduler=fake_guard())
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(result["submitted"][0]["job_id"], str(job_id))
        return result

    def test_prepare_launch_repair_repair_again_resume_complete(self) -> None:
        run = _idle_run(self.step1, "OH50_run")  # generation 0 exactly as step1-prepare wrote it
        write_manifest(self.step1, [run])

        # 1. Never launched -> recover submits it (launch).
        first = self.recover(5001, "launch")
        self.assertEqual(first["submitted"][0]["generation_id"], GEN0_ID)
        self.assertEqual(first["changed"], [])

        # 2. The job ran and the energy ran away at step 25 -> repair generation 1.
        _write_segment(run, steps=30, start_x=h_position(0), drift=H_STEP, energies=_runaway(25))
        plan = plan_step1_recovery(self.step1, scheduler=fake_guard())
        self.assertIn("repair g1: rewind to segment step 16 (cumulative 16/400), NSW=384", render_recovery_plan(plan))
        second = self.recover(5002, "repair")
        g1 = second["changed"][0]["generation_id"]
        self.assertTrue(g1.startswith("g1-repair-"))
        record = read_json(run / REPAIR_RECORD)
        self.assertEqual((record["generation"], record["safe_prefix_steps"], record["repair_nsw"]), (1, 16, 384))
        self.assertEqual(record["status"], "SUBMITTED")
        incar = parse_incar(run / "INCAR")
        self.assertEqual((incar["NSW"], incar["POTIM"], incar["TEBEG"], incar["TEEND"]), ("384", "0.5", "100", "300"))
        self.assertIn(_PRECONDITION_MARKER, (run / "runvasp.sh").read_text(encoding="utf-8"))
        self.assertAlmostEqual(_poscar_x(run), h_position(16))

        # 3. The repaired segment fails again after 52 safe steps -> generation 2, no manual renaming.
        _write_segment(
            run, steps=70, start_x=_poscar_x(run), energies=_runaway(61), temperatures=linear_ramp(100, 300, 384)
        )
        third = self.recover(5003, "repair")
        g2 = third["changed"][0]["generation_id"]
        self.assertTrue(g2.startswith("g2-repair-"))
        record = read_json(run / REPAIR_RECORD)
        self.assertEqual(
            (record["generation"], record["parent_generation_id"], record["previous_safe_prefix_steps"]), (2, g1, 16)
        )
        self.assertEqual(
            (record["safe_segment_steps"], record["safe_prefix_steps"], record["repair_nsw"]), (52, 68, 332)
        )
        self.assertEqual(
            [(row["steps"], row["potim_fs"]) for row in record["accepted_segments"]], [(16, 1.0), (52, 0.5)]
        )
        self.assertEqual(parse_incar(run / "INCAR")["NSW"], "332")
        self.assertAlmostEqual(_poscar_x(run), h_position(16) + SEGMENT_DRIFT * 52)

        # 4. Generation 2 is interrupted while healthy after 100 steps -> resume (generation 3).
        _write_segment(run, steps=100, start_x=_poscar_x(run), temperatures=linear_ramp(100, 300, 332))
        plan = plan_step1_recovery(self.step1, scheduler=fake_guard())
        action = plan["categories"]["resume"][0]["action"]
        tebeg = format_temperature(100.0 + 200.0 * 100 / 332)
        self.assertEqual(tebeg, "160.24")
        self.assertEqual(
            (action["restart_source"], action["resume_nsw"], action["accepted_prefix_steps"]), ("CONTCAR", 232, 168)
        )
        self.assertEqual(action["incar_changes"]["TEBEG"], tebeg)
        fourth = self.recover(5004, "resume")
        g3 = fourth["changed"][0]["generation_id"]
        self.assertTrue(g3.startswith("g3-resume-"))
        self.assertFalse((run / REPAIR_RECORD).exists())
        record = read_json(run / RESUME_RECORD)
        self.assertEqual((record["generation"], record["parent_generation_id"], record["resume_nsw"]), (3, g2, 232))
        self.assertEqual(
            [(row["steps"], row["potim_fs"]) for row in record["accepted_segments"]], [(16, 1.0), (52, 0.5), (100, 0.5)]
        )
        self.assertEqual(record["electronic_start"]["mode"], "precondition")
        incar = parse_incar(run / "INCAR")
        self.assertEqual(
            (incar["NSW"], incar["TEBEG"], incar["TEEND"], incar["POTIM"], incar["ALGO"]),
            ("232", tebeg, "300", "0.5", "Normal"),
        )

        # 5. The resumed segment completes -> done; nothing is left to do.
        _write_segment(
            run,
            steps=232,
            start_x=_poscar_x(run),
            temperatures=linear_ramp(float(tebeg), 300.0, 232),
            outcar=FINISHED_OUTCAR,
        )
        plan = plan_step1_recovery(self.step1, scheduler=fake_guard())
        self.assertEqual(_names(plan, "done"), ["OH50_run"], plan["counts"])
        self.assertEqual(plan["categories"]["done"][0]["progress"], "400/400")
        journal_before = (self.step1 / RECOVER_JOURNAL).read_bytes()
        self.assertEqual(execute_step1_recovery(self.step1, scheduler=fake_guard())["status"], "NOTHING_TO_DO")
        self.assertEqual((self.step1 / RECOVER_JOURNAL).read_bytes(), journal_before)

        # The ledger keeps every historical submission with its generation id; the journal every execution.
        ledger = [
            (row["status"], row["job_id"], row["generation"], row["generation_id"]) for row in _ledger_rows(self.step1)
        ]
        self.assertEqual(
            ledger,
            [
                ("SUBMITTED", "5001", 0, GEN0_ID),
                ("SUBMITTED", "5002", 1, g1),
                ("SUBMITTED", "5003", 2, g2),
                ("SUBMITTED", "5004", 3, g3),
            ],
        )
        executions = _journal(self.step1)["executions"]
        self.assertEqual(
            [(item["runs"][0]["category"], item["runs"][0]["generation_id"]) for item in executions],
            [("launch", GEN0_ID), ("repair", g1), ("repair", g2), ("resume", g3)],
        )


# --------------------------------------------------------------------------- #
# (8) precondition blocker, (9) untrustworthy lineage / ledger
# --------------------------------------------------------------------------- #


class RecoverReviewRoutingTests(RecoverTestCase):
    def test_unwrappable_launcher_with_precondition_needs_review_and_nothing_is_mutated(self) -> None:
        repair = write_step1_run(self.step1, "a_repair", steps=30, energies=_runaway(25))
        resume = write_step1_run(self.step1, "b_resume", steps=33)
        for run in (repair, resume):
            (run / "runvasp.sh").write_text(TWO_VASP_LINES, encoding="utf-8")
            set_age(run, 10.0)
        before = tree_snapshot(self.root)

        plan = plan_step1_recovery(self.step1, scheduler=fake_guard(), resume_options={"precondition": True})
        self.assertEqual(_names(plan, "review"), ["a_repair", "b_resume"])
        for name in ("a_repair", "b_resume"):
            entry = _entry(plan, name)
            self.assertIn("cannot precondition", entry["reason"])
            self.assertIn("exactly one line that runs vasp", entry["reason"])
            self.assertIn(entry["status_category"], ("repair", "resume"))
        self.assertIn("(step1-status classified it 'repair'; not acted on)", render_recovery_plan(plan))

        with mock_sbatch() as calls:
            result = execute_step1_recovery(self.step1, scheduler=fake_guard(), resume_options={"precondition": True})
        self.assertEqual((result["status"], calls), ("NOTHING_TO_DO", []))
        self.assertEqual(tree_snapshot(self.root), before)

        # Without preconditioning the same runs are ordinary repair / resume entries.
        relaxed = plan_step1_recovery(self.step1, scheduler=fake_guard(), repair_options={"precondition": False})
        self.assertEqual((_names(relaxed, "repair"), _names(relaxed, "resume")), (["a_repair"], ["b_resume"]))

    def test_unreadable_or_conflicting_records_and_unreadable_ledgers_are_reviewed_never_mutated(self) -> None:
        unreadable = write_step1_run(self.step1, "a_unreadable", steps=33)
        (unreadable / REPAIR_RECORD).write_text("{truncated", encoding="utf-8")
        conflict = write_step1_run(self.step1, "b_conflict", nsw=388, potim=0.5, steps=33)
        write_legacy_repair_record(conflict)  # schema 1 (not generation-aware) ...
        atomic_write_json(  # ... next to a generation-aware resume record
            conflict / RESUME_RECORD,
            {"generation": 2, "generation_id": "g2-resume-20260922T000000Z", "status": "SUBMITTED",
             "segment_kind": "resume", "original_nsw": 400, "accepted_prefix_steps": 20},
        )  # fmt: skip
        set_age(conflict, 10.0)
        ledger_run = write_step1_run(self.step1, "c_bad_ledger", steps=33)
        (ledger_run / LAUNCH_LEDGER).write_text('{"runs": [', encoding="utf-8")
        set_age(ledger_run, 10.0)
        fresh = _idle_run(self.step1, "d_fresh_bad_ledger")
        (fresh / LAUNCH_LEDGER).write_text("[]", encoding="utf-8")
        set_age(fresh, 48.0)
        healthy = write_step1_run(self.step1, "e_healthy", steps=33)
        write_manifest(self.step1, [fresh])
        reviewed = ("a_unreadable", "b_conflict", "c_bad_ledger", "d_fresh_bad_ledger")
        before = {name: tree_snapshot(self.step1 / name) for name in reviewed}

        plan = plan_step1_recovery(self.step1, scheduler=fake_guard())
        self.assertEqual(_names(plan, "review"), list(reviewed))
        self.assertEqual(_names(plan, "resume"), ["e_healthy"])
        self.assertIn("segment record UNREADABLE", _entry(plan, "a_unreadable")["reason"])
        self.assertIn("segment record CONFLICT", _entry(plan, "b_conflict")["reason"])
        self.assertIn("is unreadable", _entry(plan, "c_bad_ledger")["reason"])
        self.assertIn("is unreadable", _entry(plan, "d_fresh_bad_ledger")["reason"])

        with mock_sbatch() as calls:
            result = execute_step1_recovery(self.step1, scheduler=fake_guard())
        self.assertEqual([cwd for _command, cwd in calls], [str(healthy)])
        self.assertEqual([row["relative_path"] for row in result["execution"]["runs"]], ["e_healthy"])
        for name in reviewed:
            self.assertEqual(tree_snapshot(self.step1 / name), before[name], name)

    def test_explicit_launcher_that_bypasses_the_preconditioner_needs_review(self) -> None:
        repair = write_step1_run(self.step1, "a_repair", steps=30, energies=_runaway(25))
        wrapped = write_step1_run(self.step1, "b_resume_wrapped", steps=33, launcher="precondition")
        for run in (repair, wrapped):
            (run / "run.slurm").write_text("#!/bin/bash\nsrun vasp_std\n", encoding="utf-8")
            set_age(run, 10.0)
        before = tree_snapshot(self.root)
        plan = plan_step1_recovery(self.step1, scheduler=fake_guard(), launcher="run.slurm")
        self.assertEqual(_names(plan, "review"), ["a_repair", "b_resume_wrapped"])
        for name in ("a_repair", "b_resume_wrapped"):
            self.assertIn(
                "--launcher run.slurm would be submitted, but the preconditioning static SCF is wrapped into "
                "runvasp.sh",
                _entry(plan, name)["reason"],
            )
        # The default launcher (the wrapped one) is fine; so is run.slurm without preconditioning.
        default = plan_step1_recovery(self.step1, scheduler=fake_guard())
        self.assertEqual((_names(default, "repair"), _names(default, "resume")), (["a_repair"], ["b_resume_wrapped"]))
        plain = plan_step1_recovery(
            self.step1, scheduler=fake_guard(), launcher="run.slurm", repair_options={"precondition": False}
        )
        self.assertEqual(_names(plain, "repair"), ["a_repair"])
        self.assertEqual(tree_snapshot(self.root), before)

    def test_gen0_run_whose_manifest_is_below_the_root_is_reviewed_with_a_hint(self) -> None:
        run = _idle_run(self.step1, "fresh")
        write_manifest(self.step1, [run])
        # Invoked on the campaign directory above Step1/: step1-launch reads gen-0 hashes from the
        # manifest of the root it is invoked on, so recover does not guess.
        plan = plan_step1_recovery(self.root, scheduler=fake_guard())
        self.assertEqual(_names(plan, "review"), ["Step1/fresh"])
        reason = _entry(plan, "Step1/fresh")["reason"]
        self.assertIn("launch preflight refused", reason)
        self.assertIn(f"run step1-recover (or step1-launch) on {self.step1} to launch it", reason)
        self.assertEqual(_names(plan_step1_recovery(self.step1, scheduler=fake_guard()), "launch"), ["fresh"])

    def test_interrupted_mutation_is_reviewed(self) -> None:
        from interfaceforge.step1_lineage import archive_step1_state

        run = write_step1_run(self.step1, "half_done", steps=33)
        archive = archive_step1_state(run, "step1_resume_g1")  # left IN_PROGRESS on purpose
        plan = plan_step1_recovery(self.step1, scheduler=fake_guard())
        self.assertEqual(_names(plan, "review"), ["half_done"])
        self.assertIn(str(archive), _entry(plan, "half_done")["reason"])

    def test_stricter_diagnostic_thresholds_turn_a_done_run_into_review(self) -> None:
        # Complete and clean under the defaults; a 20 K limit calls every step too hot.
        write_step1_run(self.step1, "done", steps=400, outcar="finished")
        self.assertEqual(_names(plan_step1_recovery(self.step1, scheduler=fake_guard()), "done"), ["done"])
        plan = plan_step1_recovery(self.step1, scheduler=fake_guard(), diagnostic_options={"max_temperature_k": 20.0})
        self.assertEqual(_names(plan, "review"), ["done"])
        self.assertIn("the requested diagnostic thresholds report unstable", _entry(plan, "done")["reason"])

    def test_render_survives_a_cp1252_stdout(self) -> None:
        runs = self.mixed_tree()
        text = render_recovery_plan(plan_step1_recovery(self.step1, scheduler=self.mixed_guard(runs)))
        buffer = io.BytesIO()
        stream = io.TextIOWrapper(buffer, encoding="cp1252")
        with redirect_stdout(stream):
            print(text)
        stream.flush()
        self.assertIn("Dry run: nothing changed.", buffer.getvalue().decode("cp1252"))
        stream.detach()


if __name__ == "__main__":
    unittest.main()
