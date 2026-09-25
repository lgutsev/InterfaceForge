"""Tests for ``step1-resume``: healthy interrupted Step1 runs continue from their latest trustworthy state.

Covers spec section 6: eligibility (every skip status), the restart source
(trusted CONTCAR, XDATCAR frame fallback, segment-start POSCAR), NSW = exactly
the remaining steps, TEBEG continuation of an interrupted ramp (also on top of
a schema-2 or a legacy schema-1 repair record and of an earlier resume), the
electronic-start modes, generation records and ledgers, and the safety rules
of spec section 2 (dry-run zero mutation, the Slurm guard, fingerprint
re-checks, archive-before-mutation, stop-at-first-failure, launch-ledger
sealing).  Every trajectory is synthetic (tests/step1_fixtures); nothing here
is real NiO data.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from interfaceforge.errors import SafetyError
from interfaceforge.step1_launch import launch_step1_runs
from interfaceforge.step1_lineage import (
    ARCHIVE_MANIFEST,
    GEN0_ID,
    REPAIR_RECORD,
    RESUME_RECORD,
    build_segment_record,
    current_generation,
    format_temperature,
    interrupted_archive,
    read_json,
    run_fingerprint,
)
from interfaceforge.step1_repair import plan_repair_run, prepare_step1_repair
from interfaceforge.step1_resume import (
    RESUME_RUNTIME_OUTPUTS,
    execute_resume_plan,
    plan_resume_run,
    prepare_step1_resume,
)
from interfaceforge.step1_status import step1_status
from interfaceforge.vasp import _PRECONDITION_MARKER, parse_incar

_TESTS = str(Path(__file__).resolve().parent)
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)
from step1_fixtures import (  # noqa: E402
    O_POSITION,
    PLAIN_LAUNCHER,
    RUNNING_OUTCAR,
    _header,
    contcar_text,
    fake_guard,
    h_position,
    linear_ramp,
    oszicar_text,
    sequence_guard,
    set_age,
    tree_snapshot,
    write_legacy_launch_ledger,
    write_legacy_repair_record,
    write_manifest,
    write_step1_run,
)

REFERENCE_F = -10.0
RUNAWAY_EV = 110.0  # sustained departure, well below the 500 eV catastrophic limit
REPAIR_G1_ID = "g1-repair-20260920T101500Z"
# INCAR tags a resume must never touch (on top of every tag it does not name).
_RETAINED_TAGS = ("ALGO", "EDIFF", "NELM", "POTIM", "SMASS", "LDAUU", "MAGMOM", "LDAU", "ISPIN", "ENCUT", "NBLOCK")


def _startup_transient(step: int) -> float:
    """The real false positive: step 1 at F_ref + 77 eV, a quiet trajectory afterwards."""

    return REFERENCE_F + (77.0 if step == 1 else 0.05 * ((step * 7) % 5))


def _runaway(first_bad: int) -> Any:
    return lambda step: REFERENCE_F + (RUNAWAY_EV if step >= first_bad else 0.02 * (step % 3))


def _isolated_spike(step: int) -> float:
    """One post-grace step 100 eV off with both neighbours in band: a review-level warning (W2)."""

    return REFERENCE_F + (100.0 if step == 20 else 0.0)


def _coordinates(poscar: Path) -> list[list[str]]:
    """The x/y/z tokens of every ion in a two-ion POSCAR (flags dropped)."""

    lines = poscar.read_text(encoding="utf-8").splitlines()
    start = next(index for index, line in enumerate(lines) if line.strip().lower().startswith("direct")) + 1
    return [line.split()[:3] for line in lines[start : start + 2]]


def _frame_row(step: int) -> list[str]:
    return [f"{h_position(step):.8f}", "0.10000000", "0.50000000"]


def _o_row() -> list[str]:
    return [f"{O_POSITION[0]:.8f}", f"{O_POSITION[1]:.8f}", f"{O_POSITION[2]:.8f}"]


def _incar_lines(path: Path) -> dict[str, str]:
    """Raw INCAR line per tag (byte-level comparison of untouched tags)."""

    lines: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            lines[line.split("=", 1)[0].strip().upper()] = line
    return lines


def _segment_outputs(
    run: Path,
    *,
    steps: int,
    offset: int,
    nblock: int = 4,
    temperatures: Any = None,
    energies: Any = None,
) -> None:
    """Let a prepared segment 'run' on the cluster: fresh outputs continuing from ion position ``offset``.

    Segment step ``j`` puts the H ion at ``h_position(offset + j)``; the
    CONTCAR holds the next position (as VASP writes it) with a velocity block.
    """

    frames = []
    for index, step in enumerate(range(nblock, steps + 1, nblock), start=1):
        frames.append(
            f"Direct configuration= {index:6d}\n"
            f"  {h_position(offset + step):.8f}  0.10000000  0.50000000\n"
            f"  {O_POSITION[0]:.8f}  {O_POSITION[1]:.8f}  {O_POSITION[2]:.8f}\n"
        )
    (run / "XDATCAR").write_text(_header("step1 fixture") + "".join(frames), encoding="utf-8")
    (run / "OSZICAR").write_text(oszicar_text(steps, temperatures=temperatures, energies=energies), encoding="utf-8")
    (run / "CONTCAR").write_text(contcar_text(offset + steps), encoding="utf-8")
    (run / "OUTCAR").write_text(RUNNING_OUTCAR, encoding="utf-8")
    set_age(run, 10.0)


def _repaired_ramp_run(root: Path, name: str = "OH25_run", *, incar_nsw: int = 388) -> Path:
    """Gen-1 repair segment (NSW 388 @ 0.5 fs, 100->300 K) that ran 150 steps and was killed on wall time."""

    return write_step1_run(
        root,
        name,
        nsw=incar_nsw,
        potim=0.5,
        tebeg=100.0,
        teend=300.0,
        algo="Normal",
        steps=150,
        temperatures=linear_ramp(100.0, 300.0, incar_nsw),
    )


def _write_new_repair_record(run: Path, *, status: str = "SUBMITTED") -> dict[str, Any]:
    """A schema-2 step1_repair.json for generation 1: 12 accepted @ 1.0 fs, segment NSW 388 @ 0.5 fs."""

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
    (run / REPAIR_RECORD).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    set_age(run, 10.0)
    return record


def _leaf_launch_row(run: Path, job_id: str) -> dict[str, Any]:
    """A schema-1 row written when the leaf was launched as its own root (relative_path '.')."""

    return {
        "status": "SUBMITTED",
        "job_id": job_id,
        "kind": "prepared",
        "root": str(run),
        "relative_path": ".",
        "directory": str(run),
        "launcher": "runvasp.sh",
        "notes": "",
        "detail": f"Submitted batch job {job_id}",
    }


class ResumeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()
        self.step1 = self.root / "Step1"

    def healthy(self, name: str = "OH50_run", **kwargs: Any) -> Path:
        """Gen 0, NSW 400, POTIM 1.0, 300 -> 300 K, 33 steps, good CONTCAR, no timing footer, 10 h old."""

        return write_step1_run(self.step1, name, **kwargs)

    def plan(self, run: Path, guard: Any = None, **kwargs: Any) -> dict[str, Any]:
        guard = guard if guard is not None else fake_guard()
        return plan_resume_run(run, snapshot=guard.snapshot, stale_hours=None, **kwargs)

    def execute(self, run: Path, **kwargs: Any) -> dict[str, Any]:
        guard = fake_guard()
        plan = self.plan(run, guard, **kwargs)
        self.assertEqual(plan["status"], "READY", plan["skip_reason"])
        return execute_resume_plan(plan, guard=guard)


class HealthyStalledResumeTests(ResumeTestCase):
    def test_healthy_stalled_original_run_resumes_from_its_contcar(self) -> None:
        run = self.healthy()
        write_manifest(self.step1, [run])
        originals = {name: (run / name).read_bytes() for name in ("OSZICAR", "OUTCAR", "XDATCAR", "CONTCAR", "INCAR")}
        originals["POSCAR"] = (run / "POSCAR").read_bytes()
        incar_before = _incar_lines(run / "INCAR")
        row = step1_status(self.step1, stale_hours=None, scheduler=fake_guard())["runs"][0]
        self.assertEqual(row["recovery"]["category"], "resume")

        guard = fake_guard()
        dry = prepare_step1_resume(self.step1, scheduler=guard)
        self.assertEqual((dry["mode"], dry["resumable"], dry["skipped"], dry["skipped_active"]), ("dry-run", 1, [], 0))
        self.assertEqual(dry["format"], "interfaceforge-step1-resume-plan")
        self.assertTrue(dry["scheduler"]["verified"])
        self.assertEqual(dry["settings"]["stale_hours"], 0.1)  # Slurm verified: 6 min settle window
        plan = dry["runs"][0]
        self.assertEqual(plan["status"], "READY")
        self.assertEqual((plan["restart_source"], plan["restart_frame"]), ("CONTCAR", None))
        self.assertEqual((plan["segment_completed_steps"], plan["segment_accepted_steps"]), (33, 33))
        self.assertEqual((plan["accepted_prefix_steps"], plan["resume_nsw"], plan["original_nsw"]), (33, 367, 400))
        check = plan["contcar_check"]
        self.assertTrue(check["trusted"], check["reason"])
        self.assertAlmostEqual(check["max_displacement_angstrom"], 0.01, places=6)  # CONTCAR (step 34) vs frame 8
        self.assertAlmostEqual(check["tolerance_angstrom"], 1.1, places=6)  # 1.0 + 0.05 * (33 + 1 - 32) * 1.0
        self.assertNotIn("TEBEG", plan["incar_changes"])  # constant-temperature segment: TEBEG unchanged
        self.assertEqual(plan["temperature_continuation"]["resumed_tebeg_k"], 300.0)
        json.dumps(dry)  # the CLI prints it

        payload = prepare_step1_resume(self.step1, execute=True, scheduler=guard)
        self.assertEqual((payload["mode"], payload["resumable"]), ("prepared", 1))
        prepared = payload["runs"][0]
        self.assertEqual(prepared["status"], "PREPARED")
        gid = prepared["generation_id"]
        self.assertTrue(gid.startswith("g1-resume-"))
        archive = Path(prepared["archive"])
        self.assertTrue(archive.name.startswith("step1_resume_g1_"))
        for name, content in originals.items():
            self.assertEqual((archive / name).read_bytes(), content, name)
        for name in RESUME_RUNTIME_OUTPUTS:
            self.assertFalse((run / name).exists(), name)
        manifest = read_json(archive / ARCHIVE_MANIFEST)
        self.assertEqual((manifest["status"], manifest["generation_id"]), ("COMPLETE", gid))
        self.assertEqual(manifest["operation"], "step1_resume_g1")

        # POSCAR <- CONTCAR verbatim: velocities (and any predictor-corrector block) continue the dynamics.
        self.assertEqual((run / "POSCAR").read_bytes(), originals["CONTCAR"])
        self.assertIn("0.01234567", (run / "POSCAR").read_text(encoding="utf-8"))

        incar = parse_incar(run / "INCAR")
        self.assertEqual((incar["NSW"], incar["TEBEG"], incar["TEEND"]), ("367", "300", "300"))
        incar_after = _incar_lines(run / "INCAR")
        for tag in _RETAINED_TAGS:
            self.assertEqual(incar_after[tag], incar_before[tag], tag)
        # Nothing but NSW (and the electronic start's ISTART) changed; nothing was added.
        changed = {tag for tag in incar_before if incar_after.get(tag) != incar_before[tag]}
        self.assertEqual(changed, {"NSW", "ISTART"})
        self.assertEqual(set(incar_after), set(incar_before))

        record = read_json(run / RESUME_RECORD)
        self.assertEqual((record["format"], record["schema_version"]), ("interfaceforge-step1-resume", 1))
        self.assertEqual((record["status"], record["generation"], record["generation_id"]), ("PREPARED", 1, gid))
        self.assertEqual((record["parent_generation_id"], record["segment_kind"]), (GEN0_ID, "resume"))
        self.assertEqual((record["accepted_prefix_steps"], record["original_nsw"]), (33, 400))
        self.assertEqual((record["resume_nsw"], record["segment_nsw"], record["segment_potim_fs"]), (367, 367, 1.0))
        self.assertEqual((record["restart_source"], record["segment_accepted_steps"]), ("CONTCAR", 33))
        self.assertEqual([(row["steps"], row["potim_fs"]) for row in record["accepted_segments"]], [(33, 1.0)])
        self.assertEqual(record["accepted_segments"][0]["generation_id"], GEN0_ID)
        self.assertAlmostEqual(record["accepted_ps"], 0.033, places=9)
        self.assertTrue(record["ledger_exact"])
        self.assertEqual(record["archive"], str(archive))
        self.assertTrue(record["contcar_check"]["trusted"])
        self.assertEqual(record["electronic_start"]["mode"], "fresh")  # ISTART=1 but no WAVECAR
        self.assertNotIn("fingerprint", record)

        generation = current_generation(run)
        self.assertEqual((generation.kind, generation.generation, generation.status), ("resume", 1, "PREPARED"))
        self.assertEqual((generation.accepted_prefix_steps, generation.segment_nsw), (33, 367))

        # The prepared run is launchable as resume-prepared; status agrees.
        row = step1_status(self.step1, stale_hours=None, scheduler=fake_guard())["runs"][0]
        self.assertEqual((row["state"], row["recovery"]["category"]), ("resume-prepared", "launch"))
        launch = launch_step1_runs([self.step1], scheduler=fake_guard(), only_resumed=True, progress=lambda _: None)
        self.assertEqual(
            [(item["kind"], item["generation_id"]) for item in launch["planned"]], [("resume-prepared", gid)]
        )

    def test_healthy_stalled_run_is_not_a_repair_candidate(self) -> None:
        run = self.healthy()
        guard = fake_guard()
        before = tree_snapshot(self.root)
        self.assertIsNone(plan_repair_run(run, snapshot=guard.snapshot, stale_hours=None))
        payload = prepare_step1_repair(self.step1, scheduler=guard)
        self.assertEqual((payload["runs"], payload["repairable"]), ([], 0))
        self.assertEqual(tree_snapshot(self.root), before)

    def test_benign_startup_transient_still_resumes(self) -> None:
        # Real failure 4 on an interrupted run: step 1 at F_ref + 77 eV, settled and downhill.
        run = self.healthy(steps=40, energies=_startup_transient)
        plan = self.plan(run)
        self.assertEqual(plan["status"], "READY", plan["skip_reason"])
        self.assertTrue(plan["diagnostic"]["benign_warnings_only"])
        self.assertEqual(plan["accepted_warnings"], [])
        self.assertEqual((plan["segment_accepted_steps"], plan["resume_nsw"]), (40, 360))


class RestartSourceTests(ResumeTestCase):
    def test_untrusted_contcar_falls_back_to_the_latest_xdatcar_frame(self) -> None:
        reasons = {"truncated": "too short", "far": "beyond", "nan": "velocity", "missing": "missing"}
        for kind, reason in reasons.items():
            with self.subTest(contcar=kind):
                run = self.healthy(f"run_{kind}", contcar=kind)
                plan = self.plan(run)
                self.assertEqual(plan["status"], "READY", plan["skip_reason"])
                self.assertFalse(plan["contcar_check"]["trusted"])
                self.assertIn(reason, plan["contcar_check"]["reason"])
                # 33 steps at NBLOCK 4 -> frame 8 = segment step 32.
                self.assertEqual(
                    (plan["restart_source"], plan["restart_file"], plan["restart_frame"]), ("XDATCAR", "XDATCAR", 8)
                )
                self.assertEqual((plan["segment_accepted_steps"], plan["resume_nsw"]), (32, 368))
                self.assertFalse(plan["restart_velocities"])

                guard = fake_guard()
                prepared = execute_resume_plan(self.plan(run, guard), guard=guard)
                self.assertEqual(_coordinates(run / "POSCAR"), [_frame_row(32), _o_row()])
                lines = [line for line in (run / "POSCAR").read_text(encoding="utf-8").splitlines() if line.strip()]
                self.assertEqual(len(lines), 11)  # header + two coordinate rows, no velocity block
                self.assertIn("T  T  T", lines[9])  # selective-dynamics flags kept from the segment-start POSCAR
                self.assertEqual(parse_incar(run / "INCAR")["NSW"], "368")
                record = read_json(run / RESUME_RECORD)
                self.assertEqual((record["restart_source"], record["restart_frame"]), ("XDATCAR", 8))
                self.assertEqual((record["accepted_prefix_steps"], record["segment_accepted_steps"]), (32, 32))
                self.assertEqual(record["accepted_segments"][0]["restart_source"], "XDATCAR frame 8")
                self.assertEqual(Path(prepared["archive"]).parent.parent.parent, run)

    def test_no_trustworthy_state_restarts_the_segment_from_its_poscar(self) -> None:
        cases = {
            # A good CONTCAR cannot be checked against a frame that lags it by 33 steps.
            "no_xdatcar": {"xdatcar": False},
            "no_xdatcar_no_contcar": {"xdatcar": False, "contcar": "missing"},
        }
        for name, kwargs in cases.items():
            with self.subTest(name):
                run = self.healthy(name, poscar_velocities=True, **kwargs)
                poscar = (run / "POSCAR").read_bytes()
                plan = self.plan(run)
                self.assertEqual(plan["status"], "READY", plan["skip_reason"])
                self.assertEqual((plan["restart_source"], plan["segment_accepted_steps"]), ("POSCAR", 0))
                self.assertEqual((plan["accepted_prefix_steps"], plan["resume_nsw"]), (0, 400))
                if name == "no_xdatcar":
                    self.assertIn("lags", plan["contcar_check"]["reason"])
                self.execute(run)
                self.assertEqual((run / "POSCAR").read_bytes(), poscar)  # left unchanged
                self.assertEqual(parse_incar(run / "INCAR")["NSW"], "400")
                self.assertEqual(read_json(run / RESUME_RECORD)["accepted_segments"][0]["steps"], 0)

    def test_contcar_with_other_species_or_lattice_is_not_trusted(self) -> None:
        cases = {
            "species": contcar_text(33).replace("H O\n", "Ni O\n"),
            "lattice": contcar_text(33).replace("10.000000 0.000000 0.000000", "10.100000 0.000000 0.000000"),
            "short_velocities": contcar_text(33).rstrip("\n").rsplit("\n", 1)[0] + "\n",
        }
        for name, text in cases.items():
            with self.subTest(name):
                run = self.healthy(name)
                (run / "CONTCAR").write_text(text, encoding="utf-8")
                set_age(run, 10.0)
                plan = self.plan(run)
                self.assertFalse(plan["contcar_check"]["trusted"], plan["contcar_check"]["reason"])
                self.assertEqual((plan["restart_source"], plan["segment_accepted_steps"]), ("XDATCAR", 32))

    def test_tolerance_option_widens_the_trust_window(self) -> None:
        run = self.healthy(contcar="far")  # 4 A off the trajectory
        self.assertEqual(self.plan(run)["restart_source"], "XDATCAR")
        plan = self.plan(run, contcar_tolerance_angstrom=5.0)
        self.assertEqual((plan["restart_source"], plan["segment_accepted_steps"]), ("CONTCAR", 33))
        with self.assertRaises(ValueError):
            self.plan(run, contcar_tolerance_angstrom=-1.0)


class TemperatureContinuationTests(ResumeTestCase):
    def assert_repaired_ramp_resume(self, plan: dict[str, Any], *, parent_id: str) -> None:
        self.assertEqual(plan["status"], "READY", plan["skip_reason"])
        self.assertEqual((plan["parent_generation_id"], plan["parent_segment_kind"]), (parent_id, "repair"))
        self.assertEqual((plan["previous_accepted_prefix_steps"], plan["segment_accepted_steps"]), (12, 150))
        self.assertEqual((plan["accepted_prefix_steps"], plan["resume_nsw"], plan["generation"]), (162, 238, 2))
        tebeg = format_temperature(100.0 + 200.0 * 150 / 388)
        self.assertEqual(tebeg, "177.32")
        self.assertEqual(plan["incar_changes"]["TEBEG"], tebeg)
        self.assertNotIn("TEEND", plan["incar_changes"])  # the explicit TEEND line stays as it is
        self.assertNotIn("POTIM", plan["incar_changes"])
        self.assertIsNone(plan["temperature_continuation"]["note"])  # 388 - 150 == 238: same ramp rate
        ledger = plan["accepted_segments"]
        self.assertEqual([(row["steps"], row["potim_fs"]) for row in ledger], [(12, 1.0), (150, 0.5)])
        self.assertEqual([row["generation"] for row in ledger], [0, 1])
        self.assertEqual(ledger[1]["generation_id"], parent_id)
        self.assertEqual((ledger[1]["tebeg_k"], ledger[1]["teend_k"]), (100.0, 177.32))
        self.assertAlmostEqual(plan["accepted_ps"], 0.087, places=9)  # 12 * 1.0 fs + 150 * 0.5 fs
        self.assertTrue(plan["ledger_exact"])

    def assert_repaired_ramp_prepared(self, run: Path, prepared: dict[str, Any], record_bytes: bytes) -> None:
        incar = parse_incar(run / "INCAR")
        self.assertEqual(
            (incar["NSW"], incar["TEBEG"], incar["TEEND"], incar["POTIM"]), ("238", "177.32", "300", "0.5")
        )
        self.assertFalse((run / REPAIR_RECORD).exists())
        self.assertEqual((Path(prepared["archive"]) / REPAIR_RECORD).read_bytes(), record_bytes)
        record = read_json(run / RESUME_RECORD)
        self.assertEqual((record["generation"], record["accepted_prefix_steps"], record["resume_nsw"]), (2, 162, 238))
        self.assertEqual(record["retired_records"], [REPAIR_RECORD])
        self.assertEqual(
            [(row["steps"], row["potim_fs"]) for row in record["accepted_segments"]], [(12, 1.0), (150, 0.5)]
        )
        self.assertAlmostEqual(record["accepted_ps"], 0.087, places=9)
        self.assertEqual(record["segment_potim_fs"], 0.5)
        self.assertEqual(record["segment_schedule"]["tebeg_k"], 177.32)
        self.assertEqual(record["segment_schedule"]["teend_k"], 300.0)
        generation = current_generation(run)
        self.assertEqual((generation.kind, generation.generation, generation.accepted_prefix_steps), ("resume", 2, 162))

    def test_interrupted_repaired_conservative_ramp_continues(self) -> None:
        run = _repaired_ramp_run(self.step1)
        _write_new_repair_record(run)
        record_bytes = (run / REPAIR_RECORD).read_bytes()
        plan = self.plan(run)
        self.assert_repaired_ramp_resume(plan, parent_id=REPAIR_G1_ID)
        self.assertEqual(plan["restart_source"], "CONTCAR")
        prepared = self.execute(run)
        self.assert_repaired_ramp_prepared(run, prepared, record_bytes)
        self.assertEqual(read_json(run / RESUME_RECORD)["parent_generation_id"], REPAIR_G1_ID)

    def test_legacy_schema1_repair_record_resumes_as_generation_2(self) -> None:
        run = _repaired_ramp_run(self.step1)
        write_legacy_repair_record(run)  # 12 accepted @ 1.0 fs, repair NSW 388 @ 0.5 fs (schema 1)
        set_age(run, 10.0)
        record_bytes = (run / REPAIR_RECORD).read_bytes()
        plan = self.plan(run)
        self.assert_repaired_ramp_resume(plan, parent_id="legacy-repair-g1-20260901T000000Z")
        self.assertTrue(plan["parent_legacy_record"])
        prepared = self.execute(run)
        self.assert_repaired_ramp_prepared(run, prepared, record_bytes)

    def test_interrupted_nio_profile_original_ramp_continues_at_150_k(self) -> None:
        run = self.healthy(
            nsw=400, potim=0.5, tebeg=100.0, teend=300.0, steps=100, temperatures=linear_ramp(100, 300, 400)
        )
        plan = self.plan(run)
        self.assertEqual(plan["status"], "READY", plan["skip_reason"])
        self.assertEqual((plan["incar_changes"]["TEBEG"], plan["resume_nsw"]), ("150", 300))
        self.execute(run)
        incar = parse_incar(run / "INCAR")
        self.assertEqual((incar["TEBEG"], incar["TEEND"], incar["NSW"], incar["POTIM"]), ("150", "300", "300", "0.5"))
        record = read_json(run / RESUME_RECORD)
        continuation = record["temperature_continuation"]
        self.assertEqual((continuation["previous_tebeg_k"], continuation["resumed_tebeg_k"]), (100.0, 150.0))
        self.assertEqual((continuation["teend_k"], continuation["accepted_in_segment"]), (300.0, 100))
        self.assertAlmostEqual(continuation["rate_k_per_step_before"], 0.5)
        self.assertAlmostEqual(continuation["rate_k_per_step_after"], 0.5)
        self.assertEqual(record["accepted_segments"][0]["teend_k"], 150.0)

    def test_resume_of_a_resume_accumulates_and_continues_linearly(self) -> None:
        run = self.healthy(
            nsw=400, potim=0.5, tebeg=100.0, teend=300.0, steps=100, temperatures=linear_ramp(100, 300, 400)
        )
        first = self.execute(run)
        g1 = first["generation_id"]
        g1_record = (run / RESUME_RECORD).read_bytes()
        # The resumed segment (NSW 300, 150 -> 300 K) runs 60 steps from the CONTCAR position (ion step 101).
        _segment_outputs(run, steps=60, offset=101, temperatures=linear_ramp(150, 300, 300))

        plan = self.plan(run)
        self.assertEqual(plan["status"], "READY", plan["skip_reason"])
        self.assertEqual((plan["parent_generation_id"], plan["parent_segment_kind"]), (g1, "resume"))
        self.assertEqual(
            (plan["generation"], plan["segment_accepted_steps"], plan["accepted_prefix_steps"]), (2, 60, 160)
        )
        self.assertEqual((plan["resume_nsw"], plan["incar_changes"]["TEBEG"]), (240, "180"))  # 100 + 200 * 160 / 400
        second = execute_resume_plan(plan, guard=fake_guard())
        self.assertTrue(second["generation_id"].startswith("g2-resume-"))
        self.assertTrue(Path(second["archive"]).name.startswith("step1_resume_g2_"))
        self.assertEqual((Path(second["archive"]) / RESUME_RECORD).read_bytes(), g1_record)
        record = read_json(run / RESUME_RECORD)
        self.assertEqual((record["generation"], record["parent_generation_id"]), (2, g1))
        self.assertEqual(record["retired_records"], [RESUME_RECORD])
        ledger = record["accepted_segments"]
        self.assertEqual(
            [(row["steps"], row["potim_fs"], row["kind"]) for row in ledger],
            [(100, 0.5, "original"), (60, 0.5, "resume")],
        )
        self.assertEqual([(row["tebeg_k"], row["teend_k"]) for row in ledger], [(100.0, 150.0), (150.0, 180.0)])
        self.assertAlmostEqual(record["accepted_ps"], 0.08, places=9)
        incar = parse_incar(run / "INCAR")
        self.assertEqual((incar["NSW"], incar["TEBEG"], incar["TEEND"]), ("240", "180", "300"))
        self.assertAlmostEqual(record["temperature_continuation"]["rate_k_per_step_after"], 0.5)

    def test_ramp_rate_note_when_the_segment_nsw_disagrees_with_the_remaining_steps(self) -> None:
        # Hand-edited INCAR: the repair record says segment NSW 388, but the INCAR ran NSW 300.
        run = _repaired_ramp_run(self.step1, incar_nsw=300)
        _write_new_repair_record(run)
        plan = self.plan(run)
        self.assertEqual(plan["status"], "READY", plan["skip_reason"])
        self.assertEqual((plan["resume_nsw"], plan["incar_changes"]["TEBEG"]), (238, "200"))  # 100 + 200 * 150 / 300
        continuation = plan["temperature_continuation"]
        self.assertIn("ramp rate changes", continuation["note"])
        self.assertAlmostEqual(continuation["rate_k_per_step_before"], 200.0 / 300.0)
        self.assertAlmostEqual(continuation["rate_k_per_step_after"], 100.0 / 238.0)
        self.assertEqual(continuation["teend_k"], 300.0)


class ElectronicStartTests(ResumeTestCase):
    def test_precondition_wrapped_launcher_removes_wavecar_and_keeps_istart_1(self) -> None:
        run = self.healthy(launcher="precondition", wavecar=True)
        launcher = (run / "runvasp.sh").read_bytes()
        precondition_incar = (run / "INCAR.precondition").read_bytes()
        plan = self.plan(run)
        self.assertEqual((plan["electronic_start"]["mode"], plan["electronic_start"]["istart"]), ("precondition", 1))
        self.assertNotIn("ISTART", plan["incar_changes"])
        prepared = self.execute(run)
        self.assertFalse((run / "WAVECAR").exists())
        self.assertEqual(parse_incar(run / "INCAR")["ISTART"], "1")
        self.assertEqual((run / "runvasp.sh").read_bytes(), launcher)
        self.assertEqual((Path(prepared["archive"]) / "INCAR.precondition").read_bytes(), precondition_incar)
        self.assertEqual(read_json(run / RESUME_RECORD)["electronic_start"]["mode"], "precondition")

    def test_plain_launcher_with_istart_1_keeps_the_wavecar(self) -> None:
        run = self.healthy(wavecar=True)
        wavecar = (run / "WAVECAR").read_bytes()
        plan = self.plan(run)
        self.assertEqual((plan["electronic_start"]["mode"], plan["electronic_start"]["wavecar"]), ("wavecar", "keep"))
        self.execute(run)
        self.assertEqual((run / "WAVECAR").read_bytes(), wavecar)
        self.assertEqual(parse_incar(run / "INCAR")["ISTART"], "1")
        start = read_json(run / RESUME_RECORD)["electronic_start"]
        self.assertEqual((start["mode"], start["istart"], start["private_copy"]), ("wavecar", 1, False))

    def test_fresh_start_sets_istart_0_removes_wavecar_and_icharg(self) -> None:
        run = self.healthy(wavecar=True, incar_extra={"ICHARG": 1})
        payload = prepare_step1_resume(self.step1, execute=True, scheduler=fake_guard(), fresh_start=True)
        self.assertEqual(payload["runs"][0]["electronic_start"]["mode"], "fresh")
        self.assertFalse((run / "WAVECAR").exists())
        incar = parse_incar(run / "INCAR")
        self.assertEqual(incar["ISTART"], "0")
        self.assertNotIn("ICHARG", incar)

    def test_precondition_option_wraps_a_plain_launcher_after_the_incar_update(self) -> None:
        run = self.healthy(
            wavecar=True,
            istart=0,
            nsw=400,
            potim=0.5,
            tebeg=100.0,
            teend=300.0,
            steps=100,
            temperatures=linear_ramp(100, 300, 400),
        )
        plan = self.plan(run, precondition=True)
        self.assertEqual(plan["status"], "READY", plan["skip_reason"])
        self.assertEqual((plan["electronic_start"]["mode"], plan["incar_changes"]["ISTART"]), ("precondition", 1))
        execute_resume_plan(plan, guard=fake_guard())
        self.assertIn(_PRECONDITION_MARKER, (run / "runvasp.sh").read_text(encoding="utf-8"))
        self.assertFalse((run / "WAVECAR").exists())
        self.assertEqual(parse_incar(run / "INCAR")["ISTART"], "1")
        static = parse_incar(run / "INCAR.precondition")
        self.assertEqual((static["NSW"], static["ISTART"]), ("0", "0"))
        self.assertNotIn("TEBEG", static)

    def test_unwrappable_launcher_with_precondition_needs_review(self) -> None:
        run = self.healthy(launcher=None)
        (run / "runvasp.sh").write_text(PLAIN_LAUNCHER + "srun -n4 vasp_std\n", encoding="utf-8")  # two VASP lines
        set_age(run, 10.0)
        before = tree_snapshot(self.root)
        plan = self.plan(run, precondition=True)
        self.assertEqual(plan["status"], "REVIEW")
        self.assertIn("cannot precondition", plan["skip_reason"])
        with self.assertRaises(SafetyError):
            execute_resume_plan(plan, guard=fake_guard())
        self.assertEqual(tree_snapshot(self.root), before)

    def test_hard_linked_wavecar_is_replaced_by_a_private_copy(self) -> None:
        run = self.healthy(wavecar=True)
        other = self.root / "elsewhere"
        other.mkdir()
        try:
            os.link(run / "WAVECAR", other / "WAVECAR")
        except (OSError, NotImplementedError, AttributeError) as exc:  # pragma: no cover - FS without hard links
            self.skipTest(f"hard links unsupported here: {exc}")
        if (run / "WAVECAR").stat().st_nlink < 2:  # pragma: no cover - link silently became a copy
            self.skipTest("filesystem does not report hard links")
        content = (run / "WAVECAR").read_bytes()
        self.execute(run)
        self.assertEqual((run / "WAVECAR").read_bytes(), content)
        self.assertEqual((run / "WAVECAR").stat().st_nlink, 1)
        self.assertEqual((other / "WAVECAR").stat().st_nlink, 1)
        self.assertEqual((other / "WAVECAR").read_bytes(), content)
        self.assertTrue(read_json(run / RESUME_RECORD)["electronic_start"]["private_copy"])

    def test_precondition_and_fresh_start_together_are_rejected(self) -> None:
        self.healthy()
        with self.assertRaises(ValueError):
            prepare_step1_resume(self.step1, scheduler=fake_guard(), precondition=True, fresh_start=True)

    def test_wavecar_that_disappears_after_planning_is_refused(self) -> None:
        run = self.healthy(wavecar=True)
        guard = fake_guard()
        plan = self.plan(run, guard)
        self.assertEqual(plan["electronic_start"]["mode"], "wavecar")
        (run / "WAVECAR").unlink()
        before = tree_snapshot(self.root)
        with self.assertRaises(SafetyError) as caught:
            execute_resume_plan(plan, guard=guard)
        self.assertIn("electronic start changed since planning", str(caught.exception))
        self.assertEqual(tree_snapshot(self.root), before)


class EligibilityTests(ResumeTestCase):
    def test_skip_statuses(self) -> None:
        unstable = self.healthy("unstable", steps=30, energies=_runaway(25))
        complete = self.healthy("complete", steps=400, outcar="finished")
        not_started = self.healthy("not_started", steps=0, outcar=None)
        no_progress = self.healthy("no_progress", steps=0, partial_scf_tail=5)
        missing = self.healthy("missing")
        (missing / "KPOINTS").unlink()
        error = self.healthy("error", outcar="error")
        recent = self.healthy("recent", age_hours=0.0)
        guard = fake_guard()
        expected = {
            unstable: ("UNSTABLE", "use step1-repair"),
            complete: ("COMPLETE", "400/400"),
            not_started: ("NOT_STARTED", "not started"),
            no_progress: ("NO_PROGRESS", "no completed ionic step"),
            missing: ("MISSING_INPUTS", "KPOINTS"),
            error: ("REVIEW", "ZBRENT"),
            recent: ("ACTIVE_OR_RECENT", "h ago"),
        }
        before = tree_snapshot(self.root)
        for run, (status, text) in expected.items():
            with self.subTest(run.name):
                plan = self.plan(run, guard)
                self.assertEqual(plan["status"], status, plan["skip_reason"])
                self.assertIn(text, plan["skip_reason"])
        payload = prepare_step1_resume(self.step1, execute=True, scheduler=guard)
        self.assertEqual((payload["runs"], payload["resumable"], payload["skipped_active"]), ([], 0, 1))
        self.assertEqual(
            sorted((row["relative_path"], row["status"]) for row in payload["skipped"]),
            sorted((run.name, status) for run, (status, _) in expected.items()),
        )
        self.assertEqual(tree_snapshot(self.root), before)  # skipped runs are never touched

    def test_isolated_spike_needs_review_unless_accepted(self) -> None:
        run = self.healthy(steps=40, energies=_isolated_spike)
        plan = self.plan(run)
        self.assertEqual(plan["status"], "REVIEW")
        self.assertIn("isolated energy spike", plan["skip_reason"])
        self.assertIn("--accept-warnings", plan["skip_reason"])
        accepted = self.plan(run, accept_warnings=True)
        self.assertEqual(accepted["status"], "READY", accepted["skip_reason"])
        self.assertTrue(accepted["accepted_warnings"])
        execute_resume_plan(accepted, guard=fake_guard())
        self.assertTrue(read_json(run / RESUME_RECORD)["accepted_warnings"])

    def test_unreadable_record_and_interrupted_archive_need_review(self) -> None:
        unreadable = self.healthy("unreadable")
        (unreadable / RESUME_RECORD).write_text("{ not json", encoding="utf-8")
        interrupted = self.healthy("interrupted")
        pending = interrupted / ".interfaceforge" / "archive" / "step1_resume_g1_20260923T000000Z"
        pending.mkdir(parents=True)
        (pending / ARCHIVE_MANIFEST).write_text(
            json.dumps({"status": "IN_PROGRESS", "created_at": "2026-09-23T00:00:00+00:00"}), encoding="utf-8"
        )
        set_age(unreadable, 10.0)
        set_age(interrupted, 10.0)
        plan = self.plan(unreadable)
        self.assertEqual(plan["status"], "REVIEW")
        self.assertIn("UNREADABLE", plan["skip_reason"])
        plan = self.plan(interrupted)
        self.assertEqual(plan["status"], "REVIEW")
        self.assertIn("interrupted recovery mutation", plan["skip_reason"])

    def test_stale_hours_resolve_from_the_scheduler(self) -> None:
        run = self.healthy(age_hours=0.2)
        verified = self.plan(run, fake_guard())
        self.assertEqual(verified["status"], "READY", verified["skip_reason"])  # 0.1 h settle window
        unverified = self.plan(run, fake_guard(verified=False))
        self.assertEqual(unverified["status"], "ACTIVE_OR_RECENT")  # 6 h file-age guard
        payload = prepare_step1_resume(self.step1, scheduler="none")
        self.assertEqual((payload["settings"]["stale_hours"], payload["skipped_active"]), (6.0, 1))
        explicit = plan_resume_run(run, snapshot=fake_guard(verified=False).snapshot, stale_hours=0.1)
        self.assertEqual(explicit["status"], "READY")  # an explicit window wins

    def test_torn_final_oszicar_line_is_ignored(self) -> None:
        run = self.healthy()
        with (run / "OSZICAR").open("a", encoding="utf-8") as handle:
            handle.write("   34 T=   300.0 E= -0.9000")  # killed while VASP wrote step 34
        set_age(run, 10.0)
        plan = self.plan(run)
        self.assertEqual(plan["status"], "READY", plan["skip_reason"])
        self.assertEqual((plan["segment_completed_steps"], plan["restart_source"]), (33, "CONTCAR"))

    def test_single_run_root_and_payload_shape(self) -> None:
        run = self.healthy()
        payload = prepare_step1_resume(run, scheduler=fake_guard())
        for key in (
            "format",
            "schema_version",
            "mode",
            "root",
            "scheduler",
            "settings",
            "runs",
            "skipped",
            "resumable",
            "skipped_active",
        ):
            self.assertIn(key, payload)
        self.assertEqual((payload["root"], payload["resumable"]), (str(run), 1))
        with self.assertRaises(FileNotFoundError):
            prepare_step1_resume(self.root / "absent", scheduler=fake_guard())


class ResumeSafetyTests(ResumeTestCase):
    def test_active_slurm_workdir_is_skipped_and_never_mutated(self) -> None:
        busy = self.healthy("OH25_run")  # files 10 h old: only squeue knows it is running
        guard = fake_guard({busy: "RUNNING"})
        before = tree_snapshot(self.root)
        payload = prepare_step1_resume(self.step1, scheduler=guard)
        self.assertEqual(payload["skipped"][0]["status"], "ACTIVE_SLURM")
        self.assertIn("active in Slurm (job 9000 RUNNING)", payload["skipped"][0]["skip_reason"])
        self.assertEqual((payload["resumable"], payload["skipped_active"]), (0, 1))
        self.assertEqual(payload["scheduler"]["active_jobs"], 1)

        executed = prepare_step1_resume(self.step1, execute=True, scheduler=guard)
        self.assertEqual(executed["runs"], [])
        active_plan = self.plan(busy, guard)
        self.assertEqual(active_plan["status"], "ACTIVE_SLURM")
        with self.assertRaises(SafetyError) as caught:
            execute_resume_plan(active_plan, guard=fake_guard())
        self.assertIn("not READY", str(caught.exception))
        # A READY plan made before the job appeared is refused by the pre-mutation check.
        ready_plan = self.plan(busy, fake_guard())
        with self.assertRaises(SafetyError) as caught:
            execute_resume_plan(ready_plan, guard=guard)
        self.assertIn("Refusing to mutate", str(caught.exception))
        self.assertEqual(tree_snapshot(self.root), before)
        self.assertFalse((busy / ".interfaceforge").exists())

    def test_job_that_starts_between_planning_and_mutation_is_refused_before_any_write(self) -> None:
        run = self.healthy()
        before = tree_snapshot(self.root)
        with self.assertRaises(SafetyError) as caught:
            prepare_step1_resume(self.step1, execute=True, scheduler=sequence_guard([{}, {run: "PENDING"}]))
        message = str(caught.exception)
        self.assertIn("Refusing to mutate", message)
        self.assertIn("it was not modified", message)
        self.assertIn("Already prepared: none", message)
        self.assertEqual(tree_snapshot(self.root), before)

        guard = sequence_guard([{}, {run: "PENDING"}])
        plan = self.plan(run, guard)
        self.assertEqual(plan["status"], "READY")
        with self.assertRaises(SafetyError):
            execute_resume_plan(plan, guard=guard)
        self.assertEqual(tree_snapshot(self.root), before)

    def test_multi_run_execution_stops_at_the_first_failure(self) -> None:
        first = self.healthy("OH25_run")
        second = self.healthy("OH50_run")
        third = self.healthy("OH75_run")
        second_before = tree_snapshot(second)
        third_before = tree_snapshot(third)
        # Snapshot 0 plans; 1 is OH25's pre-mutation check; 2 finds OH50 queued.
        guard = sequence_guard([{}, {}, {second: "PENDING"}])
        with self.assertRaises(SafetyError) as caught:
            prepare_step1_resume(self.step1, execute=True, scheduler=guard)
        message = str(caught.exception)
        self.assertIn("stopped at OH50_run", message)
        self.assertIn("Already prepared: OH25_run", message)
        self.assertIn("Not attempted: OH75_run", message)
        self.assertTrue((first / RESUME_RECORD).is_file())
        self.assertEqual(tree_snapshot(second), second_before)
        self.assertEqual(tree_snapshot(third), third_before)

    def test_dry_run_performs_zero_mutation(self) -> None:
        healthy = self.healthy("OH25_run", poscar_velocities=True, wavecar=True)
        wrapped = self.healthy("OH50_run", launcher="precondition", wavecar=True, contcar="far")
        repaired = _repaired_ramp_run(self.step1, "OH75_run")
        write_legacy_repair_record(repaired)
        self.healthy("unstable", steps=30, energies=_runaway(25))
        self.healthy("spike", steps=40, energies=_isolated_spike)
        write_manifest(self.step1, [healthy, wrapped])
        write_legacy_launch_ledger(repaired, [_leaf_launch_row(repaired, "4242")])
        write_legacy_launch_ledger(self.step1, [{**_leaf_launch_row(healthy, "4243"), "relative_path": "OH25_run"}])
        set_age(repaired, 10.0)
        before = tree_snapshot(self.root)

        payload = prepare_step1_resume(self.step1, scheduler=fake_guard(), precondition=True, accept_warnings=True)
        self.assertEqual(payload["mode"], "dry-run")
        self.assertEqual(
            sorted(Path(plan["run"]).name for plan in payload["runs"]), ["OH25_run", "OH50_run", "OH75_run", "spike"]
        )
        json.dumps(payload)
        prepare_step1_resume(self.step1, scheduler="none", stale_hours=0.0, fresh_start=True)
        for run in sorted(path for path in self.step1.iterdir() if path.is_dir()):
            self.plan(run, fake_guard(), contcar_tolerance_angstrom=0.0)
        self.assertEqual(tree_snapshot(self.root), before)
        self.assertEqual(list(self.root.rglob(".interfaceforge")), [])

    def test_fingerprint_change_between_plan_and_execute_is_refused(self) -> None:
        run = self.healthy()
        guard = fake_guard()
        plan = self.plan(run, guard)
        self.assertEqual(plan["status"], "READY")
        with (run / "OSZICAR").open("a", encoding="utf-8") as handle:
            handle.write(oszicar_text(1))  # the job wrote one more step after planning
        before = tree_snapshot(self.root)
        with self.assertRaises(SafetyError) as caught:
            execute_resume_plan(plan, guard=guard)
        self.assertIn("changed since planning", str(caught.exception))
        self.assertEqual(tree_snapshot(self.root), before)
        self.assertFalse((run / ".interfaceforge").exists())

    def test_historical_leaf_ledger_is_sealed_archived_and_does_not_block_launch(self) -> None:
        # Real failure 1: a leaf launched as its own root keeps a schema-1 SUBMITTED row.
        run = self.healthy()
        ledger = write_legacy_launch_ledger(run, [_leaf_launch_row(run, "7001")], age_hours=12.0)
        set_age(run, 10.0)
        ledger_bytes = ledger.read_bytes()
        prepared = self.execute(run)
        gid = prepared["generation_id"]
        archive = Path(prepared["archive"])
        self.assertEqual((archive / "step1_launch.json").read_bytes(), ledger_bytes)
        rows = read_json(ledger)["runs"]
        self.assertEqual([(row["job_id"], row.get("superseded_by_generation_id")) for row in rows], [("7001", gid)])
        self.assertIn(str(ledger), read_json(run / RESUME_RECORD)["sealed_ledgers"])
        launch = launch_step1_runs([run], scheduler=fake_guard(), progress=lambda _: None)
        self.assertEqual(
            [(item["kind"], item["generation_id"]) for item in launch["planned"]], [("resume-prepared", gid)]
        )

    def test_failure_after_archiving_leaves_an_in_progress_archive_and_no_record(self) -> None:
        run = self.healthy()
        with patch("interfaceforge.step1_resume.update_incar", side_effect=OSError("disk full")):
            with self.assertRaises(SafetyError) as caught:
                prepare_step1_resume(self.step1, execute=True, scheduler=fake_guard())
        self.assertIn("IN_PROGRESS", str(caught.exception))
        archive = interrupted_archive(run)
        self.assertIsNotNone(archive)
        self.assertTrue((archive / "OSZICAR").is_file())
        self.assertFalse((run / RESUME_RECORD).exists())
        plan = self.plan(run)
        self.assertEqual(plan["status"], "REVIEW")  # the half-done run is never resumed automatically
        self.assertIn("interrupted recovery mutation", plan["skip_reason"])

    def test_extra_runtime_outputs_are_archived_before_removal(self) -> None:
        run = self.healthy()
        (run / "DOSCAR").write_text("fixture DOSCAR\n", encoding="utf-8")
        (run / "CHGCAR").write_text("fixture CHGCAR\n", encoding="utf-8")
        set_age(run, 10.0)
        prepared = self.execute(run)
        archive = Path(prepared["archive"])
        self.assertEqual((archive / "DOSCAR").read_text(encoding="utf-8"), "fixture DOSCAR\n")
        self.assertFalse((archive / "CHGCAR").exists())  # regenerable, listed as not archived
        manifest = read_json(archive / ARCHIVE_MANIFEST)
        self.assertIn("DOSCAR", [item["name"] for item in manifest["files"]])
        self.assertIn("CHGCAR", manifest["not_archived"])
        self.assertFalse((run / "DOSCAR").exists())
        self.assertFalse((run / "CHGCAR").exists())
        self.assertEqual(prepared["archived_extra_outputs"], ["DOSCAR"])

    def test_generation_change_between_plan_and_execute_is_refused(self) -> None:
        run = self.healthy()
        guard = fake_guard()
        plan = dict(self.plan(run, guard))
        # Another tool prepared a repair after planning; refresh only the fingerprint so the lineage check fires.
        write_legacy_repair_record(run)
        plan["fingerprint"] = run_fingerprint(run)
        before = tree_snapshot(self.root)
        with self.assertRaises(SafetyError) as caught:
            execute_resume_plan(plan, guard=guard)
        self.assertIn("current generation changed since planning", str(caught.exception))
        self.assertEqual(tree_snapshot(self.root), before)


class CompletionTests(ResumeTestCase):
    def test_all_steps_done_is_complete_even_without_a_contcar(self) -> None:
        run = self.healthy(steps=400, outcar="finished", contcar="missing")
        plan = self.plan(run)
        self.assertEqual(plan["status"], "COMPLETE")
        self.assertEqual((plan["restart_source"], plan["accepted_prefix_steps"]), ("XDATCAR", 400))
        self.assertTrue(math.isclose(plan["contcar_check"]["tolerance_angstrom"], 1.05))  # 1 + 0.05 * 1 * 1.0

    def test_complete_accounting_includes_the_accepted_prefix(self) -> None:
        # A repaired segment (12 accepted before it) that ran all its 388 steps completes the 400-step target.
        run = write_step1_run(
            self.step1,
            "OH25_run",
            nsw=388,
            potim=0.5,
            tebeg=100.0,
            teend=300.0,
            steps=388,
            temperatures=linear_ramp(100.0, 300.0, 388),
            outcar="finished",
        )
        _write_new_repair_record(run)
        plan = self.plan(run)
        self.assertEqual(plan["status"], "COMPLETE", plan["skip_reason"])
        self.assertIn("400/400", plan["skip_reason"])


if __name__ == "__main__":
    unittest.main()
