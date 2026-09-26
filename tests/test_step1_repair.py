"""Tests for ``step1-repair``: rewind planning, generation-aware execution and its safety rules.

The first class keeps the original schema-1 regression tests unchanged.  The
later classes cover the plan/execute split (spec section 7): cumulative repair
over generations (the real 16 -> +52 -> 68/332 case, also on top of a legacy
schema-1 record and of a resume record), the schema-2 record, archive-before-
mutation, launch-ledger sealing, the Slurm guard, fingerprint re-checks,
dry-run zero mutation and the temperature schedule.  Every trajectory is
synthetic (tests/step1_fixtures); nothing here is real NiO data.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

from interfaceforge.cli import main
from interfaceforge.errors import SafetyError
from interfaceforge.step1_launch import launch_step1_runs
from interfaceforge.step1_lineage import (
    ARCHIVE_MANIFEST,
    GEN0_ID,
    LEGACY_REPAIR_KEYS,
    REPAIR_RECORD,
    RESUME_RECORD,
    archive_step1_state,
    build_segment_record,
    current_generation,
    interrupted_archive,
    read_json,
)
from interfaceforge.step1_repair import (
    _xdatcar_frames,
    diagnose_step1_run,
    execute_repair_plan,
    plan_repair_run,
    prepare_step1_repair,
)
from interfaceforge.step1_scheduler import SchedulerGuard, SchedulerSnapshot
from interfaceforge.step1_status import step1_status
from interfaceforge.vasp import parse_incar

_TESTS = str(Path(__file__).resolve().parent)
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)
from step1_fixtures import (  # noqa: E402
    O_POSITION,
    RUNNING_OUTCAR,
    fake_guard,
    h_position,
    oszicar_text,
    poscar_text,
    sequence_guard,
    set_age,
    tree_snapshot,
    write_legacy_launch_ledger,
    write_legacy_repair_record,
    write_manifest,
    write_step1_run,
    xdatcar_text,
)


def _poscar() -> str:
    return (
        "repair fixture\n"
        "1.0\n"
        "10 0 0\n0 10 0\n0 0 20\n"
        "H O\n1 1\n"
        "Selective dynamics\nDirect\n"
        "0.10 0.10 0.50 T T T\n"
        "0.20 0.20 0.50 F F F\n"
    )


def _xdatcar() -> str:
    header = "repair fixture\n1.0\n10 0 0\n0 10 0\n0 0 20\nH O\n1 1\n"
    frames = []
    for index in range(1, 6):
        frames.append(
            f"Direct configuration= {index:6d}\n"
            f"{0.10 + index / 100:.8f} 0.10 0.50\n"
            "0.20 0.20 0.50\n"
        )
    return header + "".join(frames)


def _oszicar(*, scf_ceiling: bool = False) -> str:
    lines = []
    for step in range(1, 23):
        iterations = 60 if scf_ceiling else 3
        for electronic in range(1, iterations + 1):
            lines.append(
                f"RMM: {electronic:3d} -0.100000E+02 -0.1E-04 -0.1E-04 10 0.1E-03\n"
            )
        energy = 100.0 if step >= 21 else -10.0
        lines.append(
            f"{step:5d} T=   300. E= {energy + 1:.8E} F= {energy:.8E} "
            f"E0= {energy:.8E} EK= 0.1E+01\n"
        )
    return "".join(lines)


def _run(root: Path, *, scf_ceiling: bool = False) -> Path:
    run = root / "Step1" / "OH25_run"
    run.mkdir(parents=True)
    (run / "INCAR").write_text(
        "IBRION=0\nNSW=400\nPOTIM=1.0\nNBLOCK=4\nTEBEG=300\nTEEND=300\n"
        "SMASS=-1\nALGO=Fast\nNELM=60\nISTART=1\n",
        encoding="utf-8",
    )
    (run / "POSCAR").write_text(_poscar(), encoding="utf-8")
    (run / "KPOINTS").write_text("Gamma\n0\nGamma\n1 1 1\n0 0 0\n", encoding="utf-8")
    (run / "POTCAR").write_text("fixture\n", encoding="utf-8")
    (run / "OSZICAR").write_text(_oszicar(scf_ceiling=scf_ceiling), encoding="utf-8")
    (run / "XDATCAR").write_text(_xdatcar(), encoding="utf-8")
    (run / "runvasp.sh").write_text(
        "#!/bin/bash\n#SBATCH -N 1\nmodule load vasp\nsrun -n4 vasp_std\n", encoding="utf-8"
    )
    old = time.time() - 10 * 3600
    for name in ("OSZICAR", "XDATCAR"):  # all activity files quiet for 10 h
        os.utime(run / name, (old, old))
    return run


class Step1RepairTests(unittest.TestCase):
    def setUp(self) -> None:
        # These schema-1 tests call the old keyword API (scheduler="auto" by default).
        # With no squeue on PATH "auto" resolves to the unverified file-age guard,
        # so no test can reach a real Slurm installation on a cluster login node.
        patcher = patch("interfaceforge.step1_scheduler.shutil.which", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_diagnoses_energy_runaway_and_scf_ceiling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = _run(Path(tmp), scf_ceiling=True)
            row = diagnose_step1_run(run)
            self.assertEqual(row["first_bad_step"], 21)
            self.assertTrue(row["scf_unreliable"])
            self.assertEqual(row["scf_ceiling_fraction"], 1.0)
            self.assertTrue(row["unstable"])

    def test_dry_run_rewinds_before_bad_frame_without_mutating(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = _run(Path(tmp))
            before = (run / "INCAR").read_text(encoding="utf-8")
            payload = prepare_step1_repair(run)
            plan = payload["runs"][0]
            self.assertEqual(payload["mode"], "dry-run")
            self.assertEqual(plan["status"], "READY")
            self.assertEqual(plan["safe_prefix_steps"], 12)
            self.assertEqual(plan["rewind_frame"], 3)
            self.assertEqual(plan["repair_nsw"], 388)
            self.assertEqual((run / "INCAR").read_text(encoding="utf-8"), before)
            self.assertFalse((run / ".interfaceforge").exists())

    def test_execute_archives_rewinds_and_uses_robust_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = _run(Path(tmp))
            payload = prepare_step1_repair(run, execute=True)
            plan = payload["runs"][0]
            self.assertEqual(plan["status"], "PREPARED")
            archive = Path(plan["archive"])
            self.assertTrue((archive / "OSZICAR").is_file())
            self.assertTrue((archive / "XDATCAR").is_file())
            self.assertFalse((run / "OSZICAR").exists())
            self.assertFalse((run / "XDATCAR").exists())
            incar = parse_incar(run / "INCAR")
            self.assertEqual(incar["ISTART"], "0")
            self.assertEqual(incar["ALGO"], "Normal")
            self.assertEqual(incar["POTIM"], "0.5")
            self.assertEqual(incar["NSW"], "388")
            poscar = (run / "POSCAR").read_text(encoding="utf-8")
            self.assertIn("0.13000000  0.10  0.50  T  T  T", poscar)
            self.assertIn("0.20  0.20  0.50  F  F  F", poscar)
            # the recovery segment always tightens the electronic loop
            self.assertEqual(incar["EDIFF"], "1E-5")
            self.assertEqual(incar["NELM"], "120")
            self.assertEqual(incar["NELMIN"], "6")
            record = json.loads((run / "step1_repair.json").read_text(encoding="utf-8"))
            self.assertEqual(record["safe_prefix_steps"], 12)
            status = step1_status(run)["runs"][0]
            self.assertEqual(status["state"], "repair-prepared")
            self.assertEqual(status["frames_oszicar"], 12)
            self.assertEqual(status["frames_oszicar_segment"], 0)
            self.assertEqual(status["nsw_target"], 400)
            self.assertEqual(status["nsw_segment_target"], 388)

    def test_execute_langevin_and_ramp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = _run(Path(tmp))
            prepare_step1_repair(
                run, execute=True, langevin_gamma=15.0, ramp_from=120.0
            )
            incar = parse_incar(run / "INCAR")
            self.assertEqual(incar["MDALGO"], "3")
            self.assertEqual(incar["LANGEVIN_GAMMA"], "15 15")  # H O -> 2 species
            self.assertNotIn("SMASS", incar)
            self.assertEqual(incar["TEBEG"], "120")

    def test_execute_precondition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = _run(Path(tmp))
            prepare_step1_repair(run, execute=True, precondition=True)
            incar = parse_incar(run / "INCAR")
            self.assertEqual(incar["ISTART"], "1")
            pre = parse_incar(run / "INCAR.precondition")
            self.assertEqual(pre["NSW"], "0")
            self.assertEqual(pre["ISTART"], "0")
            launcher = (run / "runvasp.sh").read_text(encoding="utf-8")
            self.assertIn("InterfaceForge --precondition", launcher)
            self.assertEqual(launcher.count("srun -n4 vasp_std"), 2)
            self.assertFalse((run / "WAVECAR").exists())

    def test_completed_unstable_run_is_repairable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = _run(Path(tmp))
            (run / "INCAR").write_text(
                (run / "INCAR").read_text(encoding="utf-8").replace("NSW=400", "NSW=22"),
                encoding="utf-8",
            )
            payload = prepare_step1_repair(run, stale_hours=0.0)
            self.assertEqual(payload["repairable"], 1)
            plan = payload["runs"][0]
            self.assertEqual(plan["original_nsw"], 22)
            self.assertLess(plan["safe_prefix_steps"], 22)
            self.assertGreater(plan["repair_nsw"], 0)

    def test_repeat_repair_preserves_cumulative_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = _run(Path(tmp))
            first = prepare_step1_repair(run, execute=True, stale_hours=0.0)["runs"][0]
            self.assertEqual(first["safe_prefix_steps"], 12)

            # Simulate the prepared repair itself running into another late runaway.
            (run / "OSZICAR").write_text(_oszicar(), encoding="utf-8")
            (run / "XDATCAR").write_text(_xdatcar(), encoding="utf-8")
            old = time.time() - 10 * 3600
            for name in ("OSZICAR", "XDATCAR"):
                os.utime(run / name, (old, old))

            second = prepare_step1_repair(run, stale_hours=0.0)["runs"][0]
            self.assertEqual(second["previous_safe_prefix_steps"], 12)
            self.assertGreaterEqual(second["safe_prefix_steps"], 12)
            self.assertEqual(
                second["repair_nsw"],
                second["original_nsw"] - second["safe_prefix_steps"],
            )

    def test_cli_is_dry_run_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = _run(Path(tmp))
            self.assertEqual(main(["vasp", "step1-repair", str(run)]), 0)
            self.assertTrue((run / "OSZICAR").exists())


# --------------------------------------------------------------------------- #
# Generation-aware repair (spec section 7)
# --------------------------------------------------------------------------- #

REFERENCE_F = -10.0
RUNAWAY_EV = 110.0  # sustained departure, well below the 500 eV catastrophic limit
SEGMENT_DRIFT = 0.0003  # fractional x per step in a repair segment (the original drifts 0.0005)
LEGACY_ARCHIVE = "step1_repair_20260901T000000Z"


def _runaway(first_bad: int) -> Callable[[int], float]:
    """Quiet energies near F_ref, then a sustained +110 eV departure from ``first_bad`` onward."""

    return lambda step: REFERENCE_F + (RUNAWAY_EV if step >= first_bad else 0.02 * (step % 3))


def _startup_transient(step: int) -> float:
    """The real false positive: step 1 at F_ref + 77 eV, a quiet trajectory afterwards."""

    return REFERENCE_F + (77.0 if step == 1 else 0.05 * ((step * 7) % 5))


def _segment_xdatcar(start_x: float, n_steps: int, *, nblock: int = 4) -> str:
    """XDATCAR of a repair segment whose H ion starts at ``start_x`` and drifts SEGMENT_DRIFT per step."""

    frames = []
    for index, step in enumerate(range(nblock, n_steps + 1, nblock), start=1):
        frames.append(
            f"Direct configuration= {index:6d}\n"
            f"  {start_x + SEGMENT_DRIFT * step:.8f}  0.10000000  0.50000000\n"
            f"  {O_POSITION[0]:.8f}  {O_POSITION[1]:.8f}  {O_POSITION[2]:.8f}\n"
        )
    return xdatcar_text(0) + "".join(frames)  # xdatcar_text(0) is the header alone


def _coordinates(poscar: Path) -> list[list[str]]:
    """The x/y/z tokens of every ion in a two-ion POSCAR (flags dropped)."""

    lines = poscar.read_text(encoding="utf-8").splitlines()
    start = next(index for index, line in enumerate(lines) if line.strip().lower().startswith("direct")) + 1
    return [line.split()[:3] for line in lines[start : start + 2]]


def _frame_coordinates(xdatcar: Path, frame: int) -> list[list[str]]:
    return [line.split()[:3] for line in _xdatcar_frames(xdatcar, 2)[frame - 1]]


def _run_segment(run: Path, *, steps: int, first_bad: int, start_x: float) -> None:
    """Let the prepared segment 'run' on the cluster: fresh outputs that fail at ``first_bad``."""

    (run / "OSZICAR").write_text(oszicar_text(steps, energies=_runaway(first_bad)), encoding="utf-8")
    (run / "XDATCAR").write_text(_segment_xdatcar(start_x, steps), encoding="utf-8")
    (run / "OUTCAR").write_text(RUNNING_OUTCAR, encoding="utf-8")
    set_age(run, 10.0)


def _leaf_launch_row(run: Path, job_id: str) -> dict[str, Any]:
    """A schema-1 row written when the leaf was launched as its own root (relative_path '.')."""

    return {
        "status": "SUBMITTED",
        "job_id": job_id,
        "kind": "repair-prepared",
        "root": str(run),
        "relative_path": ".",
        "directory": str(run),
        "launcher": "runvasp.sh",
        "notes": "",
        "detail": f"Submitted batch job {job_id}",
    }


class RepairTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()
        self.step1 = self.root / "Step1"

    def unstable_run(self, name: str = "OH50_run", *, first_bad: int = 25, steps: int = 30, **kwargs: Any) -> Path:
        """A gen-0 run (NSW 400, POTIM 1.0, NBLOCK 4) whose energy runs away from ``first_bad``."""

        return write_step1_run(self.step1, name, steps=steps, energies=_runaway(first_bad), **kwargs)

    def first_repair(self, run: Path) -> dict[str, Any]:
        """Repair 1 of the real case: 16 accepted steps at 1.0 fs, then a 0.5 fs 100->300 K segment."""

        payload = prepare_step1_repair(run, execute=True, scheduler=fake_guard(), ramp_from=100.0)
        first = payload["runs"][0]
        self.assertEqual(first["status"], "PREPARED")
        self.assertEqual(
            (first["safe_prefix_steps"], first["safe_segment_steps"], first["rewind_frame"], first["repair_nsw"]),
            (16, 16, 4, 384),
        )
        self.assertEqual(first["generation"], 1)
        return first


class RepeatedRepairTests(RepairTestCase):
    """Real failure 5 (must not regress): 16 accepted, then 52 more -> cumulative 68, remaining 332."""

    def assert_second_repair_numbers(self, plan: dict[str, Any], *, legacy_parent: bool) -> None:
        self.assertEqual(plan["previous_safe_prefix_steps"], 16)
        self.assertEqual(plan["safe_segment_steps"], 52)
        self.assertEqual(plan["safe_prefix_steps"], 68)
        self.assertEqual(plan["repair_nsw"], 332)
        self.assertEqual(plan["original_nsw"], 400)
        self.assertEqual(plan["rewind_frame"], 13)
        self.assertEqual(plan["source"], "XDATCAR")
        self.assertEqual(plan["generation"], 2)
        self.assertEqual(plan["original_potim_fs"], 0.5)  # POTIM of the segment being rewound
        self.assertEqual(plan["repair_potim_fs"], 0.5)
        self.assertEqual(plan["parent_legacy_record"], legacy_parent)
        ledger = plan["accepted_segments"]
        self.assertEqual([(row["steps"], row["potim_fs"]) for row in ledger], [(16, 1.0), (52, 0.5)])
        self.assertEqual([row["generation"] for row in ledger], [0, 1])
        self.assertEqual(ledger[0]["generation_id"], GEN0_ID)
        self.assertEqual(ledger[0]["kind"], "original")
        self.assertEqual(ledger[1]["kind"], "repair")
        self.assertEqual(ledger[1]["generation_id"], plan["parent_generation_id"])
        self.assertEqual(ledger[1]["restart_source"], "XDATCAR frame 13")
        self.assertEqual(ledger[1]["tebeg_k"], 100.0)
        self.assertAlmostEqual(ledger[1]["teend_k"], 127.08, places=2)  # 100 + 200 * 52 / 384
        self.assertAlmostEqual(plan["accepted_ps"], 0.042, places=9)
        self.assertTrue(plan["ledger_exact"])

    def test_real_repeated_repair_numbers_geometry_record_and_ledger_sealing(self) -> None:
        run = self.unstable_run()
        write_manifest(self.step1, [run])
        first = self.first_repair(run)
        g1 = first["generation_id"]
        self.assertTrue(g1.startswith("g1-repair-"))
        self.assertEqual(_coordinates(run / "POSCAR")[0], [f"{h_position(16):.8f}", "0.10000000", "0.50000000"])
        incar = parse_incar(run / "INCAR")
        self.assertEqual((incar["TEBEG"], incar["TEEND"], incar["POTIM"], incar["NSW"]), ("100", "300", "0.5", "384"))

        # Repair 1 is launched as its own root (schema-1 leaf ledger), then fails 52 steps later.
        write_legacy_launch_ledger(run, [_leaf_launch_row(run, "4242")])
        _run_segment(run, steps=70, first_bad=61, start_x=h_position(16))
        self.assertEqual(diagnose_step1_run(run)["first_bad_step"], 61)
        first_record_bytes = (run / REPAIR_RECORD).read_bytes()
        ledger_bytes = (run / "step1_launch.json").read_bytes()

        dry = prepare_step1_repair(run, scheduler=fake_guard())
        plan = dry["runs"][0]
        self.assertEqual(plan["status"], "READY")
        self.assertEqual(plan["parent_generation_id"], g1)
        self.assert_second_repair_numbers(plan, legacy_parent=False)

        second = prepare_step1_repair(run, execute=True, scheduler=fake_guard())["runs"][0]
        self.assertEqual(second["status"], "PREPARED")
        self.assert_second_repair_numbers(second, legacy_parent=False)
        archive = Path(second["archive"])
        self.assertTrue(archive.name.startswith("step1_repair_g2_"))

        # Geometry: frame 13 (= 52 / NBLOCK 4) of the CURRENT segment's XDATCAR, not the original's.
        segment_frame = _frame_coordinates(archive / "XDATCAR", 13)
        self.assertEqual(segment_frame[0], [f"{h_position(16) + SEGMENT_DRIFT * 52:.8f}", "0.10000000", "0.50000000"])
        self.assertEqual(_coordinates(run / "POSCAR"), segment_frame)
        original_frames = [
            [line.split()[:3] for line in frame] for frame in _xdatcar_frames(Path(first["archive"]) / "XDATCAR", 2)
        ]
        self.assertEqual(len(original_frames), 7)  # the original run only reached step 30
        self.assertNotIn(segment_frame, original_frames)
        self.assertIn("T T T", " ".join((run / "POSCAR").read_text(encoding="utf-8").split()))  # flags kept

        # INCAR: 332 remaining; without ramp_from the 100->300 K ramp continues from the rewind point.
        incar = parse_incar(run / "INCAR")
        self.assertEqual(incar["NSW"], "332")
        self.assertEqual(incar["POTIM"], "0.5")
        self.assertEqual(incar["TEBEG"], "127.08")
        self.assertEqual(incar["TEEND"], "300")
        for name in ("OSZICAR", "OUTCAR", "XDATCAR", "CONTCAR"):
            self.assertFalse((run / name).exists(), name)

        # The schema-2 record keeps every schema-1 key with its original meaning.
        record = read_json(run / REPAIR_RECORD)
        self.assertEqual(record["format"], "interfaceforge-step1-repair")
        self.assertEqual(record["schema_version"], 2)
        self.assertEqual(record["status"], "PREPARED")
        self.assertEqual(record["generation"], 2)
        self.assertEqual(record["generation_id"], second["generation_id"])
        self.assertTrue(record["generation_id"].startswith("g2-repair-"))
        self.assertEqual(record["parent_generation_id"], g1)
        self.assertEqual(record["segment_kind"], "repair")
        for key in (*LEGACY_REPAIR_KEYS, "format", "schema_version", "run", "status", "original_nsw", "archive"):
            self.assertIn(key, record)
        self.assertEqual((record["safe_prefix_steps"], record["accepted_prefix_steps"]), (68, 68))
        self.assertEqual((record["repair_nsw"], record["segment_nsw"]), (332, 332))
        self.assertEqual((record["repair_potim_fs"], record["segment_potim_fs"]), (0.5, 0.5))
        self.assertEqual((record["previous_safe_prefix_steps"], record["safe_segment_steps"]), (16, 52))
        self.assertEqual((record["rewind_frame"], record["source"], record["original_potim_fs"]), (13, "XDATCAR", 0.5))
        self.assertEqual(record["repair_algo"], "Normal")
        self.assertIsNone(record["repair_ramp_from_k"])
        self.assertEqual(record["diagnostic"]["first_bad_step"], 61)
        self.assertEqual(record["archive"], str(archive))
        self.assertEqual(record["submissions"], [])
        self.assertAlmostEqual(record["accepted_ps"], 0.042, places=9)
        ledger = [(row["steps"], row["potim_fs"]) for row in record["accepted_segments"]]
        self.assertEqual(ledger, [(16, 1.0), (52, 0.5)])
        self.assertEqual(record["segment_schedule"]["tebeg_k"], 127.08)
        self.assertEqual(record["segment_schedule"]["teend_k"], 300.0)
        self.assertEqual(record["segment_schedule"]["nsw"], 332)
        self.assertTrue(record["segment_schedule"]["ramp"])
        for key in ("fingerprint", "skip_reason", "review_reasons", "active_jobs"):
            self.assertNotIn(key, record)

        # The previous generation's record was archived first, then replaced.
        self.assertEqual((archive / REPAIR_RECORD).read_bytes(), first_record_bytes)
        self.assertEqual(record["retired_records"], [REPAIR_RECORD])
        manifest = read_json(archive / ARCHIVE_MANIFEST)
        self.assertEqual((manifest["status"], manifest["generation_id"]), ("COMPLETE", record["generation_id"]))
        self.assertIsNone(interrupted_archive(run))

        # Real failure 1: the historical leaf ledger was copied to the archive, then sealed.
        self.assertEqual((archive / "step1_launch.json").read_bytes(), ledger_bytes)
        sealed = read_json(run / "step1_launch.json")
        self.assertEqual(sealed["runs"][0]["job_id"], "4242")
        self.assertEqual(sealed["runs"][0]["superseded_by_generation_id"], record["generation_id"])
        self.assertIn("superseded_at", sealed["runs"][0])
        self.assertEqual(record["sealed_ledgers"], [str((run / "step1_launch.json").resolve())])

        # The lineage reads it back; launch plans generation 2 despite the old SUBMITTED row.
        generation = current_generation(run)
        self.assertEqual((generation.generation, generation.accepted_prefix_steps), (2, 68))
        segment = (generation.original_nsw, generation.segment_nsw, generation.segment_potim_fs)
        self.assertEqual(segment, (400, 332, 0.5))
        before = tree_snapshot(self.root)
        launch = launch_step1_runs([run], only_repaired=True, scheduler=fake_guard())
        self.assertEqual([row["generation_id"] for row in launch["planned"]], [record["generation_id"]])
        self.assertEqual(tree_snapshot(self.root), before)

    def test_second_repair_on_a_legacy_schema1_first_repair_gives_the_same_numbers(self) -> None:
        run = self.unstable_run()
        write_manifest(self.step1, [run])
        first = self.first_repair(run)
        # Rewrite repair 1's provenance exactly as the pre-generation code left it:
        # an archive named step1_repair_<stamp> without manifest and a schema-1 record.
        legacy_archive = Path(first["archive"]).with_name(LEGACY_ARCHIVE)
        Path(first["archive"]).rename(legacy_archive)
        (legacy_archive / ARCHIVE_MANIFEST).unlink()
        write_legacy_repair_record(
            run,
            safe_prefix_steps=16,
            safe_segment_steps=16,
            previous_safe_prefix_steps=0,
            rewind_frame=4,
            original_nsw=400,
            repair_nsw=384,
            original_potim_fs=1.0,
            repair_potim_fs=0.5,
            repair_ramp_from_k=100.0,
            repair_precondition=False,
            archive=str(legacy_archive),
        )
        legacy_bytes = (run / REPAIR_RECORD).read_bytes()
        _run_segment(run, steps=70, first_bad=61, start_x=h_position(16))

        plan = prepare_step1_repair(run, scheduler=fake_guard())["runs"][0]
        self.assertEqual(plan["status"], "READY")
        self.assertEqual(plan["parent_generation_id"], "legacy-repair-g1-20260901T000000Z")
        self.assert_second_repair_numbers(plan, legacy_parent=True)
        self.assertTrue(plan["accepted_segments"][0]["legacy"])  # reconstructed from the archive chain

        second = prepare_step1_repair(run, execute=True, scheduler=fake_guard())["runs"][0]
        self.assert_second_repair_numbers(second, legacy_parent=True)
        archive = Path(second["archive"])
        self.assertEqual((archive / REPAIR_RECORD).read_bytes(), legacy_bytes)
        self.assertEqual(_coordinates(run / "POSCAR"), _frame_coordinates(archive / "XDATCAR", 13))
        record = read_json(run / REPAIR_RECORD)
        self.assertEqual((record["generation"], record["schema_version"]), (2, 2))
        self.assertEqual(record["parent_generation_id"], "legacy-repair-g1-20260901T000000Z")
        self.assertEqual((record["safe_prefix_steps"], record["repair_nsw"]), (68, 332))
        generation = current_generation(run)
        self.assertEqual((generation.generation, generation.accepted_prefix_steps), (2, 68))
        self.assertEqual([row["steps"] for row in generation.ledger], [16, 52])

    def test_repair_on_top_of_a_resume_record_uses_its_accepted_prefix(self) -> None:
        # Generation 1 is a resume that kept 40 accepted steps; its segment (NSW 360) then fails at step 21.
        run = self.unstable_run("OH25_run", first_bad=21, nsw=360)
        write_manifest(self.step1, [run])
        resume = build_segment_record(
            "resume",
            run=run,
            generation=1,
            generation_id="g1-resume-20260920T101500Z",
            parent=current_generation(run),
            prepared_at="2026-09-20T10:15:00+00:00",
            original_nsw=400,
            accepted_prefix_steps=40,
            accepted_segments=[
                {"generation": 0, "generation_id": GEN0_ID, "kind": "original", "steps": 40, "potim_fs": 1.0}
            ],
            ledger_exact=True,
            segment_nsw=360,
            segment_potim_fs=1.0,
            segment_schedule={"tebeg_k": 300.0, "teend_k": 300.0, "nsw": 360, "thermostat": "x", "ramp": False},
            archive=str(run / ".interfaceforge" / "archive" / "step1_resume_g1_20260920T101500Z"),
            extra={"restart_source": "CONTCAR", "restart_frame": None},
        )
        resume["status"] = "SUBMITTED"
        (run / RESUME_RECORD).write_text(json.dumps(resume, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        resume_bytes = (run / RESUME_RECORD).read_bytes()
        set_age(run, 10.0)

        plan = prepare_step1_repair(run, scheduler=fake_guard())["runs"][0]
        self.assertEqual(plan["status"], "READY")
        self.assertEqual(plan["parent_generation_id"], "g1-resume-20260920T101500Z")
        self.assertEqual(plan["parent_segment_kind"], "resume")
        self.assertEqual(
            (plan["previous_safe_prefix_steps"], plan["safe_segment_steps"], plan["safe_prefix_steps"]), (40, 12, 52)
        )
        self.assertEqual((plan["original_nsw"], plan["repair_nsw"], plan["generation"]), (400, 348, 2))
        self.assertEqual(
            [(row["steps"], row["kind"]) for row in plan["accepted_segments"]], [(40, "original"), (12, "resume")]
        )

        second = prepare_step1_repair(run, execute=True, scheduler=fake_guard())["runs"][0]
        archive = Path(second["archive"])
        self.assertFalse((run / RESUME_RECORD).exists())
        self.assertEqual((archive / RESUME_RECORD).read_bytes(), resume_bytes)
        record = read_json(run / REPAIR_RECORD)
        self.assertEqual(record["retired_records"], [RESUME_RECORD])
        self.assertEqual((record["generation"], record["parent_generation_id"]), (2, "g1-resume-20260920T101500Z"))
        self.assertEqual((record["safe_prefix_steps"], record["repair_nsw"]), (52, 348))
        self.assertEqual(parse_incar(run / "INCAR")["NSW"], "348")
        self.assertEqual(_coordinates(run / "POSCAR"), _frame_coordinates(archive / "XDATCAR", 3))
        generation = current_generation(run)
        self.assertEqual((generation.kind, generation.generation, generation.accepted_prefix_steps), ("repair", 2, 52))

    def test_root_level_ledger_rows_for_the_run_are_sealed_too(self) -> None:
        run = self.unstable_run()
        other = self.unstable_run("OH25_run")
        write_manifest(self.step1, [run, other])
        row = {**_leaf_launch_row(run, "7001"), "root": str(self.step1), "relative_path": "OH50_run"}
        foreign = {**_leaf_launch_row(other, "7002"), "root": str(self.step1), "relative_path": "OH25_run"}
        write_legacy_launch_ledger(self.step1, [row, foreign])
        set_age(self.step1, 10.0)

        payload = prepare_step1_repair(self.step1, execute=True, scheduler=fake_guard())
        self.assertEqual([plan["status"] for plan in payload["runs"]], ["PREPARED", "PREPARED"])
        ledger = read_json(self.step1 / "step1_launch.json")
        sealed = {item["job_id"]: item.get("superseded_by_generation_id") for item in ledger["runs"]}
        by_run = {Path(plan["run"]).name: plan["generation_id"] for plan in payload["runs"]}
        self.assertEqual(sealed, {"7001": by_run["OH50_run"], "7002": by_run["OH25_run"]})


class RepairSafetyTests(RepairTestCase):
    def test_active_slurm_workdir_is_active_slurm_and_execute_refuses_the_tree(self) -> None:
        busy = self.unstable_run("OH25_run")  # files 10 h old: only squeue knows it is running
        idle = self.unstable_run("OH50_run")
        guard = fake_guard({busy: "RUNNING"})
        before = tree_snapshot(self.root)

        payload = prepare_step1_repair(self.step1, scheduler=guard)
        plans = {Path(plan["run"]).name: plan for plan in payload["runs"]}
        self.assertEqual(plans["OH25_run"]["status"], "ACTIVE_SLURM")
        self.assertIn("active in Slurm (job 9000 RUNNING)", plans["OH25_run"]["skip_reason"])
        self.assertEqual(plans["OH50_run"]["status"], "READY")
        self.assertEqual((payload["skipped_active_or_recent"], payload["repairable"]), (1, 1))
        self.assertTrue(payload["scheduler"]["verified"])
        self.assertEqual(payload["scheduler"]["active_jobs"], 1)

        with self.assertRaises(SafetyError) as caught:
            prepare_step1_repair(self.step1, execute=True, scheduler=guard)
        self.assertIn("Refusing to partially mutate the tree", str(caught.exception))
        self.assertIn("active in Slurm", str(caught.exception))
        self.assertEqual(tree_snapshot(self.root), before)
        self.assertFalse((idle / ".interfaceforge").exists())

    def test_job_that_starts_between_planning_and_mutation_is_refused_before_any_write(self) -> None:
        run = self.unstable_run()
        before = tree_snapshot(self.root)
        with self.assertRaises(SafetyError) as caught:
            prepare_step1_repair(self.step1, execute=True, scheduler=sequence_guard([{}, {run: "RUNNING"}]))
        message = str(caught.exception)
        self.assertIn("Refusing to mutate", message)
        self.assertIn("it was not modified", message)
        self.assertIn("Already prepared: none", message)
        self.assertEqual(tree_snapshot(self.root), before)

        # The same race through the public plan/execute pair.
        guard = sequence_guard([{}, {run: "PENDING"}])
        plan = plan_repair_run(run, snapshot=guard.snapshot, stale_hours=None)
        self.assertEqual(plan["status"], "READY")
        with self.assertRaises(SafetyError):
            execute_repair_plan(plan, guard=guard)
        self.assertEqual(tree_snapshot(self.root), before)

    def test_changed_since_planning_is_refused(self) -> None:
        run = self.unstable_run()
        guard = fake_guard()
        plan = plan_repair_run(run, snapshot=guard.snapshot, stale_hours=None)
        self.assertEqual(plan["status"], "READY")
        with (run / "OSZICAR").open("a", encoding="utf-8") as handle:
            handle.write(oszicar_text(1))  # the job wrote one more step after planning
        before = tree_snapshot(self.root)
        with self.assertRaises(SafetyError) as caught:
            execute_repair_plan(plan, guard=guard)
        self.assertIn("changed since planning", str(caught.exception))
        self.assertEqual(tree_snapshot(self.root), before)

    def test_change_during_the_final_scheduler_check_is_refused(self) -> None:
        run = self.unstable_run()
        calls = {"count": 0}

        def factory() -> SchedulerSnapshot:
            calls["count"] += 1
            if calls["count"] == 2:  # the pre-mutation squeue: a write lands while it runs
                stamp = time.time_ns()
                os.utime(run / "OSZICAR", ns=(stamp, stamp))
            return SchedulerSnapshot("slurm", "slurm", True, "fixture", "2026-09-23T00:00:00+00:00", [])

        guard = SchedulerGuard("slurm", recheck_seconds=0.0, snapshot_factory=factory)
        plan = plan_repair_run(run, snapshot=guard.snapshot, stale_hours=0.0)
        poscar, incar = (run / "POSCAR").read_bytes(), (run / "INCAR").read_bytes()
        with self.assertRaises(SafetyError) as caught:
            execute_repair_plan(plan, guard=guard)
        self.assertIn("changed since planning", str(caught.exception))
        self.assertEqual(calls["count"], 2)
        self.assertFalse((run / ".interfaceforge").exists())
        self.assertEqual(((run / "POSCAR").read_bytes(), (run / "INCAR").read_bytes()), (poscar, incar))

    def test_execute_refuses_a_plan_that_is_not_ready(self) -> None:
        run = self.unstable_run()
        guard = fake_guard({run: "RUNNING"})
        plan = plan_repair_run(run, snapshot=guard.snapshot, stale_hours=None)
        self.assertEqual(plan["status"], "ACTIVE_SLURM")
        before = tree_snapshot(self.root)
        with self.assertRaises(SafetyError) as caught:
            execute_repair_plan(plan, guard=fake_guard())
        self.assertIn("not READY", str(caught.exception))
        self.assertEqual(tree_snapshot(self.root), before)

    def test_dry_run_performs_zero_mutation(self) -> None:
        self.unstable_run("OH25_run", poscar_velocities=True)
        repaired = self.unstable_run("OH50_run")
        write_step1_run(self.step1, "OH75_run", steps=40, energies=_startup_transient)  # healthy
        write_manifest(self.step1, [self.step1 / "OH25_run", repaired, self.step1 / "OH75_run"])
        write_legacy_repair_record(
            repaired, safe_prefix_steps=16, safe_segment_steps=16, rewind_frame=4, repair_nsw=384
        )
        write_legacy_launch_ledger(repaired, [_leaf_launch_row(repaired, "4242")])
        write_legacy_launch_ledger(self.step1, [{**_leaf_launch_row(repaired, "4243"), "relative_path": "OH50_run"}])
        set_age(repaired, 10.0)
        before = tree_snapshot(self.root)

        payload = prepare_step1_repair(
            self.step1, scheduler=fake_guard(), precondition=True, langevin_gamma=10.0, ramp_from=100.0
        )
        self.assertEqual(payload["mode"], "dry-run")
        self.assertEqual(sorted(Path(plan["run"]).name for plan in payload["runs"]), ["OH25_run", "OH50_run"])
        self.assertEqual(payload["repairable"], 2)
        json.dumps(payload)  # the CLI prints it
        for name in ("OH25_run", "OH50_run", "OH75_run"):
            plan_repair_run(self.step1 / name, snapshot=fake_guard().snapshot, stale_hours=None, precondition=True)
        prepare_step1_repair(self.step1, scheduler="none")
        self.assertEqual(tree_snapshot(self.root), before)

    def test_startup_transient_run_is_not_repairable(self) -> None:
        # Real failure 4: completed 400/400 at ~302 K, step 1 at F_ref + 77 eV, clean afterwards.
        run = write_step1_run(
            self.step1,
            "OH50_run",
            steps=400,
            outcar="finished",
            energies=_startup_transient,
            temperatures=lambda step: 302.0 + 4.0 * math.sin(step),
        )
        diagnostic = diagnose_step1_run(run)
        self.assertFalse(diagnostic["unstable"])
        self.assertTrue(diagnostic["benign_warnings_only"])
        guard = fake_guard()
        self.assertIsNone(plan_repair_run(run, snapshot=guard.snapshot, stale_hours=None))
        before = tree_snapshot(self.root)
        payload = prepare_step1_repair(self.step1, execute=True, scheduler=guard)
        self.assertEqual((payload["runs"], payload["repairable"]), ([], 0))
        self.assertEqual(tree_snapshot(self.root), before)

    def test_diagnostic_options_pass_through(self) -> None:
        run = write_step1_run(self.step1, "OH50_run", steps=40, energies=_startup_transient)
        guard = fake_guard()
        # A 60 eV catastrophic limit makes the +77 eV step 1 hard (H4): rewind to the segment start.
        payload = prepare_step1_repair(self.step1, scheduler=guard, catastrophic_energy_ev=60.0)
        self.assertEqual(payload["settings"]["catastrophic_energy_ev"], 60.0)
        plan = payload["runs"][0]
        self.assertEqual(plan["diagnostic"]["catastrophic_energy_limit_ev"], 60.0)
        self.assertEqual((plan["safe_segment_steps"], plan["source"], plan["repair_nsw"]), (0, "POSCAR", 400))
        # Without a grace window a lone step-1 departure is an isolated spike: a warning, not repairable.
        payload = prepare_step1_repair(self.step1, scheduler=guard, startup_grace_steps=0)
        self.assertEqual((payload["settings"]["startup_grace_steps"], payload["runs"]), (0, []))
        options = {"startup_grace_steps": None, "catastrophic_energy_ev": 60.0}
        plan = plan_repair_run(run, snapshot=guard.snapshot, stale_hours=None, diagnostic_options=options)
        self.assertEqual(plan["diagnostic"]["startup_grace_steps"], 10)
        with self.assertRaises(ValueError):
            plan_repair_run(run, snapshot=guard.snapshot, stale_hours=None, diagnostic_options={"bogus": 1})

    def test_stale_hours_none_resolves_from_the_scheduler(self) -> None:
        self.unstable_run(age_hours=1.0)  # quiet for an hour
        verified = prepare_step1_repair(self.step1, scheduler=fake_guard())
        self.assertEqual(verified["settings"]["stale_hours"], 0.1)
        self.assertIsNone(verified["settings"]["stale_hours_requested"])
        self.assertIn("Slurm verified", verified["settings"]["stale_hours_reason"])
        self.assertEqual(verified["runs"][0]["status"], "READY")

        unverified = prepare_step1_repair(self.step1, scheduler="none")
        self.assertEqual(unverified["settings"]["stale_hours"], 6.0)
        self.assertIn("not verified", unverified["settings"]["stale_hours_reason"])
        self.assertFalse(unverified["scheduler"]["verified"])
        self.assertEqual(unverified["runs"][0]["status"], "ACTIVE_OR_RECENT")
        self.assertEqual(unverified["skipped_active_or_recent"], 1)

        explicit = prepare_step1_repair(self.step1, scheduler="none", stale_hours=0.5)
        self.assertEqual((explicit["settings"]["stale_hours"], explicit["runs"][0]["status"]), (0.5, "READY"))

    def test_recently_modified_unverified_run_blocks_the_whole_tree(self) -> None:
        self.unstable_run("OH25_run", age_hours=1.0)
        self.unstable_run("OH50_run")
        before = tree_snapshot(self.root)
        with self.assertRaises(SafetyError) as caught:
            prepare_step1_repair(self.step1, execute=True, scheduler="none")
        self.assertIn("active/recent (<6 h): OH25_run", str(caught.exception))
        self.assertEqual(tree_snapshot(self.root), before)

    def test_unwrappable_launcher_unreadable_record_and_interrupted_archive_need_review(self) -> None:
        no_launcher = self.unstable_run("A_run", launcher=None)
        unreadable = self.unstable_run("B_run")
        (unreadable / REPAIR_RECORD).write_text("{not json", encoding="utf-8")
        interrupted = self.unstable_run("C_run")
        pending = archive_step1_state(interrupted, "step1_repair_g1")  # left IN_PROGRESS, as after a crash
        for run in (no_launcher, unreadable, interrupted):
            set_age(run, 10.0)
        before = tree_snapshot(self.root)

        payload = prepare_step1_repair(self.step1, scheduler=fake_guard(), precondition=True)
        plans = {Path(plan["run"]).name: plan for plan in payload["runs"]}
        self.assertEqual({plan["status"] for plan in plans.values()}, {"REVIEW"})
        self.assertIn("cannot precondition", plans["A_run"]["skip_reason"])
        self.assertIn("current generation is UNREADABLE", plans["B_run"]["skip_reason"])
        self.assertIn(f"interrupted recovery mutation; inspect {pending}", plans["C_run"]["skip_reason"])
        self.assertEqual((payload["skipped_review"], payload["repairable"]), (3, 0))
        with self.assertRaises(SafetyError) as caught:
            prepare_step1_repair(self.step1, execute=True, scheduler=fake_guard(), precondition=True)
        self.assertIn("need review before repair", str(caught.exception))
        self.assertEqual(tree_snapshot(self.root), before)

    def test_failure_after_archiving_leaves_an_in_progress_archive_and_no_record(self) -> None:
        run = self.unstable_run()
        incar = (run / "INCAR").read_bytes()
        with patch("interfaceforge.step1_repair.update_incar", side_effect=OSError("simulated disk full")):
            with self.assertRaises(SafetyError) as caught:
                prepare_step1_repair(self.step1, execute=True, scheduler=fake_guard())
        message = str(caught.exception)
        self.assertIn("simulated disk full", message)
        self.assertIn("IN_PROGRESS", message)
        archive = interrupted_archive(run)
        self.assertIsNotNone(archive)
        self.assertIn(str(archive), message)
        self.assertEqual(read_json(archive / ARCHIVE_MANIFEST)["status"], "IN_PROGRESS")
        self.assertEqual((archive / "INCAR").read_bytes(), incar)
        self.assertTrue((archive / "OSZICAR").is_file())
        self.assertFalse((run / REPAIR_RECORD).exists())

    def test_multi_run_execution_stops_at_the_first_failure(self) -> None:
        first = self.unstable_run("OH25_run")
        second = self.unstable_run("OH50_run")
        second_before = tree_snapshot(second)
        # plan snapshot, pre-mutation check of OH25_run, pre-mutation check of OH50_run (now running)
        guard = sequence_guard([{}, {}, {second: "RUNNING"}])
        with self.assertRaises(SafetyError) as caught:
            prepare_step1_repair(self.step1, execute=True, scheduler=guard)
        message = str(caught.exception)
        self.assertIn("stopped at OH50_run", message)
        self.assertIn("Already prepared: OH25_run", message)
        self.assertIn("it was not modified", message)
        self.assertEqual(read_json(first / REPAIR_RECORD)["generation"], 1)
        self.assertIsNone(interrupted_archive(first))
        self.assertEqual(tree_snapshot(second), second_before)


class RepairTemperatureAndGeometryTests(RepairTestCase):
    def test_rewind_to_the_segment_start_strips_the_velocity_block(self) -> None:
        # SCF-only failure (H6): no rewind anchor, so the repair restarts from the segment's POSCAR.
        run = write_step1_run(self.step1, "OH50_run", steps=30, scf_iterations=60, poscar_velocities=True)
        payload = prepare_step1_repair(run, execute=True, scheduler=fake_guard())
        plan = payload["runs"][0]
        self.assertIsNone(plan["diagnostic"]["first_bad_step"])
        self.assertEqual((plan["safe_segment_steps"], plan["rewind_frame"], plan["source"]), (0, None, "POSCAR"))
        self.assertEqual((run / "POSCAR").read_text(encoding="utf-8"), poscar_text(0))
        archive = Path(plan["archive"])
        self.assertEqual((archive / "POSCAR").read_text(encoding="utf-8"), poscar_text(0, velocities=True))
        self.assertEqual(plan["accepted_segments"][-1]["restart_source"], "POSCAR")
        self.assertEqual(plan["accepted_segments"][-1]["steps"], 0)

    def test_tebeg_continues_the_schedule_of_a_ramp_segment(self) -> None:
        run = self.unstable_run(tebeg=100.0, teend=300.0)
        plan = prepare_step1_repair(run, execute=True, scheduler=fake_guard())["runs"][0]
        self.assertEqual(plan["safe_segment_steps"], 16)
        self.assertEqual(plan["repair_tebeg_k"], 108.0)  # 100 + 200 * 16 / 400
        self.assertEqual(plan["rewound_segment"]["temperature_at_rewind_k"], 108.0)
        self.assertEqual(plan["accepted_segments"][-1]["teend_k"], 108.0)
        incar = parse_incar(run / "INCAR")
        self.assertEqual((incar["TEBEG"], incar["TEEND"], incar["NSW"]), ("108", "300", "384"))

    def test_constant_temperature_segment_keeps_tebeg(self) -> None:
        run = self.unstable_run()
        plan = prepare_step1_repair(run, execute=True, scheduler=fake_guard())["runs"][0]
        self.assertNotIn("TEBEG", plan["incar_changes"])
        incar = parse_incar(run / "INCAR")
        self.assertEqual((incar["TEBEG"], incar["TEEND"]), ("300", "300"))
        self.assertFalse(plan["segment_schedule"]["ramp"])

    def test_ramp_from_is_honoured_and_an_absent_teend_is_written(self) -> None:
        ramp = self.unstable_run("ramp_run", tebeg=100.0, teend=300.0)
        implicit = self.unstable_run("implicit_run", teend=None)
        prepare_step1_repair(self.step1, execute=True, scheduler=fake_guard(), ramp_from=150.0)
        incar = parse_incar(ramp / "INCAR")
        self.assertEqual((incar["TEBEG"], incar["TEEND"]), ("150", "300"))  # not the 108 K schedule point
        incar = parse_incar(implicit / "INCAR")
        # Without an explicit TEEND VASP would hold 150 K; the original 300 K endpoint is kept.
        self.assertEqual((incar["TEBEG"], incar["TEEND"]), ("150", "300"))
        record = read_json(implicit / REPAIR_RECORD)
        self.assertEqual(record["repair_ramp_from_k"], 150.0)
        self.assertEqual((record["segment_schedule"]["tebeg_k"], record["segment_schedule"]["teend_k"]), (150.0, 300.0))


if __name__ == "__main__":
    unittest.main()
