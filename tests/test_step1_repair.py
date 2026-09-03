from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from interfaceforge.cli import main
from interfaceforge.step1_repair import diagnose_step1_run, prepare_step1_repair
from interfaceforge.step1_status import step1_status
from interfaceforge.vasp import parse_incar


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
    old = time.time() - 10 * 3600
    os.utime(run / "OSZICAR", (old, old))
    return run


class Step1RepairTests(unittest.TestCase):
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

    def test_cli_is_dry_run_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = _run(Path(tmp))
            self.assertEqual(main(["vasp", "step1-repair", str(run)]), 0)
            self.assertTrue((run / "OSZICAR").exists())


if __name__ == "__main__":
    unittest.main()
