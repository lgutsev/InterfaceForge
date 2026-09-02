from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from interfaceforge.cli import main
from interfaceforge.step1_status import render, step1_status

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


class Step1StatusTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
