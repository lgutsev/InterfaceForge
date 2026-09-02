from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from interfaceforge.cli import main
from interfaceforge.step2_status import render, step2_status

_INCAR = (
    "ENCUT=520\nPREC=Med\nALGO=Fast\nLREAL=Auto\nISMEAR=0\nSIGMA=0.1\n"
    "ISPIN=2\nLDAU=.TRUE.\nLDAUU=4.6 0.0\nLMAXMIX=4\nIBRION=0\nNSW=1000\n"
    "POTIM=1.0\nMDALGO=2\nSMASS=1.0\nNBLOCK=4\nTEBEG=300\nTEEND=300\nLWAVE=.FALSE.\n"
)


def _oszicar(steps: int, temp: int = 300) -> str:
    return "".join(
        f"   {i} F= -.1E1 E0= -.1E1  d E =0  T= {temp} \n" for i in range(1, steps + 1)
    )


def _tree(root: Path, label: str, *, nsw: int = 1000) -> Path:
    tree = root / f"Step2_{label}K"
    tree.mkdir()
    (tree / "step2_manifest.json").write_text(
        json.dumps(
            {
                "format": "interfaceforge-step2-series",
                "protocol": "training",
                "temperature_k": float(label),
                "nsw": nsw,
            }
        ),
        encoding="utf-8",
    )
    running = tree / "runA"
    running.mkdir()
    (running / "INCAR").write_text(_INCAR, encoding="utf-8")
    (running / "POTCAR").write_text("ENMAX  =  400.000; ENMIN\n", encoding="utf-8")
    (running / "OSZICAR").write_text(_oszicar(812, int(label)), encoding="utf-8")
    (running / "OUTCAR").write_text("running, no timing block\n", encoding="utf-8")
    (running / "XDATCAR").write_text(
        "".join(f"Direct configuration=  {i}\n0 0 0\n" for i in range(1, 813)), encoding="utf-8"
    )
    done = tree / "runB"
    done.mkdir()
    (done / "INCAR").write_text(_INCAR, encoding="utf-8")
    (done / "OSZICAR").write_text(_oszicar(nsw, int(label)), encoding="utf-8")
    (done / "OUTCAR").write_text(
        "General timing and accounting informations for this job\n", encoding="utf-8"
    )
    return tree


class Step2StatusTests(unittest.TestCase):
    def test_multi_tree_states_frames_and_incar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _tree(root, "300")
            _tree(root, "450")

            payload = step2_status(root)
            self.assertEqual(payload["state_tally"], {"running": 2, "done": 2})
            self.assertEqual([t["tree"] for t in payload["trees"]], ["Step2_300K", "Step2_450K"])

            t300 = payload["trees"][0]
            self.assertEqual(t300["protocol"], "training")
            self.assertEqual(t300["nsw_target"], 1000)
            rows = {r["run"]: r for r in t300["runs"]}
            self.assertEqual(rows["runA"]["state"], "running")
            self.assertEqual(rows["runA"]["frames_oszicar"], 812)
            self.assertEqual(rows["runA"]["frames_xdatcar"], 812)
            self.assertEqual(rows["runA"]["percent_complete"], 81.2)
            self.assertEqual(rows["runA"]["incar"]["encut_ev"], 520.0)
            self.assertEqual(rows["runA"]["incar"]["encut_over_enmax"], 1.3)
            self.assertEqual(rows["runA"]["incar"]["mdalgo"], "2")
            self.assertEqual(rows["runB"]["state"], "done")

    def test_single_tree_and_single_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tree = _tree(root, "600")

            self.assertEqual(len(step2_status(tree)["trees"]), 1)

            one = step2_status(tree / "runB")
            self.assertEqual(len(one["trees"]), 1)
            self.assertEqual(one["trees"][0]["runs"][0]["state"], "done")

    def test_done_early_and_error_and_stalled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tree = root / "Step2_300K"
            tree.mkdir()
            (tree / "step2_manifest.json").write_text(
                json.dumps({"protocol": "training", "nsw": 1000}), encoding="utf-8"
            )

            short = tree / "short"
            short.mkdir()
            (short / "INCAR").write_text(_INCAR, encoding="utf-8")
            (short / "OSZICAR").write_text(_oszicar(640), encoding="utf-8")
            (short / "OUTCAR").write_text(
                "General timing and accounting informations for this job\n", encoding="utf-8"
            )

            bad = tree / "bad"
            bad.mkdir()
            (bad / "INCAR").write_text(_INCAR, encoding="utf-8")
            (bad / "OSZICAR").write_text(_oszicar(12), encoding="utf-8")
            (bad / "OUTCAR").write_text("ZBRENT: fatal error in bracketing\n", encoding="utf-8")

            stalled = tree / "stalled"
            stalled.mkdir()
            (stalled / "INCAR").write_text(_INCAR, encoding="utf-8")
            (stalled / "OSZICAR").write_text(_oszicar(50), encoding="utf-8")
            (stalled / "OUTCAR").write_text("no timing block\n", encoding="utf-8")
            old = time.time() - 10 * 3600
            for name in ("OSZICAR", "OUTCAR"):
                os.utime(stalled / name, (old, old))

            states = {r["run"]: r["state"] for r in step2_status(root, stale_hours=6.0)["trees"][0]["runs"]}
            self.assertEqual(states["short"], "done-early")
            self.assertEqual(states["bad"], "error")
            self.assertEqual(states["stalled"], "stalled?")

    def test_render_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _tree(root, "300")
            text = render(step2_status(root))
            self.assertIn("Step2 status:", text)
            self.assertIn("Step2_300K", text)
            self.assertIn("frames 812/1000", text)
            self.assertIn("runs total", text)
            self.assertEqual(main(["vasp", "step2-status", str(root), "--json"]), 0)


if __name__ == "__main__":
    unittest.main()
