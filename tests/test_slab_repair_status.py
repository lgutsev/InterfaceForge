from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from interfaceforge.slab_repair_status import slab_repair_status


def _outcar(*, nsw: int, ibrion: int, reached: bool, finished: bool, force: float) -> str:
    lines = [
        f"   NSW    =     {nsw}    number of steps for IOM",
        f"   IBRION =      {ibrion}    ionic relax",
        "   NELM   =     200;   NELMIN=  2; NELMDL=  0",
        "   EDIFF  = 0.1E-06   stopping-criterion for ELM",
        "   EDIFFG = -0.3E-01   stopping-criterion for IOM",
        "     AMIN     =   0.01",
        "----- Iteration      1(   1)  -----",
        "------------------------ aborting loop because EDIFF is reached -----",
        " POSITION                                       TOTAL-FORCE (eV/Angst)",
        " " + "-" * 83,
        f"      0.0 0.0 0.0      {force:.6f} 0.000000 0.000000",
        " " + "-" * 83,
    ]
    if reached:
        lines.append(" reached required accuracy - stopping structural energy minimisation")
    if finished:
        lines.append(" General timing and accounting informations for this job:")
    return "\n".join(lines) + "\n"


class SlabRepairStatusTests(unittest.TestCase):
    def test_live_campaign_classification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tight = root / "tight_scf" / "A"
            tight.mkdir(parents=True)
            (tight / "INCAR").write_text("NSW=0\n", encoding="utf-8")
            (tight / "OUTCAR").write_text(
                _outcar(nsw=0, ibrion=-1, reached=False, finished=True, force=0.0),
                encoding="utf-8",
            )
            (tight / "LOCPOT").write_text("ready\n", encoding="utf-8")

            running = root / "relax_continue" / "B"
            running.mkdir(parents=True)
            (running / "INCAR").write_text("NSW=200\n", encoding="utf-8")
            (running / "OUTCAR").write_text(
                _outcar(nsw=200, ibrion=2, reached=False, finished=False, force=0.08),
                encoding="utf-8",
            )

            converged = root / "relax_continue" / "C"
            converged.mkdir(parents=True)
            (converged / "INCAR").write_text("NSW=200\n", encoding="utf-8")
            (converged / "OUTCAR").write_text(
                _outcar(nsw=200, ibrion=2, reached=True, finished=True, force=0.02),
                encoding="utf-8",
            )

            result = slab_repair_status(root)
            rows = {(row["family"], row["folder"]): row for row in result["rows"]}
            self.assertEqual(rows[("tight_scf", "A")]["status"], "STATIC_CONVERGED")
            self.assertTrue(rows[("tight_scf", "A")]["ready_for_workfunction_audit"])
            self.assertEqual(rows[("relax_continue", "B")]["status"], "RUNNING_RELAX")
            self.assertEqual(rows[("relax_continue", "C")]["status"], "RELAX_CONVERGED")
            self.assertTrue(rows[("relax_continue", "C")]["ready_for_final_static"])
            self.assertEqual(result["workfunction_ready"], 1)
            self.assertEqual(result["final_static_ready"], 1)
            self.assertTrue((root / "slab_repair_status.txt").is_file())
            self.assertTrue((root / "slab_repair_status.json").is_file())


if __name__ == "__main__":
    unittest.main()
