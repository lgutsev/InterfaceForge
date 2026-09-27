from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from interfaceforge.profile_cli import main
from interfaceforge.vasp import parse_incar

_OPT_INCAR = (
    "ENCUT = 520\nPREC = Accurate\nEDIFF = 1E-6\nGGA = PE\n"
    "ISPIN = 2\nLASPH = .TRUE.\nMAGMOM = 2*2.0 3*-2.0\n"
    "LDAU = .TRUE.\nLDAUTYPE = 2\nLDAUL = 2 -1\nLDAUU = 4.6 0.0\n"
    "LDAUJ = 0.0 0.0\nLMAXMIX = 4\nIBRION = 2\nISIF = 2\nNSW = 200\n"
)


def _nio_opt_tree(root: Path) -> Path:
    opt = root / "OPT"
    run = opt / "NiO_m110_Big_U46"
    run.mkdir(parents=True)
    (opt / "KPOINTS").write_text("Gamma\n0\nGamma\n2 2 1\n0 0 0\n", encoding="utf-8")
    launcher = opt / "runvasp.sh"
    launcher.write_text(
        "#!/usr/bin/env bash\nmodule load vasp6/6.5.1-cpu\nsrun -n8 vasp_std\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    (run / "INCAR").write_text(_OPT_INCAR, encoding="utf-8")
    coords = "\n".join(f"0.1 0.1 {0.30 + i * 0.02:.4f}" for i in range(5))
    (run / "CONTCAR").write_text(
        f"opt\n1.0\n10 0 0\n0 10 0\n0 0 40\nNi O\n3 2\nDirect\n{coords}\n",
        encoding="utf-8",
    )
    (run / "POTCAR").write_text("licensed fixture Ni O\n", encoding="utf-8")
    return opt


class Step1ProfileTests(unittest.TestCase):
    def test_nio_profile_expands_to_conservative_preconditioned_ramp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            opt = _nio_opt_tree(root)
            rc = main(
                [
                    "vasp",
                    "step1-prepare",
                    str(opt),
                    "--profile",
                    "nio",
                    "--protocol",
                    "training",
                    "--fresh-start",
                ]
            )
            self.assertEqual(rc, 0)

            step1 = root / "Step1"
            run = step1 / "NiO_m110_Big_U46"
            incar = parse_incar(run / "INCAR")
            self.assertEqual(incar["POTIM"], "0.5")
            self.assertEqual(incar["ALGO"], "Normal")
            self.assertEqual(incar["EDIFF"], "1E-5")
            self.assertEqual(incar["NELM"], "120")
            self.assertEqual(incar["NELMIN"], "6")
            self.assertEqual(incar["TEBEG"], "100")
            self.assertEqual(incar["TEEND"], "300")
            self.assertEqual(incar["ISTART"], "1")
            self.assertTrue((run / "INCAR.precondition").is_file())

            manifest = json.loads((step1 / "step1_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["profile"], "nio")
            self.assertEqual(manifest["profile_settings"]["algo"], "Normal")
            self.assertEqual(manifest["profile_settings"]["ramp_from_k"], 100.0)
            self.assertTrue(manifest["profile_settings"]["conservative"])
            self.assertTrue(manifest["profile_settings"]["precondition"])
            self.assertIsNone(manifest["profile_settings"]["langevin_gamma"])

            audit = json.loads((step1 / "step1_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["profile"], "nio")
            self.assertEqual(audit["profile_settings"], manifest["profile_settings"])

    def test_nio_profile_allows_explicit_algo_ramp_and_langevin_escalation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            opt = _nio_opt_tree(root)
            rc = main(
                [
                    "vasp",
                    "step1-prepare",
                    str(opt),
                    "--profile",
                    "nio",
                    "--protocol",
                    "training",
                    "--fresh-start",
                    "--algo",
                    "All",
                    "--ramp-from",
                    "150",
                    "--langevin",
                    "--langevin-gamma",
                    "12",
                ]
            )
            self.assertEqual(rc, 0)
            incar = parse_incar(root / "Step1" / "NiO_m110_Big_U46" / "INCAR")
            self.assertEqual(incar["ALGO"], "All")
            self.assertEqual(incar["TEBEG"], "150")
            self.assertEqual(incar["MDALGO"], "3")
            self.assertEqual(incar["LANGEVIN_GAMMA"], "12 12")
            self.assertNotIn("SMASS", incar)

            manifest = json.loads(
                (root / "Step1" / "step1_manifest.json").read_text(encoding="utf-8")
            )
            settings = manifest["profile_settings"]
            self.assertEqual(settings["algo"], "All")
            self.assertEqual(settings["ramp_from_k"], 150.0)
            self.assertEqual(settings["langevin_gamma"], 12.0)


if __name__ == "__main__":
    unittest.main()
