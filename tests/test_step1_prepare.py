from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from interfaceforge.cli import main
from interfaceforge.errors import SafetyError
from interfaceforge.vasp import parse_incar, prepare_step1_series

_OPT_INCAR = (
    "ENCUT = 520\nPREC = Accurate\nEDIFF = 1E-6\nGGA = PE\n"
    "ISPIN = 2\nLASPH = .TRUE.\nMAGMOM = 2*2.0 3*-2.0\n"
    "LDAU = .TRUE.\nLDAUTYPE = 2\nLDAUL = 2 -1\nLDAUU = 4.6 0.0\n"
    "LDAUJ = 0.0 0.0\nLDAUPRINT = 1\nLMAXMIX = 4\nIBRION = 2\nISIF = 2\nNSW = 200\n"
)


def _opt_tree(
    root: Path,
    *,
    magmom: str = "2*2.0 3*-2.0",
    wavecar: bool = True,
    potcar: bool = True,
    launcher_dir: Path | None = None,
) -> Path:
    opt = root / "OPT"
    run = opt / "NiO_m110_Big_U46"
    run.mkdir(parents=True)
    (opt / "KPOINTS").write_text("Gamma\n0\nGamma\n2 2 1\n0 0 0\n", encoding="utf-8")
    launcher = (launcher_dir if launcher_dir is not None else opt) / "runvasp.sh"
    launcher.write_text("#!/usr/bin/env bash\nsbatch payload\n", encoding="utf-8")
    launcher.chmod(0o755)
    (run / "INCAR").write_text(_OPT_INCAR.replace("2*2.0 3*-2.0", magmom), encoding="utf-8")
    coords = "\n".join(f"0.1 0.1 {0.30 + i * 0.02:.4f}" for i in range(5))
    (run / "CONTCAR").write_text(
        f"opt\n1.0\n10 0 0\n0 10 0\n0 0 40\nNi O\n3 2\nDirect\n{coords}\n", encoding="utf-8"
    )
    if potcar:
        (run / "POTCAR").write_text("licensed fixture Ni O\n", encoding="utf-8")
    if wavecar:
        (run / "WAVECAR").write_text("x" * 4096, encoding="utf-8")
    return opt


class Step1PrepareTests(unittest.TestCase):
    def test_inherits_opt_spin_and_hubbard_verbatim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            opt = _opt_tree(root)

            result = prepare_step1_series(opt, protocol="academic")

            self.assertEqual(result["mode"], "prepared-and-audited")
            self.assertEqual(result["nsw"], 2000)
            incar = parse_incar(root / "Step1" / "NiO_m110_Big_U46" / "INCAR")
            self.assertEqual(incar["ISPIN"], "2")
            self.assertEqual(incar["MAGMOM"], "2*2.0 3*-2.0")
            self.assertEqual(incar["LASPH"], ".TRUE.")
            self.assertEqual(incar["GGA"], "PE")
            self.assertEqual(incar["LDAUU"], "4.6 0.0")
            self.assertEqual(incar["LMAXMIX"], "4")
            # preheat MD tags + restart come from the template
            self.assertEqual(incar["IBRION"], "0")
            self.assertEqual(incar["SMASS"], "-1")
            self.assertEqual(incar["NSW"], "2000")
            self.assertEqual(incar["ISTART"], "1")
            self.assertEqual(incar["TEBEG"], "300")
            # template convergence settings, not the OPT's
            self.assertEqual(incar["ENCUT"], "400")
            self.assertNotIn("LDAUPRINT", incar)  # LDAUPRINT is not inherited

    def test_wavecar_is_brought_across_and_audited(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            opt = _opt_tree(root)
            prepare_step1_series(opt, protocol="training")
            wc = root / "Step1" / "NiO_m110_Big_U46" / "WAVECAR"
            self.assertTrue(wc.is_file() and wc.stat().st_size)
            audit = json.loads(
                (root / "Step1" / "step1_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(audit["status"], "PASS")
            self.assertEqual(audit["runs"][0]["preheat_ps"], 0.4)

    def test_missing_wavecar_falls_back_to_fresh_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            opt = _opt_tree(root, wavecar=False)
            result = prepare_step1_series(opt, protocol="training")
            self.assertEqual(result["mode"], "prepared-and-audited")
            self.assertEqual(result["fresh_start_runs"], 1)
            self.assertTrue(result["warnings"])
            run = root / "Step1" / "NiO_m110_Big_U46"
            incar = parse_incar(run / "INCAR")
            self.assertEqual(incar["ISTART"], "0")
            # Hubbard U + spin are still inherited verbatim from the OPT INCAR
            self.assertEqual(incar["ISPIN"], "2")
            self.assertEqual(incar["MAGMOM"], "2*2.0 3*-2.0")
            self.assertEqual(incar["LDAUU"], "4.6 0.0")
            self.assertEqual(incar["LMAXMIX"], "4")
            self.assertEqual(incar["GGA"], "PE")
            self.assertFalse((run / "WAVECAR").exists())
            audit = json.loads((root / "Step1" / "step1_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["status"], "PASS")

    def test_prepares_without_potcar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            opt = _opt_tree(root, potcar=False)
            result = prepare_step1_series(opt, protocol="training")
            self.assertEqual(result["mode"], "prepared-and-audited")
            self.assertTrue(any("POTCAR" in w for w in result["warnings"]))
            run = root / "Step1" / "NiO_m110_Big_U46"
            self.assertTrue((run / "INCAR").is_file())
            self.assertTrue((run / "KPOINTS").is_file())
            self.assertFalse((run / "POTCAR").exists())
            audit = json.loads((root / "Step1" / "step1_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["status"], "PASS")

    def test_launcher_is_found_in_the_invocation_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            opt = _opt_tree(root, launcher_dir=root)  # runvasp.sh sits above OPT/, not inside it
            cwd = Path.cwd()
            os.chdir(root)
            try:
                result = prepare_step1_series(opt, protocol="training")
            finally:
                os.chdir(cwd)
            self.assertEqual(result["mode"], "prepared-and-audited")
            launcher = root / "Step1" / "NiO_m110_Big_U46" / "runvasp.sh"
            self.assertTrue(launcher.is_file())
            self.assertTrue(os.access(launcher, os.X_OK))

    def test_missing_wavecar_is_rejected_with_require_wavecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            opt = _opt_tree(root, wavecar=False)
            with self.assertRaises(SafetyError):
                prepare_step1_series(opt, require_wavecar=True)

    def test_fresh_start_ignores_existing_wavecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            opt = _opt_tree(root)
            result = prepare_step1_series(opt, protocol="training", fresh_start=True)
            run = root / "Step1" / "NiO_m110_Big_U46"
            self.assertEqual(parse_incar(run / "INCAR")["ISTART"], "0")
            self.assertFalse((run / "WAVECAR").exists())
            self.assertEqual(result["fresh_start_runs"], 1)

    def test_magmom_length_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            opt = _opt_tree(root, magmom="2*2.0 2*-2.0")  # 4 for 5 ions
            with self.assertRaises(SafetyError):
                prepare_step1_series(opt)

    def test_dry_run_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            opt = _opt_tree(root)
            result = prepare_step1_series(opt, dry_run=True)
            self.assertEqual(result["mode"], "dry-run")
            self.assertFalse((root / "Step1").exists())

    def test_refuses_to_overwrite_existing_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            opt = _opt_tree(root)
            prepare_step1_series(opt)
            with self.assertRaises(SafetyError):
                prepare_step1_series(opt)

    def test_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            opt = _opt_tree(root)
            self.assertEqual(main(["vasp", "step1-prepare", str(opt), "--protocol", "training"]), 0)
            self.assertEqual(
                parse_incar(root / "Step1" / "NiO_m110_Big_U46" / "INCAR")["NSW"], "400"
            )


if __name__ == "__main__":
    unittest.main()
