from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from interfaceforge.aimd import (
    audit_step1_incar,
    integrated_autocorrelation_time,
    plan_step2_retention,
    sample_step2_runs,
    select_decorrelated_frames,
    switch_step1_protocol,
)
from interfaceforge.cli import main
from interfaceforge.errors import SafetyError
from interfaceforge.vasp import parse_incar, prepare_step2_series


def _ar1_series(n: int, phi: float, *, seed: int = 7) -> list[float]:
    import numpy as np

    rng = np.random.default_rng(seed)
    noise = rng.normal(size=n)
    out = [0.0]
    for i in range(1, n):
        out.append(phi * out[-1] + noise[i])
    return [-808.0 + 0.01 * value for value in out]


def _oszicar(energies: list[float]) -> str:
    lines = []
    for step, energy in enumerate(energies, start=1):
        lines.append(
            f"{step:6d} T=  300. E= {energy:.5E} F= {energy:.5E} E0= {energy:.5E} "
            f"EK= 0.50000 SP 0.00E+00 SK 0.00E+00"
        )
    return "\n".join(lines) + "\n"


class AutocorrelationTests(unittest.TestCase):
    def test_white_noise_tau_near_one(self) -> None:
        tau = integrated_autocorrelation_time(_ar1_series(4000, 0.0))
        self.assertLess(tau, 3.0)

    def test_correlated_series_has_large_tau(self) -> None:
        # AR(1) with phi=0.9 has analytic tau = (1+phi)/(1-phi) = 19.
        tau = integrated_autocorrelation_time(_ar1_series(8000, 0.9))
        self.assertGreater(tau, 8.0)

    def test_training_selection_lands_in_target_band(self) -> None:
        plan = select_decorrelated_frames(
            _ar1_series(3000, 0.9), burn_in_frames=150, target_min=15, target_max=40
        )
        self.assertTrue(plan["within_target"])
        self.assertGreaterEqual(plan["kept_frames"], 15)
        self.assertLessEqual(plan["kept_frames"], 40)
        self.assertEqual(plan["burn_in_frames"], 150)
        self.assertTrue(all(index >= 150 for index in plan["indices"]))

    def test_academic_retention_is_dense_nblock_stride(self) -> None:
        plan = plan_step2_retention(
            "academic", _ar1_series(3000, 0.9), potim_fs=1.0, nblock=4
        )
        self.assertEqual(plan["stride"], 4)
        self.assertEqual(plan["kept_frames"], 750)

    def test_training_retention_is_small_and_decorrelated(self) -> None:
        plan = plan_step2_retention(
            "training", _ar1_series(3000, 0.9), potim_fs=1.0, nblock=4
        )
        self.assertLess(plan["kept_frames"], 60)
        self.assertGreater(plan["stride"], 4)
        self.assertEqual(plan["burn_in_frames"], 50)


_ACADEMIC_STEP1 = (
    "ISTART = 1\nIBRION = 0\nNSW = 2000\nPOTIM = 1.0\nSMASS = -1\nNBLOCK = 4\n"
    "TEBEG = 300\nTEEND = 300\nISIF = 2\nLWAVE = .TRUE.\n"
    "LDAU = .TRUE.\nLDAUL = 2 -1\nLDAUU = 4.6 0.0\nLDAUJ = 0.0 0.0\nLMAXMIX = 4\n"
)


class Step1ProtocolTests(unittest.TestCase):
    def test_switch_academic_incar_to_training_rewrites_only_nsw(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            incar = Path(tmp) / "INCAR"
            incar.write_text(_ACADEMIC_STEP1, encoding="utf-8")

            result = switch_step1_protocol(incar, "training")

            self.assertEqual(result["mode"], "switched")
            self.assertEqual(result["changed"], 1)
            parsed = parse_incar(incar)
            self.assertEqual(parsed["NSW"], "400")
            # Everything else is untouched.
            self.assertEqual(parsed["SMASS"], "-1")
            self.assertEqual(parsed["TEBEG"], "300")
            self.assertEqual(parsed["LDAUU"], "4.6 0.0")

    def test_audit_only_does_not_write_and_flags_length_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            incar = Path(tmp) / "INCAR"
            incar.write_text(_ACADEMIC_STEP1, encoding="utf-8")

            result = switch_step1_protocol(incar, "training", audit_only=True)

            self.assertEqual(result["mode"], "audited")
            self.assertEqual(result["changed"], 0)
            self.assertEqual(parse_incar(incar)["NSW"], "2000")
            self.assertEqual(result["status"], "WARN")
            self.assertTrue(
                any("outside the training range" in note for note in result["runs"][0]["audit"]["notes"])
            )

    def test_academic_incar_passes_academic_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            incar = Path(tmp) / "INCAR"
            incar.write_text(_ACADEMIC_STEP1, encoding="utf-8")
            audit = audit_step1_incar(incar, "academic")
            self.assertEqual(audit["status"], "PASS")
            self.assertAlmostEqual(audit["preheat_ps"], 2.0)

    def test_tree_root_switches_every_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Step1"
            for name in ("NiO_m110_Big_U46", "NiO_m110_Big_U46_Me4PACz"):
                (root / name).mkdir(parents=True)
                (root / name / "INCAR").write_text(_ACADEMIC_STEP1, encoding="utf-8")
            (root / "archive" / "old").mkdir(parents=True)
            (root / "archive" / "old" / "INCAR").write_text(_ACADEMIC_STEP1, encoding="utf-8")

            result = switch_step1_protocol(root, "training")

            self.assertEqual(result["incars"], 2)  # archive/ skipped
            self.assertEqual(result["changed"], 2)
            for name in ("NiO_m110_Big_U46", "NiO_m110_Big_U46_Me4PACz"):
                self.assertEqual(parse_incar(root / name / "INCAR")["NSW"], "400")
            self.assertEqual(parse_incar(root / "archive" / "old" / "INCAR")["NSW"], "2000")

    def test_cli_step1_protocol_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            incar = Path(tmp) / "INCAR"
            incar.write_text(_ACADEMIC_STEP1, encoding="utf-8")
            self.assertEqual(
                main(["vasp", "step1-protocol", str(incar), "--protocol", "training"]), 0
            )
            self.assertEqual(parse_incar(incar)["NSW"], "400")


def _step1_tree(root: Path, *, nsw: int) -> Path:
    step1 = root / "Step1"
    step1.mkdir()
    (step1 / "KPOINTS").write_text("Gamma\n0\nGamma\n1 1 1\n0 0 0\n", encoding="utf-8")
    launcher = step1 / "runvasp.sh"
    launcher.write_text("#!/usr/bin/env bash\nsbatch payload\n", encoding="utf-8")
    launcher.chmod(0o755)
    run = step1 / "NiO_m110_Big_U46"
    run.mkdir()
    (run / "INCAR").write_text(
        f"IBRION = 0\nNSW = {nsw}\nPOTIM = 1.0\nSMASS = -1\nTEBEG = 300\nTEEND = 300\n"
        "LWAVE = .TRUE.\nLDAU = .TRUE.\nLDAUL = 2 -1\nLDAUU = 4.6 0.0\n"
        "LDAUJ = 0.0 0.0\nLMAXMIX = 4\n",
        encoding="utf-8",
    )
    (run / "CONTCAR").write_text(
        "step1\n1.0\n10 0 0\n0 10 0\n0 0 10\nNi O\n1 1\nDirect\n0 0 0\n0.5 0.5 0.5\n",
        encoding="utf-8",
    )
    (run / "POTCAR").write_text("licensed fixture Ni O\n", encoding="utf-8")
    return step1


def _step1_tree_spin(root: Path, *, magmom: str = "2.0 -2.0") -> Path:
    step1 = root / "Step1"
    step1.mkdir()
    (step1 / "KPOINTS").write_text("Gamma\n0\nGamma\n1 1 1\n0 0 0\n", encoding="utf-8")
    launcher = step1 / "runvasp.sh"
    launcher.write_text("#!/usr/bin/env bash\nsbatch payload\n", encoding="utf-8")
    launcher.chmod(0o755)
    run = step1 / "NiO_m110_Big_U46"
    run.mkdir()
    (run / "INCAR").write_text(
        "IBRION = 0\nNSW = 400\nPOTIM = 1.0\nSMASS = -1\nTEBEG = 300\nTEEND = 300\n"
        "ISPIN = 2\nLASPH = .TRUE.\n"
        f"MAGMOM = {magmom}\n"
        "LWAVE = .TRUE.\nLDAU = .TRUE.\nLDAUL = 2 -1\nLDAUU = 4.6 0.0\n"
        "LDAUJ = 0.0 0.0\nLDAUPRINT = 1\nLMAXMIX = 4\n",
        encoding="utf-8",
    )
    (run / "CONTCAR").write_text(
        "step1\n1.0\n10 0 0\n0 10 0\n0 0 10\nNi O\n1 1\nDirect\n0 0 0\n0.5 0.5 0.5\n",
        encoding="utf-8",
    )
    (run / "POTCAR").write_text("licensed fixture Ni O\n", encoding="utf-8")
    return step1


# A deliberately heavy custom Step2 template (publication-grade output).
_HEAVY_TEMPLATE = (
    "ENCUT = 520\nPREC = Accurate\nLREAL = .FALSE.\nEDIFF = 1E-6\nADDGRID = .TRUE.\n"
    "ISMEAR = 0\nSIGMA = 0.05\nIBRION = 0\nNSW = 1\nNBLOCK = 1\nPOTIM = 1.0\n"
    "MDALGO = 2\nSMASS = 1.0\nTEBEG = 1\nTEEND = 1\nISIF = 2\n"
    "LORBIT = 11\nLDAUPRINT = 1\nLDIPOL = .TRUE.\nIDIPOL = 3\nDIPOL = 0.5 0.5 0.5\n"
    "NWRITE = 2\nLWAVE = .FALSE.\nLCHARG = .FALSE.\nNCORE = 4\n"
)


class Step2SpinInheritanceTests(unittest.TestCase):
    def _template(self, root: Path) -> Path:
        path = root / "INCAR_STEP2"
        path.write_text(_HEAVY_TEMPLATE, encoding="utf-8")
        return path

    def test_academic_inherits_spin_verbatim_and_keeps_template_otherwise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            step1 = _step1_tree_spin(root)

            prepare_step2_series(
                step1, temperatures=[300], template=self._template(root)
            )  # academic default

            incar = parse_incar(root / "Step2_300K" / "NiO_m110_Big_U46" / "INCAR")
            self.assertEqual(incar["ISPIN"], "2")
            self.assertEqual(incar["MAGMOM"], "2.0 -2.0")
            self.assertEqual(incar["LASPH"], ".TRUE.")
            # DFT+U inherited verbatim; academic does not trim the template.
            self.assertEqual(incar["LDAUU"], "4.6 0.0")
            self.assertEqual(incar["LDAUPRINT"], "1")
            self.assertEqual(incar["LORBIT"], "11")
            self.assertEqual(incar["LDIPOL"], ".TRUE.")

    def test_training_inherits_spin_but_trims_output_and_dipole(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            step1 = _step1_tree_spin(root)

            prepare_step2_series(
                step1, temperatures=[300], protocol="training", template=self._template(root)
            )

            incar = parse_incar(root / "Step2_300K" / "NiO_m110_Big_U46" / "INCAR")
            self.assertEqual(incar["ISPIN"], "2")
            self.assertEqual(incar["MAGMOM"], "2.0 -2.0")
            # real DFT+U parameters are still inherited untouched...
            self.assertEqual(incar["LDAUU"], "4.6 0.0")
            self.assertEqual(incar["LDAUL"], "2 -1")
            self.assertEqual(incar["LMAXMIX"], "4")
            # ...only print verbosity / output tags are forced or stripped.
            self.assertEqual(incar["LDAUPRINT"], "0")  # Step1 had LDAUPRINT=1
            self.assertEqual(incar["NWRITE"], "1")
            self.assertNotIn("LORBIT", incar)
            self.assertNotIn("LDIPOL", incar)
            self.assertNotIn("IDIPOL", incar)
            self.assertNotIn("DIPOL", incar)
            audit = json.loads(
                (root / "Step2_300K" / "step2_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(audit["status"], "PASS")  # over-convergence is a note, not a fail
            notes = " ".join(n for row in audit["runs"] for n in row["notes"])
            self.assertIn("over-converged for training", notes)
            self.assertIn("LREAL", notes)

    def test_set_protocol_rewrites_existing_incar_and_keeps_hubbards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            step1 = _step1_tree_spin(root)
            tmpl = self._template(root)

            # Prepared once in academic mode.
            prepare_step2_series(step1, temperatures=[300], template=tmpl)
            before = parse_incar(root / "Step2_300K" / "NiO_m110_Big_U46" / "INCAR")
            self.assertEqual(before["LORBIT"], "11")
            self.assertEqual(before["LDAUPRINT"], "1")

            # Convert the existing tree to training in place.
            result = prepare_step2_series(
                step1, temperatures=[300], template=tmpl, protocol="training",
                reprotocol=True,
            )
            self.assertEqual(result["mode"], "reprotocoled-and-audited")

            after = parse_incar(root / "Step2_300K" / "NiO_m110_Big_U46" / "INCAR")
            self.assertNotIn("LORBIT", after)
            self.assertNotIn("LDIPOL", after)
            self.assertEqual(after["LDAUPRINT"], "0")
            self.assertEqual(after["NWRITE"], "1")
            # DFT+U and spin untouched.
            self.assertEqual(after["LDAUU"], "4.6 0.0")
            self.assertEqual(after["LDAUL"], "2 -1")
            self.assertEqual(after["LMAXMIX"], "4")
            self.assertEqual(after["ISPIN"], "2")
            self.assertEqual(after["MAGMOM"], "2.0 -2.0")
            manifest = json.loads(
                (root / "Step2_300K" / "step2_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["sampling"]["protocol"], "training")
            # POSCAR / inputs are left alone.
            self.assertTrue((root / "Step2_300K" / "NiO_m110_Big_U46" / "POTCAR").is_file())

    def test_set_protocol_round_trips_back_to_academic_byte_for_byte(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            step1 = _step1_tree_spin(root)
            tmpl = self._template(root)
            incar = root / "Step2_300K" / "NiO_m110_Big_U46" / "INCAR"

            prepare_step2_series(step1, temperatures=[300], template=tmpl)
            original = incar.read_text(encoding="utf-8")

            prepare_step2_series(
                step1, temperatures=[300], template=tmpl, protocol="training", reprotocol=True
            )
            self.assertNotEqual(incar.read_text(encoding="utf-8"), original)

            prepare_step2_series(
                step1, temperatures=[300], template=tmpl, protocol="academic", reprotocol=True
            )
            self.assertEqual(incar.read_text(encoding="utf-8"), original)

    def test_set_protocol_refuses_after_a_run_started(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            step1 = _step1_tree_spin(root)
            tmpl = self._template(root)
            prepare_step2_series(step1, temperatures=[300], template=tmpl)
            (root / "Step2_300K" / "NiO_m110_Big_U46" / "OSZICAR").write_text("1 T= 300\n")

            with self.assertRaises(SafetyError):
                prepare_step2_series(
                    step1, temperatures=[300], template=tmpl, protocol="training",
                    reprotocol=True,
                )

    def test_magmom_length_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            step1 = _step1_tree_spin(root, magmom="2.0 -2.0 0.0")  # 3 for 2 ions

            with self.assertRaises(SafetyError):
                prepare_step2_series(step1, temperatures=[300])


class Step2ProtocolTests(unittest.TestCase):
    def test_training_protocol_recorded_in_manifest_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            step1 = _step1_tree(root, nsw=2000)

            result = prepare_step2_series(step1, temperatures=[300], protocol="training")

            self.assertEqual(result["protocol"], "training")
            self.assertIsNone(result["sampling"]["training_frames_per_run"])
            manifest = json.loads(
                (root / "Step2_300K" / "step2_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["sampling"]["protocol"], "training")
            self.assertEqual(
                manifest["sampling"]["retention"]["method"], "energy-autocorrelation"
            )
            audit_md = (root / "Step2_300K" / "step2_audit.md").read_text(encoding="utf-8")
            self.assertIn("Protocol: **training**", audit_md)
            # Long academic preheat flagged (informational, still PASS).
            audit = json.loads(
                (root / "Step2_300K" / "step2_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(audit["status"], "PASS")
            self.assertTrue(any(row["notes"] for row in audit["runs"]))

    def test_academic_protocol_is_unchanged_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            step1 = _step1_tree(root, nsw=2000)

            result = prepare_step2_series(step1, temperatures=[300])

            self.assertEqual(result["protocol"], "academic")
            self.assertEqual(result["sampling"]["training_frames_per_run"], 1250)
            audit = json.loads(
                (root / "Step2_300K" / "step2_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(audit["sampling"]["nsw"], 5000)
            self.assertEqual(audit["sampling"]["nblock"], 4)
            self.assertTrue(all(row["training_frames"] == 1250 for row in audit["runs"]))

    def test_step2_sample_produces_small_decorrelated_count_for_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            step1 = _step1_tree(root, nsw=400)
            prepare_step2_series(step1, temperatures=[300], protocol="training")

            run_dir = root / "Step2_300K" / "NiO_m110_Big_U46"
            (run_dir / "OSZICAR").write_text(
                _oszicar(_ar1_series(3000, 0.9)), encoding="utf-8"
            )

            summary = sample_step2_runs([root / "Step2_300K"])

            self.assertEqual(summary["mode"], "written")
            payload = json.loads(
                (root / "Step2_300K" / "step2_sample.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["protocol"], "training")
            row = payload["runs"][0]
            self.assertEqual(row["status"], "OK")
            self.assertGreaterEqual(row["kept_frames"], 15)
            self.assertLessEqual(row["kept_frames"], 40)
            self.assertEqual(row["burn_in_frames"], 50)

    def test_step2_sample_reports_pending_when_no_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            step1 = _step1_tree(root, nsw=400)
            prepare_step2_series(step1, temperatures=[300], protocol="training")

            summary = sample_step2_runs([root / "Step2_300K"], dry_run=True)

            self.assertEqual(summary["mode"], "dry-run")
            self.assertEqual(summary["roots"][0]["pending"], 1)
            self.assertFalse((root / "Step2_300K" / "step2_sample.json").exists())


if __name__ == "__main__":
    unittest.main()
