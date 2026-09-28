from __future__ import annotations

import importlib.util
import math
import subprocess
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = ROOT / "launch_scripts" / "uma_zero_shot_audit.py"
AUDIT_SLURM = ROOT / "launch_scripts" / "uma_zero_shot_audit.sbatch"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("uma_zero_shot_audit", AUDIT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class UmaZeroShotAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = _load_audit_module()

    def test_reference_virial_to_stress_sign_and_units(self) -> None:
        virial = np.diag([-0.08, -0.16, -0.24])

        class FakeAtoms:
            info = {"REF_virial": virial.reshape(-1)}

            @staticmethod
            def get_volume() -> float:
                return 8.0

        stress = self.audit._reference_stress_gpa(FakeAtoms())
        expected = (-virial / 8.0) * self.audit.EV_A3_TO_GPA
        np.testing.assert_allclose(stress, expected)

    def test_composition_centering_removes_only_fixed_composition_offset(self) -> None:
        rows = [
            {
                "composition": "N1_Si1",
                "natoms": 2,
                "ref_energy_ev": -10.0,
                "uma_energy_ev": -9.8,
            },
            {
                "composition": "N1_Si1",
                "natoms": 2,
                "ref_energy_ev": -9.0,
                "uma_energy_ev": -8.7,
            },
            {
                "composition": "N1_Ti1",
                "natoms": 2,
                "ref_energy_ev": -8.0,
                "uma_energy_ev": -7.0,
            },
        ]
        used = self.audit._apply_composition_centering(rows)
        self.assertEqual(used, 2)
        self.assertAlmostEqual(
            rows[0]["centered_energy_error_mev_per_atom"], -25.0
        )
        self.assertAlmostEqual(
            rows[1]["centered_energy_error_mev_per_atom"], 25.0
        )
        self.assertTrue(
            math.isnan(rows[2]["centered_energy_error_mev_per_atom"])
        )

    def test_gate_is_diagnostic_and_handles_missing_metric(self) -> None:
        self.assertEqual(self.audit._gate(0.10, 0.15)["status"], "PASS")
        self.assertEqual(self.audit._gate(0.20, 0.15)["status"], "FAIL")
        self.assertEqual(self.audit._gate(math.nan, 0.15)["status"], "NA")

    def test_loni_launcher_has_valid_bash_syntax_and_safe_defaults(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(AUDIT_SLURM)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        script = AUDIT_SLURM.read_text(encoding="utf-8")
        self.assertIn(
            'UMA_MODEL="${UMA_MODEL:-uma-s-1p2p1}"', script
        )
        self.assertIn('UMA_TASK="${UMA_TASK:-omat}"', script)
        self.assertIn(
            'UMA_INFERENCE_SETTINGS="${UMA_INFERENCE_SETTINGS:-batch}"',
            script,
        )
        self.assertIn(
            'UMA_ENV="${UMA_ENV:-/project/lgutsev/env/lgutsev_dev}"', script
        )
        self.assertIn("UMA_RESUME", script)

    def test_python_audit_uses_fairchem_checkpoint_loader_and_canonical_labels(
        self,
    ) -> None:
        script = AUDIT_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("FAIRChemCalculator.from_model_checkpoint", script)
        self.assertIn('"REF_energy"', script)
        self.assertIn('"REF_forces"', script)
        self.assertIn('"REF_virial"', script)
        self.assertIn("composition-centered", script)
        self.assertIn(
            'datasets" / "canonical" / "test.extxyz', script
        )


if __name__ == "__main__":
    unittest.main()
