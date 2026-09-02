from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from interfaceforge.progress import mlip_progress, render


def _build_campaign(root: Path) -> Path:
    camp = root / "camp"
    dp = camp / "models" / "deepmd"

    # dpa2: 2-member committee, both at the 1000-step target with checkpoints
    for index in range(2):
        model = dp / "dpa2" / f"model_{index:03d}"
        (model / "set").mkdir(parents=True)
        (model / "input.json").write_text(
            json.dumps({"training": {"numb_steps": 1000}}), encoding="utf-8"
        )
        (model / "lcurve.out").write_text(
            "#  step  rmse_val  rmse_trn  rmse_e_val  rmse_e_trn  rmse_f_val  rmse_f_trn  lr\n"
            "500  0.3  0.1  1e-3  1e-3  0.12  0.04  1e-6\n"
            "1000  0.28  0.1  1e-3  1e-3  0.11  0.04  1e-6\n",
            encoding="utf-8",
        )
        (model / "model.ckpt.pt").write_bytes(b"ckpt")
        (model / "frozen_model.pth").write_bytes(b"frozen")

    # dpa3: one member partway, one not started
    for index in (0, 1):
        model = dp / "dpa3" / f"model_{index:03d}"
        model.mkdir(parents=True)
        (model / "input.json").write_text(
            json.dumps({"training": {"numb_steps": 1000}}), encoding="utf-8"
        )
        if index == 0:
            (model / "lcurve.out").write_text(
                "#  step  rmse_val  rmse_trn  rmse_e_val  rmse_e_trn  rmse_f_val  rmse_f_trn  lr\n"
                "400  0.5  0.2  2e-3  2e-3  0.20  0.09  1e-5\n",
                encoding="utf-8",
            )
            (model / "model.ckpt.pt").write_bytes(b"ckpt")

    # dpa2 evaluation: 3 systems complete for the 2-member committee
    job = dp / "evaluation" / "dpa2" / "job_1000286"
    for system in range(3):
        system_dir = job / "by_system" / f"system_{system:03d}"
        system_dir.mkdir(parents=True)
        for member in range(2):
            (system_dir / f"model_{member:03d}_detail.e_peratom.out").write_text("1 1\n", encoding="utf-8")
            (system_dir / f"model_{member:03d}_detail.f.out").write_text("1 1 1 1 1 1\n", encoding="utf-8")
    (job / "rmse_overall.csv").write_text("architecture,model\n", encoding="utf-8")

    # MACE: from-scratch complete, fine-tune still running
    mace = camp / "models" / "mace_committee_520eV"
    for base, finished in (("mace_committee", True), ("mace_finetune_committee", False)):
        for seed in (11, 23):
            seed_dir = mace / base / f"seed_{seed}"
            (seed_dir / "logs").mkdir(parents=True)
            (seed_dir / "mace_model").mkdir()
            (seed_dir / "logs" / "run.log").write_text(
                "INFO: Initial: RMSE_F= 283.26 meV / A\n"
                "INFO: Epoch 5: head: Default, RMSE_E_per_atom= 4.1 meV, RMSE_F= 61.3 meV / A\n",
                encoding="utf-8",
            )
            if finished:
                (seed_dir / "mace_model" / f"x_seed{seed}_stagetwo.model").write_bytes(b"model")

    # comparison prepared and finalized
    out = camp / "audit" / "mlip_compare"
    out.mkdir(parents=True)
    (out / "comparison_manifest.json").write_text(
        json.dumps(
            {
                "deepmd_architecture": "dpa2",
                "systems": [{"system_id": f"system_{i:03d}"} for i in range(3)],
                "models": [{"model": "model_000"}, {"model": "model_001"}],
            }
        ),
        encoding="utf-8",
    )
    for i in range(3):
        for member in ("model_000", "model_001"):
            member_dir = out / "predictions" / "mace" / member
            member_dir.mkdir(parents=True, exist_ok=True)
            (member_dir / f"system_{i:03d}.npz").write_bytes(b"npz")
    (out / "comparison.json").write_text("{}", encoding="utf-8")
    return camp


class TestMlipProgress(unittest.TestCase):
    def test_reports_training_evaluation_and_comparison_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            camp = _build_campaign(Path(temporary))
            payload = mlip_progress(camp)

            training = {row["architecture"]: row for row in payload["deepmd_training"]}
            self.assertEqual(set(training), {"dpa2", "dpa3"})
            self.assertTrue(training["dpa2"]["complete"])
            self.assertFalse(training["dpa3"]["complete"])
            self.assertEqual(training["dpa2"]["target_steps"], 1000)
            self.assertEqual(training["dpa2"]["members"][0]["step"], 1000)
            self.assertAlmostEqual(
                training["dpa2"]["members"][0]["rmse_f_val_ev_ang"], 0.11
            )
            self.assertTrue(training["dpa2"]["members"][0]["frozen"])
            self.assertEqual(training["dpa3"]["members"][1]["step"], None)
            self.assertFalse(training["dpa3"]["members"][1]["checkpoint"])

            evaluation = payload["deepmd_evaluation"]
            self.assertEqual(len(evaluation), 1)
            self.assertEqual(evaluation[0]["architecture"], "dpa2")
            self.assertEqual(evaluation[0]["systems_complete"], 3)
            self.assertTrue(evaluation[0]["rmse_overall"])

            mace = {row["committee"]: row for row in payload["mace_committees"]}
            self.assertTrue(mace["mace_committee"]["complete"])
            self.assertFalse(mace["mace_finetune_committee"]["complete"])
            self.assertEqual(mace["mace_committee"]["members"][0]["epoch"], 5)
            self.assertAlmostEqual(
                mace["mace_committee"]["members"][0]["rmse_f_mev_ang"], 61.3
            )

            comparison = payload["comparisons"][0]
            self.assertEqual(comparison["deepmd_architecture"], "dpa2")
            self.assertEqual(comparison["mace_predictions"], "6/6")
            self.assertTrue(comparison["finalized"])

            text = render(payload)
            self.assertIn("DeePMD training", text)
            self.assertIn("[OK] dpa2", text)
            self.assertIn("[..] dpa3", text)
            self.assertIn("mace_finetune_committee", text)

    def test_empty_campaign_renders_without_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = mlip_progress(Path(temporary))
            self.assertEqual(payload["deepmd_training"], [])
            self.assertIn("no models/deepmd", render(payload))


if __name__ == "__main__":
    unittest.main()
