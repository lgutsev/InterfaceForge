from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from test_config_scheduler import write_campaign

from interfaceforge.config import load_campaign
from interfaceforge.errors import SafetyError
from interfaceforge.training import (
    deepmd_input,
    generate_deepmd_training,
    validate_deepmd_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
MACE_FINETUNE = ROOT / "launch_scripts" / "mace_finetune_committee.sh"
MACE_TRAIN = ROOT / "launch_scripts" / "mace_train_committee.sh"
MACE_VIRIAL_A1 = ROOT / "launch_scripts" / "mace_virial_finetune.sbatch"
DEEPMD_VIRIAL_A1 = ROOT / "launch_scripts" / "deepmd_virial_finetune.sbatch"


def _make_deepmd_system(root: Path, split: str, *, with_virial: bool) -> None:
    system = root / "datasets" / "canonical" / "deepmd" / split / "system"
    set_dir = system / "set.000"
    set_dir.mkdir(parents=True)
    (system / "type.raw").write_text("0\n1\n", encoding="utf-8")
    (system / "type_map.raw").write_text("A\nB\n", encoding="utf-8")
    np.save(set_dir / "coord.npy", np.zeros((1, 6)))
    np.save(set_dir / "box.npy", np.eye(3).reshape(1, 9))
    np.save(set_dir / "energy.npy", np.zeros((1, 1)))
    np.save(set_dir / "force.npy", np.zeros((1, 6)))
    if with_virial:
        np.save(set_dir / "virial.npy", np.zeros((1, 9)))


class VirialTrainingTests(unittest.TestCase):
    def test_mace_finetune_requires_explicit_virial_loss(self) -> None:
        script = MACE_FINETUNE.read_text(encoding="utf-8")
        self.assertIn('LOSS="${MACE_LOSS:-weighted}"', script)
        self.assertIn('VIRIALS_KEY="${MACE_VIRIALS_KEY:-}"', script)
        self.assertIn('MACE_VIRIALS_KEY requires MACE_LOSS=virials', script)
        self.assertIn('--virials_key "$VIRIALS_KEY"', script)
        self.assertIn('--virials_weight "$VIRIALS_WEIGHT"', script)
        self.assertIn('--loss "$LOSS"', script)
        self.assertNotIn('--loss "weighted"', script)

    def test_mace_scratch_launcher_has_same_virial_guard(self) -> None:
        script = MACE_TRAIN.read_text(encoding="utf-8")
        self.assertIn('MACE_VIRIALS_KEY requires MACE_LOSS=virials', script)
        self.assertIn('--virials_key "$VIRIALS_KEY"', script)
        self.assertIn('--loss "$LOSS"', script)
        self.assertNotIn('--loss "weighted"', script)

    def test_mace_virial_a1_wrapper_is_isolated_and_uses_original_split(self) -> None:
        script = MACE_VIRIAL_A1.read_text(encoding="utf-8")
        self.assertIn('DATASET="$CAMP/models/mace_committee_520eV"', script)
        self.assertIn('OUTPUT="$CAMP/models/mace_committee_520eV_$RUN_TAG"', script)
        self.assertIn('MACE_LOSS="${MACE_LOSS:-virials}"', script)
        self.assertIn('MACE_VIRIALS_KEY="${MACE_VIRIALS_KEY:-REF_virial}"', script)
        self.assertIn('MACE_USE_STAGE_TWO="${MACE_USE_STAGE_TWO:-False}"', script)
        self.assertIn('MACE_LR="${MACE_LR:-0.001}"', script)
        self.assertIn('MACE_MAX_EPOCHS="${MACE_MAX_EPOCHS:-30}"', script)

    def test_virial_a1_slurm_wrappers_have_valid_bash_syntax(self) -> None:
        for launcher in (MACE_VIRIAL_A1, DEEPMD_VIRIAL_A1):
            with self.subTest(launcher=launcher.name):
                result = subprocess.run(
                    ["bash", "-n", str(launcher)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_deepmd_input_accepts_virial_preferences(self) -> None:
        payload = deepmd_input(
            dataset_root=Path("/tmp/example"),
            type_map=["A", "B"],
            architecture="dpa2",
            backend="pt_expt",
            seed=11,
            numb_steps=100,
            batch_atoms=64,
            systems={"train": ["train"], "valid": ["valid"], "test": ["test"]},
            loss_settings={"start_pref_v": 1.0, "limit_pref_v": 1.0},
        )
        self.assertEqual(payload["loss"]["start_pref_v"], 1.0)
        self.assertEqual(payload["loss"]["limit_pref_v"], 1.0)
        self.assertEqual(payload["loss"]["start_pref_f"], 1000.0)

    def test_deepmd_dataset_requires_virial_when_loss_is_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "valid", "test"):
                _make_deepmd_system(root, split, with_virial=False)
            dataset = root / "datasets" / "canonical" / "deepmd"
            with self.assertRaisesRegex(SafetyError, "virial.npy is missing"):
                validate_deepmd_dataset(dataset, require_virial=True)

    def test_deepmd_virial_campaign_records_loss_and_generates_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "valid", "test"):
                _make_deepmd_system(root, split, with_virial=True)
            deepmd = {
                "enabled": True,
                "profile": "deepmd_gpu",
                "backend": "pt_expt",
                "architectures": ["dpa2"],
                "committee": 1,
                "seeds": [11],
                "numb_steps": 100,
                "batch_atoms": 64,
                "max_concurrent": 1,
                "loss": {
                    "start_pref_v": 1.0,
                    "limit_pref_v": 1.0,
                },
            }
            campaign = load_campaign(write_campaign(root, deepmd=deepmd))
            manifest = generate_deepmd_training(campaign)
            self.assertTrue(manifest["virial_loss_enabled"])
            self.assertEqual(manifest["loss"]["start_pref_v"], 1.0)
            self.assertEqual(manifest["loss"]["limit_pref_v"], 1.0)

            payload = json.loads(
                (root / "models/deepmd/dpa2/model_000/input.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(payload["loss"]["start_pref_v"], 1.0)
            self.assertEqual(payload["loss"]["limit_pref_v"], 1.0)


if __name__ == "__main__":
    unittest.main()
