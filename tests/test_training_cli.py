from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from test_config_scheduler import write_campaign

from interfaceforge.cli import build_parser, main
from interfaceforge.config import load_campaign
from interfaceforge.training import generate_deepmd_training


def make_deepmd_system(root: Path, split: str) -> None:
    system = root / "datasets" / "canonical" / "deepmd" / split / "system"
    set_dir = system / "set.000"
    set_dir.mkdir(parents=True)
    (system / "type.raw").write_text("0\n1\n", encoding="utf-8")
    (system / "type_map.raw").write_text("A\nB\n", encoding="utf-8")
    np.save(set_dir / "coord.npy", np.zeros((1, 6)))
    np.save(set_dir / "box.npy", np.eye(3).reshape(1, 9))
    np.save(set_dir / "energy.npy", np.zeros((1, 1)))
    np.save(set_dir / "force.npy", np.zeros((1, 6)))


class TrainingTests(unittest.TestCase):
    def test_modern_deepmd_campaign_generates_valid_shell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "valid", "test"):
                make_deepmd_system(root, split)
            deepmd = {
                "enabled": True,
                "profile": "deepmd_gpu",
                "backend": "pt_expt",
                "architectures": ["dpa2", "dpa3", "dpa4"],
                "committee": 2,
                "seeds": [11, 23],
                "numb_steps": 100,
                "batch_atoms": 64,
                "max_concurrent": 1,
            }
            campaign = load_campaign(write_campaign(root, deepmd=deepmd))
            manifest = generate_deepmd_training(campaign)
            self.assertEqual(
                manifest["execution_order"],
                [
                    "run_preflight.slurm",
                    "run_smoke.slurm",
                    "run_ensemble.slurm",
                    "run_evaluate.slurm",
                ],
            )
            self.assertEqual(manifest["backend_flag"], "--pt")
            dpa4 = json.loads(
                (root / "models/deepmd/dpa4/model_000/input.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(dpa4["model"]["descriptor"]["type"], "dpa4")
            self.assertEqual(dpa4["model"]["fitting_net"]["type"], "dpa4_ener")
            for name in manifest["execution_order"]:
                result = subprocess.run(
                    ["bash", "-n", root / "models/deepmd" / name],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_pack_parser_uses_unambiguous_output(self) -> None:
        args = build_parser().parse_args(
            ["vasp", "pack", "archive.zip", "--root", "runs"]
        )
        self.assertEqual(args.output, "archive.zip")
        self.assertEqual(args.root, "runs")

    def test_init_writes_campaign_and_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "new_campaign"
            self.assertEqual(main(["init", str(target)]), 0)
            self.assertTrue((target / "campaign.yaml").is_file())
            self.assertTrue((target / "profiles/loni.yaml").is_file())


if __name__ == "__main__":
    unittest.main()
