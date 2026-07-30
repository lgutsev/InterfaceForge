from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from interfaceforge.config import load_campaign
from interfaceforge.errors import ConfigurationError, SafetyError
from interfaceforge.scheduler import render_job


def write_campaign(root: Path, *, deepmd: dict | None = None, preserve: bool = True) -> Path:
    profile = {
        "name": "test",
        "scheduler": "slurm",
        "jobs": {
            "deepmd_gpu": {
                "partition": "gpu",
                "account": "allocation",
                "nodes": 1,
                "ntasks": 1,
                "cpus_per_task": 2,
                "gpus": 1,
                "time": "00:10:00",
                "command": "srun -n {ntasks} dp --version",
            },
            "mace_gpu": {
                "partition": "gpu",
                "account": "allocation",
                "nodes": 1,
                "ntasks": 1,
                "cpus_per_task": 2,
                "gpus": 1,
                "time": "00:10:00",
                "command": "python -V",
            },
            "vasp_workq": {
                "partition": "workq",
                "account": "allocation",
                "nodes": 1,
                "ntasks": 64,
                "cpus_per_task": 1,
                "time": "00:10:00",
                "command": "srun -n {ntasks} vasp_std",
            },
        },
    }
    (root / "profile.yaml").write_text(yaml.safe_dump(profile), encoding="utf-8")
    payload = {
        "schema_version": 1,
        "project": {"name": "test-interface"},
        "profile": "profile.yaml",
        "reference": {"engine": "vasp", "inputs": {}},
        "systems": [{"id": "interface", "kind": "interface", "structure": "POSCAR"}],
        "stages": {"vasp_mlff": {"enabled": False}},
        "dataset": {
            "strategy": "grouped",
            "stride": 1,
            "ratios": [0.8, 0.1, 0.1],
            "preserve_raw_forces": preserve,
        },
        "models": {"deepmd": deepmd or {"enabled": False}},
    }
    path = root / "campaign.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


class ConfigTests(unittest.TestCase):
    def test_rejects_zeroing_reference_forces(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(SafetyError):
                load_campaign(write_campaign(Path(temporary), preserve=False))

    def test_modern_dpa_requires_pytorch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            deepmd = {
                "enabled": True,
                "backend": "tensorflow",
                "architectures": ["dpa2"],
                "committee": 1,
                "seeds": [11],
            }
            with self.assertRaises(ConfigurationError):
                load_campaign(write_campaign(Path(temporary), deepmd=deepmd))

    def test_shell_braces_are_not_python_formatted(self) -> None:
        profile = {
            "scheduler": "slurm",
            "jobs": {
                "gpu": {
                    "partition": "gpu",
                    "account": "allocation",
                    "ntasks": 2,
                    "cpus_per_task": 1,
                    "command": "unused",
                }
            },
        }
        script = render_job(
            profile,
            "gpu",
            command='VALUE="${SLURM_ARRAY_TASK_ID:?missing}"\nf() { echo "$VALUE"; }\nsrun -n {ntasks} true',
            array="0-1",
        )
        self.assertIn('${SLURM_ARRAY_TASK_ID:?missing}', script)
        self.assertIn("f() { echo", script)
        self.assertIn("srun -n 2 true", script)


if __name__ == "__main__":
    unittest.main()
