from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from interfaceforge.ai2kit import adapter_status, export_adapter, preflight_adapter, run_adapter
from interfaceforge.config import load_campaign
from interfaceforge.errors import ConfigurationError


def write_tesla_fixture(root: Path) -> Path:
    for relative, content in {
        "inputs/INCAR": "ENCUT = 400\nEDIFF = 1E-5\n",
        "inputs/KPOINTS": "Gamma\n0\nGamma\n1 1 1\n0 0 0\n",
        "structures/interface.vasp": "placeholder\n",
        "datasets/train.extxyz": "placeholder\n",
        "datasets/valid.extxyz": "placeholder\n",
        "potcars/Ti/POTCAR": "TITEL = Ti\n",
        "potcars/N/POTCAR": "TITEL = N\n",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    model_paths = []
    for index in range(4):
        path = root / f"models/model-{index}.model"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"model-{index}".encode())
        model_paths.append(str(path.relative_to(root)))
    profile = {
        "name": "test-loni",
        "scheduler": "slurm",
        "ai2kit": {
            "commands": {
                "controller_python": "python",
                "ai2kit": "ai2-kit",
                "omb": "omb",
                "python": "python",
                "mace": "mace_run_train",
            },
            "jobs": {"train": "mace_gpu", "explore": "openmm_gpu", "label": "vasp_cpu"},
            "potcar_source": {
                "Ti": str(root / "potcars/Ti/POTCAR"),
                "N": str(root / "potcars/N/POTCAR"),
            },
        },
        "jobs": {
            "mace_gpu": {
                "partition": "gpu2", "account": "account", "nodes": 1,
                "ntasks": 1, "cpus_per_task": 4, "gpus": 1, "time": "01:00:00",
                "modules": [],
            },
            "openmm_gpu": {
                "partition": "gpu2", "account": "account", "nodes": 1,
                "ntasks": 1, "cpus_per_task": 4, "gpus": 1, "time": "01:00:00",
                "modules": [],
            },
            "vasp_cpu": {
                "partition": "workq", "account": "account", "nodes": 1,
                "ntasks": 64, "cpus_per_task": 1, "time": "01:00:00",
                "modules": [], "command": "srun -n {ntasks} vasp_gam",
            },
        },
    }
    (root / "profile.yaml").write_text(yaml.safe_dump(profile), encoding="utf-8")
    campaign = {
        "schema_version": 1,
        "project": {"name": "tesla-test"},
        "profile": "profile.yaml",
        "reference": {"engine": "vasp", "inputs": {"INCAR": "inputs/INCAR", "KPOINTS": "inputs/KPOINTS"}},
        "systems": [{"id": "interface", "kind": "interface", "structure": "structures/interface.vasp"}],
        "dataset": {"strategy": "grouped", "ratios": [0.8, 0.1, 0.1], "type_map": ["Ti", "N"]},
        "models": {"mace": {"enabled": True, "profile": "mace_gpu", "batch_size": 8, "max_num_epochs": 10}},
        "exploration": {"temperatures": [300, 450], "strains": [-0.01, 0.0, 0.01], "replicas": 2},
        "active_learning": {
            "enabled": True,
            "engine": "ai2kit",
            "approval_required": True,
            "max_iterations": 1,
            "output_root": "runs/active_learning/ai2kit",
            "ai2kit": {
                "workflow": "tesla_mace",
                "version": "1.0.9",
                "omb_version": "0.7.2",
                "executor_name": "loni",
                "trainer": "mace",
                "explorer": "openmm",
                "labeler": "vasp",
                "selector": "model_deviation",
                "model_count": 4,
                "committee_models": model_paths,
                "committee_seeds": [101, 211, 307, 419],
                "training_artifacts": ["datasets/train.extxyz"],
                "validation_artifacts": ["datasets/valid.extxyz"],
                "exploration_artifacts": ["structures/interface.vasp"],
                "trust_force_low": 0.1,
                "trust_force_high": 0.25,
                "selection_limit": 12,
                "md_steps": 100,
                "sample_frequency": 10,
                "equilibration_frames": 1,
                "timestep_fs": 0.5,
                "default_dtype": "float64",
            },
        },
    }
    path = root / "campaign.yaml"
    path.write_text(yaml.safe_dump(campaign, sort_keys=False), encoding="utf-8")
    return path


class Ai2KitTeslaTests(unittest.TestCase):
    def test_export_uses_ready_committee_ai2kit_and_omb(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = load_campaign(write_tesla_fixture(root))
            manifest = export_adapter(campaign)
            self.assertEqual(manifest["workflow"], "tesla_mace")
            generated = root / "runs/active_learning/ai2kit/generated"
            iteration = (generated / "01-workflow/iter-mace-openmm-vasp.sh").read_text(encoding="utf-8")
            self.assertIn("omb combo", iteration)
            self.assertIn("omb job slurm submit", iteration)
            self.assertIn("ai2-kit tool model_devi", iteration)
            self.assertIn("committee-models.txt", iteration)
            self.assertIn("model_devi.out", iteration)
            self.assertIn('NEW_DATASET_DIR="$ITER_DIR/new-dataset"', iteration)
            self.assertNotIn("+  --name", (generated / "00-config/mace/run.sh").read_text(encoding="utf-8"))
            header = (generated / "00-config/openmm/slurm-header.sh").read_text(encoding="utf-8")
            self.assertLess(header.index('PBS_O_WORKDIR:='), header.index("set -euo pipefail"))
            self.assertEqual(set(manifest["reference_inputs"]), {"INCAR", "KPOINTS"})
            self.assertEqual(set(manifest["potcars"]), {"Ti", "N"})
            for script in generated.rglob("*.sh"):
                result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, f"{script}: {result.stderr}")
            for script in generated.rglob("*.py"):
                compile(script.read_text(encoding="utf-8"), str(script), "exec")

    def test_run_is_dry_by_default_and_status_identifies_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = load_campaign(write_tesla_fixture(Path(temporary)))
            export_adapter(campaign)
            result = run_adapter(campaign)
            self.assertFalse(result["executed"])
            self.assertEqual(result["command"][0], "bash")
            status = adapter_status(campaign)
            self.assertEqual(status["workflow"], "tesla_mace")
            self.assertEqual(status["state"], "exported")

    def test_committee_model_count_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = write_tesla_fixture(Path(temporary))
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            payload["active_learning"]["ai2kit"]["committee_models"].pop()
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "committee_models must match"):
                load_campaign(path)

    def test_preflight_checks_python_and_source_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = load_campaign(write_tesla_fixture(root))
            export_adapter(campaign)
            (root / "potcars/Ti/POTCAR").write_text("tampered\n", encoding="utf-8")

            def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                if "-c" in command and any("importlib.metadata" in token for token in command):
                    return subprocess.CompletedProcess(command, 0, "3.12\n1.0.9\n0.7.2\n", "")
                return subprocess.CompletedProcess(command, 0, "ok\n", "")

            with (
                patch("interfaceforge.ai2kit_tesla.shutil.which", return_value="/mock/executable"),
                patch("interfaceforge.ai2kit_tesla.subprocess.run", side_effect=fake_run),
            ):
                report = preflight_adapter(campaign)

            checks = {item["name"]: item for item in report["checks"]}
            self.assertFalse(report["passed"])
            self.assertFalse(checks["controller_python"]["ok"])
            self.assertFalse(checks["potcar:Ti"]["ok"])
            self.assertTrue(checks["potcar:N"]["ok"])
            self.assertIn("command:sbatch", checks)
            self.assertIn("openmm_mace_system", checks)


if __name__ == "__main__":
    unittest.main()
