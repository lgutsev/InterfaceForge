from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from interfaceforge.ai2kit import (
    adapter_status,
    approve_round,
    export_adapter,
    preflight_adapter,
    run_adapter,
    stage_import,
)
from interfaceforge.cli import build_parser
from interfaceforge.config import load_campaign
from interfaceforge.errors import ConfigurationError, SafetyError

_HAS_ASE = importlib.util.find_spec("ase") is not None


def write_fixture(root: Path, *, enabled: bool = True, ai2kit_command: str = "ai2-kit") -> Path:
    for path, content in (
        (root / "inputs/INCAR", "ENCUT = 400\nEDIFF = 1E-5\n"),
        (root / "inputs/KPOINTS", "Gamma\n0\nGamma\n1 1 1\n0 0 0\n"),
        (root / "structures/interface.vasp", "synthetic structure placeholder\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    for split in ("train", "valid"):
        (root / f"datasets/canonical/deepmd/{split}").mkdir(parents=True)
    profile = {
        "name": "test",
        "scheduler": "slurm",
        "jobs": {
            "deepmd_gpu": {
                "partition": "gpu", "account": "account", "nodes": 1,
                "ntasks": 1, "cpus_per_task": 4, "gpus": 1, "time": "01:00:00",
                "modules": [],
            },
            "lammps_gpu": {
                "partition": "gpu", "account": "account", "nodes": 1,
                "ntasks": 1, "cpus_per_task": 4, "gpus": 1, "time": "01:00:00",
                "modules": [],
            },
            "vasp_cpu": {
                "partition": "cpu", "account": "account", "nodes": 1,
                "ntasks": 64, "cpus_per_task": 1, "time": "02:00:00",
                "modules": [],
            },
        },
        "ai2kit": {
            "ssh": {"host": "user@login.example"},
            "work_dir": "/remote/work/interfaceforge",
            "python_cmd": "/remote/bin/python",
            "commands": {"ai2kit": ai2kit_command, "deepmd": "dp", "lammps": "lmp", "vasp": "vasp_std"},
            "jobs": {"train": "deepmd_gpu", "explore": "lammps_gpu", "label": "vasp_cpu"},
            "potcar_source": {"H": "/licensed/H/POTCAR", "O": "/licensed/O/POTCAR"},
        },
    }
    (root / "profiles").mkdir()
    (root / "profiles/test.yaml").write_text(yaml.safe_dump(profile), encoding="utf-8")
    campaign = {
        "schema_version": 1,
        "project": {"name": "adapter-test"},
        "profile": "profiles/test.yaml",
        "reference": {"engine": "vasp", "inputs": {"INCAR": "inputs/INCAR", "KPOINTS": "inputs/KPOINTS"}},
        "systems": [{"id": "interface", "kind": "interface", "structure": "structures/interface.vasp"}],
        "dataset": {"strategy": "grouped", "ratios": [0.8, 0.1, 0.1], "type_map": ["H", "O"]},
        "models": {"deepmd": {
            "enabled": True, "profile": "deepmd_gpu", "backend": "tensorflow",
            "descriptor": "se_e2_a", "architectures": ["se_e2_a"],
            "committee": 2, "seeds": [11, 23], "numb_steps": 100,
        }},
        "exploration": {"temperatures": [300, 450], "strains": [0.0], "replicas": 1},
        "active_learning": {
            "enabled": enabled, "engine": "ai2kit", "approval_required": True,
            "max_iterations": 1, "output_root": "runs/active_learning/ai2kit",
            "ai2kit": {
                "version": "1.0.9", "executor_name": "loni", "trainer": "deepmd",
                "explorer": "lammps", "labeler": "vasp", "selector": "model_deviation",
                "architecture": "se_e2_a", "backend": "tensorflow", "model_count": 2,
                "training_artifacts": ["datasets/canonical/deepmd/train"],
                "validation_artifacts": ["datasets/canonical/deepmd/valid"],
                "exploration_artifacts": ["structures/interface.vasp"],
                "trust_force_low": 0.15, "trust_force_high": 0.30, "selection_limit": 20,
            },
        },
    }
    path = root / "campaign.yaml"
    path.write_text(yaml.safe_dump(campaign, sort_keys=False), encoding="utf-8")
    return path


class Ai2KitAdapterTests(unittest.TestCase):
    def test_disabled_configuration_loads_without_ai2kit_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = load_campaign(write_fixture(Path(temporary), enabled=False))
            self.assertFalse(campaign.active_learning["enabled"])

    def test_trust_thresholds_are_required_and_ordered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = write_fixture(Path(temporary))
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            del payload["active_learning"]["ai2kit"]["trust_force_low"]
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "trust_force_low"):
                load_campaign(path)

    def test_disabled_configuration_still_rejects_unknown_adapter_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = write_fixture(Path(temporary), enabled=False)
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            payload["active_learning"]["ai2kit"]["typo"] = True
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "Unknown active_learning.ai2kit"):
                load_campaign(path)

    def test_model_count_must_match_deepmd_committee(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = write_fixture(Path(temporary))
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            payload["active_learning"]["ai2kit"]["model_count"] = 1
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "must match"):
                load_campaign(path)

    def test_export_is_deterministic_and_does_not_copy_potcar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = load_campaign(write_fixture(root))
            first = export_adapter(campaign)
            generated = root / "runs/active_learning/ai2kit/generated"
            contents = {name: (generated / name).read_text(encoding="utf-8") for name in (
                "artifacts.yml", "executor.yml", "workflow.yml"
            )}
            second = export_adapter(campaign, force=True)
            self.assertEqual(first["export_fingerprint"], second["export_fingerprint"])
            self.assertEqual(contents, {name: (generated / name).read_text(encoding="utf-8") for name in contents})
            workflow = yaml.safe_load(contents["workflow.yml"])["workflow"]
            self.assertEqual(workflow["general"]["mass_map"], [1.008, 15.999])
            self.assertEqual(workflow["train"]["deepmd"]["model_num"], 2)
            self.assertFalse(any(path.name == "POTCAR" for path in (generated.parent).rglob("*")))

    def test_export_refuses_nonempty_output_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = load_campaign(write_fixture(Path(temporary)))
            export_adapter(campaign)
            with self.assertRaisesRegex(SafetyError, "not empty"):
                export_adapter(campaign)

    def test_dry_run_never_executes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = load_campaign(write_fixture(Path(temporary)))
            export_adapter(campaign)
            with mock.patch("subprocess.run") as run:
                result = run_adapter(campaign)
            run.assert_not_called()
            self.assertFalse(result["executed"])
            self.assertIn("cll-mlp-training", result["command"])

    def test_nested_slurm_execution_is_refused_before_submission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = load_campaign(write_fixture(Path(temporary)))
            export_adapter(campaign)
            with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "123"}):
                with self.assertRaisesRegex(SafetyError, "child jobs"):
                    run_adapter(campaign, execute=True)

    def test_local_preflight_with_fake_ai2kit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "bin/ai2-kit"
            executable.parent.mkdir()
            executable.write_text(
                "#!/usr/bin/env bash\n"
                "if [[ $1 == --version ]]; then echo 'ai2-kit 1.0.9'; exit 0; fi\n"
                "echo 'cll-mlp-training --checkpoint'; exit 0\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            campaign = load_campaign(write_fixture(root, ai2kit_command=str(executable)))
            export_adapter(campaign)
            report = preflight_adapter(campaign)
            self.assertTrue(report["passed"], report)
            self.assertFalse(report["remote_checked"])

    def test_status_does_not_deserialize_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = load_campaign(write_fixture(root))
            export_adapter(campaign)
            checkpoint = root / "runs/active_learning/ai2kit/checkpoints/cll"
            checkpoint.mkdir(parents=True)
            (checkpoint / "opaque.pkl").write_bytes(b"not a pickle")
            result = adapter_status(campaign)
            self.assertTrue(result["checkpoint_present"])
            self.assertIn("opaque", result["note"])

    def test_approval_rejects_tampered_import_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = load_campaign(write_fixture(root))
            export_adapter(campaign)
            destination = root / "runs/active_learning/ai2kit/imports/round_000"
            destination.mkdir(parents=True)
            import_manifest = destination / "import_manifest.json"
            import_manifest.write_text('{"accepted": 1}\n', encoding="utf-8")
            (destination / "approval.json").write_text(
                '{"approved": false, "import_manifest_sha256": "stale"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SafetyError, "changed after validation"):
                approve_round(campaign, round_number=0)

    def test_cli_surface_parses(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["active-learning", "ai2kit", "export", "-c", "campaign.yaml"])
        self.assertEqual(args.ai2kit_command, "export")
        args = parser.parse_args(["active-learning", "ai2kit", "import", "--round", "0", "--result-root", "results"])
        self.assertEqual(args.round_number, 0)

    @unittest.skipUnless(_HAS_ASE, "ASE is required for import staging")
    def test_import_is_staged_idempotently_without_canonical_mutation(self) -> None:
        import numpy as np
        from ase import Atoms
        from ase.calculators.singlepoint import SinglePointCalculator
        from ase.io import write

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = load_campaign(write_fixture(root))
            export_adapter(campaign)
            result_root = root / "results"
            result_root.mkdir()
            atoms = Atoms("HO", positions=[[0, 0, 0], [0, 0, 1]], cell=[5, 5, 5], pbc=True)
            atoms.calc = SinglePointCalculator(atoms, energy=-1.0, forces=np.zeros((2, 3)))
            write(result_root / "labels.extxyz", atoms, format="extxyz")
            canonical_before = sorted((root / "datasets/canonical").rglob("*"))
            first = stage_import(campaign, round_number=0, result_root=result_root)
            second = stage_import(campaign, round_number=0, result_root=result_root)
            self.assertEqual(first["accepted"], 1)
            self.assertTrue(second["idempotent"])
            self.assertEqual(canonical_before, sorted((root / "datasets/canonical").rglob("*")))


if __name__ == "__main__":
    unittest.main()
