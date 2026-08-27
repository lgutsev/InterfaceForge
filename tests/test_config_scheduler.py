from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from interfaceforge.campaign import build_plan
from interfaceforge.config import load_campaign
from interfaceforge.errors import ConfigurationError, SafetyError
from interfaceforge.scheduler import render_job


def write_campaign(
    root: Path,
    *,
    deepmd: dict | None = None,
    mace: dict | None = None,
    preserve: bool = True,
    scheduler: str = "slurm",
) -> Path:
    profile = {
        "name": "test",
        "scheduler": scheduler,
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
        "models": {
            "deepmd": deepmd or {"enabled": False},
            "mace": mace or {"enabled": False},
        },
    }
    path = root / "campaign.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


class ConfigTests(unittest.TestCase):
    def test_missing_vasp_mlff_stage_defaults_to_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = write_campaign(root)
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            data.pop("stages")
            path.write_text(yaml.safe_dump(data), encoding="utf-8")

            plan = build_plan(load_campaign(path))

            self.assertFalse(any(task["engine"] == "vasp_mlff" for task in plan["tasks"]))

    def test_defect_is_a_valid_system_kind(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = write_campaign(root)
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            data["systems"][0]["kind"] = "defect"
            path.write_text(yaml.safe_dump(data), encoding="utf-8")

            campaign = load_campaign(path)

            self.assertEqual(campaign.systems[0].kind, "defect")

    def test_run_glob_is_parsed_for_geometry_class_matching(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = write_campaign(root)
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            data["systems"][0]["run_glob"] = "*/interface_*/*"
            path.write_text(yaml.safe_dump(data), encoding="utf-8")

            campaign = load_campaign(path)

            self.assertEqual(campaign.systems[0].run_glob, "*/interface_*/*")

    def test_run_glob_defaults_to_none(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = load_campaign(write_campaign(Path(temporary)))

            self.assertIsNone(campaign.systems[0].run_glob)

    def test_run_glob_must_be_a_string(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = write_campaign(root)
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            data["systems"][0]["run_glob"] = 12345
            path.write_text(yaml.safe_dump(data), encoding="utf-8")

            with self.assertRaises(ConfigurationError):
                load_campaign(path)

    def test_system_id_may_be_nested_to_mirror_a_source_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = write_campaign(root)
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            data["systems"][0]["id"] = "Real/N_Term/SiN_TiN_N-term_O_x0.25"
            path.write_text(yaml.safe_dump(data), encoding="utf-8")

            campaign = load_campaign(path)

            self.assertEqual(campaign.systems[0].id, "Real/N_Term/SiN_TiN_N-term_O_x0.25")

    def test_system_id_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = write_campaign(root)
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            data["systems"][0]["id"] = "Real/../../etc"
            path.write_text(yaml.safe_dump(data), encoding="utf-8")

            with self.assertRaises(ConfigurationError):
                load_campaign(path)

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

    def test_local_scheduler_rejects_array_jobs(self) -> None:
        # A local job script has no SLURM_ARRAY_TASK_ID; silently dropping
        # the `array` request (as the old code did) produced a script that
        # fails confusingly at runtime instead of at generation time.
        profile = {
            "scheduler": "local",
            "jobs": {"gpu": {"command": "dp train input.json"}},
        }
        with self.assertRaises(ConfigurationError):
            render_job(profile, "gpu", array="0-3")

    def test_local_scheduler_without_array_still_works(self) -> None:
        profile = {
            "scheduler": "local",
            "jobs": {"gpu": {"command": "dp train input.json"}},
        }
        script = render_job(profile, "gpu")
        self.assertIn("dp train input.json", script)


if __name__ == "__main__":
    unittest.main()
