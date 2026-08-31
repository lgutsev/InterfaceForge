from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from test_config_scheduler import write_campaign

from interfaceforge.campaign import prepare_campaign
from interfaceforge.cli import build_parser, main
from interfaceforge.config import load_campaign
from interfaceforge.errors import ConfigurationError, SafetyError
from interfaceforge.training import (
    _deepmd_shell_prefix,
    generate_deepmd_training,
    generate_mace_training,
    validate_deepmd_dataset,
)
from interfaceforge.vasp import parse_incar


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
            campaign_path = write_campaign(root, deepmd=deepmd)
            profile_path = root / "profile.yaml"
            profile_data = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
            profile_data["jobs"]["deepmd_gpu"]["modules"] = [
                "deepmd-kit/r9.3-deepmd3.2.0.b.0-gpu"
            ]
            profile_path.write_text(
                yaml.safe_dump(profile_data, sort_keys=False), encoding="utf-8"
            )
            campaign = load_campaign(campaign_path)
            manifest = generate_deepmd_training(campaign)
            self.assertEqual(manifest["runtime"]["profile"], "deepmd_gpu")
            self.assertEqual(
                manifest["runtime"]["modules"],
                ["deepmd-kit/r9.3-deepmd3.2.0.b.0-gpu"],
            )
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

    def test_dpa2_finetune_requires_pretrained_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "valid", "test"):
                make_deepmd_system(root, split)
            deepmd = {
                "enabled": True,
                "profile": "deepmd_gpu",
                "backend": "pt_expt",
                "architectures": ["dpa2_ft"],
                "committee": 1,
                "seeds": [11],
            }
            campaign_path = write_campaign(root, deepmd=deepmd)
            with self.assertRaisesRegex(ConfigurationError, "finetune.pretrained"):
                load_campaign(campaign_path)

    def test_dpa2_finetune_emits_gated_finetune_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "valid", "test"):
                make_deepmd_system(root, split)
            pretrained = root / "pretrained" / "dpa2_openlam.pt"
            pretrained.parent.mkdir(parents=True)
            pretrained.write_bytes(b"placeholder checkpoint")
            deepmd = {
                "enabled": True,
                "profile": "deepmd_gpu",
                "backend": "pt_expt",
                "architectures": ["dpa2", "dpa2_ft"],
                "committee": 2,
                "seeds": [11, 23],
                "numb_steps": 100,
                "batch_atoms": 64,
                "max_concurrent": 1,
                "finetune": {
                    "pretrained": str(pretrained),
                    "model_branch": "Domains_Anode",
                },
            }
            campaign = load_campaign(write_campaign(root, deepmd=deepmd))
            manifest = generate_deepmd_training(campaign)
            self.assertEqual(
                manifest["finetune"],
                {
                    "architecture": "dpa2_ft",
                    "pretrained": str(pretrained),
                    "model_branch": "Domains_Anode",
                },
            )
            ensemble = (root / "models/deepmd/run_ensemble.slurm").read_text(
                encoding="utf-8"
            )
            smoke = (root / "models/deepmd/run_smoke.slurm").read_text(encoding="utf-8")
            self.assertIn('elif [[ "$ARCH" == "dpa2_ft" ]]', ensemble)
            self.assertIn("--finetune", ensemble)
            self.assertIn("--model-branch Domains_Anode", ensemble)
            self.assertIn("--model-branch Domains_Anode", smoke)
            # scratch dpa2 must not pick up the finetune flag
            self.assertNotIn("dpa2/model_${MODEL_ID} --finetune", ensemble)
            ft_input = json.loads(
                (root / "models/deepmd/dpa2_ft/model_000/input.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(ft_input["model"]["descriptor"]["type"], "dpa2")
            for name in manifest["execution_order"]:
                result = subprocess.run(
                    ["bash", "-n", root / "models/deepmd" / name],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_pytorch_audit_uses_checkpoint_and_records_export_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "valid", "test"):
                make_deepmd_system(root, split)
            deepmd = {
                "enabled": True,
                "profile": "deepmd_gpu",
                "backend": "pt_expt",
                "architectures": ["dpa4"],
                "committee": 1,
                "seeds": [11],
                "numb_steps": 100,
                "batch_atoms": 64,
                "max_concurrent": 1,
            }
            campaign = load_campaign(write_campaign(root, deepmd=deepmd))
            manifest = generate_deepmd_training(campaign)
            ensemble = (root / "models/deepmd/run_ensemble.slurm").read_text(encoding="utf-8")
            evaluation = (root / "models/deepmd/run_evaluate.slurm").read_text(encoding="utf-8")
            self.assertEqual(manifest["evaluation_model_name"], "model.ckpt.pt")
            self.assertIn("freeze failed", ensemble)
            self.assertIn("DPA-4 freeze failed", ensemble)
            self.assertIn("exit 3", ensemble)
            self.assertIn("MODEL_FILE=model.ckpt.pt", ensemble)
            self.assertIn("model_${MODEL_ID}/model.ckpt.pt", evaluation)
            self.assertIn("summarize_deepmd.py", evaluation)
            self.assertEqual(
                manifest["evaluation_reports"],
                ["rmse_by_system.csv", "rmse_overall.csv", "rmse_audit.json"],
            )

    def test_deepmd_shell_prefix_avoids_nested_srun(self) -> None:
        direct_settings: dict = {}
        slurm_prefix = _deepmd_shell_prefix(direct_settings, "pytorch", scheduler="slurm")
        local_prefix = _deepmd_shell_prefix(direct_settings, "pytorch", scheduler="local")
        self.assertNotIn("srun", slurm_prefix)
        self.assertIn('dp_exec() { dp "$@"; }', slurm_prefix)
        self.assertNotIn("srun", local_prefix)
        self.assertIn('dp_exec() { dp "$@"; }', local_prefix)

    def test_deepmd_shell_prefix_container_path_also_drops_srun_locally(self) -> None:
        settings = {"container_image": "/opt/images/deepmd.sif"}
        local_prefix = _deepmd_shell_prefix(settings, "pytorch", scheduler="local")
        self.assertNotIn("srun", local_prefix)
        self.assertIn('exec --nv "${DEEPMD_BIND_ARGS[@]}" "$DEEPMD_IMAGE" dp "$@"', local_prefix)

    def test_local_profile_deepmd_committee_fails_fast_not_silently_broken(self) -> None:
        # DeePMD committees are distributed via a Slurm array task id; a
        # local profile has no such mechanism. This must fail loudly at
        # generation time rather than emit a script that hangs on
        # ${SLURM_ARRAY_TASK_ID:?Submit with sbatch} when actually run.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "valid", "test"):
                make_deepmd_system(root, split)
            deepmd = {"enabled": True, "profile": "deepmd_gpu", "committee": 1, "seeds": [11]}
            campaign = load_campaign(
                write_campaign(root, deepmd=deepmd, scheduler="local")
            )
            with self.assertRaises(ConfigurationError):
                generate_deepmd_training(campaign)

    def test_mace_stage2_refuses_to_run_before_stage1_model_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = load_campaign(write_campaign(root, mace={"enabled": True}))
            manifest = generate_mace_training(campaign)
            stage2_script = Path(manifest["stages"][1]["launcher"]).read_text(encoding="utf-8")
            self.assertIn("--max_num_epochs=300", stage2_script)

            guard_start = stage2_script.index("if [[ ! -d")
            guard_end = stage2_script.index("fi\n", guard_start) + len("fi\n")
            guard = stage2_script[guard_start:guard_end]
            harness_path = root / "guard.sh"
            harness_path.write_text(
                "#!/usr/bin/env bash\nset -u\n" + guard + 'echo "guard passed"\n',
                encoding="utf-8",
            )

            # artifacts/ does not exist yet (stage1 never ran): must fail.
            missing_result = subprocess.run(
                ["bash", str(harness_path)], capture_output=True, text=True
            )
            self.assertEqual(missing_result.returncode, 2, missing_result.stderr)
            self.assertIn("ERROR", missing_result.stderr)

            # Once stage1 has produced *something* in model_dir, the guard
            # must let stage2 proceed.
            (root / "models/mace/artifacts").mkdir(parents=True, exist_ok=True)
            (root / "models/mace/artifacts" / "checkpoint.pt").write_text("x", encoding="utf-8")
            present_result = subprocess.run(
                ["bash", str(harness_path)], capture_output=True, text=True
            )
            self.assertEqual(present_result.returncode, 0, present_result.stderr)
            self.assertIn("guard passed", present_result.stdout)

    def test_mace_stage2_epoch_limit_adds_refinement_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = load_campaign(
                write_campaign(
                    root,
                    mace={
                        "enabled": True,
                        "max_num_epochs": 200,
                        "stage2_max_num_epochs": 100,
                    },
                )
            )
            manifest = generate_mace_training(campaign)

            self.assertIn("--max_num_epochs=200", manifest["stages"][0]["command"])
            self.assertIn("--max_num_epochs=300", manifest["stages"][1]["command"])

    def test_validate_deepmd_dataset_accepts_well_formed_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "valid", "test"):
                make_deepmd_system(root, split)
            type_map, inventory = validate_deepmd_dataset(root / "datasets/canonical/deepmd")
            self.assertEqual(type_map, ["A", "B"])
            self.assertEqual(set(inventory), {"train", "valid", "test"})

    def test_validate_deepmd_dataset_rejects_nan_forces(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "valid", "test"):
                make_deepmd_system(root, split)
            force_path = root / "datasets/canonical/deepmd/train/system/set.000/force.npy"
            bad = np.load(force_path)
            bad[0, 0] = np.nan
            np.save(force_path, bad)
            with self.assertRaisesRegex(SafetyError, "Non-finite"):
                validate_deepmd_dataset(root / "datasets/canonical/deepmd")

    def test_validate_deepmd_dataset_rejects_degenerate_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "valid", "test"):
                make_deepmd_system(root, split)
            box_path = root / "datasets/canonical/deepmd/train/system/set.000/box.npy"
            np.save(box_path, np.zeros((1, 9)))
            with self.assertRaisesRegex(SafetyError, "Degenerate"):
                validate_deepmd_dataset(root / "datasets/canonical/deepmd")

    def test_validate_deepmd_dataset_rejects_wrong_force_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "valid", "test"):
                make_deepmd_system(root, split)
            force_path = root / "datasets/canonical/deepmd/train/system/set.000/force.npy"
            np.save(force_path, np.zeros((1, 9)))  # 3 atoms worth, but type.raw says 2
            with self.assertRaisesRegex(SafetyError, "columns"):
                validate_deepmd_dataset(root / "datasets/canonical/deepmd")

    def test_validate_deepmd_dataset_rejects_mismatched_frame_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "valid", "test"):
                make_deepmd_system(root, split)
            energy_path = root / "datasets/canonical/deepmd/train/system/set.000/energy.npy"
            np.save(energy_path, np.zeros((2, 1)))  # coord/box/force all have 1 frame
            with self.assertRaisesRegex(SafetyError, "Inconsistent frame counts"):
                validate_deepmd_dataset(root / "datasets/canonical/deepmd")

    def test_validate_deepmd_dataset_rejects_missing_set_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "valid", "test"):
                make_deepmd_system(root, split)
            system = root / "datasets/canonical/deepmd/train/system"
            for path in (system / "set.000").iterdir():
                path.unlink()
            (system / "set.000").rmdir()
            with self.assertRaisesRegex(SafetyError, r"No set\.\* data directories"):
                validate_deepmd_dataset(root / "datasets/canonical/deepmd")

    def test_validate_deepmd_dataset_rejects_wrong_energy_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "valid", "test"):
                make_deepmd_system(root, split)
            energy_path = root / "datasets/canonical/deepmd/train/system/set.000/energy.npy"
            np.save(energy_path, np.zeros((1, 2)))
            with self.assertRaisesRegex(SafetyError, "energy.npy.*expected 1"):
                validate_deepmd_dataset(root / "datasets/canonical/deepmd")

    def test_validate_deepmd_dataset_rejects_out_of_range_type_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "valid", "test"):
                make_deepmd_system(root, split)
            type_path = root / "datasets/canonical/deepmd/train/system/type.raw"
            type_path.write_text("0\n2\n", encoding="utf-8")
            with self.assertRaisesRegex(SafetyError, "outside type_map.raw range"):
                validate_deepmd_dataset(root / "datasets/canonical/deepmd")

    def test_pack_parser_uses_unambiguous_output(self) -> None:
        args = build_parser().parse_args(
            ["vasp", "pack", "archive.zip", "--root", "runs"]
        )
        self.assertEqual(args.output, "archive.zip")
        self.assertEqual(args.root, "runs")

    def test_archive_models_parser_uses_unambiguous_output(self) -> None:
        args = build_parser().parse_args(
            ["vasp", "archive-models", "models.zip", "--root", "successful"]
        )
        self.assertEqual(args.output, "models.zip")
        self.assertEqual(args.root, "successful")

    def test_archive_models_parser_defaults_to_current_folder_and_generated_name(self) -> None:
        args = build_parser().parse_args(["vasp", "archive-models"])
        self.assertIsNone(args.output)
        self.assertEqual(args.root, ".")

    def test_archive_models_parser_accepts_folder_exclusion_list(self) -> None:
        args = build_parser().parse_args(
            [
                "vasp",
                "archive-models",
                "--exclude-folders",
                "omit_300",
                "omit_450",
            ]
        )
        self.assertEqual(args.exclude_folders, ["omit_300", "omit_450"])
        self.assertFalse(args.recursive)

    def test_init_writes_campaign_and_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "new_campaign"
            self.assertEqual(main(["init", str(target)]), 0)
            self.assertTrue((target / "campaign.yaml").is_file())
            self.assertTrue((target / "profiles/loni.yaml").is_file())
            campaign = yaml.safe_load((target / "campaign.yaml").read_text(encoding="utf-8"))
            self.assertEqual(
                campaign["stages"]["vasp_mlff"]["accuracy_profile"], "accurate"
            )
            self.assertFalse(campaign["stages"]["vasp_mlff"]["enabled"])

            prepare_campaign(load_campaign(target / "campaign.yaml"))
            self.assertFalse((target / "runs/vasp/bulk_a_300k/train").exists())

            campaign["stages"]["vasp_mlff"]["enabled"] = True
            (target / "campaign.yaml").write_text(
                yaml.safe_dump(campaign, sort_keys=False), encoding="utf-8"
            )
            prepare_campaign(load_campaign(target / "campaign.yaml"))
            train = parse_incar(target / "runs/vasp/bulk_a_300k/train/INCAR")
            refit = parse_incar(target / "runs/vasp/bulk_a_300k/refit/INCAR")
            self.assertEqual(train["ML_IALGO_LINREG"], "1")
            self.assertEqual(train["ML_SION1"], "0.3")
            self.assertEqual(train["ML_MRB2"], "12")
            self.assertEqual(refit["ML_IALGO_LINREG"], "4")
            self.assertEqual(refit["ML_SION1"], "0.5")
            self.assertEqual(refit["ML_EPS_LOW"], "1E-11")


if __name__ == "__main__":
    unittest.main()
