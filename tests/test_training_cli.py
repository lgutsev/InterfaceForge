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
from interfaceforge.errors import ConfigurationError, SafetyError
from interfaceforge.training import (
    _deepmd_shell_prefix,
    generate_deepmd_training,
    generate_mace_training,
    validate_deepmd_dataset,
)


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

    def test_dpa4_freeze_failure_exits_nonzero_not_success(self) -> None:
        # A DPA-4 freeze failure must not report job success: downstream
        # monitoring/automation reads the exit code to decide whether a
        # deployable model exists.
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
            generate_deepmd_training(campaign)
            script = (root / "models/deepmd/run_ensemble.slurm").read_text(encoding="utf-8")
            self.assertNotIn("]] && exit 0", script)

            guard_lines = [
                line
                for line in script.splitlines()
                if line.strip().startswith('[[ "$ARCH" == "dpa4"')
            ]
            self.assertEqual(len(guard_lines), 1)
            harness = (
                "#!/usr/bin/env bash\n"
                "set -u\n"
                "ARCH=dpa4\n"
                f"{guard_lines[0].strip()}\n"
                'echo "guard did not stop execution"\n'
            )
            harness_path = root / "guard.sh"
            harness_path.write_text(harness, encoding="utf-8")
            result = subprocess.run(
                ["bash", str(harness_path)], capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
            self.assertIn("ERROR", result.stdout)

    def test_deepmd_shell_prefix_uses_srun_only_under_slurm(self) -> None:
        direct_settings: dict = {}
        slurm_prefix = _deepmd_shell_prefix(direct_settings, "pytorch", scheduler="slurm")
        local_prefix = _deepmd_shell_prefix(direct_settings, "pytorch", scheduler="local")
        self.assertIn("srun -n 1 dp", slurm_prefix)
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
