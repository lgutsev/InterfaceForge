from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write
from test_config_scheduler import write_campaign

from interfaceforge.cli import build_parser
from interfaceforge.config import load_campaign
from interfaceforge.errors import ConfigurationError, SafetyError
from interfaceforge.mace_roi import (
    CYCLE_COEFFICIENT_KEY,
    CYCLE_ID_KEY,
    CYCLE_SCALE_KEY,
    CYCLE_SIZE_KEY,
    ROI_MASK_KEY,
    ROI_WEIGHT_KEY,
    CycleBatchSampler,
    compute_roi_weights,
    cycle_mse_numpy,
    evaluate_mace_roi_predictions,
    prepare_mace_roi_dataset,
)
from interfaceforge.state import sha256_file
from interfaceforge.training import generate_mace_training


def interface_frame(source_run: str, energy: float = 0.0, natoms: int = 4) -> Atoms:
    atoms = Atoms(
        f"H{natoms}",
        positions=[[0.0, 0.0, float(index)] for index in range(natoms)],
        cell=[20.0, 20.0, 20.0],
        pbc=False,
    )
    atoms.info.update(
        {
            "REF_energy": energy,
            "source_run": source_run,
            "source_path": f"runs/{source_run}/OUTCAR",
            "source_frame": 0,
        }
    )
    atoms.arrays["REF_forces"] = np.zeros((natoms, 3))
    atoms.arrays["move_mask"] = np.ones(natoms, dtype=np.int8)
    return atoms


def write_cycle_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "source_run",
                "source_frame",
                "cycle_id",
                "coefficient",
                "scale_ev",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


class MaceRoiTests(unittest.TestCase):
    def test_interface_atoms_are_upweighted_and_mean_is_preserved(self) -> None:
        atoms = interface_frame("interface_a")
        weights, mask = compute_roi_weights(
            atoms,
            np.asarray([0, 0, 1, 1]),
            cutoff=1.1,
            interface_multiplier=4.0,
        )
        np.testing.assert_array_equal(mask, [0, 1, 1, 0])
        np.testing.assert_allclose(weights, [0.4, 1.6, 1.6, 0.4])
        self.assertAlmostEqual(float(weights.mean()), 1.0)

    def test_shell_expansion_adds_neighboring_layers(self) -> None:
        atoms = interface_frame("interface_a")
        weights, mask = compute_roi_weights(
            atoms,
            np.asarray([0, 0, 1, 1]),
            cutoff=1.1,
            interface_multiplier=4.0,
            shell_depth=1,
        )
        np.testing.assert_array_equal(mask, np.ones(4, dtype=np.int8))
        np.testing.assert_allclose(weights, np.ones(4))

    def test_cycle_batch_sampler_never_splits_a_cycle(self) -> None:
        dataset = [
            {"if_cycle_id": cycle_id}
            for cycle_id in (0, 1, 0, -1, 1, -1)
        ]
        batches = list(CycleBatchSampler(dataset, 3, shuffle=False))
        for cycle_members in ({0, 2}, {1, 4}):
            self.assertTrue(any(cycle_members.issubset(set(batch)) for batch in batches))
        self.assertEqual(sorted(index for batch in batches for index in batch), list(range(6)))

    def test_cycle_residual_uses_energy_differences_and_scale(self) -> None:
        loss = cycle_mse_numpy(
            reference=[10.0, 4.0],
            predicted=[11.0, 6.0],
            cycle_ids=[0, 0],
            coefficients=[1.0, -1.0],
            scales=[2.0, 2.0],
        )
        self.assertAlmostEqual(loss, 0.25)

    def test_evaluation_separates_roi_and_cycle_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = [interface_frame("interface_a", 1.0), interface_frame("interface_b", 2.0)]
            for index, atoms in enumerate(frames):
                atoms.arrays[ROI_MASK_KEY] = np.asarray([0, 1, 1, 0], dtype=np.int8)
                predicted = np.ones((4, 3))
                predicted[1:3] = 2.0
                atoms.arrays["MACE_forces"] = predicted
                atoms.info["MACE_energy"] = atoms.info["REF_energy"] + (1.0 if index == 0 else 0.0)
                atoms.info[CYCLE_ID_KEY] = 0
                atoms.info[CYCLE_COEFFICIENT_KEY] = 1.0 if index == 0 else -1.0
                atoms.info[CYCLE_SCALE_KEY] = 2.0
                atoms.info[CYCLE_SIZE_KEY] = 2
            prediction_path = root / "predictions.extxyz"
            output_path = root / "mace-roi-metrics.json"
            write(prediction_path, frames, format="extxyz")

            payload = evaluate_mace_roi_predictions(prediction_path, output_path)
            self.assertTrue(output_path.is_file())
            self.assertAlmostEqual(payload["force_component_ev_a"]["roi"]["rmse"], 2.0)
            self.assertAlmostEqual(payload["force_component_ev_a"]["non_roi"]["rmse"], 1.0)
            self.assertAlmostEqual(payload["cycle_residual_ev"]["rmse"], 1.0)
            self.assertAlmostEqual(payload["cycle_residual_scaled"]["rmse"], 0.5)

    def test_prepare_writes_derived_data_without_touching_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "datasets/canonical"
            source.mkdir(parents=True)
            for split in ("train", "valid", "test"):
                frames = [
                    interface_frame(f"interface_{split}_a", 1.0),
                    interface_frame(f"interface_{split}_b", 2.0),
                    interface_frame(f"bulk_{split}", 3.0),
                ]
                write(source / f"{split}.extxyz", frames, format="extxyz")
            canonical_text = (source / "train.extxyz").read_text(encoding="utf-8")
            canonical_hash = sha256_file(source / "train.extxyz")

            cycles = root / "cycles.csv"
            write_cycle_manifest(
                cycles,
                [
                    {
                        "split": "train",
                        "source_run": "interface_train_a",
                        "source_frame": 0,
                        "cycle_id": "adhesion_0",
                        "coefficient": 1,
                        "scale_ev": 2,
                    },
                    {
                        "split": "train",
                        "source_run": "interface_train_b",
                        "source_frame": 0,
                        "cycle_id": "adhesion_0",
                        "coefficient": -1,
                        "scale_ev": 2,
                    },
                ],
            )
            mace = {
                "enabled": True,
                "max_num_epochs": 20,
                "stage2_max_num_epochs": 10,
                "roi": {
                    "enabled": True,
                    "component_ranges": {"interface_*": [[0, 2], [2, 4]]},
                    "cutoff": 1.1,
                    "interface_multiplier": 4,
                    "cycle_manifest": "cycles.csv",
                    "stage1_cycle_weight": 0.25,
                    "stage2_cycle_weight": 1.0,
                },
            }
            campaign = load_campaign(write_campaign(root, mace=mace))
            payload = prepare_mace_roi_dataset(campaign)

            self.assertEqual(sha256_file(source / "train.extxyz"), canonical_hash)
            self.assertEqual(payload["cycles"]["groups"], 1)
            derived = read(root / "datasets/mace_roi/train.extxyz", index=0)
            self.assertIn(ROI_WEIGHT_KEY, derived.arrays)
            self.assertIn(ROI_MASK_KEY, derived.arrays)
            self.assertEqual(derived.info[CYCLE_ID_KEY], 0)
            self.assertAlmostEqual(float(derived.arrays[ROI_WEIGHT_KEY].mean()), 1.0)
            bulk = read(root / "datasets/mace_roi/train.extxyz", index=2)
            np.testing.assert_array_equal(bulk.arrays[ROI_MASK_KEY], np.zeros(4))
            np.testing.assert_allclose(bulk.arrays[ROI_WEIGHT_KEY], np.ones(4))

            training = generate_mace_training(campaign)
            stage1 = training["stages"][0]["command"]
            stage2 = training["stages"][1]["command"]
            self.assertIn("iface-mace-roi", stage1)
            self.assertIn("--if-cycle-weight=0.25", stage1)
            self.assertIn("--if-cycle-weight=1.0", stage2)
            self.assertIn("--max_num_epochs=30", stage2)
            self.assertEqual(training["method"], "mace-roi")

            canonical_path = source / "train.extxyz"
            canonical_path.write_text(canonical_text + "# modified\n", encoding="utf-8")
            with self.assertRaisesRegex(SafetyError, "Canonical source changed"):
                generate_mace_training(campaign, force=True)
            canonical_path.write_text(canonical_text, encoding="utf-8")

            derived_path = root / "datasets/mace_roi/train.extxyz"
            derived_path.write_text(
                derived_path.read_text(encoding="utf-8") + "# modified\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SafetyError, "does not match"):
                generate_mace_training(campaign, force=True)

    def test_output_cannot_contain_the_canonical_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "datasets/canonical"
            source.mkdir(parents=True)
            marker = source / "keep-me"
            marker.write_text("canonical", encoding="utf-8")
            campaign = load_campaign(
                write_campaign(
                    root,
                    mace={
                        "enabled": True,
                        "roi": {
                            "enabled": True,
                            "component_ranges": {"interface_*": [[0, 2], [2, 4]]},
                        },
                    },
                )
            )
            with self.assertRaisesRegex(SafetyError, "must be separate"):
                prepare_mace_roi_dataset(
                    campaign,
                    output_root=root / "datasets",
                    force=True,
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "canonical")

    def test_cycle_id_cannot_cross_data_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "datasets/canonical"
            source.mkdir(parents=True)
            for split in ("train", "valid", "test"):
                write(
                    source / f"{split}.extxyz",
                    interface_frame(f"interface_{split}"),
                    format="extxyz",
                )
            cycles = root / "cycles.csv"
            write_cycle_manifest(
                cycles,
                [
                    {
                        "split": "train",
                        "source_run": "interface_train",
                        "source_frame": 0,
                        "cycle_id": "leaking_cycle",
                        "coefficient": 1,
                        "scale_ev": 1,
                    },
                    {
                        "split": "valid",
                        "source_run": "interface_valid",
                        "source_frame": 0,
                        "cycle_id": "leaking_cycle",
                        "coefficient": -1,
                        "scale_ev": 1,
                    },
                ],
            )
            campaign = load_campaign(
                write_campaign(
                    root,
                    mace={
                        "enabled": True,
                        "roi": {
                            "enabled": True,
                            "component_ranges": {"interface_*": [[0, 2], [2, 4]]},
                            "cycle_manifest": "cycles.csv",
                        },
                    },
                )
            )
            with self.assertRaisesRegex(SafetyError, "may not cross"):
                prepare_mace_roi_dataset(campaign)

    def test_invalid_roi_configuration_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ConfigurationError):
                load_campaign(
                    write_campaign(
                        Path(temporary),
                        mace={
                            "enabled": True,
                            "roi": {"enabled": True, "interface_multiplier": 0.5},
                        },
                    )
                )

    def test_cli_exposes_prepare_command(self) -> None:
        args = build_parser().parse_args(["mace-roi", "prepare", "--cycles", "cycles.csv"])
        self.assertEqual(args.mace_roi_command, "prepare")
        self.assertEqual(args.cycles, "cycles.csv")
        evaluate = build_parser().parse_args(
            ["mace-roi", "evaluate", "predictions.extxyz", "metrics.json"]
        )
        self.assertEqual(evaluate.mace_roi_command, "evaluate")
        self.assertEqual(evaluate.predicted_energy_key, "MACE_energy")


if __name__ == "__main__":
    unittest.main()
