# ruff: noqa: E501
from __future__ import annotations

import csv
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

from interfaceforge.errors import SafetyError
from interfaceforge.mlip_compare import (
    DEFAULT_SEEDS,
    MACE_EVALUATOR,
    _groups,
    _metrics,
    comparison_status,
    finalize_comparison,
    prepare_comparison,
    validate_membership,
)


def _write_system(
    deepmd_root: Path,
    leaf: str,
    frames: list[Atoms],
    source_frames: list[int],
) -> None:
    system = deepmd_root / leaf
    set_dir = system / "set.000"
    set_dir.mkdir(parents=True)
    (system / "type_map.raw").write_text("Si\nN\n", encoding="utf-8")
    (system / "type.raw").write_text("0\n1\n", encoding="utf-8")
    np.save(set_dir / "coord.npy", np.asarray([frame.positions.reshape(-1) for frame in frames]))
    np.save(set_dir / "box.npy", np.asarray([frame.cell.array.reshape(-1) for frame in frames]))
    np.save(set_dir / "energy.npy", np.asarray([[frame.info["REF_energy"]] for frame in frames]))
    np.save(set_dir / "force.npy", np.asarray([frame.arrays["REF_forces"].reshape(-1) for frame in frames]))
    with (system / "frame_map.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["local_frame", "source_frame", "source_path", "relative_leaf"])
        for local, source_frame in enumerate(source_frames):
            writer.writerow([local, source_frame, "OUTCAR", leaf])


def _campaign(tmp_path: Path) -> Path:
    campaign = tmp_path / "campaign"
    canonical = campaign / "datasets" / "canonical"
    deepmd_test = canonical / "deepmd" / "test"
    canonical.mkdir(parents=True)
    all_frames: list[Atoms] = []
    specifications = (
        ("bulk/SiN-Bulk_300K", 0.0),
        ("interface/450K/Real/N_Term/O_x_0.25", 0.4),
    )
    for leaf, base in specifications:
        frames = []
        source_frames = [2, 7]
        for source_frame, shift in zip(source_frames, (0.0, 0.1), strict=True):
            atoms = Atoms(
                "SiN",
                positions=[
                    [base + shift, 0.0, 0.0],
                    [1.5 + base + shift, 1.5, 1.5],
                ],
                cell=np.eye(3) * (5.0 + base),
                pbc=True,
            )
            atoms.info.update(
                {
                    "REF_energy": -10.0 + base + shift,
                    "IF_leaf": leaf,
                    "source_frame": source_frame,
                }
            )
            atoms.arrays["REF_forces"] = np.full((2, 3), base + shift)
            frames.append(atoms)
            all_frames.append(atoms)
        _write_system(deepmd_test, leaf, frames, source_frames)
    write(canonical / "test.extxyz", all_frames, format="extxyz")

    model_root = campaign / "models" / "mace_committee_520eV" / "mace_committee"
    for seed in DEFAULT_SEEDS:
        directory = model_root / f"seed_{seed}" / "mace_model"
        directory.mkdir(parents=True)
        (directory / f"synthetic_seed{seed}_stagetwo.model").write_text(
            "synthetic model placeholder\n", encoding="utf-8"
        )
    return campaign


def _materialize_predictions(campaign: Path, manifest: dict[str, object]) -> Path:
    output = campaign / "audit" / "mlip_compare"
    dpa_root = campaign / "models" / "deepmd" / "evaluation" / "dpa2" / "job_1002"
    models = manifest["models"]
    systems = manifest["systems"]
    assert isinstance(models, list)
    assert isinstance(systems, list)
    for model_index, model in enumerate(models):
        label = model["model"]
        energy_delta = 0.001 * (model_index + 1)
        force_delta = 0.01 * (model_index + 1)
        for system in systems:
            frames = read(system["mace_input"], index=":")
            natoms = int(system["natoms"])
            ref_total_e = np.asarray([frame.info["REF_energy"] for frame in frames], dtype=float)
            ref_e = ref_total_e / natoms
            ref_f = np.asarray([frame.arrays["REF_forces"] for frame in frames], dtype=float)
            mace_dir = output / "predictions" / "mace" / label
            mace_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                mace_dir / f'{system["system_id"]}.npz',
                energy=ref_total_e + energy_delta * natoms,
                forces=ref_f + force_delta,
            )
            prefix = dpa_root / "by_system" / system["system_id"] / f"{label}_detail"
            prefix.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(
                Path(str(prefix) + ".e_peratom.out"),
                np.column_stack((ref_e, ref_e + energy_delta)),
                fmt="%.18e",
            )
            force_rows = ref_f.reshape(-1, 3)
            np.savetxt(
                Path(str(prefix) + ".f.out"),
                np.column_stack((force_rows, force_rows + force_delta)),
                fmt="%.18e",
            )
    return dpa_root


class TestMLIPComparison(unittest.TestCase):
    def test_membership_proves_geometry_labels_and_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = _campaign(Path(temporary))
            canonical = campaign / "datasets" / "canonical"
            systems, summary = validate_membership(
                canonical / "test.extxyz",
                canonical / "deepmd" / "test",
                campaign / "grouped",
            )
            self.assertTrue(summary["exact_membership"])
            self.assertEqual(summary["frames"], 4)
            self.assertEqual(summary["systems"], 2)
            self.assertEqual({row["relative_leaf"] for row in systems}, {
                "bulk/SiN-Bulk_300K",
                "interface/450K/Real/N_Term/O_x_0.25",
            })

    def test_membership_rejects_silent_coordinate_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = _campaign(Path(temporary))
            canonical = campaign / "datasets" / "canonical"
            coord_path = (
                canonical
                / "deepmd"
                / "test"
                / "bulk"
                / "SiN-Bulk_300K"
                / "set.000"
                / "coord.npy"
            )
            coord = np.load(coord_path)
            coord[0, 0] += 1.0e-3
            np.save(coord_path, coord)
            with self.assertRaisesRegex(SafetyError, "Canonical data mismatch"):
                validate_membership(
                    canonical / "test.extxyz",
                    canonical / "deepmd" / "test",
                    campaign / "grouped",
                )

    def test_centered_energy_and_force_metric_definitions(self) -> None:
        metrics = _metrics(
            np.asarray([0.0, 1.0]),
            np.asarray([0.5, 1.5]),
            np.zeros((2, 1, 3)),
            np.ones((2, 1, 3)) * 0.1,
        )
        self.assertAlmostEqual(metrics["energy_rmse_mev_per_atom"], 500.0)
        self.assertAlmostEqual(metrics["energy_centered_rmse_mev_per_atom"], 0.0)
        self.assertAlmostEqual(metrics["force_rmse_mev_per_angstrom"], 100.0)
        self.assertAlmostEqual(
            metrics["force_vector_rmse_mev_per_angstrom"], 100.0 * 3**0.5
        )

    def test_generated_mace_evaluator_is_valid_python(self) -> None:
        compile(MACE_EVALUATOR, "evaluate_mace.py", "exec")
        self.assertIn('default_dtype="float32"', MACE_EVALUATOR)

    def test_oxidation_grouping_is_canonical(self) -> None:
        cases = {
            "bulk/SiN-Bulk_300K": "NA",
            "bulk/TiN-Bulk_450K/O_x1.00": "NA",
            "interface/450K/Real/N_Term": "0",
            "interface/300K/Ideal/Ti_Term/O_x0.25": "0.25",
            "interface/600K/Real/N_Term/O_x_0.5": "0.5",
            "interface/450K/Real/Ti_Term/O_x0.75": "0.75",
            "interface/450K/Real/N_Term/O_x1.0": "1",
            "interface/600K/Ideal/Ti_Term/O_x1.00": "1",
        }
        for leaf, expected in cases.items():
            self.assertEqual(_groups(leaf)["oxidation"], expected, leaf)
        self.assertEqual(_groups("interface/300K/Ideal/Ti_Term/O_x1.0")["temperature"], "300K")
        self.assertEqual(_groups("interface/450K/Real/N_Term")["family"], "Real")
        self.assertEqual(_groups("interface/450K/Real/N_Term")["termination"], "N_Term")

    def test_prepare_status_finalize_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = _campaign(Path(temporary))
            manifest = prepare_comparison(campaign)
            self.assertEqual(manifest["mace_inference_dtype"], "float32")
            self.assertTrue(manifest["validation"]["exact_membership"])
            dpa_root = _materialize_predictions(campaign, manifest)
            status = comparison_status(campaign, deepmd_eval_root=dpa_root)
            self.assertEqual(status["status"], "READY_TO_FINALIZE")
            self.assertEqual(set(status["mace"].values()), {2})
            self.assertEqual(set(status["deepmd"].values()), {2})
            report = finalize_comparison(campaign, deepmd_eval_root=dpa_root)
            self.assertEqual(report["status"], "OK")
            self.assertEqual(report["mace_inference_dtype"], "float32")
            output = campaign / "audit" / "mlip_compare"
            for name in (
                "metrics_by_system.csv",
                "metrics_overall.csv",
                "metrics_by_group.csv",
                "uncertainty_calibration.csv",
                "comparison.json",
                "comparison.md",
                "comparison.svg",
                "force_rmse_heatmap_mace.png",
                "force_rmse_heatmap_mace.svg",
                "force_rmse_heatmap_dpa2.png",
                "force_rmse_heatmap_dpa2.svg",
                "force_rmse_heatmaps.png",
                "force_rmse_heatmaps.svg",
            ):
                self.assertTrue((output / name).is_file(), name)
                self.assertGreater((output / name).stat().st_size, 0, name)
            for name in (
                "force_rmse_heatmap_mace.svg",
                "force_rmse_heatmap_dpa2.svg",
                "force_rmse_heatmaps.svg",
            ):
                ET.parse(output / name)
            self.assertEqual(
                set(report["outputs"]) & {
                    "force_heatmap_mace_png",
                    "force_heatmap_dpa2_png",
                    "force_heatmaps_png",
                },
                {
                    "force_heatmap_mace_png",
                    "force_heatmap_dpa2_png",
                    "force_heatmaps_png",
                },
            )
            with (output / "metrics_overall.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len([row for row in rows if row["engine"] == "MACE"]), 10)
            self.assertEqual(len([row for row in rows if row["engine"] == "DPA2"]), 10)

    def test_prepare_refuses_unforced_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = _campaign(Path(temporary))
            prepare_comparison(campaign)
            with self.assertRaisesRegex(SafetyError, "not empty"):
                prepare_comparison(campaign)

    def test_prepare_requires_exact_model_arity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = _campaign(Path(temporary))
            missing = (
                campaign
                / "models"
                / "mace_committee_520eV"
                / "mace_committee"
                / "seed_53"
                / "mace_model"
                / "synthetic_seed53_stagetwo.model"
            )
            missing.unlink()
            with self.assertRaisesRegex(SafetyError, "Expected one stage-two model"):
                prepare_comparison(campaign)

    def test_finalize_refuses_incomplete_committees(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = _campaign(Path(temporary))
            prepare_comparison(campaign)
            dpa_root = campaign / "models" / "deepmd" / "evaluation" / "dpa2" / "job_1002"
            with self.assertRaisesRegex(SafetyError, "incomplete"):
                finalize_comparison(campaign, deepmd_eval_root=dpa_root)

    def test_finalize_rejects_corrupted_deepmd_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = _campaign(Path(temporary))
            manifest = prepare_comparison(campaign)
            dpa_root = _materialize_predictions(campaign, manifest)
            detail = (
                dpa_root
                / "by_system"
                / "system_000"
                / "model_000_detail.e_peratom.out"
            )
            values = np.loadtxt(detail)
            values[0, 0] += 1.0e-3
            np.savetxt(detail, values, fmt="%.18e")
            with self.assertRaisesRegex(
                SafetyError, "DeePMD detail references differ"
            ):
                finalize_comparison(campaign, deepmd_eval_root=dpa_root)


if __name__ == "__main__":
    unittest.main()
