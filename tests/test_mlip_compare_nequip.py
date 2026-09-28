"""Matched-frame comparison with NequIP alongside MACE and DeePMD/DPA."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase.io import read
from nio_fixture import build_nio_tree
from test_mlip_compare import _campaign, _materialize_predictions

from interfaceforge.errors import SafetyError
from interfaceforge.mlip_compare import (
    MACE_EVALUATOR,
    NEQUIP_EVALUATOR,
    comparison_status,
    finalize_comparison,
    parse_combine_entry,
    prepare_comparison,
)
from interfaceforge.nio_dataset import ExportConfig, export_dataset

REPO = Path(__file__).resolve().parents[1]


def _nequip_root(campaign: Path, seeds=(11, 23, 37, 53)) -> Path:
    root = campaign / "models" / "nequip"
    for seed in seeds:
        final = root / f"seed_{seed}" / "final"
        final.mkdir(parents=True)
        (final / "model.nequip.pt2").write_text(f"compiled {seed}\n")
        (final / "model.nequip.zip").write_text(f"package {seed}\n")
        (root / f"seed_{seed}" / "config.yaml").write_text("run: [train, test]\n")
    (root / "training_manifest.json").write_text(
        json.dumps({"seeds": list(seeds), "compile": {"compiled_name": "model.nequip.pt2"}})
    )
    return root


def _profile(campaign: Path) -> Path:
    target = campaign / "profiles" / "loni.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "profiles" / "loni.yaml", target)
    return target


def _write_nequip_predictions(manifest: dict, delta: float = 0.002) -> None:
    output = Path(manifest["output_root"])
    for index, model in enumerate(manifest["engines"]["NEQUIP"]):
        for system in manifest["systems"]:
            frames = read(system["mace_input"], index=":")
            natoms = int(system["natoms"])
            energy = np.asarray([frame.info["REF_energy"] for frame in frames]) + delta * (index + 1) * natoms
            forces = np.asarray([frame.arrays["REF_forces"] for frame in frames]) + 5 * delta * (index + 1)
            target = output / "predictions" / "nequip" / model["model"] / f"{system['system_id']}.npz"
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, energy=energy, forces=forces)


class ThreeBackendComparisonTests(unittest.TestCase):
    def test_mace_dpa_nequip_on_identical_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = _campaign(Path(temporary))
            nequip_root = _nequip_root(campaign)
            profile = _profile(campaign)
            manifest = prepare_comparison(campaign, nequip_models_root=nequip_root, profile_path=profile)
            self.assertEqual(manifest["backends"], ["mace", "deepmd", "nequip"])
            self.assertEqual(set(manifest["engines"]), {"MACE", "DPA2", "NEQUIP"})
            self.assertEqual(manifest["engine_display"]["NEQUIP"], "NequIP")
            self.assertIn("not compared", manifest["stress_comparison"])
            launcher = Path(manifest["launchers"]["NEQUIP"]).read_text()
            self.assertIn("#SBATCH --partition=gpu2", launcher)
            self.assertIn("NEQUIP_ACTIVATE_SCRIPT", launcher)
            self.assertNotIn("loni_perovsk27", launcher)
            self.assertNotIn("/home/lgutsev", launcher)
            subprocess.run(["bash", "-n", manifest["launchers"]["NEQUIP"]], check=True)

            dpa_root = _materialize_predictions(campaign, manifest)
            status = comparison_status(campaign, deepmd_eval_root=dpa_root)
            self.assertEqual(status["status"], "INCOMPLETE")
            self.assertTrue(any("NequIP inference incomplete" in hint for hint in status["hints"]))
            with self.assertRaisesRegex(SafetyError, "NequIP"):
                finalize_comparison(campaign, deepmd_eval_root=dpa_root)

            _write_nequip_predictions(manifest)
            status = comparison_status(campaign, deepmd_eval_root=dpa_root)
            self.assertEqual(status["status"], "READY_TO_FINALIZE")
            self.assertEqual(set(status["nequip"].values()), {2})
            report = finalize_comparison(campaign, deepmd_eval_root=dpa_root)
            self.assertEqual(report["engines"], ["MACE", "DPA2", "NEQUIP"])
            output = campaign / "audit" / "mlip_compare"

            with (output / "metrics_overall.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len([row for row in rows if row["engine"] == "NEQUIP"]), 10)
            ensemble = next(
                row
                for row in rows
                if row["engine"] == "NEQUIP" and row["model"] == "ensemble_mean" and row["averaging"] == "micro"
            )
            self.assertAlmostEqual(float(ensemble["energy_rmse_mev_per_atom"]), 5.0)
            self.assertAlmostEqual(float(ensemble["force_rmse_mev_per_angstrom"]), 25.0)

            with (output / "matched_frames.csv").open() as handle:
                frames = list(csv.DictReader(handle))
            self.assertEqual(len(frames), 4)
            for column in (
                "frame_id",
                "dft_energy_per_atom_ev",
                "mace_energy_per_atom_ev",
                "deepmd_energy_per_atom_ev",
                "nequip_energy_per_atom_ev",
                "nequip_force_rmse_mev_per_angstrom",
                "nequip_energy_spread_mev_per_atom",
                "nequip_force_disagreement_mev_per_angstrom",
            ):
                self.assertIn(column, frames[0])
            self.assertEqual(len({row["frame_id"] for row in frames}), 4)
            with (output / "matched_frames_members.csv").open() as handle:
                members = list(csv.DictReader(handle))
            self.assertEqual(len(members), 3 * 4 * 4)
            self.assertEqual({row["engine"] for row in members}, {"MACE", "DPA2", "NEQUIP"})
            for key in ("force_heatmap_nequip_png", "force_heatmaps_png", "matched_frames", "publication_by_group"):
                self.assertTrue(Path(report["outputs"][key]).is_file(), key)
            markdown = (output / "comparison.md").read_text()
            self.assertIn("MACE versus DPA-2 versus NequIP audit", markdown)
            self.assertIn("| NequIP |", markdown)
            self.assertIn("Stress: not compared", markdown)
            self.assertIn("meV/atom", report["energy_normalization"])

    def test_backend_subset_does_not_need_deepmd_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = _campaign(Path(temporary))
            nequip_root = _nequip_root(campaign, seeds=(11, 23))
            manifest = prepare_comparison(
                campaign,
                backends=("mace", "nequip"),
                nequip_models_root=nequip_root,
                profile_path=_profile(campaign),
            )
            self.assertEqual(set(manifest["engines"]), {"MACE", "NEQUIP"})
            output = Path(manifest["output_root"])
            for model in manifest["engines"]["MACE"]:
                for system in manifest["systems"]:
                    frames = read(system["mace_input"], index=":")
                    target = output / "predictions" / "mace" / model["model"] / f"{system['system_id']}.npz"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        target,
                        energy=np.asarray([frame.info["REF_energy"] for frame in frames]),
                        forces=np.asarray([frame.arrays["REF_forces"] for frame in frames]),
                    )
            _write_nequip_predictions(manifest)
            report = finalize_comparison(campaign)
            self.assertEqual(report["engines"], ["MACE", "NEQUIP"])
            self.assertEqual(report["deepmd_reference_max_absolute_delta"], {"energy": 0.0, "force": 0.0})

    def test_missing_compiled_member_and_bad_backend_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = _campaign(Path(temporary))
            root = _nequip_root(campaign, seeds=(11, 23))
            (root / "seed_23" / "final" / "model.nequip.pt2").unlink()
            with self.assertRaisesRegex(SafetyError, r"seed\(s\) \[23\]"):
                prepare_comparison(campaign, nequip_models_root=root, profile_path=_profile(campaign), force=True)
            with self.assertRaisesRegex(SafetyError, "backends"):
                prepare_comparison(campaign, backends=("allegro",), force=True)

    def test_evaluators_never_apply_constraints_to_predictions(self) -> None:
        self.assertIn("get_forces(apply_constraint=False)", MACE_EVALUATOR)
        self.assertIn("get_forces(apply_constraint=False)", NEQUIP_EVALUATOR)
        compile(NEQUIP_EVALUATOR, "evaluate_nequip.py", "exec")
        self.assertEqual(parse_combine_entry("nequip_v1=out"), ("nequip_v1", "NEQUIP", "out"))
        self.assertEqual(parse_combine_entry("x:NEQUIP=out")[1], "NEQUIP")


class NiOCanonicalComparisonTests(unittest.TestCase):
    def test_nio_dataset_feeds_matched_comparison_through_generated_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            head = build_nio_tree(root / "NiO_head", temperatures=(300, 450))
            campaign = root / "campaign"
            export_dataset([head], campaign / "datasets" / "canonical", ExportConfig())
            nequip_root = _nequip_root(campaign, seeds=(11, 23))
            manifest = prepare_comparison(
                campaign, backends=("nequip",), nequip_models_root=nequip_root, profile_path=_profile(campaign)
            )
            self.assertTrue(manifest["validation"]["exact_membership"])
            first_system = manifest["systems"][0]
            self.assertIn("stage", first_system["metadata"])
            self.assertEqual(len(first_system["frame_ids"]), first_system["frames"])

            stub = root / "stubpy" / "nequip" / "integrations"
            stub.mkdir(parents=True)
            (root / "stubpy" / "nequip" / "__init__.py").write_text("")
            (stub / "__init__.py").write_text("")
            (stub / "ase.py").write_text(
                "import numpy as np\n"
                "from ase.calculators.calculator import Calculator, all_changes\n"
                "class NequIPCalculator(Calculator):\n"
                "    implemented_properties = ['energy', 'forces']\n"
                "    @classmethod\n"
                "    def from_compiled_model(cls, path, device='cpu', **kw):\n"
                "        return cls()\n"
                "    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):\n"
                "        super().calculate(atoms, properties, system_changes)\n"
                "        self.results = {'energy': float(atoms.info['REF_energy']) + 0.01 * len(atoms),\n"
                "                        'forces': np.asarray(atoms.arrays['REF_forces']) + 0.05}\n"
            )
            env = {**os.environ, "PYTHONPATH": f"{root / 'stubpy'}:{os.environ.get('PYTHONPATH', '')}"}
            output = Path(manifest["output_root"])
            for task in range(2):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(output / "evaluate_nequip.py"),
                        "--root",
                        str(output),
                        "--task",
                        str(task),
                        "--device",
                        "cpu",
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
            report = finalize_comparison(campaign)
            self.assertEqual(report["engines"], ["NEQUIP"])
            # Frozen atoms keep raw DFT forces, and the stub adds 0.05 eV/A to every component:
            ensemble = report["headline"][0]
            self.assertAlmostEqual(ensemble["force_rmse_mev_per_angstrom"], 50.0, places=6)
            self.assertAlmostEqual(ensemble["energy_rmse_mev_per_atom"], 10.0, places=6)
            self.assertIn("publication", report["views_skipped"])
            self.assertIn("oxidation", report["views_skipped"])
            self.assertTrue(Path(report["outputs"]["temperature_by_group"]).is_file())
            self.assertTrue(Path(report["outputs"]["ligand_by_group"]).is_file())
            with Path(report["outputs"]["temperature_by_group"]).open() as handle:
                groups = {row["temperature_group"] for row in csv.DictReader(handle)}
            self.assertLessEqual(groups, {"Overall", "300 K", "450 K"})


if __name__ == "__main__":
    unittest.main()
