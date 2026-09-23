"""Backend-agnostic committee evaluation: alignment by frame identity and metric definitions."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import write

from interfaceforge.committee_eval import evaluate_committee, write_predictions
from interfaceforge.errors import SafetyError


def _reference(path: Path) -> list[Atoms]:
    frames = []
    for index, (leaf, natoms) in enumerate((("sysA", 2), ("sysA", 2), ("sysB", 3))):
        atoms = Atoms(
            "Ni" * natoms, positions=np.arange(natoms * 3).reshape(natoms, 3) * 0.7, cell=np.eye(3) * 6, pbc=True
        )
        atoms.info.update(
            {"REF_energy": -10.0 * natoms, "frame_id": f"{leaf}:{index}", "IF_leaf": leaf, "IF_stage": "Step2"}
        )
        atoms.arrays["REF_forces"] = np.zeros((natoms, 3))
        if natoms == 3:
            atoms.set_constraint(FixAtoms(indices=[0]))
        frames.append(atoms)
    write(path, frames, format="extxyz")
    return frames


class CommitteeEvalTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.reference = self.root / "test.extxyz"
        self.frames = _reference(self.reference)

    def _member(self, name: str, energy_offset_per_atom: float, force_offset: float, *, order=(0, 1, 2)) -> Path:
        frames = [self.frames[index] for index in order]
        return write_predictions(
            self.root / name / "predictions.npz",
            frame_ids=[frame.info["frame_id"] for frame in frames],
            natoms=[len(frame) for frame in frames],
            energy=[frame.info["REF_energy"] + energy_offset_per_atom * len(frame) for frame in frames],
            forces=[np.full((len(frame), 3), force_offset) for frame in frames],
        )

    def test_metric_definitions_ensemble_and_disagreement(self) -> None:
        members = [
            {"label": "model_000", "seed": 11, "predictions": self._member("a", 0.001, 0.01)},
            # Same frames written in a different order: alignment is by frame_id.
            {"label": "model_001", "seed": 23, "predictions": self._member("b", 0.003, 0.03, order=(2, 0, 1))},
        ]
        summary = evaluate_committee(self.reference, members, self.root / "eval", backend="NequIP")
        per_model = {row["model"]: row for row in summary["per_model"]}
        self.assertAlmostEqual(per_model["model_000"]["energy_rmse_mev_per_atom"], 1.0)
        self.assertAlmostEqual(per_model["model_001"]["energy_mae_mev_per_atom"], 3.0)
        self.assertAlmostEqual(per_model["model_000"]["force_rmse_mev_per_angstrom"], 10.0)
        self.assertAlmostEqual(per_model["ensemble_mean"]["energy_rmse_mev_per_atom"], 2.0)
        self.assertAlmostEqual(per_model["ensemble_mean"]["force_mae_mev_per_angstrom"], 20.0)
        self.assertAlmostEqual(per_model["ensemble_mean"]["energy_centered_rmse_mev_per_atom"], 0.0, places=9)
        self.assertAlmostEqual(per_model["model_000"]["force_rmse_mobile_mev_per_angstrom"], 10.0)
        self.assertAlmostEqual(per_model["model_000"]["atoms"], 7.0)
        self.assertAlmostEqual(summary["disagreement"]["energy_spread_mean_mev_per_atom"], 1.0)
        self.assertAlmostEqual(summary["disagreement"]["force_disagreement_mean_mev_per_angstrom"], 10.0 * 3**0.5)
        self.assertFalse(summary["disagreement"]["calibrated"])
        self.assertIn("NOT a calibrated uncertainty", summary["notes"]["spread"])
        with (self.root / "eval" / "per_frame.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["frame_id"] for row in rows], ["sysA:0", "sysA:1", "sysB:2"])
        self.assertAlmostEqual(float(rows[2]["ensemble_energy_error_mev_per_atom"]), 2.0)
        with (self.root / "eval" / "per_system.csv").open() as handle:
            systems = {(row["system"], row["model"]) for row in csv.DictReader(handle)}
        self.assertIn(("sysB", "ensemble_mean"), systems)
        self.assertTrue(json.loads((self.root / "eval" / "summary.json").read_text())["reference"]["sha256"])

    def test_missing_extra_and_mismatched_frames_are_errors(self) -> None:
        short = write_predictions(
            self.root / "short.npz", frame_ids=["sysA:0"], natoms=[2], energy=[-20.0], forces=[np.zeros((2, 3))]
        )
        with self.assertRaisesRegex(SafetyError, "missing from predictions"):
            evaluate_committee(self.reference, [{"label": "m", "predictions": short}], self.root / "e1", backend="X")
        extra = write_predictions(
            self.root / "extra.npz",
            frame_ids=["sysA:0", "sysA:1", "sysB:2", "other:9"],
            natoms=[2, 2, 3, 1],
            energy=[-20.0, -20.0, -30.0, -1.0],
            forces=[np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((3, 3)), np.zeros((1, 3))],
        )
        with self.assertRaisesRegex(SafetyError, "not in the reference"):
            evaluate_committee(self.reference, [{"label": "m", "predictions": extra}], self.root / "e2", backend="X")
        wrong = write_predictions(
            self.root / "wrong.npz",
            frame_ids=["sysA:0", "sysA:1", "sysB:2"],
            natoms=[2, 2, 2],
            energy=[-20.0, -20.0, -30.0],
            forces=[np.zeros((2, 3))] * 3,
        )
        with self.assertRaisesRegex(SafetyError, "atom count mismatch"):
            evaluate_committee(self.reference, [{"label": "m", "predictions": wrong}], self.root / "e3", backend="X")
        with self.assertRaisesRegex(SafetyError, "Duplicate"):
            evaluate_committee(
                self.reference,
                [
                    {"label": "m", "predictions": self._member("c", 0, 0)},
                    {"label": "m", "predictions": self._member("d", 0, 0)},
                ],
                self.root / "e4",
                backend="X",
            )


if __name__ == "__main__":
    unittest.main()
