from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from interfaceforge.data import SourceTrajectory, assign_grouped, guarded_assignments
from interfaceforge.selection import select_indices
from interfaceforge.validation import (
    EV_A2_TO_J_M2,
    parity_from_csv,
    work_of_adhesion,
)


class SplitTests(unittest.TestCase):
    def test_grouped_split_keeps_each_trajectory_whole(self) -> None:
        sources = [
            SourceTrajectory(Path(f"/tmp/run_{index}/OUTCAR"), f"run_{index}", "bulk", "bulk")
            for index in range(9)
        ]
        assignment = assign_grouped(sources, [0.7, 0.2, 0.1], seed=7)
        self.assertEqual(set(assignment), {source.path for source in sources})
        self.assertEqual(set(assignment.values()), {"train", "valid", "test"})

    def test_guarded_blocks_drop_boundary_frames(self) -> None:
        assignment, guards = guarded_assignments(36, [0.6, 0.2, 0.2], 3)
        self.assertEqual(len(guards), 6)
        self.assertFalse(set(assignment).intersection(guards))
        self.assertEqual(len(assignment) + len(guards), 36)


class SelectionTests(unittest.TestCase):
    def test_selection_combines_uncertainty_and_diversity(self) -> None:
        chosen, diagnostics = select_indices(
            [0.9, 0.8, 0.7, 0.6],
            2,
            features=np.asarray([[0, 0], [0.01, 0.01], [1, 1], [0.5, 0.5]]),
            uncertainty_weight=0.5,
        )
        self.assertEqual(chosen[0], 0)
        self.assertIn(2, chosen)
        self.assertEqual(len(diagnostics), 2)


class ValidationTests(unittest.TestCase):
    def test_work_of_adhesion_units(self) -> None:
        ev_a2, j_m2 = work_of_adhesion(-12, -5, -5, 2)
        self.assertAlmostEqual(ev_a2, 1.0)
        self.assertAlmostEqual(j_m2, EV_A2_TO_J_M2)

    def test_constant_parity_writes_strict_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "parity.csv"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["model", "reference", "predicted"]
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"model": "a", "reference": 1, "predicted": 1.1},
                        {"model": "a", "reference": 1, "predicted": 0.9},
                    ]
                )
            parity_from_csv(source, root / "metrics.csv")
            payload = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
            self.assertIsNone(payload["groups"][0]["r2"])


if __name__ == "__main__":
    unittest.main()
