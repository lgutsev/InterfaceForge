from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import interfaceforge.data as data_module
from interfaceforge.config import load_campaign
from interfaceforge.data import (
    SourceTrajectory,
    _has_constraint_source,
    assign_grouped,
    collect_dataset,
    discover_outcars,
    guarded_assignments,
)
from interfaceforge.selection import select_indices
from interfaceforge.validation import (
    EV_A2_TO_J_M2,
    parity_from_csv,
    work_of_adhesion,
)

sys.path.insert(0, str(Path(__file__).parent))
from test_config_scheduler import write_campaign  # noqa: E402


class _FakeCell:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array


class _FakeAtoms:
    """Minimal ASE-Atoms stand-in so collect_dataset can be exercised
    end-to-end without depending on the optional `ase` package."""

    def __init__(self, symbols: list[str], positions: np.ndarray, cell: np.ndarray) -> None:
        self._symbols = list(symbols)
        self.arrays: dict[str, np.ndarray] = {"positions": np.asarray(positions, dtype=float)}
        self.cell = _FakeCell(np.asarray(cell, dtype=float))
        self.info: dict = {}
        self.constraints: list = []
        self.calc = None

    def __len__(self) -> int:
        return len(self._symbols)

    @property
    def positions(self) -> np.ndarray:
        return self.arrays["positions"]

    def get_chemical_symbols(self) -> list[str]:
        return list(self._symbols)

    def get_potential_energy(self) -> float:
        return -1.0

    def get_forces(self, apply_constraint: bool = True) -> np.ndarray:
        return np.zeros((len(self), 3))

    def get_volume(self) -> float:
        return float(np.linalg.det(self.cell.array))

    def copy(self) -> _FakeAtoms:
        clone = _FakeAtoms(self._symbols, self.arrays["positions"].copy(), self.cell.array.copy())
        clone.info = dict(self.info)
        clone.constraints = list(self.constraints)
        return clone


def _fake_ase_io():
    def fake_iread(path: str, index: str = ":"):
        yield _FakeAtoms(["H"], np.array([[0.0, 0.0, 0.0]]), np.eye(3) * 5.0)

    def fake_write(path: str, atoms: _FakeAtoms, *, format: str, append: bool) -> None:
        mode = "a" if append else "w"
        with open(path, mode, encoding="utf-8") as handle:
            handle.write(f"frame natoms={len(atoms)}\n")

    return fake_iread, fake_write


class ConstraintSourceTests(unittest.TestCase):
    def test_missing_contcar_and_poscar_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            outcar = run / "OUTCAR"
            outcar.write_text("", encoding="utf-8")
            self.assertFalse(_has_constraint_source(outcar))
            (run / "POSCAR").write_text("", encoding="utf-8")
            self.assertTrue(_has_constraint_source(outcar))

    def test_collect_dataset_warns_when_constraints_are_unrecoverable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = load_campaign(write_campaign(root))
            source_root = root / "vasp_runs" / "run_a"
            source_root.mkdir(parents=True)
            (source_root / "OUTCAR").write_text("", encoding="utf-8")
            # Deliberately no CONTCAR/POSCAR beside OUTCAR.

            with patch.object(data_module, "_ase_io", side_effect=_fake_ase_io):
                payload = collect_dataset(
                    campaign,
                    source_root=root / "vasp_runs",
                    output_root=root / "canonical",
                )

            with (Path(payload["output_root"]) / "manifest.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertIn("Selective Dynamics constraints", rows[0]["warnings"])
            self.assertIn("move_mask marks every atom as mobile", rows[0]["warnings"])


class RunIdCollisionTests(unittest.TestCase):
    def test_colliding_sanitized_names_get_unique_run_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # "system_A" and "system A" both sanitize to run_id "system_A".
            (root / "system_A").mkdir()
            (root / "system_A" / "OUTCAR").write_text("", encoding="utf-8")
            (root / "system A").mkdir()
            (root / "system A" / "OUTCAR").write_text("", encoding="utf-8")

            sources = discover_outcars(root)

            self.assertEqual(len(sources), 2)
            run_ids = [source.run_id for source in sources]
            self.assertEqual(len(set(run_ids)), 2, f"run_ids collided: {run_ids}")
            self.assertIn("system_A", run_ids)
            self.assertTrue(any(run_id.startswith("system_A__dup") for run_id in run_ids))


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
