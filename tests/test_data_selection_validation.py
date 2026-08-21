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
    stratified_parity_from_csv,
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

    def test_collect_dataset_writes_geometry_class_columns(self) -> None:
        """kind/tebeg_k/high_temperature/coordination degrade gracefully end
        to end when nothing is declared/available, rather than crashing
        collect_dataset -- the fixture's system has no run_glob and the
        fake OUTCAR has no sibling INCAR."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = load_campaign(write_campaign(root))
            source_root = root / "vasp_runs" / "run_a"
            source_root.mkdir(parents=True)
            (source_root / "OUTCAR").write_text("", encoding="utf-8")
            (source_root / "POSCAR").write_text("", encoding="utf-8")

            with patch.object(data_module, "_ase_io", side_effect=_fake_ase_io):
                payload = collect_dataset(
                    campaign,
                    source_root=root / "vasp_runs",
                    output_root=root / "canonical",
                )

            self.assertEqual(payload["kind_counts"], {"unclassified": 1})
            with (Path(payload["output_root"]) / "frames.csv").open(newline="", encoding="utf-8") as handle:
                frame_rows = list(csv.DictReader(handle))
            self.assertEqual(len(frame_rows), 1)
            self.assertEqual(frame_rows[0]["kind"], "unclassified")
            self.assertEqual(frame_rows[0]["tebeg_k"], "")
            self.assertEqual(frame_rows[0]["high_temperature"], "False")
            self.assertEqual(frame_rows[0]["min_coordination_number"], "")


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

    def test_stratified_parity_reports_per_class_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "predictions.csv"
            fieldnames = ["reference", "predicted", "kind", "high_temperature", "min_coordination_number"]
            rows = [
                {"reference": -100.0, "predicted": -99.9, "kind": "interface",
                 "high_temperature": "False", "min_coordination_number": 6},
                {"reference": -100.0, "predicted": -99.5, "kind": "interface",
                 "high_temperature": "True", "min_coordination_number": 3},
                {"reference": -50.0, "predicted": -50.3, "kind": "bulk",
                 "high_temperature": "False", "min_coordination_number": 6},
                {"reference": -20.0, "predicted": -19.0, "kind": "surface",
                 "high_temperature": "False", "min_coordination_number": 2},
            ]
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            result = stratified_parity_from_csv(
                source, root / "stratified.csv", low_coordination_percentile=30
            )

            by_class = {row["class"]: row for row in result["classes"]}
            self.assertEqual(by_class["overall"]["count"], 4)
            self.assertEqual(by_class["kind=interface"]["count"], 2)
            self.assertEqual(by_class["kind=bulk"]["count"], 1)
            self.assertEqual(by_class["kind=surface"]["count"], 1)
            self.assertEqual(by_class["high_temperature"]["count"], 1)
            # Only the surface frame (CN=2) falls at/below the 30th percentile of [6,3,6,2].
            self.assertEqual(by_class["low_coordination"]["count"], 1)
            self.assertAlmostEqual(by_class["low_coordination"]["mae"], 1.0)
            self.assertTrue(Path(result["output"]).is_file())
            self.assertTrue(Path(result["output"]).with_suffix(".json").is_file())

    def test_stratified_parity_omits_absent_columns_without_erroring(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "predictions.csv"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["reference", "predicted"])
                writer.writeheader()
                writer.writerow({"reference": 1.0, "predicted": 1.1})

            result = stratified_parity_from_csv(source, root / "stratified.csv")

            self.assertEqual([row["class"] for row in result["classes"]], ["overall"])

    def test_stratified_parity_requires_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "empty.csv"
            with source.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=["reference", "predicted"]).writeheader()

            with self.assertRaises(ValueError):
                stratified_parity_from_csv(source, root / "out.csv")


class GeometryClassificationTests(unittest.TestCase):
    def test_classify_kind_matches_declared_run_glob(self) -> None:
        from interfaceforge.config import SystemSpec
        from interfaceforge.data import classify_kind

        root = Path("/campaign/runs/vasp")
        systems = [
            SystemSpec(id="iface", kind="interface", structure=Path("x"), run_glob="*/interface_*/*"),
            SystemSpec(id="bulk", kind="bulk", structure=Path("x"), run_glob="*/bulk_*/*"),
        ]
        self.assertEqual(
            classify_kind(root / "interface_300K" / "OUTCAR", root, systems), "interface"
        )
        self.assertEqual(classify_kind(root / "bulk_500K" / "OUTCAR", root, systems), "bulk")
        self.assertEqual(classify_kind(root / "unrelated_run" / "OUTCAR", root, systems), "unclassified")

    def test_classify_kind_with_no_run_glob_declared_is_unclassified(self) -> None:
        from interfaceforge.config import SystemSpec
        from interfaceforge.data import classify_kind

        root = Path("/campaign/runs/vasp")
        systems = [SystemSpec(id="iface", kind="interface", structure=Path("x"))]
        self.assertEqual(classify_kind(root / "anything" / "OUTCAR", root, systems), "unclassified")

    def test_read_tebeg_from_sibling_incar(self) -> None:
        from interfaceforge.data import _read_tebeg

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outcar = root / "OUTCAR"
            outcar.write_text("", encoding="utf-8")
            (root / "INCAR").write_text("TEBEG = 600\n", encoding="utf-8")
            self.assertEqual(_read_tebeg(outcar), 600.0)

    def test_read_tebeg_missing_incar_returns_none(self) -> None:
        from interfaceforge.data import _read_tebeg

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outcar = root / "sub" / "OUTCAR"
            outcar.parent.mkdir()
            outcar.write_text("", encoding="utf-8")
            self.assertIsNone(_read_tebeg(outcar))

    def test_coordination_stats_on_fake_atoms_degrades_to_none(self) -> None:
        from interfaceforge.data import _coordination_stats

        # _FakeAtoms above stands in for real ase.Atoms in other tests here;
        # ase.neighborlist needs the real interface, so this must degrade
        # gracefully rather than raise.
        atoms = _FakeAtoms(["H"], np.array([[0.0, 0.0, 0.0]]), np.eye(3) * 5.0)
        self.assertEqual(_coordination_stats(atoms), (None, None))


if __name__ == "__main__":
    unittest.main()
