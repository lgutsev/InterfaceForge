from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ase")
from ase import Atoms
from ase.io import write

from interfaceforge.errors import SafetyError
from interfaceforge.mlip_compare import MACE_EVALUATOR, _metrics, validate_membership


def _dataset(tmp_path: Path) -> tuple[Path, Path]:
    mace = tmp_path / "test.extxyz"
    deepmd = tmp_path / "deepmd" / "test"
    system = deepmd / "bulk" / "sample"
    set_dir = system / "set.000"
    set_dir.mkdir(parents=True)
    frames = []
    for source_frame, shift in ((2, 0.0), (7, 0.1)):
        atoms = Atoms(
            "SiN",
            positions=[[0.0 + shift, 0.0, 0.0], [1.5 + shift, 1.5, 1.5]],
            cell=np.eye(3) * 5.0,
            pbc=True,
        )
        atoms.info.update(
            {
                "REF_energy": -10.0 + shift,
                "IF_leaf": "bulk/sample",
                "source_frame": source_frame,
            }
        )
        atoms.arrays["REF_forces"] = np.full((2, 3), shift)
        frames.append(atoms)
    write(mace, frames, format="extxyz")

    (system / "type_map.raw").write_text("Si\nN\n", encoding="utf-8")
    (system / "type.raw").write_text("0\n1\n", encoding="utf-8")
    np.save(set_dir / "coord.npy", np.asarray([frame.positions.reshape(-1) for frame in frames]))
    np.save(set_dir / "box.npy", np.asarray([frame.cell.array.reshape(-1) for frame in frames]))
    np.save(set_dir / "energy.npy", np.asarray([[frame.info["REF_energy"]] for frame in frames]))
    np.save(set_dir / "force.npy", np.asarray([frame.arrays["REF_forces"].reshape(-1) for frame in frames]))
    with (system / "frame_map.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["local_frame", "source_frame", "source_path", "relative_leaf"])
        writer.writerow([0, 2, "OUTCAR", "bulk/sample"])
        writer.writerow([1, 7, "OUTCAR", "bulk/sample"])
    return mace, deepmd


def test_membership_proves_geometry_labels_and_identity(tmp_path: Path) -> None:
    mace, deepmd = _dataset(tmp_path)
    systems, summary = validate_membership(mace, deepmd, tmp_path / "grouped")
    assert summary["exact_membership"] is True
    assert summary["frames"] == 2
    assert summary["systems"] == 1
    assert systems[0]["relative_leaf"] == "bulk/sample"
    assert Path(systems[0]["mace_input"]).is_file()


def test_membership_rejects_silent_coordinate_drift(tmp_path: Path) -> None:
    mace, deepmd = _dataset(tmp_path)
    coord_path = deepmd / "bulk" / "sample" / "set.000" / "coord.npy"
    coord = np.load(coord_path)
    coord[0, 0] += 1.0e-3
    np.save(coord_path, coord)
    with pytest.raises(SafetyError, match="Canonical data mismatch"):
        validate_membership(mace, deepmd, tmp_path / "grouped")


def test_centered_energy_and_force_metric_definitions() -> None:
    ref_e = np.asarray([0.0, 1.0])
    pred_e = np.asarray([0.5, 1.5])
    ref_f = np.zeros((2, 1, 3))
    pred_f = np.ones((2, 1, 3)) * 0.1
    metrics = _metrics(ref_e, pred_e, ref_f, pred_f)
    assert metrics["energy_rmse_mev_per_atom"] == pytest.approx(500.0)
    assert metrics["energy_centered_rmse_mev_per_atom"] == pytest.approx(0.0)
    assert metrics["force_rmse_mev_per_angstrom"] == pytest.approx(100.0)
    assert metrics["force_vector_rmse_mev_per_angstrom"] == pytest.approx(100.0 * 3**0.5)


def test_generated_mace_evaluator_is_valid_python() -> None:
    compile(MACE_EVALUATOR, "evaluate_mace.py", "exec")
