from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ase = pytest.importorskip("ase")
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write

from interfaceforge.derivative_probe import (
    PAPER,
    evaluate_derivative_probe,
    prepare_derivative_probe,
)
from interfaceforge.errors import SafetyError


def _structure(path: Path) -> Path:
    atoms = Atoms(
        "Si2",
        positions=[[1.0, 1.2, 1.4], [3.0, 2.8, 2.6]],
        cell=np.eye(3) * 5.0,
        pbc=True,
    )
    write(path, atoms, format="vasp", direct=True, vasp5=True)
    return path


def _case(root: Path, fragment: str) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return next(row for row in manifest["structures"] if fragment in row["structure_id"])


class HarmonicCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, spring: float = 2.0):
        super().__init__()
        self.spring = spring

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = np.asarray(self.atoms.positions)
        self.results = {
            "energy": 0.5 * self.spring * float(np.sum(positions**2)),
            "forces": -self.spring * positions,
            "stress": np.zeros(6),
        }


def test_prepare_uses_paper_defaults_and_central_pairs(tmp_path: Path) -> None:
    source = _structure(tmp_path / "POSCAR")
    root = tmp_path / "probe"
    result = prepare_derivative_probe([f"bulk={source}"], root)

    assert result["structures"] == 9
    assert result["citation"]["doi"] == PAPER["doi"]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    protocol = manifest["protocol"]
    assert protocol["displacement_component_stdev_a"] == pytest.approx(0.03)
    assert protocol["strains"] == [-0.01, 0.0, 0.01]
    assert protocol["paired_displacements"] is True
    assert "InterfaceForge extension" in protocol["interfaceforge_extension"]
    assert (root / "derivative_probe.extxyz").is_file()
    assert len((root / "runs.txt").read_text(encoding="utf-8").splitlines()) == 9

    center_row = _case(root, "strain_p0p000000_center")
    plus_row = _case(root, "strain_p0p000000_rattle_001_plus")
    minus_row = _case(root, "strain_p0p000000_rattle_001_minus")
    center = read(root / center_row["relative_directory"] / "POSCAR")
    plus = read(root / plus_row["relative_directory"] / "POSCAR")
    minus = read(root / minus_row["relative_directory"] / "POSCAR")
    assert (plus.positions + minus.positions) / 2.0 == pytest.approx(center.positions)
    assert np.sqrt(np.mean((plus.positions - center.positions) ** 2)) == pytest.approx(
        plus_row["displacement_realized_rms_a"]
    )


def test_volume_strain_changes_volume_not_each_vector_by_one_percent(tmp_path: Path) -> None:
    source = _structure(tmp_path / "cell.vasp")
    root = tmp_path / "probe"
    prepare_derivative_probe([str(source)], root)
    minus = _case(root, "strain_m0p010000_center")
    zero = _case(root, "strain_p0p000000_center")
    plus = _case(root, "strain_p0p010000_center")
    volumes = [
        read(root / row["relative_directory"] / "POSCAR").get_volume()
        for row in (minus, zero, plus)
    ]
    assert volumes[0] / volumes[1] == pytest.approx(0.99)
    assert volumes[2] / volumes[1] == pytest.approx(1.01)


def test_prepare_is_deterministic_and_rejects_foreign_force_target(tmp_path: Path) -> None:
    source = _structure(tmp_path / "cell.vasp")
    first = tmp_path / "first"
    second = tmp_path / "second"
    prepare_derivative_probe([f"sample={source}"], first, seed=41)
    prepare_derivative_probe([f"sample={source}"], second, seed=41)

    first_rows = json.loads((first / "manifest.json").read_text(encoding="utf-8"))["structures"]
    second_rows = json.loads((second / "manifest.json").read_text(encoding="utf-8"))["structures"]
    assert [row["input_sha256"]["POSCAR"] for row in first_rows] == [
        row["input_sha256"]["POSCAR"] for row in second_rows
    ]

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(SafetyError, match="not a recognized"):
        prepare_derivative_probe([str(source)], foreign, force=True)
    assert (foreign / "keep.txt").read_text(encoding="utf-8") == "user data"


def test_vasp_template_is_static_and_hashed(tmp_path: Path) -> None:
    source = _structure(tmp_path / "cell.vasp")
    template = tmp_path / "template"
    template.mkdir()
    (template / "INCAR").write_text(
        "ENCUT = 520\nIBRION = 0\nNSW = 3000\nPOTIM = 1.0\n"
        "TEBEG = 450\nML_LMLFF = .TRUE.\n",
        encoding="utf-8",
    )
    (template / "KPOINTS").write_text(
        "mesh\n0\nGamma\n2 2 2\n0 0 0\n", encoding="utf-8"
    )
    (template / "POTCAR").write_text("test pseudopotential\n", encoding="utf-8")
    root = tmp_path / "probe"
    result = prepare_derivative_probe(
        [str(source)],
        root,
        strains=(0.0,),
        vasp_template=template,
    )

    assert result["vasp_ready"] is True
    row = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["structures"][0]
    run = root / row["relative_directory"]
    incar = (run / "INCAR").read_text(encoding="utf-8")
    assert "ENCUT = 520" in incar
    assert "IBRION          = -1" in incar
    assert "NSW             = 0" in incar
    assert "ISYM            = 0" in incar
    assert "POTIM" not in incar and "TEBEG" not in incar and "ML_LMLFF" not in incar
    assert (run / "POTCAR").read_text(encoding="utf-8") == "test pseudopotential\n"
    assert set(row["input_sha256"]) == {"POSCAR", "INCAR", "KPOINTS", "POTCAR"}


def test_evaluate_reports_exact_harmonic_curvature(tmp_path: Path) -> None:
    source = _structure(tmp_path / "cell.vasp")
    root = tmp_path / "probe"
    prepare_derivative_probe([f"bulk={source}"], root)
    result = evaluate_derivative_probe(
        root, _test_calculators={"harmonic": HarmonicCalculator(2.0)}
    )

    assert result["status"] == "INCOMPLETE"
    assert result["dft_completed"] == 0
    assert result["models"] == ["harmonic"]
    responses = [
        row
        for row in __import__("csv").DictReader(
            (root / "responses.csv").open(encoding="utf-8")
        )
    ]
    assert len(responses) == 3
    for row in responses:
        assert float(row["energy_curvature_ev_a2"]) == pytest.approx(2.0)
        assert float(row["force_curvature_ev_a2"]) == pytest.approx(2.0)


def test_evaluate_rejects_changed_probe_input(tmp_path: Path) -> None:
    source = _structure(tmp_path / "cell.vasp")
    root = tmp_path / "probe"
    prepare_derivative_probe([str(source)], root, strains=(0.0,))
    row = _case(root, "center")
    poscar = root / row["relative_directory"] / "POSCAR"
    poscar.write_text(poscar.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(SafetyError, match="changed since prepare"):
        evaluate_derivative_probe(root, _test_calculators={"harmonic": HarmonicCalculator()})
