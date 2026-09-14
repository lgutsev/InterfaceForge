from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
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


class DerivativeProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_prepare_uses_paper_defaults_and_central_pairs(self) -> None:
        source = _structure(self.root / "POSCAR")
        root = self.root / "probe"
        result = prepare_derivative_probe([f"bulk={source}"], root)

        self.assertEqual(result["structures"], 9)
        self.assertEqual(result["citation"]["doi"], PAPER["doi"])
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        protocol = manifest["protocol"]
        self.assertAlmostEqual(protocol["displacement_component_stdev_a"], 0.03)
        self.assertEqual(protocol["strains"], [-0.01, 0.0, 0.01])
        self.assertTrue(protocol["paired_displacements"])
        self.assertIn("InterfaceForge extension", protocol["interfaceforge_extension"])
        self.assertTrue((root / "derivative_probe.extxyz").is_file())
        self.assertEqual(
            len((root / "runs.txt").read_text(encoding="utf-8").splitlines()), 9
        )

        center_row = _case(root, "strain_p0p000000_center")
        plus_row = _case(root, "strain_p0p000000_rattle_001_plus")
        minus_row = _case(root, "strain_p0p000000_rattle_001_minus")
        center = read(root / center_row["relative_directory"] / "POSCAR")
        plus = read(root / plus_row["relative_directory"] / "POSCAR")
        minus = read(root / minus_row["relative_directory"] / "POSCAR")
        np.testing.assert_allclose(
            (plus.positions + minus.positions) / 2.0, center.positions
        )
        realized = np.sqrt(np.mean((plus.positions - center.positions) ** 2))
        self.assertAlmostEqual(realized, plus_row["displacement_realized_rms_a"])

    def test_volume_strain_changes_volume_not_each_vector_by_one_percent(self) -> None:
        source = _structure(self.root / "cell.vasp")
        root = self.root / "probe"
        prepare_derivative_probe([str(source)], root)
        minus = _case(root, "strain_m0p010000_center")
        zero = _case(root, "strain_p0p000000_center")
        plus = _case(root, "strain_p0p010000_center")
        volumes = [
            read(root / row["relative_directory"] / "POSCAR").get_volume()
            for row in (minus, zero, plus)
        ]
        self.assertAlmostEqual(volumes[0] / volumes[1], 0.99)
        self.assertAlmostEqual(volumes[2] / volumes[1], 1.01)

    def test_prepare_is_deterministic_and_guards_foreign_force_target(self) -> None:
        source = _structure(self.root / "cell.vasp")
        first = self.root / "first"
        second = self.root / "second"
        prepare_derivative_probe([f"sample={source}"], first, seed=41)
        prepare_derivative_probe([f"sample={source}"], second, seed=41)

        first_rows = json.loads(
            (first / "manifest.json").read_text(encoding="utf-8")
        )["structures"]
        second_rows = json.loads(
            (second / "manifest.json").read_text(encoding="utf-8")
        )["structures"]
        self.assertEqual(
            [row["input_sha256"]["POSCAR"] for row in first_rows],
            [row["input_sha256"]["POSCAR"] for row in second_rows],
        )

        foreign = self.root / "foreign"
        foreign.mkdir()
        (foreign / "keep.txt").write_text("user data", encoding="utf-8")
        with self.assertRaisesRegex(SafetyError, "not a recognized"):
            prepare_derivative_probe([str(source)], foreign, force=True)
        self.assertEqual(
            (foreign / "keep.txt").read_text(encoding="utf-8"), "user data"
        )

    def test_vasp_template_is_static_hashed_and_species_checked(self) -> None:
        source = _structure(self.root / "cell.vasp")
        template = self.root / "template"
        template.mkdir()
        (template / "INCAR").write_text(
            "ENCUT = 520\nIBRION = 0\nNSW = 3000\nPOTIM = 1.0\n"
            "TEBEG = 450\nML_LMLFF = .TRUE.\n",
            encoding="utf-8",
        )
        (template / "KPOINTS").write_text(
            "mesh\n0\nGamma\n2 2 2\n0 0 0\n", encoding="utf-8"
        )
        (template / "POTCAR").write_text(
            "VRHFIN =Si: s2p2\n", encoding="utf-8"
        )
        root = self.root / "probe"
        result = prepare_derivative_probe(
            [str(source)],
            root,
            strains=(0.0,),
            vasp_template=template,
        )

        self.assertTrue(result["vasp_ready"])
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        row = manifest["structures"][0]
        run = root / row["relative_directory"]
        incar = (run / "INCAR").read_text(encoding="utf-8")
        self.assertIn("ENCUT = 520", incar)
        self.assertIn("IBRION          = -1", incar)
        self.assertIn("NSW             = 0", incar)
        self.assertIn("ISYM            = 0", incar)
        self.assertNotIn("POTIM", incar)
        self.assertNotIn("TEBEG", incar)
        self.assertNotIn("ML_LMLFF", incar)
        self.assertEqual(
            (run / "POTCAR").read_text(encoding="utf-8"),
            "VRHFIN =Si: s2p2\n",
        )
        self.assertEqual(
            set(row["input_sha256"]), {"POSCAR", "INCAR", "KPOINTS", "POTCAR"}
        )

        bad = self.root / "bad_template"
        bad.mkdir()
        (bad / "INCAR").write_text("ENCUT=520\n", encoding="utf-8")
        (bad / "KPOINTS").write_text("Gamma\n", encoding="utf-8")
        (bad / "POTCAR").write_text("VRHFIN =Ti: d2s2\n", encoding="utf-8")
        with self.assertRaisesRegex(SafetyError, "does not match"):
            prepare_derivative_probe(
                [str(source)], self.root / "bad_probe", vasp_template=bad
            )

    def test_evaluate_reports_exact_harmonic_curvature(self) -> None:
        source = _structure(self.root / "cell.vasp")
        root = self.root / "probe"
        prepare_derivative_probe([f"bulk={source}"], root)
        result = evaluate_derivative_probe(
            root, _test_calculators={"harmonic": HarmonicCalculator(2.0)}
        )

        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["dft_completed"], 0)
        self.assertEqual(result["models"], ["harmonic"])
        with Path(result["outputs"]["responses"]).open(encoding="utf-8") as handle:
            responses = list(csv.DictReader(handle))
        self.assertEqual(len(responses), 3)
        for row in responses:
            self.assertAlmostEqual(float(row["energy_curvature_ev_a2"]), 2.0)
            self.assertAlmostEqual(float(row["force_curvature_ev_a2"]), 2.0)

    def test_evaluate_rejects_changed_probe_input(self) -> None:
        source = _structure(self.root / "cell.vasp")
        root = self.root / "probe"
        prepare_derivative_probe([str(source)], root, strains=(0.0,))
        row = _case(root, "center")
        poscar = root / row["relative_directory"] / "POSCAR"
        poscar.write_text(
            poscar.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )

        with self.assertRaisesRegex(SafetyError, "changed since prepare"):
            evaluate_derivative_probe(
                root, _test_calculators={"harmonic": HarmonicCalculator()}
            )


if __name__ == "__main__":
    unittest.main()
