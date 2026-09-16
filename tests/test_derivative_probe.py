from __future__ import annotations

import csv
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms
from ase.io import read, write

from interfaceforge.derivative_probe import (
    PAPER,
    _calculators,
    _prediction,
    _static_incar,
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


def _write_dft(run: Path, *, spring: float = 2.0, offset: float = 0.0) -> None:
    """Synthetic OUTCAR with analytic values, parsed through the real ASE reader."""
    atoms = read(run / "POSCAR")
    energy = 0.5 * spring * float(np.sum(atoms.positions**2)) + offset
    lines = [
        "vasp.6.5.1 synthetic regression fixture",
        "POTCAR: PAW_PBE Si 05Jan2001", "POTCAR: PAW_PBE Si 05Jan2001",
        "TITEL = PAW_PBE Si 05Jan2001", "ions per type = 2",
        "NKPTS = 1", "ISPIN = 1", "ENCUT = 520", "IBRION = -1", "NSW = 0", "ISYM = 0",
        "Iteration 1(1)", "aborting loop because EDIFF is reached",
        "in kB 0 0 0 0 0 0",
        "direct lattice vectors",
        *[" ".join(str(x) for x in row) for row in atoms.cell.array],
        "POSITION                                       TOTAL-FORCE (eV/Angst)",
        "-------------------------------------------------------------------",
        *[" ".join(f"{x:.12f}" for x in (*pos, *(-spring * pos))) for pos in atoms.positions],
        "-------------------------------------------------------------------",
        "FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)",
        "---------------------------------------------------",
        f"free energy TOTEN = {energy:.12f} eV", "",
        f"energy without entropy= {energy:.12f} energy(sigma->0) = {energy:.12f}",
        "General timing and accounting informations for this job",
    ]
    (run / "OUTCAR").write_text("\n".join(lines) + "\n")
    (run / "INCAR").write_text("ENCUT=520; IBRION=-1; NSW=0; ISYM=0\n")
    (run / "KPOINTS").write_text("mesh\n0\nGamma\n1 1 1\n0 0 0\n")
    (run / "POTCAR").write_text("TITEL = PAW_PBE Si 05Jan2001\n")


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

    def _dft_tree(self):
        source = _structure(self.root / "POSCAR")
        root = self.root / "probe"
        prepare_derivative_probe([f"bulk={source}"], root, strains=(0.,))
        for poscar in root.glob("bulk/*/POSCAR"):
            _write_dft(poscar.parent, offset=100.)
        return root

    def test_semicolon_incar_preserves_electronic_tags_and_removes_ml(self):
        source = self.root / "INCAR"
        source.write_text("NSW=3000; ENCUT=520 ! comment; ENCUT=1\n"
                          "PREC=Accurate; ML_LMLFF=.TRUE.; POTIM=1\n"
                          "ENCUT=600 # last assignment wins\n")
        text = _static_incar(source)
        self.assertIn("ENCUT = 600", text)
        self.assertIn("PREC = Accurate", text)
        self.assertNotIn("ML_LMLFF", text)
        self.assertNotIn("POTIM", text)
        self.assertNotIn("3000", text)

    def test_constraints_do_not_mask_forces_or_curvature(self):
        source = _structure(self.root / "POSCAR")
        atoms = read(source)
        atoms.set_constraint(FixAtoms(indices=[0, 1]))
        _, forces, _ = _prediction(atoms, HarmonicCalculator())
        np.testing.assert_allclose(forces, -2 * atoms.positions)
        write(source, atoms, format="vasp")
        root = self.root / "probe"
        prepare_derivative_probe([str(source)], root, strains=(0.,))
        self.assertFalse(read(next(root.glob("*/*/POSCAR"))).constraints)
        result = evaluate_derivative_probe(root, _test_calculators={"harmonic": HarmonicCalculator()})
        with Path(result["outputs"]["responses"]).open() as handle:
            row = next(csv.DictReader(handle))
        self.assertAlmostEqual(float(row["energy_curvature_ev_a2"]), 2.)
        self.assertAlmostEqual(float(row["force_curvature_ev_a2"]), 2.)

    def test_real_reader_dft_comparison_and_provenance(self):
        root = self._dft_tree()
        result = evaluate_derivative_probe(root, _test_calculators={"harmonic": HarmonicCalculator()})
        self.assertEqual(result["status"], "COMPLETE")
        summary = result["summaries"][0]
        self.assertEqual(summary["matched_structures"], 3)
        self.assertEqual(summary["matched_curvatures"], 1)
        self.assertLess(summary["relative_energy_mev_atom"]["rmse"], 1e-8)
        self.assertLess(summary["force_ev_a"]["rmse"], 1e-10)
        self.assertEqual(summary["stress_ev_a3"]["rmse"], 0.)
        self.assertLess(summary["directional_curvature_ev_a2"]["rmse"], 1e-8)
        self.assertEqual(len(result["provenance"]["manifest_sha256"]), 64)
        self.assertEqual(result["provenance"]["mace_dtype"], "float64")
        self.assertEqual(len(result["dft_evidence"]), 3)
        self.assertTrue(all(len(v["outcar_sha256"]) == 64 for v in result["dft_evidence"].values()))

    def test_nonzero_dft_model_errors_have_correct_units(self):
        root = self._dft_tree()
        result = evaluate_derivative_probe(root, _test_calculators={"stiffer": HarmonicCalculator(3.)})
        summary = result["summaries"][0]
        self.assertAlmostEqual(summary["directional_curvature_ev_a2"]["rmse"], 1.)
        atoms = [read(p) for p in root.glob("bulk/*/POSCAR")]
        center = read(root / _case(root, "center")["relative_directory"] / "POSCAR")
        reference = float(np.sum(center.positions**2))
        errors = [(float(np.sum(a.positions**2)) - reference) / 2 / len(a) * 1000 for a in atoms]
        self.assertAlmostEqual(summary["relative_energy_mev_atom"]["rmse"], float(np.sqrt(np.mean(np.square(errors)))))
        expected_force_rmse = np.sqrt(np.mean(np.concatenate([a.positions.ravel() for a in atoms])**2))
        self.assertAlmostEqual(summary["force_ev_a"]["rmse"], expected_force_rmse)

    def test_periodic_image_equivalence_is_accepted(self):
        root = self._dft_tree()
        from interfaceforge.derivative_probe import _validate_dft
        run = root / _case(root, "center")["relative_directory"]
        expected = read(run / "POSCAR")
        actual = read(run / "OUTCAR")
        actual.positions[0] += actual.cell[0]
        self.assertEqual(_validate_dft(run, expected, actual)["geometry"], "PASS")

    def test_cross_probe_input_and_setting_changes_are_rejected(self):
        root = self._dft_tree()
        run = root / _case(root, "plus")["relative_directory"]
        kpoints = run / "KPOINTS"
        original = kpoints.read_text()
        kpoints.write_text(original.replace("1 1 1", "2 2 2"))
        with self.assertRaisesRegex(SafetyError, "KPOINTS differs across"):
            evaluate_derivative_probe(root)
        kpoints.write_text(original)
        outcar = run / "OUTCAR"
        outcar.write_text(outcar.read_text().replace("ISPIN = 1", "ISPIN = 2"))
        with self.assertRaisesRegex(SafetyError, "Executed DFT settings differ"):
            evaluate_derivative_probe(root)

    def test_copied_outcar_is_rejected(self):
        root = self._dft_tree()
        center = root / _case(root, "center")["relative_directory"]
        plus = root / _case(root, "plus")["relative_directory"]
        (plus / "OUTCAR").write_bytes((center / "OUTCAR").read_bytes())
        with self.assertRaisesRegex(SafetyError, "positions differ"):
            evaluate_derivative_probe(root)

    def test_invalid_dft_evidence_is_rejected(self):
        replacements = [
            ("aborting loop because EDIFF is reached", "NELM exhausted", "convergence"),
            ("General timing and accounting informations", "truncated", "termination"),
            ("ENCUT = 520", "ENCUT = 400", "ENCUT differs"),
            ("NSW = 0", "NSW = 100", "NSW=0"),
            ("NSW = 0", "NSW = 0; ML_LMLFF = T", "ML forces"),
            ("PAW_PBE Si", "PAW_PBE Ti", "species/order"),
            ("5.0 0.0 0.0", "6.0 0.0 0.0", "cell differs"),
        ]
        root = self._dft_tree()
        outcar = next(root.glob("bulk/*/OUTCAR"))
        original = outcar.read_text()
        for old, new, message in replacements:
            with self.subTest(message=message):
                self.assertIn(old, original)
                outcar.write_text(original.replace(old, new))
                with self.assertRaisesRegex(SafetyError, message):
                    evaluate_derivative_probe(root)
        outcar.write_text(original)

    def test_missing_provenance_is_not_complete(self):
        root = self._dft_tree()
        next(root.glob("bulk/*/POTCAR")).unlink()
        result = evaluate_derivative_probe(root)
        self.assertEqual(result["status"], "CHECK")
        self.assertTrue(any("POTCAR unavailable" in warning for warning in result["warnings"]))

    def test_mace_dtype_and_model_hash_are_recorded(self):
        root = self._dft_tree()
        model = self.root / "model.model"
        model.write_bytes(b"test model identity")
        factory = MagicMock(side_effect=lambda **kwargs: HarmonicCalculator())
        module = types.ModuleType("mace.calculators")
        module.MACECalculator = factory
        with patch.dict(sys.modules, {"mace": types.ModuleType("mace"), "mace.calculators": module}):
            result = evaluate_derivative_probe(root, mace_models=[model], mace_dtype="float32")
        self.assertEqual(factory.call_args.kwargs["default_dtype"], "float32")
        self.assertEqual(len(result["provenance"]["models"][0]["sha256"]), 64)
        self.assertEqual(result["provenance"]["models"][0]["path"], str(model))
        with patch.dict(sys.modules, {"mace": types.ModuleType("mace"), "mace.calculators": module}):
            _calculators([model], [], "cpu")
        self.assertEqual(factory.call_args.kwargs["default_dtype"], "float64")


if __name__ == "__main__":
    unittest.main()
