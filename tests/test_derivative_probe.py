"""Regression tests for the derivative-sensitive MLIP probe.

Every fixture here is **synthetic or mocked**: analytic ASE calculators stand in
for MLIPs, hand-written OUTCARs stand in for VASP (they are parsed by the real
ASE reader), and MACE/DeePMD calculator construction is patched into
``sys.modules``. Nothing in this file runs VASP, MACE or DeePMD, so a green run
establishes software behavior only, never real-backend or materials validation.
"""

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
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms
from ase.io import read, write

from interfaceforge.derivative_probe import (
    CURVATURE_DEFINITIONS,
    FORCE_EXPORT_SCHEMA,
    PAPER,
    _calculators,
    _prediction,
    _sha256,
    _static_incar,
    evaluate_derivative_probe,
    prepare_derivative_probe,
)
from interfaceforge.errors import DependencyError, SafetyError


def _structure(path: Path, shift: float = 0.0) -> Path:
    atoms = Atoms(
        "Si2",
        positions=np.array([[1.0, 1.2, 1.4], [3.0, 2.8, 2.6]]) + shift,
        cell=np.eye(3) * 5.0,
        pbc=True,
    )
    write(path, atoms, format="vasp", direct=True, vasp5=True)
    return path


def _case(root: Path, fragment: str) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return next(row for row in manifest["structures"] if fragment in row["structure_id"])


class HarmonicCalculator(Calculator):
    """Synthetic isotropic harmonic well with an optional fixed Voigt stress."""

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, spring: float = 2.0, stress=None):
        super().__init__()
        self.spring = spring
        self.stress = np.zeros(6) if stress is None else np.asarray(stress, dtype=float)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = np.asarray(self.atoms.positions)
        self.results = {
            "energy": 0.5 * self.spring * float(np.sum(positions**2)),
            "forces": -self.spring * positions,
            "stress": self.stress.copy(),
        }


# Symmetric 3N x 3N Hessian for two atoms, with deliberate off-diagonal coupling
# between different atoms and between different Cartesian components.
COUPLED_HESSIAN = np.array(
    [
        [3.00, 0.40, -0.20, 0.10, 0.00, 0.05],
        [0.40, 2.50, 0.30, 0.00, 0.20, -0.10],
        [-0.20, 0.30, 4.00, 0.05, -0.15, 0.00],
        [0.10, 0.00, 0.05, 2.00, 0.25, 0.10],
        [0.00, 0.20, -0.15, 0.25, 3.50, -0.20],
        [0.05, -0.10, 0.00, 0.10, -0.20, 2.80],
    ]
)


class CoupledQuadraticCalculator(Calculator):
    """Synthetic anisotropic quadratic well E = 0.5 x^T H x with coupled modes."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, hessian=COUPLED_HESSIAN):
        super().__init__()
        self.hessian = np.asarray(hessian, dtype=float)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        flat = np.asarray(self.atoms.positions, dtype=float).reshape(-1)
        self.results = {
            "energy": 0.5 * float(flat @ self.hessian @ flat),
            "forces": (-(self.hessian @ flat)).reshape(-1, 3),
        }


class QuarticCalculator(Calculator):
    """Synthetic anharmonic well E = sum_i (k x_i^2 / 2 + c x_i^4).

    Its Taylor series terminates at fourth order, so the finite-displacement
    error of both curvature estimators is analytic rather than approximate.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, spring: float = 2.0, quartic: float = 6.0):
        super().__init__()
        self.spring = spring
        self.quartic = quartic

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = np.asarray(self.atoms.positions, dtype=float)
        self.results = {
            "energy": float(
                np.sum(0.5 * self.spring * positions**2 + self.quartic * positions**4)
            ),
            "forces": -(self.spring * positions + 4.0 * self.quartic * positions**3),
        }


DEFAULT_KPOINTS = """mesh
0
Gamma
1 1 1
0 0 0
"""


def _write_dft(
    run: Path,
    *,
    spring: float = 2.0,
    offset: float = 0.0,
    stress_kbar: tuple[float, ...] = (0.0,) * 6,
    kpoints: str = DEFAULT_KPOINTS,
) -> None:
    """Synthetic OUTCAR with analytic values, parsed through the real ASE reader."""
    atoms = read(run / "POSCAR")
    energy = 0.5 * spring * float(np.sum(atoms.positions**2)) + offset
    lines = [
        "vasp.6.5.1 synthetic regression fixture",
        "POTCAR: PAW_PBE Si 05Jan2001", "POTCAR: PAW_PBE Si 05Jan2001",
        "TITEL = PAW_PBE Si 05Jan2001", "ions per type = 2",
        "NKPTS = 1", "ISPIN = 1", "ENCUT = 520", "IBRION = -1", "NSW = 0", "ISYM = 0",
        "Iteration 1(1)", "aborting loop because EDIFF is reached",
        "in kB " + " ".join(f"{value:.8f}" for value in stress_kbar),
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
    (run / "KPOINTS").write_text(kpoints)
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


class DerivativeProbeMetricTests(unittest.TestCase):
    """Cross-probe input guards, per-metric summaries, curvature naming and export.

    Synthetic fixtures only: the DFT side is a hand-written OUTCAR and the MLIP
    side is an analytic ASE calculator.
    """

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _tree(self, *, sources: int = 1, strains=(0.0,), name: str = "probe") -> Path:
        entries = []
        for index in range(sources):
            label = f"src{index}"
            path = _structure(self.root / f"POSCAR_{label}", shift=0.1 * index)
            entries.append(f"{label}={path}")
        root = self.root / name
        prepare_derivative_probe(entries, root, strains=strains)
        return root

    def _write_all_dft(self, root: Path, **kwargs) -> None:
        for poscar in sorted(root.glob("*/*/POSCAR")):
            _write_dft(poscar.parent, offset=100.0, **kwargs)

    def _runs(self, root: Path, label: str) -> list[Path]:
        """Run directories for one source, in the manifest order evaluation uses."""
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        return [
            root / row["relative_directory"]
            for row in manifest["structures"]
            if row["label"] == label
        ]

    def _run(self, root: Path, label: str, fragment: str) -> Path:
        return next(path for path in self._runs(root, label) if fragment in path.name)

    def _responses(self, result: dict) -> list[dict]:
        with Path(result["outputs"]["responses"]).open(encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def _predictions(self, result: dict) -> list[dict]:
        with Path(result["outputs"]["predictions"]).open(encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    # --- Cross-probe input consistency ----------------------------------

    def test_missing_first_kpoints_does_not_hide_a_later_conflict(self) -> None:
        root = self._tree()
        self._write_all_dft(root)
        runs = self._runs(root, "src0")
        runs[0].joinpath("KPOINTS").unlink()
        runs[1].joinpath("KPOINTS").write_text("mesh A\n", encoding="utf-8")
        runs[2].joinpath("KPOINTS").write_text("mesh B\n", encoding="utf-8")

        with self.assertRaisesRegex(SafetyError, "KPOINTS differs across probes"):
            evaluate_derivative_probe(root)

    def test_missing_kpoints_is_a_warning_when_the_available_files_agree(self) -> None:
        root = self._tree()
        self._write_all_dft(root)
        self._runs(root, "src0")[0].joinpath("KPOINTS").unlink()

        result = evaluate_derivative_probe(
            root, _test_calculators={"harmonic": HarmonicCalculator()}
        )

        self.assertEqual(result["status"], "CHECK")
        self.assertEqual(
            sum("KPOINTS unavailable" in warning for warning in result["warnings"]), 1
        )
        self.assertEqual(result["summaries"][0]["matched_structures"], 3)

    def test_kpoints_conflict_is_rejected_whatever_the_probe_order(self) -> None:
        for position in (0, 1, 2):
            with self.subTest(differing_probe=position):
                root = self._tree(name=f"probe_{position}")
                self._write_all_dft(root)
                runs = self._runs(root, "src0")
                runs[position].joinpath("KPOINTS").write_text(
                    "a different mesh\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(SafetyError, "KPOINTS differs across probes"):
                    evaluate_derivative_probe(root)

    def test_source_labels_keep_independent_input_references(self) -> None:
        root = self._tree(sources=2)
        self._write_all_dft(root)
        for run in self._runs(root, "src1"):
            run.joinpath("KPOINTS").write_text(
                "a denser mesh for the second source\n", encoding="utf-8"
            )

        result = evaluate_derivative_probe(
            root, _test_calculators={"harmonic": HarmonicCalculator()}
        )

        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["dft_completed"], 6)

    # --- Per-metric availability ----------------------------------------

    def test_complete_run_reports_a_count_for_every_metric(self) -> None:
        root = self._tree()
        self._write_all_dft(root, stress_kbar=(1.0, 2.0, 3.0, 0.0, 0.0, 0.0))

        result = evaluate_derivative_probe(
            root, _test_calculators={"stiffer": HarmonicCalculator(3.0)}
        )

        summary = result["summaries"][0]
        for field, expected in (
            ("matched_structures", 3),
            ("force_matched_structures", 3),
            ("stress_matched_structures", 3),
            ("relative_energy_matched_structures", 3),
            ("matched_curvatures", 1),
            ("matched_energy_curvatures", 1),
            ("matched_force_curvatures", 1),
            ("force_components", 18),
        ):
            self.assertEqual(summary[field], expected, field)
        self.assertNotIn("omitted_metrics", summary)
        self.assertNotIn("metric_notes", summary)

    def test_missing_dft_center_keeps_force_and_stress_metrics(self) -> None:
        root = self._tree()
        self._write_all_dft(root, stress_kbar=(10.0, -20.0, 30.0, 1.5, -2.5, 3.5))
        self._run(root, "src0", "center").joinpath("OUTCAR").unlink()

        result = evaluate_derivative_probe(
            root, _test_calculators={"stiffer": HarmonicCalculator(3.0)}
        )

        self.assertEqual(result["status"], "INCOMPLETE")
        summary = result["summaries"][0]
        self.assertEqual(summary["matched_structures"], 2)
        self.assertEqual(summary["force_matched_structures"], 2)
        self.assertEqual(summary["stress_matched_structures"], 2)
        self.assertIn("force_ev_a", summary)
        self.assertIn("stress_ev_a3", summary)
        self.assertNotIn("relative_energy_mev_atom", summary)
        self.assertEqual(summary["relative_energy_matched_structures"], 0)
        self.assertIn(
            "zero-strain center has no matched DFT and model result",
            summary["omitted_metrics"]["relative_energy_mev_atom"],
        )
        for field in (
            "energy_curvature_ev_a2",
            "force_curvature_ev_a2",
            "directional_curvature_ev_a2",
        ):
            self.assertNotIn(field, summary)
        self.assertIn("minus/center/plus", summary["omitted_metrics"]["force_curvature_ev_a2"])
        self.assertEqual(summary["matched_curvatures"], 0)

    def test_probe_without_a_zero_strain_center_still_reports_derivatives(self) -> None:
        root = self._tree(strains=(0.01,))
        self._write_all_dft(root)

        result = evaluate_derivative_probe(
            root, _test_calculators={"stiffer": HarmonicCalculator(3.0)}
        )

        summary = result["summaries"][0]
        self.assertEqual(summary["force_matched_structures"], 3)
        self.assertNotIn("relative_energy_mev_atom", summary)
        self.assertIn(
            "no zero-strain center was prepared",
            summary["omitted_metrics"]["relative_energy_mev_atom"],
        )
        self.assertEqual(summary["matched_curvatures"], 1)
        self.assertIn("energy_curvature_ev_a2", summary)
        self.assertIn("force_curvature_ev_a2", summary)

    def test_partially_completed_source_does_not_drop_the_other_source(self) -> None:
        root = self._tree(sources=2)
        self._write_all_dft(root)
        self._run(root, "src1", "center").joinpath("OUTCAR").unlink()

        result = evaluate_derivative_probe(
            root, _test_calculators={"stiffer": HarmonicCalculator(3.0)}
        )

        summary = result["summaries"][0]
        self.assertEqual(summary["matched_structures"], 5)
        self.assertEqual(summary["force_matched_structures"], 5)
        self.assertEqual(summary["stress_matched_structures"], 5)
        self.assertEqual(summary["relative_energy_matched_structures"], 3)
        self.assertIn("relative_energy_mev_atom", summary)
        notes = summary["metric_notes"]["relative_energy_mev_atom"]
        self.assertEqual(len(notes), 1)
        self.assertIn("'src1'", notes[0])
        self.assertNotIn("omitted_metrics", summary)
        # Only the fully collected source still has a complete displacement triplet.
        self.assertEqual(summary["matched_curvatures"], 1)

    def test_model_without_stress_omits_only_the_stress_metric(self) -> None:
        root = self._tree()
        self._write_all_dft(root)

        result = evaluate_derivative_probe(
            root, _test_calculators={"coupled": CoupledQuadraticCalculator()}
        )

        summary = result["summaries"][0]
        self.assertNotIn("stress_ev_a3", summary)
        self.assertEqual(summary["stress_matched_structures"], 0)
        self.assertIn("stress", summary["omitted_metrics"]["stress_ev_a3"])
        self.assertIn("force_ev_a", summary)
        self.assertIn("relative_energy_mev_atom", summary)
        self.assertIn("force_curvature_ev_a2", summary)

    # --- Curvature naming ------------------------------------------------

    def test_curvature_estimators_are_named_and_the_alias_is_preserved(self) -> None:
        root = self._tree()
        self._write_all_dft(root)

        result = evaluate_derivative_probe(
            root, _test_calculators={"stiffer": HarmonicCalculator(3.0)}
        )

        summary = result["summaries"][0]
        # DFT spring 2.0 against model spring 3.0: both estimators are exact here.
        self.assertAlmostEqual(summary["energy_curvature_ev_a2"]["rmse"], 1.0)
        self.assertAlmostEqual(summary["force_curvature_ev_a2"]["rmse"], 1.0)
        self.assertEqual(
            summary["directional_curvature_ev_a2"], summary["force_curvature_ev_a2"]
        )
        self.assertIn("force_curvature_ev_a2", summary["curvature_alias"])
        definitions = result["curvature_definitions"]
        self.assertEqual(definitions, CURVATURE_DEFINITIONS)
        self.assertEqual(definitions["units"], "eV/Angstrom^2")
        self.assertIn("u^T H u", definitions["interpretation"])
        self.assertIn("not mass-weighted phonon frequencies", definitions["not_established"])

    # --- Targeted scientific checks --------------------------------------

    def _pair_geometry(self, root: Path, label: str, strain: float = 0.0):
        """Return (center positions, displacement d, |d|, unit direction u)."""
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        selected = {}
        for row in manifest["structures"]:
            if row["label"] != label or float(row["strain_fraction"]) != strain:
                continue
            selected[int(row["sign"])] = read(root / row["relative_directory"] / "POSCAR")
        displacement = (selected[1].positions - selected[-1].positions) / 2.0
        norm = float(np.linalg.norm(displacement))
        return selected[0].positions, displacement, norm, displacement.reshape(-1) / norm

    def test_coupled_quadratic_recovers_the_directional_hessian_projection(self) -> None:
        source = _structure(self.root / "POSCAR")
        root = self.root / "probe"
        prepare_derivative_probe([f"bulk={source}"], root)

        result = evaluate_derivative_probe(
            root, _test_calculators={"coupled": CoupledQuadraticCalculator()}
        )

        _, _, _, unit = self._pair_geometry(root, "bulk")
        expected = float(unit @ COUPLED_HESSIAN @ unit)
        diagonal_only = float(np.sum(np.diag(COUPLED_HESSIAN) * unit**2))
        # The off-diagonal Hessian terms must actually change the answer.
        self.assertGreater(abs(expected - diagonal_only), 1e-2)

        responses = self._responses(result)
        self.assertEqual(len(responses), 3)
        for row in responses:
            self.assertAlmostEqual(float(row["energy_curvature_ev_a2"]), expected, places=6)
            self.assertAlmostEqual(float(row["force_curvature_ev_a2"]), expected, places=9)

    def test_quartic_anharmonicity_has_the_expected_displacement_dependence(self) -> None:
        source = _structure(self.root / "POSCAR")
        spring, quartic = 2.0, 6.0
        calculator = QuarticCalculator(spring=spring, quartic=quartic)
        observed = {}
        for amplitude in (0.03, 0.015):
            root = self.root / f"probe_{amplitude}"
            prepare_derivative_probe(
                [f"bulk={source}"], root, strains=(0.0,), displacement_a=amplitude
            )
            result = evaluate_derivative_probe(
                root, _test_calculators={"quartic": calculator}
            )
            rows = self._responses(result)
            self.assertEqual(len(rows), 1)
            center, _, norm, unit = self._pair_geometry(root, "bulk")
            flat = center.reshape(-1)
            # Exact directional derivatives of sum_i (k x_i^2 / 2 + c x_i^4).
            second = float(np.sum((spring + 12.0 * quartic * flat**2) * unit**2))
            fourth = float(np.sum(24.0 * quartic * unit**4))
            energy_curvature = float(rows[0]["energy_curvature_ev_a2"])
            force_curvature = float(rows[0]["force_curvature_ev_a2"])
            # A quartic terminates the Taylor series, so these are equalities.
            self.assertAlmostEqual(
                energy_curvature, second + fourth * norm**2 / 12.0, places=6
            )
            self.assertAlmostEqual(
                force_curvature, second + fourth * norm**2 / 6.0, places=6
            )
            observed[amplitude] = (
                norm,
                energy_curvature - second,
                force_curvature - second,
                second,
            )

        for _norm, energy_bias, force_bias, second in observed.values():
            self.assertGreater(abs(energy_bias / second), 1e-6)
            # k_F carries exactly twice the leading finite-displacement bias of k_E.
            self.assertAlmostEqual(force_bias / energy_bias, 2.0, places=4)

        large, small = observed[0.03], observed[0.015]
        ratio = (large[0] / small[0]) ** 2
        self.assertAlmostEqual(ratio, 4.0, places=6)
        # Both estimators converge quadratically in the displacement amplitude.
        self.assertAlmostEqual(large[1] / small[1], ratio, places=4)
        self.assertAlmostEqual(large[2] / small[2], ratio, places=4)

    def test_nonzero_stress_ordering_units_and_error_aggregation(self) -> None:
        # VASP prints "in kB XX YY ZZ XY YZ ZX"; ASE returns Voigt xx yy zz yz xz xy.
        kbar = (10.0, -20.0, 30.0, 1.5, -2.5, 3.5)
        expected = (
            -np.array([kbar[0], kbar[1], kbar[2], kbar[4], kbar[5], kbar[3]])
            * 1e-1
            * units.GPa
        )
        model_stress = np.array([0.01, -0.02, 0.03, 0.004, -0.005, 0.006])
        root = self._tree()
        self._write_all_dft(root, stress_kbar=kbar)

        result = evaluate_derivative_probe(
            root, _test_calculators={"fixed": HarmonicCalculator(stress=model_stress)}
        )

        dft_rows = [row for row in self._predictions(result) if row["model"] == "DFT"]
        self.assertEqual(len(dft_rows), 3)
        for row in dft_rows:
            np.testing.assert_allclose(
                [float(row[f"stress_{index}_ev_a3"]) for index in range(6)],
                expected,
                rtol=1e-12,
            )
        # Compression in VASP's kB convention becomes a negative ASE stress.
        self.assertLess(expected[0], 0.0)

        summary = result["summaries"][0]
        self.assertEqual(summary["stress_matched_structures"], 3)
        error = np.tile(model_stress - expected, 3)
        self.assertAlmostEqual(summary["stress_ev_a3"]["mae"], float(np.mean(np.abs(error))))
        self.assertAlmostEqual(
            summary["stress_ev_a3"]["rmse"], float(np.sqrt(np.mean(error**2)))
        )
        self.assertAlmostEqual(
            summary["stress_ev_a3"]["max_abs"], float(np.max(np.abs(error)))
        )
        self.assertAlmostEqual(summary["stress_ev_a3"]["bias"], float(np.mean(error)))

    # --- Mocked DeePMD adapter -------------------------------------------

    def test_mocked_deepmd_model_is_constructed_and_recorded(self) -> None:
        """Mocked backend: no deepmd-kit is imported or executed."""
        root = self._tree()
        self._write_all_dft(root)
        model_dir = self.root / "dpa3_000"
        model_dir.mkdir()
        model = model_dir / "frozen_model.pth"
        model.write_bytes(b"synthetic deepmd model identity")
        factory = MagicMock(side_effect=lambda **kwargs: HarmonicCalculator())
        module = types.ModuleType("deepmd.calculator")
        module.DP = factory

        with patch.dict(
            sys.modules,
            {"deepmd": types.ModuleType("deepmd"), "deepmd.calculator": module},
        ):
            result = evaluate_derivative_probe(root, deepmd_models=[model])

        self.assertEqual(factory.call_args.kwargs, {"model": str(model)})
        self.assertEqual(result["models"], ["DFT", "deepmd:dpa3_000/frozen_model"])
        record = result["provenance"]["models"][0]
        self.assertEqual(record["backend"], "deepmd")
        self.assertEqual(record["path"], str(model))
        self.assertEqual(len(record["sha256"]), 64)
        self.assertEqual(result["provenance"]["deepmd_device"], "backend-managed")
        self.assertEqual(result["summaries"][0]["model"], "deepmd:dpa3_000/frozen_model")

    def test_missing_deepmd_backend_is_reported_as_a_dependency_error(self) -> None:
        model = self.root / "frozen_model.pth"
        model.write_bytes(b"synthetic deepmd model identity")
        with patch.dict(sys.modules, {"deepmd": None, "deepmd.calculator": None}):
            with self.assertRaisesRegex(DependencyError, "deepmd-kit is unavailable"):
                _calculators([], [model], "cpu")

    def test_missing_deepmd_model_file_is_rejected(self) -> None:
        module = types.ModuleType("deepmd.calculator")
        module.DP = MagicMock()
        with patch.dict(
            sys.modules,
            {"deepmd": types.ModuleType("deepmd"), "deepmd.calculator": module},
        ):
            with self.assertRaisesRegex(SafetyError, "Missing DeePMD model"):
                _calculators([], [self.root / "absent.pth"], "cpu")

    # --- Reproducible force export ---------------------------------------

    def test_exported_arrays_reproduce_the_reported_force_metrics(self) -> None:
        root = self._tree(sources=2)
        self._write_all_dft(root, stress_kbar=(5.0, 5.0, 5.0, 0.0, 0.0, 0.0))

        result = evaluate_derivative_probe(
            root, _test_calculators={"stiffer": HarmonicCalculator(3.0)}
        )

        export = result["force_export"]
        path = Path(export["path"])
        self.assertEqual(path, Path(result["outputs"]["arrays"]))
        self.assertEqual(path.name, "derivative_probe_arrays.npz")
        self.assertEqual(export["units"]["forces"], "eV/Angstrom")
        self.assertEqual(export["units"]["energy"], "eV")
        self.assertIn("without allow_pickle", export["format"])
        self.assertEqual(export["sha256"], _sha256(path))
        self.assertEqual(len(export["entries"]), 12)

        forces: dict[tuple[str, str], np.ndarray] = {}
        with np.load(path, allow_pickle=False) as data:
            self.assertEqual(data["forces_units"].item(), "eV/Angstrom")
            self.assertEqual(data["schema"].item(), FORCE_EXPORT_SCHEMA)
            models = [str(value) for value in data["model"]]
            identifiers = [str(value) for value in data["structure_id"]]
            for entry in export["entries"]:
                index = entry["index"]
                self.assertEqual(models[index], entry["model"])
                self.assertEqual(identifiers[index], entry["structure_id"])
                array = data[entry["forces_key"]]
                self.assertEqual(list(array.shape), entry["forces_shape"])
                self.assertEqual(entry["forces_shape"], [entry["natoms"], 3])
                forces[(entry["model"], entry["structure_id"])] = array

        summary = result["summaries"][0]
        shared = sorted(
            {sid for model, sid in forces if model == "DFT"}
            & {sid for model, sid in forces if model == "stiffer"}
        )
        self.assertEqual(len(shared), summary["force_matched_structures"])
        reference = np.concatenate([forces[("DFT", sid)].reshape(-1) for sid in shared])
        predicted = np.concatenate([forces[("stiffer", sid)].reshape(-1) for sid in shared])
        error = predicted - reference
        self.assertEqual(error.size, summary["force_components"])
        recomputed = {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "max_abs": float(np.max(np.abs(error))),
            "bias": float(np.mean(error)),
        }
        for key, value in recomputed.items():
            self.assertAlmostEqual(value, summary["force_ev_a"][key], places=12, msg=key)

    def test_exported_arrays_follow_the_output_stem(self) -> None:
        root = self._tree()
        self._write_all_dft(root)
        result = evaluate_derivative_probe(
            root,
            output_stem="mace_float32",
            _test_calculators={"harmonic": HarmonicCalculator()},
        )
        self.assertEqual(
            result["outputs"]["arrays"], str(root / "mace_float32_arrays.npz")
        )
        self.assertTrue((root / "mace_float32_arrays.npz").is_file())
        self.assertFalse((root / "derivative_probe_arrays.npz").exists())


if __name__ == "__main__":
    unittest.main()
