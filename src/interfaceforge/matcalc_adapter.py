"""MatCalc-specific code for ``iface properties``; nothing else imports MatCalc.

Each function takes an ASE ``Atoms`` plus an already-constructed ASE
calculator and returns a plain, JSON-ready dictionary with an explicit
``units`` map. The InterfaceForge-facing shape of these dictionaries is the
stable contract; the MatCalc calls behind them may change between releases.

Behaviour of MatCalc 0.5.1 that shapes this adapter:

* ``RelaxCalc`` reports ``is_converged`` from the per-atom forces only. With
  ``relax_cell=True`` the optimizer also has to drive the cell (stress)
  degrees of freedom below ``fmax``; if it hit ``max_steps`` the cell may be
  unconverged even though ``is_converged`` is True. The step count is not
  returned, so it is recovered from the relaxation trajectory, and the
  relaxation only counts as converged when both criteria hold.
* ``EOSCalc`` returns the scan and ``bulk_modulus_bm`` but not E0/V0/B'; they
  are obtained from the same Birch-Murnaghan fit through pymatgen's public
  EOS API. The strained-point relaxations inside the scan are not
  convergence-checked by MatCalc.
* ``EOSCalc`` scans *linear* strain on an evenly spaced grid, so the default
  ``max_abs_strain=0.1`` spans V/V0 = 0.729-1.331. That is far into the
  anharmonic regime of stiff covalent solids and biases B' low; the adapter
  records the realized window and refits the near-equilibrium points.
* ``ElasticityCalc`` defaults to normal strains of +/-0.5/1 % but shear
  strains of +/-3/6 % (pymatgen deformation-gradient components, so the
  engineering shear strain equals the value). A shear window that wide picks
  up the nonlinear stress response of soft-shear solids (on a MACE rocksalt
  TiN it put C44 ~15 GPa above the finite-difference elastic limit), so the
  adapter always passes the strain windows explicitly and records them.
* ``ElasticityCalc`` also defaults to ``relax_deformed_structures=False``:
  CLAMPED-ION constants, with the atoms moved affinely with the cell. That is
  only right when no atom has a free internal coordinate. For beta-Si3N4 it
  made C44/C66/G about 45 % too stiff, and C12/C13 too low, in every model.
  So the adapter always passes the ion treatment explicitly. With relaxed
  ions MatCalc relaxes each deformed cell at fixed shape and keeps only the
  final structure. The adapter records each of those relaxations (converged
  or not, final max force) by swapping in a recording ``RelaxCalc`` for the
  duration of the call.
* ``PhononCalc`` writes ``phonon.yaml`` into the current directory by
  default; the adapter always points it at the member directory.
* Every downstream calc is run with ``relax_structure=False`` on the
  structure from the explicit, convergence-checked relaxation, so one
  relaxed structure feeds all properties of a member.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from .errors import DependencyError

INSTALL_HINT = (
    "MatCalc property validation needs the optional 'properties' extra, which requires Python >= 3.11: "
    "pip install -e '.[properties]'"
)

EV_PER_A3_TO_GPA = 160.21766208

PHONON_IMAGINARY_TOL_THZ = -0.01


def require_matcalc() -> Any:
    """Import MatCalc lazily, raising ``DependencyError`` with an install hint."""

    try:
        import matcalc
    except ImportError as exc:
        raise DependencyError(INSTALL_HINT) from exc
    return matcalc


def _atoms(structure: Any) -> Any:
    from matcalc.utils import to_ase_atoms

    return to_ase_atoms(structure)


_LIBRARY_NOISE = (DeprecationWarning, PendingDeprecationWarning, FutureWarning)


def _caught(record: list[warnings.WarningMessage]) -> list[str]:
    """Unique, scientifically relevant warnings (e.g. a low EOS R^2); library deprecations are dropped."""

    messages = [
        f"{item.category.__name__}: {item.message}" for item in record if not issubclass(item.category, _LIBRARY_NOISE)
    ]
    return list(dict.fromkeys(messages))


def _trajectory_steps(path: Path) -> int | None:
    if not path.is_file():
        return None
    from ase.io.trajectory import Trajectory

    with Trajectory(str(path)) as trajectory:
        frames = len(trajectory)
    # The optimizer observer writes the starting geometry as frame 0.
    return max(frames - 1, 0)


def relax(
    atoms: Any, calculator: Any, *, fmax: float, max_steps: int, optimizer: str, member_dir: Path
) -> dict[str, Any]:
    """Full atoms+cell relaxation. Returns the record and the relaxed ``Atoms``."""

    from matcalc import RelaxCalc

    trajectory = member_dir / "relax.traj"
    relaxer = RelaxCalc(
        calculator,
        optimizer=optimizer,
        fmax=fmax,
        max_steps=max_steps,
        relax_atoms=True,
        relax_cell=True,
        traj_file=str(trajectory),
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = relaxer.calc(atoms.copy())
    n_steps = _trajectory_steps(trajectory)
    force_converged = bool(result["is_converged"])
    hit_step_cap = n_steps is not None and n_steps >= max_steps
    stress_gpa = np.asarray(result["stress"], dtype=float) * EV_PER_A3_TO_GPA
    relaxed = _atoms(result["final_structure"])
    n_atoms = len(relaxed)
    record_out = {
        "converged": force_converged and not hit_step_cap,
        "matcalc_is_converged": force_converged,
        "hit_max_steps": hit_step_cap,
        "n_steps": n_steps,
        "n_steps_source": "relax.traj frames - 1",
        "max_force": float(result["max_force"]),
        "fmax": float(fmax),
        "max_steps": int(max_steps),
        "optimizer": str(optimizer),
        "cell_filter": "FrechetCellFilter",
        "relax_cell": True,
        "energy": float(result["energy"]),
        "energy_per_atom": float(result["energy"]) / n_atoms,
        "volume": float(result["volume"]),
        "volume_per_atom": float(result["volume"]) / n_atoms,
        "a": float(result["a"]),
        "b": float(result["b"]),
        "c": float(result["c"]),
        "alpha": float(result["alpha"]),
        "beta": float(result["beta"]),
        "gamma": float(result["gamma"]),
        "final_stress": stress_gpa,
        "max_abs_final_stress": float(np.abs(stress_gpa).max()),
        "warnings": _caught(record),
        "units": {
            "max_force": "eV/A",
            "fmax": "eV/A",
            "energy": "eV",
            "energy_per_atom": "eV/atom",
            "volume": "A^3",
            "volume_per_atom": "A^3/atom",
            "a": "A",
            "b": "A",
            "c": "A",
            "alpha": "degree",
            "beta": "degree",
            "gamma": "degree",
            "final_stress": "GPa",
            "max_abs_final_stress": "GPa",
        },
    }
    return {"record": record_out, "atoms": relaxed}


EOS_REFIT_MIN_POINTS = 5


def _birch_murnaghan(volumes: np.ndarray, energies: np.ndarray, n_atoms: int) -> dict[str, Any]:
    from pymatgen.analysis.eos import BirchMurnaghan

    fit = BirchMurnaghan(volumes=volumes, energies=energies)
    fit.fit()
    residual = energies - np.asarray(fit.func(volumes), dtype=float)
    spread = float(np.sum((energies - energies.mean()) ** 2))
    return {
        "equilibrium_energy": float(fit.e0),
        "equilibrium_energy_per_atom": float(fit.e0) / n_atoms,
        "equilibrium_volume": float(fit.v0),
        "equilibrium_volume_per_atom": float(fit.v0) / n_atoms,
        "bulk_modulus": float(fit.b0_GPa),
        "bulk_modulus_derivative": float(fit.b1),
        "r2": 1.0 - float(np.sum(residual**2)) / spread if spread > 0 else float("nan"),
    }


def _near_equilibrium_refit(
    volumes: np.ndarray, energies: np.ndarray, reference_volume: float, window: float, n_atoms: int
) -> dict[str, Any]:
    """Birch-Murnaghan refit of the scan points with |V/V_ref - 1| <= window."""

    inside = np.abs(volumes / reference_volume - 1.0) <= window + 1e-12
    base = {
        "volume_window": float(window),
        "window_reference": "relaxed input volume (scan centre)",
        "n_points": int(inside.sum()),
        "min_points": EOS_REFIT_MIN_POINTS,
    }
    if inside.sum() < EOS_REFIT_MIN_POINTS:
        return {
            **base,
            "status": "skipped",
            "reason": (
                f"only {int(inside.sum())} scan point(s) within +/-{window:.0%} of the reference volume; "
                f"need {EOS_REFIT_MIN_POINTS} (use a smaller --eos-max-strain or more --eos-points)"
            ),
        }
    try:
        fit = _birch_murnaghan(volumes[inside], energies[inside], n_atoms)
    except Exception as exc:  # noqa: BLE001 - a failed refit must not fail the full-scan result
        return {**base, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    return {**base, "status": "ok", **fit}


def eos(
    atoms: Any,
    calculator: Any,
    *,
    fmax: float,
    max_steps: int,
    optimizer: str,
    max_abs_strain: float = 0.1,
    n_points: int = 11,
    refit_volume_window: float = 0.05,
) -> dict[str, Any]:
    from matcalc import EOSCalc

    calc = EOSCalc(
        calculator,
        optimizer=optimizer,
        fmax=fmax,
        max_steps=max_steps,
        max_abs_strain=max_abs_strain,
        n_points=n_points,
        relax_structure=False,
    )
    reference_volume = float(atoms.get_volume())
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = calc.calc(atoms.copy())
    volumes = np.asarray(result["eos"]["volumes"], dtype=float)
    energies = np.asarray(result["eos"]["energies"], dtype=float)
    order = np.argsort(volumes)
    volumes, energies = volumes[order], energies[order]
    n_atoms = len(atoms)
    fit = _birch_murnaghan(volumes, energies, n_atoms)
    ratios = volumes / reference_volume
    return {
        "fit": "birch_murnaghan",
        **fit,
        # MatCalc's own B and R^2 for the same scan; kept as the headline values for parity.
        "bulk_modulus": float(result["bulk_modulus_bm"]),
        "r2": float(result["r2_score_bm"]),
        "volumes": volumes,
        "energies": energies,
        "n_points": int(len(volumes)),
        "scan_window": {
            "max_abs_linear_strain": float(max_abs_strain),
            "n_points_requested": int(n_points),
            "reference_volume": reference_volume,
            "min_volume_ratio": float(ratios.min()),
            "max_volume_ratio": float(ratios.max()),
            "nominal_min_volume_ratio": (1.0 - max_abs_strain) ** 3,
            "nominal_max_volume_ratio": (1.0 + max_abs_strain) ** 3,
        },
        "near_equilibrium_refit": _near_equilibrium_refit(
            volumes, energies, reference_volume, refit_volume_window, n_atoms
        ),
        "strained_relaxations_convergence_checked": False,
        "warnings": _caught(record),
        "units": {
            "equilibrium_energy": "eV",
            "equilibrium_energy_per_atom": "eV/atom",
            "equilibrium_volume": "A^3",
            "equilibrium_volume_per_atom": "A^3/atom",
            "bulk_modulus": "GPa",
            "bulk_modulus_derivative": "dimensionless",
            "r2": "dimensionless",
            "volumes": "A^3",
            "energies": "eV",
            "scan_window.max_abs_linear_strain": "dimensionless (linear strain)",
            "scan_window.reference_volume": "A^3",
            "scan_window.*_volume_ratio": "V / reference_volume",
            "near_equilibrium_refit.volume_window": "fraction of reference_volume",
            "near_equilibrium_refit.*": "as the full-scan fields of the same name",
        },
    }


@contextmanager
def _recorded_deformed_relaxations() -> Iterator[list[dict[str, Any]] | None]:
    """Record every fixed-cell ``RelaxCalc`` that ``ElasticityCalc`` runs on a deformed structure.

    MatCalc 0.5.1 builds that relaxer from the ``RelaxCalc`` name in ``matcalc._elasticity`` and keeps only
    ``final_structure``. Yields ``None`` when that hook is missing, so the convergence is then reported as unchecked.
    """

    from matcalc import _elasticity

    base = getattr(_elasticity, "RelaxCalc", None)
    if not isinstance(base, type):
        yield None
        return
    records: list[dict[str, Any]] = []

    class RecordingRelaxCalc(base):  # type: ignore[misc, valid-type]
        def calc(self, structure: Any) -> dict[str, Any]:
            result = super().calc(structure)
            records.append({"is_converged": bool(result["is_converged"]), "max_force": float(result["max_force"])})
            return result

    _elasticity.RelaxCalc = RecordingRelaxCalc
    try:
        yield records
    finally:
        _elasticity.RelaxCalc = base


def _deformed_relaxation_record(
    records: list[dict[str, Any]] | None, n_structures: int, *, fmax: float, max_steps: int, optimizer: str
) -> dict[str, Any]:
    base = {"n_structures": int(n_structures), "fmax": float(fmax), "max_steps": int(max_steps),
            "optimizer": str(optimizer), "relax_cell": False}
    if records is None or len(records) != n_structures:
        return {
            **base,
            "convergence_checked": False,
            "converged": None,
            "reason": "this MatCalc version does not expose the deformed-structure relaxations"
            if records is None
            else f"recorded {len(records)} relaxations for {n_structures} deformed structures",
        }
    forces = [item["max_force"] for item in records]
    n_converged = sum(item["is_converged"] for item in records)
    return {
        **base,
        "convergence_checked": True,
        "converged": n_converged == n_structures,
        "n_converged": int(n_converged),
        "max_force": float(max(forces)) if forces else None,
        "per_structure_max_force": forces,
        # With the cell fixed, RelaxCalc's is_converged (max|F| <= fmax) is the complete criterion.
        "criterion": "max|F| <= fmax at fixed deformed cell (RelaxCalc.is_converged)",
    }


def elasticity(
    atoms: Any,
    calculator: Any,
    *,
    fmax: float,
    norm_strains: tuple[float, ...] = (-0.01, -0.005, 0.005, 0.01),
    shear_strains: tuple[float, ...] = (-0.06, -0.03, 0.03, 0.06),
    ion_relaxation: str = "relaxed",
    max_steps: int = 500,
    optimizer: str = "FIRE",
) -> dict[str, Any]:
    """Elastic tensor; ``ion_relaxation`` is ``"relaxed"`` (relax each deformed cell to ``fmax``) or ``"clamped"``."""

    from matcalc import ElasticityCalc

    if ion_relaxation not in ("relaxed", "clamped"):
        raise ValueError(f"ion_relaxation must be 'relaxed' or 'clamped', not {ion_relaxation!r}")
    relax_ions = ion_relaxation == "relaxed"
    calc = ElasticityCalc(
        calculator,
        norm_strains=tuple(float(value) for value in norm_strains),
        shear_strains=tuple(float(value) for value in shear_strains),
        fmax=fmax,
        relax_structure=False,
        relax_deformed_structures=relax_ions,
        relax_calc_kwargs={"max_steps": int(max_steps), "optimizer": optimizer} if relax_ions else None,
        units_GPa=True,
    )
    # symmetry=False (the default) deforms along every Voigt component: 6 x (normal or shear) strains.
    n_deformed = 3 * len(calc.norm_strains) + 3 * len(calc.shear_strains)
    with warnings.catch_warnings(record=True) as record, _recorded_deformed_relaxations() as relaxations:
        warnings.simplefilter("always")
        result = calc.calc(atoms.copy())
    tensor = result["elastic_tensor"]
    voigt = np.asarray(getattr(tensor, "voigt", tensor), dtype=float)
    deformed = (
        _deformed_relaxation_record(relaxations, n_deformed, fmax=fmax, max_steps=max_steps, optimizer=optimizer)
        if relax_ions
        else None
    )
    return {
        "elastic_tensor": voigt,
        "tensor_convention": "voigt_6x6_pymatgen_order_xx_yy_zz_yz_xz_xy",
        "ion_relaxation": ion_relaxation,
        "deformed_relaxations": deformed,
        "bulk_modulus_vrh": float(result["bulk_modulus_vrh"]),
        "shear_modulus_vrh": float(result["shear_modulus_vrh"]),
        "youngs_modulus": float(result["youngs_modulus"]),
        "residuals_sum": float(result["residuals_sum"]),
        "r2_score_mean": float(result["r2_score_mean"]) if "r2_score_mean" in result else None,
        "strains": {
            "norm_strains": [float(value) for value in calc.norm_strains],
            "shear_strains": [float(value) for value in calc.shear_strains],
            "convention": "pymatgen DeformedStructureSet deformation-gradient components "
                          "(shear value = engineering shear strain)",
        },
        "warnings": _caught(record),
        "units": {
            "strains": "dimensionless",
            "r2_score_mean": "dimensionless",
            "deformed_relaxations.fmax": "eV/A",
            "deformed_relaxations.max_force": "eV/A",
            "deformed_relaxations.per_structure_max_force": "eV/A",
            "elastic_tensor": "GPa",
            "bulk_modulus_vrh": "GPa",
            "shear_modulus_vrh": "GPa",
            "youngs_modulus": "GPa",
            "residuals_sum": "GPa",
        },
    }


def phonon(atoms: Any, calculator: Any, *, min_length: float, member_dir: Path) -> dict[str, Any]:
    from matcalc import PhononCalc

    calc = PhononCalc(
        calculator,
        min_length=min_length,
        relax_structure=False,
        on_imaginary_modes="warn",
        imaginary_freq_tol=PHONON_IMAGINARY_TOL_THZ,
        write_phonon=str(member_dir / "phonon.yaml"),
        write_force_constants=False,
        write_band_structure=False,
        write_total_dos=False,
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = calc.calc(atoms.copy())
    frequencies = np.asarray(result["frequencies"], dtype=float)
    thermal = {key: np.asarray(value, dtype=float) for key, value in result["thermal_properties"].items()}
    supercell = np.asarray(result["phonon"].supercell_matrix, dtype=int)
    imaginary = frequencies < PHONON_IMAGINARY_TOL_THZ
    return {
        "frequencies": frequencies,
        "frequencies_shape": list(frequencies.shape),
        "frequencies_layout": "phonopy mesh: [n_qpoints, n_bands]",
        "min_frequency": float(frequencies.min()),
        "n_imaginary_modes": int(imaginary.sum()),
        "imaginary_tolerance": PHONON_IMAGINARY_TOL_THZ,
        "dynamically_stable": not bool(imaginary.any()),
        "supercell_matrix": supercell,
        "min_length": float(min_length),
        "n_displacements": int(len(result["disp_supercells"])),
        "thermal_properties": thermal,
        "phonopy_file": "phonon.yaml",
        "warnings": _caught(record),
        "units": {
            "frequencies": "THz",
            "min_frequency": "THz",
            "imaginary_tolerance": "THz",
            "min_length": "A",
            "thermal_properties.temperatures": "K",
            "thermal_properties.free_energy": "kJ/mol",
            "thermal_properties.entropy": "J/(K*mol)",
            "thermal_properties.heat_capacity": "J/(K*mol)",
        },
    }
