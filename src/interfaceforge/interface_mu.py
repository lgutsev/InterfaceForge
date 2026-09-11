# ruff: noqa: E501
"""Grand-canonical interfacial energy gamma(mu) for vacuum-free periodic cells.

A polar interface is generally not an integer count of bulk formula units, so a
plain bulk-referenced excess is undefined: the leftover anions get valued at a
full bulk per-formula-unit energy and the answer is off by J/m^2. The standard
fix is to let the leftover exchange with a reservoir and report gamma as a
function of the anion chemical potential -- Hao, Delley, Veprek & Stampfl,
PRL 97, 086102 (2006) do exactly this for TiN(111)/Si3N4, with the mu <-> (T, p)
mapping of Reuter & Scheffler, PRB 65, 035406 (2002).

For an interface cell of composition ``{n_i}`` held in equilibrium with compound
reference phases ``C`` (each one cation species plus the shared anion X)::

    x_C   = n_cation(C) / a_C                     formula units of C in the cell
    dn    = n_X - sum_C x_C * b_C                 anion excess (0 => stoichiometric)
    gamma(dmu_X) = [ E_int - sum_C x_C g_C - dn * mu_X^0 ] / (n_interfaces * A)
                   - dn * dmu_X / (n_interfaces * A)

so gamma is a straight line in ``dmu_X = mu_X - mu_X^0`` whose **slope is exactly
the stoichiometric imbalance**. A balanced cell has slope zero and gamma collapses
to a single chemical-potential-independent number.

``mu_X^0`` is the anion-rich limit (half the energy of the X2 molecule for N or O),
and the window is bounded below by precipitation of each elemental cation phase::

    dmu_X >= max_C [ dH_f(C) / b_C ]        dmu_X <= 0

Only the interface cells and the *solid* compound references are ever handed to an
MLIP: the molecular anion reference is a vacuum structure, and a bulk-trained model
has no business being asked about it. It enters both the DFT and the MLIP gamma as
the same DFT constant, so it cancels in ``gamma^MLIP - gamma^DFT``.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Collection, Mapping, Sequence
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np

from .config import merge_interface_metadata
from .errors import DependencyError, SafetyError
from .phase_diagram import (
    MOLECULAR_MOMENT_PER_PAIR,
    MOLECULAR_REFERENCES,
    hull_report,
)
from .regime import BULK, FREE_SURFACE, MLIP_DOMAIN, measure_regime, require_regime
from .separation_energy import (
    EV_A2_TO_J_M2,
    _composition,
    _deepmd_energies,
    _dft_energy,
    _mace_energies,
    _plane_area,
    _read_atoms,
    _structure_file,
)

QUANTITY = "interface_energy_mu"


def parse_named_entry(item: str, what: str) -> tuple[str, str]:
    """``NAME=DIR`` -> (name, dir), with a message naming what was expected."""

    name, sep, directory = item.partition("=")
    if not sep or not name.strip() or not directory.strip():
        raise SafetyError(f"{what} expects NAME=DIR; got {item!r}")
    return name.strip(), directory.strip()


def _formula(composition: Mapping[str, int]) -> str:
    return "".join(f"{el}{composition[el]}" for el in sorted(composition))


#: VASP prints the electron count and, with ISPIN=2, the cell moment after it.
#: With ISPIN=1 the line simply ends, which is what a missing match means.
_OUTCAR_MOMENT = re.compile(
    "number of electron[ ]+[-+0-9.Ee]+[ ]+magnetization[ ]+([-+]?[0-9][0-9.Ee+-]*)"
)


def total_magnetization(run: Path) -> float | None:
    """Total cell moment (muB) from the last SCF step of an OUTCAR.

    ``None`` means the run was not spin-polarised: with ISPIN=1 VASP prints the
    electron count with no magnetization after it.
    """

    outcar = run / "OUTCAR"
    if not outcar.is_file():
        return None
    found = _OUTCAR_MOMENT.findall(
        outcar.read_text(encoding="utf-8", errors="ignore")
    )
    return float(found[-1]) if found else None


def audit_molecular_spin(
    name: str, run: Path, composition: Mapping[str, int], *, strict: bool = True
) -> dict[str, Any]:
    """Check a diatomic molecular reference carries its ground-state spin.

    The failure this exists for is a non-spin-polarised O2: it costs ~1 eV in
    mu_O, and because mu_O is the zero of the chemical-potential scale that error
    lands on every gamma, formation enthalpy and window downstream without
    anything else looking wrong.
    """

    element = next(iter(composition))
    natoms = int(composition[element])
    measured = total_magnetization(run)
    per_pair = MOLECULAR_MOMENT_PER_PAIR.get(element)
    row: dict[str, Any] = {
        "phase": name,
        "element": element,
        "natoms": natoms,
        "total_moment_mub": measured,
        "spin_polarised": measured is not None,
    }
    if per_pair is None or natoms % 2 or natoms < 2:
        return {
            **row,
            "status": "NOT_CHECKED",
            "expected_moment_mub": None,
            "note": (
                f"no ground-state moment on record for an isolated {element} "
                f"reference of {natoms} atom(s); check the multiplicity yourself"
            ),
        }
    expected = per_pair * natoms / 2
    row["expected_moment_mub"] = expected
    if expected == 0.0:
        status = "PASS" if measured is None or abs(measured) < 0.1 else "CHECK"
        note = (
            f"{element}2 is closed-shell, so ISPIN=1 (or ISPIN=2 converging to 0) "
            "is correct"
            if status == "PASS"
            else f"{element}2 should be closed-shell but the cell carries "
            f"{measured:.2f} muB; the singlet did not converge"
        )
        return {**row, "status": status, "note": note}
    if measured is None or abs(measured) < 0.5 * expected:
        message = (
            f"molecular reference {name!r} in {run}: expected {expected:.1f} muB "
            f"({element}2 is a triplet) but the run is "
            + (
                "not spin-polarised at all"
                if measured is None
                else f"at {measured:.2f} muB"
            )
            + f". A non-spin-polarised {element}2 is ~1 eV too high, and mu_{element}"
            " is the zero of the chemical-potential scale, so that error lands on"
            " every gamma and every window derived from it. Rerun with "
            f"{MOLECULAR_REFERENCES.get(element + '2', 'ISPIN=2, NUPDOWN=2, ISYM=0')}."
        )
        if strict:
            raise SafetyError(message)
        return {
            **row,
            "status": "OVERRIDDEN",
            "note": (
                f"accepted under --allow-spin-mismatch: mu_{element} is ~1 eV too "
                f"high, so every gamma and window below is shifted with it"
            ),
            "detail": message,
        }
    status = "PASS" if abs(abs(measured) - expected) < 0.1 else "CHECK"
    return {
        **row,
        "status": status,
        "note": (
            f"{element}2 triplet converged to {measured:.2f} muB"
            if status == "PASS"
            else f"{element}2 should carry {expected:.1f} muB but converged to "
            f"{measured:.2f}; the multiplicity is only partly resolved"
        ),
    }


def _read_phase(
    name: str,
    directory: str | Path,
    *,
    allow_vacuum: bool,
    strict_spin: bool = True,
) -> dict[str, Any]:
    """Composition and total DFT energy of one reference phase run directory."""

    run = Path(directory).expanduser().resolve()
    atoms = _read_atoms(_structure_file(run))
    composition = _composition(atoms)
    energy = _dft_energy(run)
    if energy is None:
        raise SafetyError(
            f"reference phase {name!r}: no finished DFT energy in {run} "
            "(needs an OUTCAR from a completed run)"
        )
    # An elemental reference that sits in vacuum is the molecular reservoir
    # (N2, O2): it is exempt from the bulk-regime guard and is never handed to
    # an MLIP. Anything else must be a real vacuum-free periodic cell.
    probe = measure_regime(atoms)
    molecular = len(composition) == 1 and probe["regime"] == FREE_SURFACE
    regime = probe if molecular else require_regime(
        atoms, BULK, f"reference phase {name!r}", allow_mismatch=allow_vacuum
    )
    return {
        "name": name,
        "directory": str(run),
        "composition": composition,
        "formula": _formula(composition),
        "natoms": int(sum(composition.values())),
        "energy_ev": float(energy),
        "molecular": molecular,
        "spin": (
            audit_molecular_spin(name, run, composition, strict=strict_spin)
            if molecular
            else None
        ),
        "regime": regime,
        "atoms": atoms,
    }


def _classify_phases(
    phases: Mapping[str, dict[str, Any]],
    anion: str,
    auxiliary_names: Collection[str] = (),
) -> dict[str, Any]:
    """Split references into compounds, elemental cations, and the anion reference.

    ``auxiliary_names`` forces a phase onto the hull without making it an
    interface constituent. A competing nitride such as Ti2N contains the anion
    and one cation, so it *looks* like a compound reference, but the interface is
    not made of it: treating it as one would demand it coexist with TiN and would
    double-count that cation in :func:`decompose`.
    """

    unknown = [name for name in auxiliary_names if name not in phases]
    if unknown:
        raise SafetyError(f"--aux-phase names a phase that was not supplied: {unknown}")
    compounds: dict[str, dict[str, Any]] = {}
    elemental: dict[str, dict[str, Any]] = {}
    auxiliary: dict[str, dict[str, Any]] = {}
    anion_ref: dict[str, Any] | None = None
    for name, phase in phases.items():
        elements = sorted(phase["composition"])
        if name in auxiliary_names:
            if elements == [anion]:
                raise SafetyError(
                    f"the elemental {anion} reference ({name!r}) cannot be auxiliary; "
                    "it sets the zero of the chemical-potential scale"
                )
            auxiliary[name] = phase
        elif elements == [anion]:
            if anion_ref is not None:
                raise SafetyError(
                    f"two elemental {anion} references given "
                    f"({anion_ref['name']!r} and {name!r})"
                )
            anion_ref = phase
        elif len(elements) == 1:
            if elements[0] in elemental:
                raise SafetyError(
                    f"two elemental {elements[0]} references given "
                    f"({elemental[elements[0]]['name']!r} and {name!r}); keep the ground "
                    "state as the reference and pass the other with --aux-phase"
                )
            elemental[elements[0]] = phase
        elif anion in elements:
            cations = [el for el in elements if el != anion]
            if len(cations) != 1:
                raise SafetyError(
                    f"reference compound {name!r} ({phase['formula']}) has "
                    f"{len(cations)} cation species; gamma(mu) needs one cation per "
                    "compound so its formula-unit count is determined"
                )
            clash = next(
                (other for other, existing in compounds.items()
                 if existing["cation"] == cations[0]), None
            )
            if clash is not None:
                raise SafetyError(
                    f"reference compounds {clash!r} and {name!r} share the cation "
                    f"{cations[0]}; the interface cannot be decomposed into both "
                    f"(every {cations[0]} atom would be counted twice). Keep the one "
                    "the interface is made of and pass the competing phase with "
                    "--aux-phase, which puts it on the hull without making it a "
                    "constituent."
                )
            compounds[name] = {**phase, "cation": cations[0]}
        else:
            # A phase with no anion (a silicide, an intermetallic) cannot be a
            # decomposition reference, but it belongs on the convex hull: it may
            # be what actually cuts the chemical-potential window.
            auxiliary[name] = phase
    if not compounds:
        raise SafetyError(f"no compound reference phase containing {anion} was given")
    if anion_ref is None:
        raise SafetyError(
            f"no elemental {anion} reference given (e.g. --phase N2=<N2 molecule run>); "
            "it sets the anion-rich limit of the chemical-potential window"
        )
    return {
        "compounds": compounds,
        "elemental": elemental,
        "auxiliary": auxiliary,
        "anion_ref": anion_ref,
    }


def _units(phase: Mapping[str, Any], anion: str) -> tuple[float, float, float]:
    """(cations per f.u., anions per f.u., energy per f.u.) for a compound phase."""

    composition = phase["composition"]
    divisor = 0
    for value in composition.values():
        divisor = gcd(divisor, value)
    divisor = divisor or 1
    return (
        composition[phase["cation"]] / divisor,
        composition[anion] / divisor,
        phase["energy_ev"] / divisor,
    )


def chemical_potential_window(classified: Mapping[str, Any], anion: str) -> dict[str, Any]:
    """Allowed ``dmu_anion`` range and the formation enthalpy that bounds it."""

    anion_ref = classified["anion_ref"]
    mu0 = anion_ref["energy_ev"] / anion_ref["natoms"]
    bounds: list[dict[str, Any]] = []
    for name, phase in classified["compounds"].items():
        cation = phase["cation"]
        elemental = classified["elemental"].get(cation)
        if elemental is None:
            bounds.append(
                {
                    "compound": name,
                    "cation": cation,
                    "bound_ev": None,
                    "status": f"no elemental {cation} reference; this bound is not applied",
                }
            )
            continue
        a, b, g = _units(phase, anion)
        formation = g - a * (elemental["energy_ev"] / elemental["natoms"]) - b * mu0
        bounds.append(
            {
                "compound": name,
                "cation": cation,
                "formation_enthalpy_ev_per_fu": formation,
                "anions_per_fu": b,
                "bound_ev": formation / b,
                "status": "OK",
            }
        )
    applied = [item["bound_ev"] for item in bounds if item.get("bound_ev") is not None]
    lower = max(applied) if applied else None
    binding = None
    if lower is not None:
        binding = next(item["compound"] for item in bounds if item.get("bound_ev") == lower)
    return {
        "anion": anion,
        "anion_reference": anion_ref["name"],
        "mu_anion_reference_ev": mu0,
        "dmu_min_ev": lower,
        "dmu_max_ev": 0.0,
        "binding_compound": binding,
        "bounds": bounds,
        "note": (
            f"dmu = mu({anion}) - mu0 with mu0 from {anion_ref['name']}; upper limit 0 "
            f"({anion}-rich, elemental {anion} condenses), lower limit set by "
            "precipitation of an elemental cation phase"
        ),
    }



def _hull_window(
    phases: Mapping[str, Any], classified: Mapping[str, Any], anion: str
) -> dict[str, Any]:
    """Chemical-potential window from the convex hull, shaped like the pairwise one."""

    serialisable = {
        name: {"composition": phase["composition"], "energy_ev": phase["energy_ev"]}
        for name, phase in phases.items()
    }
    report = hull_report(serialisable, list(classified["compounds"]), anion)
    window = report["chemical_potential_window"]
    return {
        "anion": anion,
        "method": "convex-hull",
        "anion_reference": classified["anion_ref"]["name"],
        "mu_anion_reference_ev": window["mu_anion_reference_ev"],
        "dmu_min_ev": window["dmu_min_ev"],
        "dmu_max_ev": window["dmu_max_ev"],
        "binding_compound": window["dmu_min_set_by"],
        "upper_bound_set_by": window["dmu_max_set_by"],
        "bounds": window["per_compound"],
        "solver": window.get("solver"),
        "endpoint_dmu_ev": window.get("endpoint_dmu_ev"),
        "competing_stable_phases": window["competing_stable_phases"],
        # surfaced here as well as under "hull": an omitted stable phase widens
        # the window, so it must not be a key the reader has to go looking for
        "missing_known_phases": [
            f"{row['formula']} ({row['mp_id']})" for row in report["missing_known_phases"]
        ],
        "hull": {k: v for k, v in report.items() if k != "chemical_potential_window"},
        "note": window["note"],
    }


def decompose(
    interface_composition: Mapping[str, int],
    classified: Mapping[str, Any],
    anion: str,
) -> dict[str, Any]:
    """Formula units of each compound in the cell, and the leftover anion count."""

    cations = {phase["cation"] for phase in classified["compounds"].values()}
    unmatched = [
        el for el in interface_composition if el != anion and el not in cations
    ]
    if unmatched:
        raise SafetyError(
            f"interface contains {unmatched} with no matching reference compound; "
            "supply one --phase per cation species"
        )
    units: dict[str, float] = {}
    accounted = 0.0
    for name, phase in classified["compounds"].items():
        cation = phase["cation"]
        if cation not in interface_composition:
            raise SafetyError(
                f"interface has no {cation}; reference compound {name!r} cannot be matched"
            )
        a, b, _ = _units(phase, anion)
        x = interface_composition[cation] / a
        units[name] = x
        accounted += x * b
    excess = interface_composition.get(anion, 0) - accounted
    return {
        "formula_units": {name: round(value, 6) for name, value in units.items()},
        "anion": anion,
        "anion_actual": interface_composition.get(anion, 0),
        "anion_from_compounds": round(accounted, 6),
        "anion_excess": round(excess, 6),
        "stoichiometric": abs(excess) < 1.0e-6,
    }


def gamma_line(
    energy_ev: float,
    decomposition: Mapping[str, Any],
    classified: Mapping[str, Any],
    window: Mapping[str, Any],
    denom: float,
    anion: str,
) -> dict[str, float]:
    """Intercept (at dmu=0) and slope of gamma(dmu), both in J/m^2."""

    reference = 0.0
    for name, x in decomposition["formula_units"].items():
        reference += x * _units(classified["compounds"][name], anion)[2]
    excess = float(decomposition["anion_excess"])
    intercept = (
        energy_ev - reference - excess * float(window["mu_anion_reference_ev"])
    ) / denom * EV_A2_TO_J_M2
    return {
        "gamma0_j_per_m2": intercept,
        "slope_j_per_m2_per_ev": -excess / denom * EV_A2_TO_J_M2,
    }


def gamma_at(line: Mapping[str, float], dmu: float) -> float:
    return float(line["gamma0_j_per_m2"] + line["slope_j_per_m2_per_ev"] * dmu)


def _family_lines(
    members: Mapping[str, Mapping[str, float]],
    keys: Mapping[str, str],
    decompositions: Mapping[str, Any],
    classified: Mapping[str, Any],
    window: Mapping[str, Any],
    denominators: Mapping[str, float],
    anion: str,
    dft_lines: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Per-member gamma(dmu) lines for one MLIP family, plus committee statistics."""

    out: dict[str, Any] = {}
    for label in keys:
        per_member = {}
        for member, energies in members.items():
            solid = {
                name: energies[f"phase::{name}"]
                for name in classified["compounds"]
            }
            local = {
                "compounds": {
                    name: {**classified["compounds"][name], "energy_ev": solid[name]}
                    for name in classified["compounds"]
                }
            }
            per_member[member] = gamma_line(
                energies[f"iface::{label}"],
                decompositions[label],
                local,
                window,
                denominators[label],
                anion,
            )
        gamma0 = np.asarray([line["gamma0_j_per_m2"] for line in per_member.values()])
        block: dict[str, Any] = {
            "members": len(per_member),
            "gamma0_members_j_per_m2": {
                name: float(line["gamma0_j_per_m2"]) for name, line in per_member.items()
            },
            "gamma0_ensemble_j_per_m2": float(gamma0.mean()),
            "committee_spread_j_per_m2": float(gamma0.std(ddof=1)) if gamma0.size > 1 else 0.0,
            "slope_j_per_m2_per_ev": next(iter(per_member.values()))["slope_j_per_m2_per_ev"],
        }
        block["delta_vs_dft_j_per_m2"] = (
            block["gamma0_ensemble_j_per_m2"] - dft_lines[label]["gamma0_j_per_m2"]
        )
        out[label] = block
    return out


def interface_mu(
    entries: Sequence[tuple[str, str | Path]],
    *,
    phases: Mapping[str, str | Path],
    auxiliary_phases: Mapping[str, str | Path] | None = None,
    anion: str = "N",
    mace_models: Sequence[str] = (),
    deepmd_models: Sequence[str] = (),
    n_interfaces: int | None = None,
    interface_metadata: Sequence[Mapping[str, Any]] | None = None,
    interfaces_equivalent: bool | None = None,
    area_axis: str | None = None,
    device: str = "cpu",
    allow_vacuum: bool = False,
    allow_spin_mismatch: bool = False,
    window_method: str = "hull",
) -> dict[str, Any]:
    """gamma(dmu_anion) for one or more vacuum-free periodic interface cells."""

    if not entries:
        raise SafetyError("interface-mu needs at least one (label, directory) entry")
    if n_interfaces is not None and (type(n_interfaces) is not int or n_interfaces < 1):
        raise SafetyError("n_interfaces must be a positive integer")
    if interfaces_equivalent is not None and type(interfaces_equivalent) is not bool:
        raise SafetyError("interfaces_equivalent must be a boolean")
    labels = [label for label, _ in entries]
    if len(set(labels)) != len(labels):
        raise SafetyError(f"duplicate interface label: {labels}")

    auxiliary_phases = dict(auxiliary_phases or {})
    overlap = sorted(set(auxiliary_phases) & set(phases))
    if overlap:
        raise SafetyError(f"{overlap} given as both --phase and --aux-phase")
    read = {
        name: _read_phase(
            name, directory,
            allow_vacuum=allow_vacuum, strict_spin=not allow_spin_mismatch,
        )
        for name, directory in {**phases, **auxiliary_phases}.items()
    }
    classified = _classify_phases(read, anion, auxiliary_names=auxiliary_phases)
    if window_method not in {"hull", "pairwise"}:
        raise SafetyError("window_method must be 'hull' or 'pairwise'")
    if window_method == "hull":
        window = _hull_window(read, classified, anion)
    else:
        window = chemical_potential_window(classified, anion)
    window.setdefault("method", "pairwise-formation-enthalpy")

    rows: list[dict[str, Any]] = []
    atoms_by_key: dict[str, Any] = {
        f"phase::{name}": phase["atoms"] for name, phase in classified["compounds"].items()
    }
    decompositions: dict[str, Any] = {}
    denominators: dict[str, float] = {}
    dft_lines: dict[str, dict[str, float]] = {}

    for label, directory in entries:
        run = Path(directory).expanduser().resolve()
        atoms = _read_atoms(_structure_file(run))
        meta = merge_interface_metadata(interface_metadata, label)
        count = n_interfaces if n_interfaces is not None else meta.get("n_interfaces")
        if type(count) is not int or count < 1:
            raise SafetyError(
                f"interface {label!r}: specify a positive --n-interfaces or "
                "validation.interfaces n_interfaces matched to the entry label"
            )
        effective_axis = area_axis or meta.get("stacking_axis")
        equivalent = interfaces_equivalent if interfaces_equivalent is not None else meta.get("interfaces_equivalent")
        if equivalent is not None and type(equivalent) is not bool:
            raise SafetyError("interfaces_equivalent metadata must be a boolean")
        interpretation = (
            "single-interface energy" if count == 1 else
            "per-interface energy (equivalence declared by user)" if equivalent else
            "average over interfaces; individual termination energies are unresolved"
        )
        regime = require_regime(
            atoms, BULK, f"interface {label!r}", axis=effective_axis or "auto",
            allow_mismatch=allow_vacuum,
        )
        composition = _composition(atoms)
        area, axis = _plane_area(np.array(atoms.cell.array, dtype=float), effective_axis)
        denom = count * area
        decomposition = decompose(composition, classified, anion)
        energy = _dft_energy(run)
        row: dict[str, Any] = {
            "label": label,
            "directory": str(run),
            "composition": composition,
            "formula": _formula(composition),
            "interface_area_ang2": area,
            "area_axis": axis,
            "n_interfaces": count,
            "n_interfaces_source": "explicit" if n_interfaces is not None else "campaign-metadata",
            "interfaces_equivalent": equivalent,
            "energy_interpretation": interpretation,
            "normalization_area_ang2": denom,
            "regime": regime,
            "decomposition": decomposition,
            "dft": {"ready": energy is not None, "energy_ev": energy},
            "mlip": {},
        }
        if energy is not None:
            line = gamma_line(energy, decomposition, classified, window, denom, anion)
            row["dft"].update(line)
            row["dft"]["gamma_anion_rich_j_per_m2"] = gamma_at(line, window["dmu_max_ev"])
            if window["dmu_min_ev"] is not None:
                row["dft"]["gamma_anion_poor_j_per_m2"] = gamma_at(line, window["dmu_min_ev"])
            dft_lines[label] = line
        atoms_by_key[f"iface::{label}"] = atoms
        decompositions[label] = decomposition
        denominators[label] = denom
        rows.append(row)

    ready = {label: line for label, line in dft_lines.items()}
    if (mace_models or deepmd_models) and len(ready) != len(rows):
        raise SafetyError(
            "every interface needs a finished DFT energy before an MLIP comparison; "
            f"missing: {[row['label'] for row in rows if not row['dft']['ready']]}"
        )
    keys = {label: label for label in labels}
    families: dict[str, dict[str, Any]] = {}
    if mace_models:
        families["mace"] = _family_lines(
            _mace_energies(mace_models, atoms_by_key, device),
            keys, decompositions, classified, window, denominators, anion, ready,
        )
    if deepmd_models:
        families["deepmd"] = _family_lines(
            _deepmd_energies(deepmd_models, atoms_by_key),
            keys, decompositions, classified, window, denominators, anion, ready,
        )
    for row in rows:
        for family, blocks in families.items():
            row["mlip"][family] = blocks[row["label"]]

    unbalanced = [row["label"] for row in rows if not row["decomposition"]["stoichiometric"]]
    return {
        "schema_version": 1,
        "quantity": QUANTITY,
        "regime": BULK,
        "mlip_validity_domain": MLIP_DOMAIN[BULK],
        "definition": (
            "gamma(dmu) = [E_int - sum_C x_C g_C - dn * mu0] / (n_interfaces * A) "
            "- dn * dmu / (n_interfaces * A); dn is the anion excess, so the slope is "
            "the stoichiometric imbalance and a balanced cell is dmu-independent"
        ),
        "anion": anion,
        "n_interfaces": n_interfaces,
        "conversion_ev_a2_to_j_m2": EV_A2_TO_J_M2,
        "chemical_potential_window": window,
        "window_method": window["method"],
        "reference_phases": {
            name: {k: v for k, v in phase.items() if k != "atoms"}
            for name, phase in read.items()
        },
        "mlip_evaluated_on": sorted(atoms_by_key),
        "mlip_note": (
            "MLIPs are evaluated only on the periodic interface cells and the solid "
            "compound references. The molecular anion reference is a vacuum structure "
            "and stays DFT-only; it enters DFT and MLIP gamma identically and cancels "
            "in the difference."
        ),
        "mace_models": [str(p) for p in mace_models],
        "deepmd_models": [str(p) for p in deepmd_models],
        "chemical_potential_dependent": sorted(unbalanced),
        "interfaces": rows,
    }


_CSV_FIELDS = (
    "interface", "formula", "source", "stoichiometric", "anion_excess",
    "n_interfaces", "interface_area_ang2", "energy_interpretation",
    "gamma0_j_per_m2", "slope_j_per_m2_per_ev", "gamma_anion_rich_j_per_m2",
    "gamma_anion_poor_j_per_m2", "committee_spread_j_per_m2", "delta_vs_dft_j_per_m2",
)

_SOURCE_COLORS = {"dft": "#111111", "mace": "#0072B2", "deepmd": "#D55E00"}


def _rows_for_csv(payload: dict[str, Any]) -> list[dict[str, Any]]:
    window = payload["chemical_potential_window"]
    low = window["dmu_min_ev"]

    def _pair(line: Mapping[str, float]) -> dict[str, Any]:
        return {
            "gamma0_j_per_m2": line["gamma0_j_per_m2"],
            "slope_j_per_m2_per_ev": line["slope_j_per_m2_per_ev"],
            "gamma_anion_rich_j_per_m2": gamma_at(line, window["dmu_max_ev"]),
            "gamma_anion_poor_j_per_m2": gamma_at(line, low) if low is not None else None,
        }

    out: list[dict[str, Any]] = []
    for row in payload["interfaces"]:
        base = {
            "interface": row["label"],
            "n_interfaces": row["n_interfaces"],
            "interface_area_ang2": row["interface_area_ang2"],
            "energy_interpretation": row["energy_interpretation"],
            "formula": row["formula"],
            "stoichiometric": row["decomposition"]["stoichiometric"],
            "anion_excess": row["decomposition"]["anion_excess"],
        }
        if row["dft"]["ready"]:
            out.append({**base, "source": "dft", **_pair(row["dft"])})
        for family, block in row["mlip"].items():
            line = {
                "gamma0_j_per_m2": block["gamma0_ensemble_j_per_m2"],
                "slope_j_per_m2_per_ev": block["slope_j_per_m2_per_ev"],
            }
            out.append({
                **base, "source": family, **_pair(line),
                "committee_spread_j_per_m2": block["committee_spread_j_per_m2"],
                "delta_vs_dft_j_per_m2": block.get("delta_vs_dft_j_per_m2"),
            })
    return out


def _write_figure(payload: dict[str, Any], out: Path) -> dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "The gamma(mu) figure requires matplotlib; install interfaceforge[report]"
        ) from exc

    window = payload["chemical_potential_window"]
    anion = payload["anion"]
    low = window["dmu_min_ev"]
    if low is None:
        raise SafetyError(
            "no chemical-potential window; give one elemental reference per cation"
        )
    rows = [row for row in payload["interfaces"] if row["dft"]["ready"] or row["mlip"]]
    if not rows:
        raise SafetyError("no interface has a finished energy to plot")
    grid = np.linspace(low, window["dmu_max_ev"], 64)
    styles = ["-", "--", "-.", ":"]
    seen: set[str] = set()

    with plt.rc_context({
        "font.family": "sans-serif", "font.size": 8.0, "axes.titlesize": 9.0,
        "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "axes.linewidth": 0.7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }):
        fig, ax = plt.subplots(figsize=(5.4, 4.0), layout="constrained")
        for index, row in enumerate(rows):
            style = styles[index % len(styles)]
            if row["dft"]["ready"]:
                seen.add("dft")
                ax.plot(grid, [gamma_at(row["dft"], x) for x in grid], style,
                        color=_SOURCE_COLORS["dft"], lw=1.4, zorder=3)
            for family, block in row["mlip"].items():
                seen.add(family)
                line = {
                    "gamma0_j_per_m2": block["gamma0_ensemble_j_per_m2"],
                    "slope_j_per_m2_per_ev": block["slope_j_per_m2_per_ev"],
                }
                values = np.asarray([gamma_at(line, x) for x in grid])
                spread = float(block["committee_spread_j_per_m2"])
                colour = _SOURCE_COLORS.get(family, "#4B5563")
                ax.plot(grid, values, style, color=colour, lw=1.2, alpha=0.95, zorder=2)
                if spread:
                    ax.fill_between(grid, values - spread, values + spread,
                                    color=colour, alpha=0.16, lw=0, zorder=1)
        ax.set_xlim(low, window["dmu_max_ev"])
        ax.set_xlabel(r"$\Delta\mu_{\mathrm{" + anion + r"}}$ (eV)")
        ax.set_ylabel(r"Mean $\gamma_{\mathrm{int}}$ (J m$^{-2}$)")
        ax.set_title("Grand-canonical interfacial energy", loc="left", fontweight="bold")
        ax.grid(color="#D1D5DB", linewidth=0.45, alpha=0.75)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        poor = anion + "-poor (" + str(window["binding_compound"]) + " limit)"
        ax.annotate(poor, xy=(low, 0.0), xycoords=("data", "axes fraction"),
                    xytext=(5, 5), textcoords="offset points", fontsize=7,
                    color="#6B7280", ha="left", va="bottom")
        ax.annotate(anion + "-rich", xy=(window["dmu_max_ev"], 0.0), xycoords=("data", "axes fraction"),
                    xytext=(-5, 5), textcoords="offset points", fontsize=7,
                    color="#6B7280", ha="right", va="bottom")
        names = {"dft": "DFT", "mace": "MACE committee", "deepmd": "DeePMD committee"}
        handles = [
            Line2D([0], [0], color=_SOURCE_COLORS.get(s, "#4B5563"), lw=1.5,
                   label=names.get(s, s))
            for s in sorted(seen)
        ]
        handles += [
            Line2D([0], [0], color="#4B5563", lw=1.2,
                   linestyle=styles[i % len(styles)], label=row["label"])
            for i, row in enumerate(rows)
        ]
        fig.legend(handles=handles, loc="outside lower center",
                   ncols=min(len(handles), 3), frameon=False,
                   handlelength=1.8, columnspacing=1.1)
        paths = {
            "figure_png": out / "interface_mu.png",
            "figure_svg": out / "interface_mu.svg",
            "figure_pdf": out / "interface_mu.pdf",
        }
        for path in paths.values():
            fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return {key: str(value) for key, value in paths.items()}


def write_reports(payload: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "interface_mu.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    rows = _rows_for_csv(payload)
    with (out / "interface_mu.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    window = payload["chemical_potential_window"]
    anion = payload["anion"]
    low = window["dmu_min_ev"]
    window_line = (
        "Window: {:.3f} <= dmu({}) <= {:.3f} eV, lower bound by {}; mu0({}) = {:.4f} eV from {}.".format(
            low, anion, window["dmu_max_ev"], window["binding_compound"], anion,
            window["mu_anion_reference_ev"], window["anion_reference"],
        )
        if low is not None
        else "Window: not bounded (no elemental cation reference was supplied)."
    )
    competing = window.get("competing_stable_phases") or []
    if competing:
        window_line += " Competing phases on the hull: {}.".format(", ".join(competing))
    spin = next(
        (
            phase["spin"] for phase in payload["reference_phases"].values()
            if phase.get("spin") and phase["spin"]["element"] == anion
        ),
        None,
    )
    spin_line = (
        "Anion reference {}: {} (expected {} muB) -- {}. {}".format(
            spin["phase"],
            "not spin-polarised" if spin["total_moment_mub"] is None
            else "{:.2f} muB".format(spin["total_moment_mub"]),
            "n/a" if spin["expected_moment_mub"] is None
            else "{:.1f}".format(spin["expected_moment_mub"]),
            spin["status"],
            spin["note"],
        )
        if spin
        else None
    )
    if spin and spin["status"] not in {"PASS", "NOT_CHECKED"}:
        spin_line = "> **WARNING** " + spin_line
    missing = window.get("missing_known_phases") or []
    # the warning has to be in the human-readable report, not only the JSON: the
    # window above reads as an answer, and with a phase missing it is a bound
    incomplete_line = (
        "> **WARNING: upper limit, not the window.** The hull was built without {}. "
        "An omitted stable phase can only make the window look too wide, so "
        "compute these at the campaign's settings and re-run before quoting a "
        "gamma at either end.".format(", ".join(missing))
        if missing
        else None
    )
    lines = [
        "# Grand-canonical interfacial energy",
        "",
        "**Regime: {} (vacuum-free).** MLIP validity: {}.".format(
            payload["regime"], payload["mlip_validity_domain"]
        ),
        "",
        payload["definition"],
        "",
        window_line,
        "",
        *[f"{row['label']}: n_interfaces={row['n_interfaces']}, "
          f"A={row['interface_area_ang2']:.4f} A^2; {row['energy_interpretation']}."
          for row in payload["interfaces"]],
        "",
        *((incomplete_line, "") if incomplete_line else ()),
        *((spin_line, "") if spin_line else ()),
        f"| Interface | Source | dn({anion}) | gamma {anion}-rich | gamma {anion}-poor | slope | committee sigma | delta vs DFT |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    def _cell(row: Mapping[str, Any], key: str, fmt: str = "{:.3f}") -> str:
        value = row.get(key)
        return "-" if value is None else fmt.format(value)

    for row in rows:
        lines.append(
            "| {} | {} | {:+.3f} | {} | {} | {} | {} | {} |".format(
                row["interface"], row["source"], row["anion_excess"],
                _cell(row, "gamma_anion_rich_j_per_m2"),
                _cell(row, "gamma_anion_poor_j_per_m2"),
                _cell(row, "slope_j_per_m2_per_ev"),
                _cell(row, "committee_spread_j_per_m2"),
                _cell(row, "delta_vs_dft_j_per_m2", "{:+.4f}"),
            )
        )
    if payload["chemical_potential_dependent"]:
        lines += [
            "",
            "> Not stoichiometric, so gamma genuinely depends on dmu({}): {}. Compare these "
            "over the whole window. Crossings compare complete cells; assigning a preferred "
            "individual termination requires equivalent interfaces in each cell. "
            "A cell with dn = 0 has zero slope and one well-defined gamma.".format(
                anion, ", ".join(payload["chemical_potential_dependent"])
            ),
        ]
    lines += ["", "MLIP note: " + payload["mlip_note"]]
    (out / "interface_mu.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    outputs = {
        "json": str(out / "interface_mu.json"),
        "csv": str(out / "interface_mu.csv"),
        "markdown": str(out / "interface_mu.md"),
    }
    try:
        outputs.update(_write_figure(payload, out))
    except (DependencyError, SafetyError) as exc:
        outputs["figure"] = f"skipped: {exc}"
    return outputs
