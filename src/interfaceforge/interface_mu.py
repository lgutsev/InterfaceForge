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
import math
import re
from collections.abc import Collection, Mapping, Sequence
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np

from .config import merge_interface_metadata
from .dft_evidence import audit_provenance, structure_evidence
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
    _dft_record,
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


K_B_EV_PER_K = 8.617333262e-5
"""Boltzmann constant, for estimating how much thermal energy a snapshot carries."""


def audit_thermal_state(name: str, run: Path, natoms: int) -> dict[str, Any]:
    """Is this a 0 K energy, or one snapshot of a finite-temperature trajectory?

    ``gamma`` subtracts bulk reference energies from the interface energy, so the
    two have to describe the same thermodynamic state. A relaxed reference is at
    its 0 K minimum; an MD snapshot sits above its own minimum by roughly
    ``(3/2) N k_B T`` (equipartition puts half the thermal energy in the
    potential). Subtract one from the other and that whole offset lands in
    ``gamma``, scaled by the cell size rather than by anything physical.
    """

    from .vasp import parse_incar

    parsed = parse_incar(run / "INCAR")
    ibrion = parsed.get("IBRION")
    nsw = parsed.get("NSW")

    def _int(value: str | None) -> int | None:
        try:
            return int(float(str(value).split()[0]))
        except (TypeError, ValueError, IndexError):
            return None

    def _float(value: str | None) -> float | None:
        try:
            return float(str(value).split()[0])
        except (TypeError, ValueError, IndexError):
            return None

    ibrion_value = _int(ibrion)
    nsw_value = _int(nsw)
    tebeg = _float(parsed.get("TEBEG"))
    molecular_dynamics = ibrion_value == 0 and (nsw_value or 0) > 0
    temperature = tebeg if molecular_dynamics else None
    if molecular_dynamics:
        state = "molecular-dynamics"
    elif ibrion_value in {1, 2, 3} and (nsw_value or 0) > 0:
        state = "relaxation"
    elif ibrion_value == -1 or nsw_value in {0, None}:
        state = "static"
    else:
        state = "unknown"
    offset = (
        1.5 * natoms * K_B_EV_PER_K * temperature
        if molecular_dynamics and temperature
        else None
    )
    return {
        "name": name,
        "state": state,
        "ibrion": ibrion_value,
        "nsw": nsw_value,
        "tebeg_k": tebeg,
        "thermostat": parsed.get("SMASS") or parsed.get("MDALGO"),
        "zero_kelvin": state in {"static", "relaxation"},
        "estimated_thermal_energy_ev": offset,
        "note": (
            f"IBRION=0 with NSW={nsw_value}: this is one frame of an MD "
            f"trajectory at {temperature:.0f} K, carrying roughly "
            f"{offset:.2f} eV of thermal energy above its own 0 K minimum "
            f"((3/2) N k_B T for N={natoms})"
            if molecular_dynamics and offset is not None
            else f"{state} run: a 0 K energy"
            if state in {"static", "relaxation"}
            else f"could not tell the thermodynamic state from IBRION={ibrion!r}, "
            f"NSW={nsw!r}; check it by hand"
        ),
    }


def check_thermal_consistency(
    interfaces: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
    denominators: Mapping[str, float],
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Refuse an MD interface energy measured against 0 K bulk references."""

    hot = [row for row in interfaces if row["state"] == "molecular-dynamics"]
    cold_references = [row for row in references if row["zero_kelvin"]]
    payload: dict[str, Any] = {
        "interfaces": list(interfaces),
        "reference_phases": list(references),
        "consistent": not (hot and cold_references),
    }
    if not hot or not cold_references:
        payload["note"] = (
            "interface and reference energies describe the same thermodynamic "
            "state"
            if payload["consistent"]
            else "could not establish the thermodynamic state of every run"
        )
        return payload
    worst = []
    for row in hot:
        offset = row.get("estimated_thermal_energy_ev")
        denominator = denominators.get(row["name"])
        if offset is None or not denominator:
            continue
        worst.append((row["name"], offset / denominator * EV_A2_TO_J_M2))
    estimate = (
        "; ".join(f"{name}: about {value:.2f} J/m^2 of it" for name, value in worst)
        if worst
        else "magnitude not estimable"
    )
    message = (
        "thermodynamic state mismatch: {} came from molecular dynamics while the "
        "reference phases {} are 0 K energies. gamma subtracts one from the other, "
        "so the snapshot's thermal energy lands in gamma as a spurious positive "
        "offset that scales with cell size, not with the interface ({}). Either "
        "use relaxed 0 K interface cells, or MD-average both sides consistently "
        "with `iface validate interface-energy`. Pass allow_thermal_mismatch to "
        "proceed anyway and keep the warning in the report.".format(
            [row["name"] for row in hot],
            [row["name"] for row in cold_references],
            estimate,
        )
    )
    if strict:
        raise SafetyError(message)
    payload["note"] = message
    payload["status"] = "OVERRIDDEN"
    payload["estimated_offset_j_per_m2"] = dict(worst)
    return payload


CANCELLATION_WARN = 0.5
"""Below this, |offset| is small only because larger terms cancelled."""


def summarize_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    """One place to read the verdict, and an ordered cascade of what to do.

    Built from the payload alone, so a downstream step derives its message from
    this object rather than restating the diagnosis. Hints are ordered
    most-specific-cause-first and each names a concrete next action; a passing
    run still gets its evidence published, and an unrun check reports
    NOT_CHECKED rather than being elided into PASS.
    """

    checks: dict[str, str] = {}
    hints: list[str] = []

    evidence = [phase.get("dft_evidence", {}) for phase in payload.get("reference_phases", {}).values()]
    evidence += [row.get("dft_evidence", {}) for row in payload.get("interfaces", [])]
    for key, component in (("dft_run_health", "run"), ("outcar_composition", "composition")):
        states = [item.get(component, {}).get("status", "NOT_CHECKED") for item in evidence]
        checks[key] = "CHECK" if "CHECK" in states else "NOT_CHECKED" if not states or "NOT_CHECKED" in states else "PASS"
    provenance = payload.get("vasp_provenance_audit") or {}
    checks["vasp_provenance"] = provenance.get("status", "NOT_CHECKED")
    if any(checks[key] != "PASS" for key in checks):
        hints.append("DFT evidence needs review before attributing an offset to MLIP training: "
                     "inspect each structure's dft_evidence and vasp_provenance_audit.")
        hints.extend(provenance.get("issues", []))

    thermal = payload.get("thermal_consistency") or {}
    if thermal.get("status") == "OVERRIDDEN":
        checks["thermal_state"] = "OVERRIDDEN"
        offsets = thermal.get("estimated_offset_j_per_m2") or {}
        worst = max(offsets.values()) if offsets else None
        hints.append(
            "An MD interface energy was accepted against 0 K references"
            + (f" (about {worst:.2f} J/m^2 of spurious gamma)" if worst else "")
            + ". The MLIP offset below is unaffected -- both legs are evaluated "
            "on the same structure -- but no ABSOLUTE gamma from this run is "
            "publishable. Relax the cells, or MD-average both sides with "
            "`iface validate interface-energy`."
        )
    elif thermal:
        checks["thermal_state"] = "PASS" if thermal.get("consistent") else "CHECK"

    spins = [
        phase["spin"] for phase in payload.get("reference_phases", {}).values()
        if phase.get("spin")
    ]
    if spins:
        worst_spin = next(
            (row for row in spins if row["status"] not in {"PASS", "NOT_CHECKED"}),
            None,
        )
        checks["molecular_reference_spin"] = (
            worst_spin["status"] if worst_spin
            else ("NOT_CHECKED" if all(r["status"] == "NOT_CHECKED" for r in spins)
                  else "PASS")
        )
        if worst_spin:
            hints.append(
                f"The {worst_spin['element']} reference is {worst_spin['status']}: "
                f"{worst_spin['note']} mu0 is the zero of the chemical-potential "
                "scale, so this shifts every gamma and the whole window with it."
            )
    else:
        checks["molecular_reference_spin"] = "NOT_CHECKED"

    missing = (payload.get("chemical_potential_window") or {}).get(
        "missing_known_phases"
    ) or []
    checks["hull_completeness"] = "CHECK" if missing else "PASS"
    if missing:
        hints.append(
            f"The hull was built without {list(missing)}, so the window is an "
            "upper limit rather than the window. This does not touch the MLIP "
            "offset, which is independent of the window."
        )

    families = 0
    for row in payload.get("interfaces", []):
        for family, block in (row.get("mlip") or {}).items():
            families += 1
            label = row["label"]
            reconstruction = block.get("offset_reconstruction") or {}
            ratio = reconstruction.get("cancellation_ratio")
            terms = reconstruction.get("terms") or []
            dominant = terms[0] if terms else None
            spread = block.get("committee_spread_j_per_m2") or 0.0
            delta = abs(block.get("delta_vs_dft_j_per_m2") or 0.0)
            if ratio is not None and ratio < CANCELLATION_WARN and dominant:
                hints.append(
                    f"{label}/{family}: delta_vs_dft is small only because the "
                    f"terms cancelled (cancellation_ratio {ratio:.2f}). The "
                    f"largest single error is {dominant['error_mev_per_atom']:+.1f} "
                    f"meV/atom on {dominant['structure']}, worth "
                    f"{dominant['contribution_j_per_m2']:+.3f} J/m^2 on its own. "
                    "Read the per-structure errors, not the offset."
                )
            elif dominant and dominant["role"] == "reference":
                hints.append(
                    f"{label}/{family}: the offset is driven by a bulk reference "
                    f"({dominant['structure']}, "
                    f"{dominant['error_mev_per_atom']:+.1f} meV/atom), not by the "
                    "interface cell. The references are relaxed 0 K bulks, which "
                    "a model fine-tuned on MD frames may represent worst -- this "
                    "may indicate training coverage only after the DFT evidence checks pass."
                )
            if spread > 0.0 and delta <= spread:
                hints.append(
                    f"{label}/{family}: |delta_vs_dft| = {delta:.3f} J/m^2 sits "
                    f"inside the committee's own spread ({spread:.3f}), so the "
                    "families disagree with each other as much as with DFT. "
                    "Treat the offset as unresolved rather than as a measured bias."
                )
    checks["mlip_comparison"] = "PASS" if families else "NOT_CHECKED"
    if not families:
        hints.append(
            "No MLIP was evaluated. Pass --mace-model / --deepmd-model to turn "
            "this into an audit; the DFT gamma(dmu) above stands on its own."
        )

    order = {"OVERRIDDEN": 3, "CHECK": 2, "NOT_CHECKED": 1, "PASS": 0}
    worst = max(checks.values(), key=lambda value: order.get(value, 0))
    status = worst
    return {
        "status": status,
        "checks": checks,
        "hints": hints or [
            "No inconsistency found: the exact identities hold, the references "
            "are in the state they claim, and nothing about the MLIP offset "
            "needs qualifying."
        ],
        "note": (
            "status is the worst of `checks`. NOT_CHECKED means a check did not "
            "run, which is ignorance and not a clean bill of health. The exact "
            "identities (slope, offset reconstruction) are not listed here: they "
            "raise rather than downgrade a status, because a violation is a data "
            "inconsistency and not a finding."
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
    dft_record = _dft_record(run)
    evidence = structure_evidence(run, composition)
    energy = dft_record["energy_ev"]
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
        "dft_evidence": {**evidence, "run": dft_record},
        "molecular": molecular,
        "spin": (
            audit_molecular_spin(name, run, composition, strict=strict_spin)
            if molecular
            else None
        ),
        "thermal": audit_thermal_state(name, run, int(sum(composition.values()))),
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


def cell_formula_units(composition: Mapping[str, int]) -> int:
    """How many formula units of the reduced formula this cell holds."""

    divisor = 0
    for value in composition.values():
        divisor = gcd(divisor, value)
    return divisor or 1


def _units(phase: Mapping[str, Any], anion: str) -> tuple[float, float, float]:
    """(cations per f.u., anions per f.u., energy per f.u.) for a compound phase."""

    composition = phase["composition"]
    divisor = cell_formula_units(composition)
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
    dft_energies: Mapping[str, float],
    natoms_by_key: Mapping[str, int],
) -> dict[str, Any]:
    """Per-member gamma(dmu) lines for one MLIP family, plus committee statistics.

    ``delta_vs_dft_j_per_m2`` alone cannot say *where* a discrepancy comes from:
    an error on the interface cell and an error on a bulk reference enter it with
    opposite signs and can cancel or compound. The per-structure errors are
    reported next to it so a large offset can be attributed, and a small one can
    be distinguished from two large errors that happened to cancel.
    """

    if not members:
        raise SafetyError(
            "an MLIP family was requested with no committee members; there is "
            "nothing to compare against DFT. Pass at least one --mace-model or "
            "--deepmd-model, and two or more for a committee spread."
        )
    out: dict[str, Any] = {}
    for label in keys:
        # A missing key would surface as a bare KeyError with no context, and a
        # NaN from a calculator -- an element outside the DeePMD type_map, a
        # broken export -- would propagate silently into gamma0 and the spread.
        wanted = [f"iface::{label}"] + [
            f"phase::{name}" for name in classified["compounds"]
        ]
        for member, energies in members.items():
            absent = [key for key in wanted if key not in energies]
            if absent:
                raise SafetyError(
                    f"committee member {member!r} returned no energy for {absent}. "
                    f"It was asked for {sorted(wanted)}. A member that cannot "
                    "evaluate one of these structures cannot contribute to "
                    "gamma; check the model covers every element in the cell."
                )
            unusable = {
                key: energies[key] for key in wanted
                if not isinstance(energies[key], (int, float))
                or not math.isfinite(float(energies[key]))
            }
            if unusable:
                raise SafetyError(
                    f"committee member {member!r} returned a non-finite energy: "
                    f"{unusable}. This would propagate into gamma0, the committee "
                    "spread and every per-structure error without any of them "
                    "looking wrong. Usual cause: an element outside the model's "
                    "type_map, or a corrupt export."
                )
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
        # NOT `keys`: that is this function's own parameter, and the loop above
        # is iterating it. Rebinding it worked only because the dict iterator
        # already existed, and would silently mislead anything added later.
        structure_keys = [f"iface::{label}"] + [
            f"phase::{name}" for name in classified["compounds"]
        ]
        errors: dict[str, Any] = {}
        for key in structure_keys:
            if key not in dft_energies:
                continue
            per_atom = np.asarray([
                (energies[key] - dft_energies[key]) / natoms_by_key[key]
                for energies in members.values()
            ])
            errors[key] = {
                "natoms": natoms_by_key[key],
                "ensemble_error_ev_per_atom": float(per_atom.mean()),
                "ensemble_error_mev_per_atom": float(per_atom.mean()) * 1000.0,
                "member_spread_mev_per_atom": (
                    float(per_atom.std(ddof=1)) * 1000.0 if per_atom.size > 1 else 0.0
                ),
            }
        block["per_structure_error"] = errors

        # EXACT IDENTITY 1 -- the slope is purely structural (anion excess and
        # area), so every member and DFT must agree bit-for-bit. Equal only by
        # construction today, because _family_lines reuses the same
        # decomposition and denominator objects for the MLIP legs as for DFT;
        # this makes that reliance explicit instead of implicit.
        dft_slope = float(dft_lines[label]["slope_j_per_m2_per_ev"])
        slope_deltas = {
            name: float(line["slope_j_per_m2_per_ev"]) - dft_slope
            for name, line in per_member.items()
        }
        worst_slope = max(abs(value) for value in slope_deltas.values())
        if worst_slope > 1.0e-12:
            raise SafetyError(
                f"the gamma(dmu) slope differs between DFT and the MLIP for "
                f"{label!r} by up to {worst_slope:.3e} J/m^2/eV. The slope is "
                "-dn/(n_interfaces*A): composition and area only, no energy. "
                "Differing means the two legs were not evaluated on the same "
                f"structure. Per-member deltas: {slope_deltas}."
            )
        block["slope_identity"] = {
            "status": "PASS",
            "dft_slope_j_per_m2_per_ev": dft_slope,
            "max_abs_delta_j_per_m2_per_ev": worst_slope,
            "note": (
                "the slope is structural, so DFT and every committee member must "
                "share it exactly; published whether or not it tripped"
            ),
        }

        # EXACT IDENTITY 2 -- gamma's offset IS the interface error minus the
        # reference errors weighted by formula-unit count. Summing the terms
        # that are already computed turns the static prose caveat into a ranked
        # attribution, and the residual catches any plumbing error between them.
        denom = float(denominators[label])
        scale = EV_A2_TO_J_M2 / denom
        terms: list[dict[str, Any]] = []
        iface_key = f"iface::{label}"
        if iface_key in errors:
            contribution = errors[iface_key]["ensemble_error_ev_per_atom"] * \
                errors[iface_key]["natoms"] * scale
            terms.append({
                "structure": iface_key,
                "role": "interface",
                "weight_formula_units": 1.0,
                "error_mev_per_atom": errors[iface_key]["ensemble_error_mev_per_atom"],
                "contribution_j_per_m2": contribution,
            })
        for name, phase in classified["compounds"].items():
            key = f"phase::{name}"
            if key not in errors:
                continue
            units = float(decompositions[label]["formula_units"].get(name, 0.0))
            per_cell_units = cell_formula_units(phase["composition"])
            contribution = -(
                units * errors[key]["ensemble_error_ev_per_atom"]
                * errors[key]["natoms"] / per_cell_units
            ) * scale
            terms.append({
                "structure": key,
                "role": "reference",
                "weight_formula_units": units,
                "error_mev_per_atom": errors[key]["ensemble_error_mev_per_atom"],
                "contribution_j_per_m2": contribution,
            })
        reconstructed = sum(term["contribution_j_per_m2"] for term in terms)
        residual = reconstructed - block["delta_vs_dft_j_per_m2"]
        if abs(residual) > 1.0e-9:
            raise SafetyError(
                f"the per-structure errors for {label!r} do not reconstruct "
                f"delta_vs_dft_j_per_m2: summed to {reconstructed:.9f} against a "
                f"reported {block['delta_vs_dft_j_per_m2']:.9f} J/m^2, residual "
                f"{residual:.3e}. These are the same numbers by construction, so "
                "a difference means the decomposition, the formula-unit divisor "
                "or the normalisation area disagree between the two paths."
            )
        ranked = sorted(terms, key=lambda t: -abs(t["contribution_j_per_m2"]))
        block["offset_reconstruction"] = {
            "status": "PASS",
            "terms": ranked,
            "reconstructed_j_per_m2": reconstructed,
            "residual_j_per_m2": residual,
            "dominant_term": ranked[0]["structure"] if ranked else None,
            "cancellation_ratio": (
                abs(reconstructed) / sum(abs(t["contribution_j_per_m2"]) for t in terms)
                if terms and sum(abs(t["contribution_j_per_m2"]) for t in terms) > 0
                else None
            ),
            "note": (
                "delta_vs_dft is the interface error minus the reference errors "
                "weighted by formula-unit count, so the terms carry opposite "
                "signs. cancellation_ratio near 0 means large errors nearly "
                "cancelled and a small offset is hiding them; near 1 means the "
                "offset is what the dominant term says it is."
            ),
        }
        block["per_structure_note"] = (
            "MLIP minus DFT on each structure, in meV/atom. The interface cell is "
            "the finite-temperature snapshot; the compound references are the "
            "relaxed 0 K bulks, which a model trained only on MD may represent "
            "less well. gamma's offset is the interface error minus the "
            "reference errors weighted by formula-unit count, so check both "
            "before reading delta_vs_dft_j_per_m2 as an interface error."
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
    allow_thermal_mismatch: bool = False,
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
    thermal_states: dict[str, dict[str, Any]] = {}
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
        thermal_states[label] = audit_thermal_state(
            label, run, int(sum(composition.values()))
        )
        area, axis = _plane_area(np.array(atoms.cell.array, dtype=float), effective_axis)
        denom = count * area
        decomposition = decompose(composition, classified, anion)
        dft_record = _dft_record(run)
        evidence = structure_evidence(run, composition)
        energy = dft_record["energy_ev"]
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
            "thermal_state": thermal_states[label]["state"],
            "normalization_area_ang2": denom,
            "regime": regime,
            "decomposition": decomposition,
            "dft": {"ready": energy is not None, "energy_ev": energy},
            "dft_evidence": {**evidence, "run": dft_record},
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
    dft_energies: dict[str, float] = {
        f"phase::{name}": float(phase["energy_ev"])
        for name, phase in classified["compounds"].items()
    }
    natoms_by_key: dict[str, int] = {
        f"phase::{name}": int(phase["natoms"])
        for name, phase in classified["compounds"].items()
    }
    for row in rows:
        if row["dft"].get("energy_ev") is not None:
            dft_energies[f"iface::{row['label']}"] = float(row["dft"]["energy_ev"])
        natoms_by_key[f"iface::{row['label']}"] = int(sum(row["composition"].values()))
    families: dict[str, dict[str, Any]] = {}
    if mace_models:
        families["mace"] = _family_lines(
            _mace_energies(mace_models, atoms_by_key, device),
            keys, decompositions, classified, window, denominators, anion, ready,
            dft_energies, natoms_by_key,
        )
    if deepmd_models:
        families["deepmd"] = _family_lines(
            _deepmd_energies(deepmd_models, atoms_by_key),
            keys, decompositions, classified, window, denominators, anion, ready,
            dft_energies, natoms_by_key,
        )
    for row in rows:
        for family, blocks in families.items():
            row["mlip"][family] = blocks[row["label"]]

    payload: dict[str, Any]
    thermal = check_thermal_consistency(
        [thermal_states[row["label"]] for row in rows],
        [phase["thermal"] for phase in read.values()],
        denominators,
        strict=not allow_thermal_mismatch,
    )
    unbalanced = [row["label"] for row in rows if not row["decomposition"]["stoichiometric"]]
    payload = {
        "thermal_consistency": thermal,
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
    all_evidence = {f"phase::{name}": phase["dft_evidence"] for name, phase in read.items()}
    all_evidence.update({f"iface::{row['label']}": row["dft_evidence"] for row in rows})
    payload["vasp_provenance_audit"] = audit_provenance(all_evidence)
    payload["audit"] = summarize_audit(payload)
    return payload


_CSV_FIELDS = (
    "interface", "formula", "source", "stoichiometric", "anion_excess",
    "n_interfaces", "interface_area_ang2", "energy_interpretation", "thermal_state",
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
            "thermal_state": row["thermal_state"],
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
    audit = payload.get("audit")
    if audit:
        lines += ["", f"## Audit: {audit['status']}", ""]
        lines += [
            "| Check | Status |", "|---|---|",
            *(f"| {name} | {value} |" for name, value in audit["checks"].items()),
            "",
        ]
        lines += [f"{index}. {hint}" for index, hint in enumerate(audit["hints"], 1)]

    evidence_rows = [(f"phase::{name}", phase.get("dft_evidence", {}))
                     for name, phase in payload["reference_phases"].items()]
    evidence_rows += [(f"iface::{row['label']}", row.get("dft_evidence", {}))
                      for row in payload["interfaces"]]
    lines += ["", "## DFT evidence", "", "| Structure | Run | Composition | Health / warnings |",
              "|---|---|---|---|"]
    for name, evidence in evidence_rows:
        run = evidence.get("run", {})
        detail = f"{run.get('health') or 'unknown'}; {run.get('warnings') or ''}"
        detail = detail.replace("|", "/").replace("\n", " ")
        lines.append(f"| {name} | {run.get('status', 'NOT_CHECKED')} | "
                     f"{evidence.get('composition', {}).get('status', 'NOT_CHECKED')} | {detail} |")
    provenance = payload.get("vasp_provenance_audit", {})
    lines += ["", *[f"- {item}" for item in provenance.get("missing_evidence", [])]]

    attributed = [
        (row, family, block)
        for row in payload["interfaces"]
        for family, block in (row.get("mlip") or {}).items()
        if block.get("offset_reconstruction")
    ]
    if attributed:
        lines += [
            "", "## Where the MLIP offset comes from", "",
            "Each row is one structure's committee-mean error and what it "
            "contributes to gamma. Interface and reference terms carry opposite "
            "signs, so they can cancel: `cancellation_ratio` near 0 means a "
            "small offset is hiding larger errors.", "",
            "| Interface | Family | Structure | Role | f.u. | Error (meV/atom) "
            "| Contribution (J/m^2) |",
            "|---|---|---|---|---:|---:|---:|",
        ]
        for row, family, block in attributed:
            for term in block["offset_reconstruction"]["terms"]:
                lines.append(
                    "| {} | {} | {} | {} | {:.3f} | {:+.2f} | {:+.4f} |".format(
                        row["label"], family, term["structure"], term["role"],
                        term["weight_formula_units"], term["error_mev_per_atom"],
                        term["contribution_j_per_m2"],
                    )
                )
        for row, family, block in attributed:
            reconstruction = block["offset_reconstruction"]
            ratio = reconstruction["cancellation_ratio"]
            lines += [
                "",
                "{}/{}: total {:+.4f} J/m^2, residual {:.1e}{}.".format(
                    row["label"], family,
                    reconstruction["reconstructed_j_per_m2"],
                    reconstruction["residual_j_per_m2"],
                    "" if ratio is None else
                    f", cancellation_ratio {ratio:.2f}",
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
