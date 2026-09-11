# ruff: noqa: E501
"""Convex-hull phase diagram and rigorous chemical-potential windows (pymatgen).

The pairwise bound ``dmu_X >= max_C dH_f(C)/b_C`` in :mod:`interfaceforge.interface_mu`
only asks whether each cation's *elemental* phase precipitates. That is the right
answer for a binary, but in a real Ti-Si-N system the window can instead be cut
by a competing ternary or a silicide (Ti5Si3, TiSi2, Ti2N, ...). Building the
convex hull and enforcing simultaneous equilibrium of its constituent compounds answers it properly, and names the phase that binds each side.

Energies must all come from **one consistent set of calculations** -- the same
functional, cutoff, dispersion and k-point density as the interface. Materials
Project entries are useful to discover *which* phases exist in the chemical
system; they must not be mixed into a hull with your own numbers, because MP's
settings and anion corrections differ. ``iface phases suggest`` does the
discovery, ``iface phases hull`` does the thermodynamics on your own runs.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

from .errors import DependencyError, SafetyError


def _pymatgen() -> dict[str, Any]:
    try:
        from pymatgen.analysis.phase_diagram import (
            GrandPotentialPhaseDiagram,
            PDEntry,
            PhaseDiagram,
        )
        from pymatgen.core import Composition, Element
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without pymatgen
        raise DependencyError(
            "Phase-diagram support needs pymatgen; install interfaceforge[phases]"
        ) from exc
    return {
        "GrandPotentialPhaseDiagram": GrandPotentialPhaseDiagram,
        "PDEntry": PDEntry,
        "PhaseDiagram": PhaseDiagram,
        "Composition": Composition,
        "Element": Element,
    }


def entries_from_phases(phases: Mapping[str, Mapping[str, Any]]) -> list[Any]:
    """pymatgen ``PDEntry`` list from the reference-phase blocks interface-mu reads.

    Each block needs ``composition`` (element -> count) and ``energy_ev`` (the
    total energy of that cell). The hull is scale-free, so the cell size does
    not matter as long as composition and energy describe the same cell.
    """

    api = _pymatgen()
    entries = []
    for name, phase in phases.items():
        composition = phase["composition"]
        if not composition:
            raise SafetyError(f"reference phase {name!r} has no composition")
        entries.append(
            api["PDEntry"](
                api["Composition"](dict(composition)),
                float(phase["energy_ev"]),
                name=name,
            )
        )
    return entries


def build_hull(phases: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Convex hull over the supplied phases, with per-phase stability."""

    api = _pymatgen()
    entries = entries_from_phases(phases)
    elements = sorted({str(el) for entry in entries for el in entry.composition.elements})
    missing = [
        el for el in elements
        if not any(
            entry.composition.is_element and str(entry.composition.elements[0]) == el
            for entry in entries
        )
    ]
    if missing:
        raise SafetyError(
            f"the hull needs an elemental reference for every element; missing {missing}. "
            "Add one --phase per element (Ti, Si, N2, ...) -- without it the "
            "chemical-potential scale for that element is undefined."
        )
    diagram = api["PhaseDiagram"](entries)
    rows = []
    for entry in entries:
        _, above = diagram.get_decomp_and_e_above_hull(entry, on_error="ignore")
        rows.append(
            {
                "phase": entry.name,
                "formula": entry.composition.reduced_formula,
                "natoms": int(entry.composition.num_atoms),
                "energy_ev": float(entry.energy),
                "energy_per_atom_ev": float(entry.energy_per_atom),
                "formation_energy_per_atom_ev": float(
                    diagram.get_form_energy_per_atom(entry)
                ),
                "e_above_hull_ev_per_atom": None if above is None else float(above),
                "stable": bool(above is not None and above < 1.0e-6),
            }
        )
    unstable = [row["phase"] for row in rows if not row["stable"]]
    return {
        "elements": elements,
        "phases": rows,
        "stable_phases": [row["phase"] for row in rows if row["stable"]],
        "unstable_phases": unstable,
        "diagram": diagram,
        "entries": entries,
        "note": (
            "A reference phase that is above the hull is not a valid reservoir: it "
            "would decompose. Check the structure or the settings before using it."
            if unstable
            else "Every supplied reference phase lies on the convex hull."
        ),
    }


DMU_FLOOR_EV = -15.0
"""Lower bracket for the open-element bisection.

Far below any oxide or nitride formation energy per anion, so a compound that is
not stable even here is not a valid reservoir at all rather than merely oxidised.
"""


def _stable_under_open_element(
    entries: Sequence[Any], element: Any, mu_absolute: float
) -> set[str]:
    """Names of the phases on the grand-potential hull at this open-element mu."""

    api = _pymatgen()
    grand = api["GrandPotentialPhaseDiagram"](entries, {element: mu_absolute})
    return {
        getattr(entry, "original_entry", entry).name for entry in grand.stable_entries
    }


def open_element_limit(
    hull: Mapping[str, Any],
    compounds: Sequence[str],
    element: str,
    *,
    floor_ev: float = DMU_FLOOR_EV,
    tolerance_ev: float = 1.0e-4,
) -> dict[str, Any]:
    """Highest ``dmu_element`` at which every named compound survives the reservoir.

    For a compound that does not contain the open element -- TiN in a mu_O
    reservoir -- the ordinary chemical-potential range is the wrong question and
    pymatgen cannot answer it (its range routine divides by the compound's amount
    of that element). The right question is the **oxidation limit**: the grand
    potential of an O-free phase is flat in mu_O while every O-bearing competitor
    falls, so above some mu_O the compound is undercut and stays undercut. That
    monotonicity is what makes a bisection valid here, and it is why this is a
    one-sided bound: removing O never destabilises a phase that contains none.
    """

    api = _pymatgen()
    entries = hull["entries"]
    el = api["Element"](element)
    reference = float(hull["diagram"].el_refs[el].energy_per_atom)
    at_floor = _stable_under_open_element(entries, el, reference + floor_ev)
    at_reference = _stable_under_open_element(entries, el, reference)
    per_compound: list[dict[str, Any]] = []
    for name in compounds:
        if name not in at_floor:
            raise SafetyError(
                f"compound {name!r} is not stable even at dmu({element}) = "
                f"{floor_ev:.1f} eV, where the {element} reservoir is as poor as it "
                "can meaningfully get. It is not a valid reservoir phase: check its "
                "structure and that every energy came from the same settings."
            )
        if name in at_reference:
            per_compound.append({
                "compound": name,
                "dmu_limit_ev": 0.0,
                "limited": False,
                "decomposition_at_limit": None,
                "note": (
                    f"{name} is still on the grand-potential hull at dmu({element}) "
                    f"= 0, i.e. in contact with the elemental {element} reservoir "
                    "itself; nothing in this phase set oxidises it"
                ),
            })
            continue
        low, high = floor_ev, 0.0  # stable at low, not stable at high
        while high - low > tolerance_ev:
            middle = 0.5 * (low + high)
            if name in _stable_under_open_element(entries, el, reference + middle):
                low = middle
            else:
                high = middle
        grand = api["GrandPotentialPhaseDiagram"](entries, {el: reference + high})
        composition = api["Composition"](
            next(row["formula"] for row in hull["phases"] if row["phase"] == name)
        )
        products = sorted(
            getattr(entry, "original_entry", entry).name
            for entry in grand.get_decomposition(composition)
        )
        per_compound.append({
            "compound": name,
            "dmu_limit_ev": low,
            "limited": True,
            "decomposition_at_limit": products,
            "note": f"above dmu({element}) = {low:.4f} eV, {name} -> {' + '.join(products)}",
        })
    binding = min(per_compound, key=lambda row: row["dmu_limit_ev"])
    return {
        "anion": element,
        "method": "grand-potential-open-element",
        "mu_anion_reference_ev": reference,
        "dmu_min_ev": None,
        "dmu_max_ev": binding["dmu_limit_ev"],
        "dmu_min_set_by": None,
        "dmu_max_set_by": binding["compound"],
        "per_compound": per_compound,
        "competing_stable_phases": sorted(
            set().union(*(row["decomposition_at_limit"] or [] for row in per_compound))
        ),
        "note": (
            f"None of {list(compounds)} contains {element}, so this is an oxidation "
            f"limit rather than a two-sided window: the highest dmu({element}) at "
            "which all of them stay on the grand-potential hull. The lower side is "
            f"unbounded -- taking {element} away cannot destabilise a phase that "
            f"contains none. Bound set by {binding['compound']}: {binding['note']}."
        ),
    }


def hull_chempot_window(
    hull: Mapping[str, Any],
    compounds: Sequence[str],
    anion: str,
    *,
    fixed_dmu: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Project the *joint* coexistence polytope onto one chemical potential.

    Every constituent satisfies n.mu = E simultaneously; every supplied phase
    satisfies n.mu <= E. Intersecting separately projected stability ranges is
    insufficient: their other chemical potentials need not agree.
    """
    import numpy as np
    from scipy.optimize import linprog

    api = _pymatgen()
    elements = list(hull["elements"])
    entries = list(hull["entries"])
    by_name = {entry.name: i for i, entry in enumerate(entries)}
    fixed = dict(fixed_dmu or {})
    if not compounds:
        raise SafetyError("at least one constituent compound is required")
    missing = set(compounds) - set(by_name)
    if missing:
        raise SafetyError(f"compounds {sorted(missing)} are not among the supplied phases")
    if anion not in elements:
        raise SafetyError(f"no elemental reference for {anion}")
    if anion in fixed or set(fixed) - set(elements):
        raise SafetyError("fixed_dmu must name other elements present in the hull")
    if not all(np.isfinite(value) for value in fixed.values()):
        raise SafetyError("fixed chemical potentials must be finite")
    contains = [entries[by_name[name]].composition[anion] > 0 for name in compounds]
    if not any(contains) and not fixed:
        result = open_element_limit(hull, compounds, anion)
        result["note"] += (
            " This is an individual-stability projection, not a fixed nitrogen "
            "reservoir slice. Use --fixed-dmu N=VALUE for joint coexistence at "
            "a specified nitrogen chemical potential."
        )
        return result
    if any(contains) and not all(contains) and not fixed:
        raise SafetyError(
            "compounds with and without the open element have no common kind of bound "
            "in this report; query separate runs or specify another reservoir with --fixed-dmu"
        )
    for name in compounds:
        row = next(row for row in hull["phases"] if row["phase"] == name)
        if not row["stable"]:
            raise SafetyError(f"compound {name!r} is above the convex hull; it cannot coexist")
    refs = np.array([
        hull["diagram"].el_refs[api["Element"](el)].energy_per_atom for el in elements
    ])
    # Normalize each constraint per atom, so supercell size cannot set tolerance.
    matrix = np.array([
        [entry.composition[el] / entry.composition.num_atoms for el in elements]
        for entry in entries
    ])
    formation = np.array([entry.energy_per_atom for entry in entries]) - matrix @ refs
    objective = np.zeros(len(elements))
    objective[elements.index(anion)] = 1.0

    def solve(names: Sequence[str], sign: float) -> tuple[float | None, list[str], dict[str, float]]:
        equality = [matrix[by_name[name]] for name in names]
        values = [formation[by_name[name]] for name in names]
        for el, value in fixed.items():
            vector = np.zeros(len(elements))
            vector[elements.index(el)] = 1.0
            equality.append(vector)
            values.append(value)
        result = linprog(
            sign * objective, A_ub=matrix, b_ub=formation,
            A_eq=np.array(equality), b_eq=np.array(values),
            bounds=[(None, None)] * len(elements), method="highs",
        )
        if result.status == 3:
            return None, [], {}
        if result.status == 2:
            raise SafetyError(
                f"the supplied compounds {list(names)} have no common chemical potentials "
                f"with fixed dmu {fixed}; they cannot coexist"
            )
        if not result.success:
            raise SafetyError(f"chemical-potential optimization failed: {result.message}")
        # Dual multipliers identify bound-setting constraints, excluding the
        # constituent equalities which are active over the entire interval.
        active = sorted(
            entry.name for i, entry in enumerate(entries)
            if entry.name not in names
            and not {str(el) for el in entry.composition.elements}.issubset(fixed)
            and abs(result.ineqlin.marginals[i]) > 1e-8
        )
        value = float(result.x[elements.index(anion)])
        if abs(value) < 1e-10:
            value = 0.0
        return value, active, dict(zip(elements, map(float, result.x), strict=True))

    low, low_by, low_mu = solve(compounds, 1.0)
    high, high_by, high_mu = solve(compounds, -1.0)
    per_compound = []
    for name in compounds:
        lo, _, _ = solve([name], 1.0)
        hi, _, _ = solve([name], -1.0)
        per_compound.append({"compound": name, "dmu_min_ev": lo, "dmu_max_ev": hi})
    competing = [
        row["phase"] for row in hull["phases"]
        if row["stable"] and row["phase"] not in compounds
        and len(api["Composition"](row["formula"]).elements) > 1
    ]
    return {
        "anion": anion,
        "method": "convex-hull",
        "solver": "joint-coexistence-linear-program",
        "fixed_dmu_ev": fixed,
        "mu_anion_reference_ev": float(refs[elements.index(anion)]),
        "dmu_min_ev": low,
        "dmu_max_ev": high,
        "dmu_min_set_by": ", ".join(low_by) or None,
        "dmu_max_set_by": ", ".join(high_by) or None,
        "lower_bound_phases": low_by,
        "upper_bound_phases": high_by,
        "endpoint_dmu_ev": {"lower": low_mu, "upper": high_mu},
        "per_compound": per_compound,
        "competing_stable_phases": competing,
        "note": (
            f"Joint coexistence of {list(compounds)}: all constituent equalities "
            "and every supplied phase inequality are enforced simultaneously. "
            "Individual ranges are diagnostic projections, not the joint window. "
            f"Fixed dmu (eV/atom): {fixed or 'none'}."
        ),
    }


def hull_report(
    phases: Mapping[str, Mapping[str, Any]],
    compounds: Sequence[str],
    anion: str,
    *,
    fixed_dmu: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Hull + window, with the non-serialisable pymatgen object dropped."""

    hull = build_hull(phases)
    window = hull_chempot_window(hull, compounds, anion, fixed_dmu=fixed_dmu)
    missing = missing_known_phases(
        hull["elements"], [row["formula"] for row in hull["phases"]]
    )
    checked = covered_subsystems(hull["elements"])
    api = _pymatgen()
    for row in hull["phases"]:
        elements = api["Composition"](row["formula"]).elements
        row["role"] = (
            "compound" if row["phase"] in compounds
            else "elemental" if len(elements) == 1
            else "auxiliary"
        )
    return {
        "schema_version": 1,
        "elements": hull["elements"],
        "phases": hull["phases"],
        "stable_phases": hull["stable_phases"],
        "unstable_phases": hull["unstable_phases"],
        "hull_note": hull["note"],
        "missing_known_phases": missing,
        "completeness_note": (
            "the hull was built without "
            + ", ".join(f"{row['formula']} ({row['mp_id']})" for row in missing)
            + ". An omitted stable phase can only make the window look too wide, "
            "so treat these bounds as an upper limit until those phases are "
            "computed at the campaign's settings and the hull re-run."
            if missing
            else "every phase in the built-in list for {} was supplied. That list "
            "is curated for this campaign's chemical systems, not a proof of "
            "exhaustiveness -- confirm against Materials Project if a window "
            "bound is load-bearing.".format(", ".join(checked))
            if checked
            else "completeness NOT checked: no built-in phase list covers {}. "
            "Verify against Materials Project that no stable phase of this system "
            "is missing -- an omitted one would make the window look too wide.".format(
                ", ".join(hull["elements"])
            )
        ),
        "chemical_potential_window": window,
    }


# ------------------------------------------------------------------ MP lookup

#: Verified Materials Project ground states for this campaign's chemical system.
#: Used to tell the user which phases to calculate when the MP API is unavailable,
#: and to warn when a hull was built without one of them (see
#: :func:`missing_known_phases`). Where a composition has several polymorphs the
#: **ground state must be listed first** -- the completeness check treats a
#: composition as covered once any polymorph of it is supplied.
KNOWN_PHASES = {
    "Ti-N": [
        ("TiN", "rocksalt B1", "Fm-3m (225)", "mp-492"),
        ("Ti2N", "epsilon-Ti2N, tetragonal", "P4_2/mnm (136)", "mp-8282"),
        ("Ti", "hcp (alpha-Ti)", "P6_3/mmc (194)", "mp-46"),
    ],
    "Si-N": [
        ("Si3N4", "beta-Si3N4", "P6_3/m (176)", "mp-988"),
        ("Si", "diamond", "Fd-3m (227)", "mp-149"),
    ],
    "Ti-Si": [
        ("Ti5Si3", "hexagonal (Mn5Si3-type, D8_8)", "P6_3/mcm (193)", "mp-2108"),
        ("TiSi2", "C54", "Fddd (70)", "mp-2582"),
        ("TiSi", "orthorhombic", "Pnma (62)", "mp-7092"),
        ("Ti5Si4", "tetragonal", "P4_12_12 (92)", "mp-505527"),
        ("Ti3Si", "tetragonal", "P4_2/n (86)", "mp-980420"),
    ],
    "Ti-O": [
        ("TiO2", "rutile", "P4_2/mnm (136)", "mp-2657"),
        ("TiO2", "anatase", "I4_1/amd (141)", "mp-390"),
        ("TiO", "rocksalt", "Fm-3m (225)", "mp-2664"),
        ("Ti2O3", "corundum", "R-3c (167)", "mp-458"),
        ("Ti3O5", "monoclinic (Magneli n=3)", "C2/m (12)", "mp-1147"),
        # borderline on MP itself (+0.007 eV/atom, -> TiO2 + Ti3O5), so at the
        # campaign's own settings it may land either side; e_above_hull decides
        ("Ti4O7", "triclinic (Magneli n=4)", "P-1 (2)", "mp-12205"),
    ],
    "Si-O": [
        ("SiO2", "alpha-quartz", "P3_121 (152)", "mp-7000"),
    ],
}

MOLECULAR_REFERENCES = {
    "N2": "singlet; ISPIN=1 (or ISPIN=2 converging to 0 muB), >=12 A box, Gamma only",
    "O2": "triplet; ISPIN=2, MAGMOM=2*1.0, NUPDOWN=2, ISYM=0, >=12 A box, Gamma only",
}

#: Which element each molecular reference sets the potential for. Keyed explicitly
#: rather than sliced off the name, so asking for Na does not pull in N2.
MOLECULAR_ELEMENT = {"N2": "N", "O2": "O"}

#: Total spin moment (muB) the diatomic molecular reference of each element must
#: carry, per X2 unit. O2 is the one that bites: its ground state is a triplet,
#: and a non-spin-polarised O2 is ~1 eV too high, which shifts mu_O and every
#: gamma, formation enthalpy and window derived from it by that much.
MOLECULAR_MOMENT_PER_PAIR = {
    "N": 0.0,
    "O": 2.0,
    "H": 0.0,
    "F": 0.0,
    "Cl": 0.0,
}


def covered_subsystems(elements: Collection[str]) -> list[str]:
    """The built-in subsystems whose every element is present in ``elements``.

    Empty means this chemical system is outside the built-in list, so silence
    from the completeness check is ignorance rather than a clean bill of health.
    """

    wanted = set(elements)
    return [system for system in KNOWN_PHASES if set(system.split("-")) <= wanted]


def missing_known_phases(
    elements: Sequence[str], supplied_formulas: Sequence[str]
) -> list[dict[str, Any]]:
    """Known ground states of these elements that were *not* handed to the hull.

    A hull is only as complete as the phases fed to it, and the error is
    one-sided: an omitted stable phase can only make the window look **too
    wide**, never too narrow. So this is a warning rather than an error --
    recompute the listed phases at the campaign's settings and re-run to see
    whether they actually cut the window.

    Only subsystems whose every element is present are considered, so a Ti-N
    hull is not scolded for lacking silicides.
    """

    api = _pymatgen()
    present = {api["Composition"](f).reduced_formula for f in supplied_formulas}
    wanted = set(elements)
    missing: dict[str, dict[str, Any]] = {}
    for system in covered_subsystems(wanted):
        for formula, structure, spacegroup, mp_id in KNOWN_PHASES[system]:
            reduced = api["Composition"](formula).reduced_formula
            if reduced in present or reduced in missing:
                continue  # a polymorph of it was supplied, or already listed
            missing[reduced] = {
                "formula": formula,
                "structure": structure,
                "spacegroup": spacegroup,
                "mp_id": mp_id,
                "system": system,
            }
    return list(missing.values())


def suggest_phases(elements: Sequence[str], api_key: str | None = None) -> dict[str, Any]:
    """Which phases to compute for a chemical system.

    Queries Materials Project for the stable entries when ``mp_api`` and a key
    are available; otherwise returns the verified built-in list. Either way the
    energies must be recomputed with your own settings -- this only answers
    *which* phases exist.
    """

    wanted = sorted({el for el in elements})
    payload: dict[str, Any] = {
        "elements": wanted,
        "source": "built-in",
        "molecular_references": {
            name: note for name, note in MOLECULAR_REFERENCES.items()
            if MOLECULAR_ELEMENT[name] in wanted
        },
        "warning": (
            "Recompute every phase with the campaign's own settings. Materials "
            "Project energies use different cutoffs, no dispersion, and anion "
            "corrections; mixing them into a hull with your numbers is invalid."
        ),
    }
    try:
        from mp_api.client import MPRester  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        payload["phases"] = [
            {"formula": formula, "structure": structure, "spacegroup": spacegroup,
             "mp_id": mp_id, "system": system}
            for system, items in KNOWN_PHASES.items()
            if all(el in wanted for el in system.split("-"))
            for formula, structure, spacegroup, mp_id in items
        ]
        payload["note"] = (
            "mp_api is not installed, so this is the verified built-in list for the "
            "SiN/TiN/TiO system. `pip install mp-api` and set MP_API_KEY for a live "
            "query of an arbitrary chemical system."
        )
        return payload
    with MPRester(api_key) as rester:  # pragma: no cover - needs network + key
        docs = rester.materials.summary.search(
            chemsys="-".join(wanted), is_stable=True,
            fields=["material_id", "formula_pretty", "symmetry", "energy_above_hull"],
        )
    payload["source"] = "materials-project"
    payload["phases"] = [
        {
            "formula": doc.formula_pretty,
            "mp_id": str(doc.material_id),
            "spacegroup": getattr(doc.symmetry, "symbol", None),
            "e_above_hull_ev_per_atom": doc.energy_above_hull,
        }
        for doc in docs
    ]
    return payload


# --------------------------------------------------------------- reports

HULL_CSV_FIELDS = (
    "phase", "formula", "role", "natoms", "energy_ev", "energy_per_atom_ev",
    "formation_energy_per_atom_ev", "e_above_hull_ev_per_atom", "stable",
)


def _hull_markdown(payload: Mapping[str, Any]) -> str:
    """Human-readable tables: the window or limit first, then every phase."""

    window = payload["chemical_potential_window"]
    def bound(value: float | None, side: str) -> str:
        return ("-inf" if side == "lower" else "+inf") if value is None else f"{value:.4f}"

    anion = window["anion"]
    limit = window["method"] == "grand-potential-open-element"
    lines = [
        "# Convex-hull phase diagram and chemical-potential "
        + ("limit" if limit else "window"),
        "",
        "Elements: {}. Method: {}. mu0({}) = {:.4f} eV/atom.".format(
            ", ".join(payload["elements"]), window["method"], anion,
            window["mu_anion_reference_ev"],
        ),
        "",
    ]
    if limit:
        compounds = ", ".join(row["compound"] for row in window["per_compound"])
        lines += [
            f"## Oxidation limit in dmu({anion})",
            "",
            f"No named compound contains {anion}, so the bound is one-sided: the "
            f"lower side is unbounded, because taking {anion} away cannot "
            "destabilise a phase that contains none.",
            "",
            "| Compound | dmu limit (eV) | Decomposes to | Bounded |",
            "|---|---:|---|---|",
        ]
        for row in window["per_compound"]:
            products = " + ".join(row["decomposition_at_limit"] or []) or "-"
            lines.append("| {} | {:.4f} | {} | {} |".format(
                row["compound"], row["dmu_limit_ev"], products,
                "yes" if row["limited"] else "no (survives dmu = 0)",
            ))
        lines += [
            "",
            "**Binding limit: dmu({}) <= {:.4f} eV, set by {}.** Above it the "
            "individual stability of {} no longer holds. This is a necessary, "
            "not sufficient, condition for joint coexistence at fixed nitrogen potential.".format(
                anion, window["dmu_max_ev"], window["dmu_max_set_by"], compounds,
            ),
        ]
    else:
        lines += [
            f"## Window in dmu({anion})",
            "",
            "| Compound | dmu min (eV) | dmu max (eV) |",
            "|---|---:|---:|",
        ]
        for row in window["per_compound"]:
            lines.append("| {} | {} | {} |".format(
                row["compound"], bound(row["dmu_min_ev"], "lower"), bound(row["dmu_max_ev"], "upper")
            ))
        lines += [
            "",
            "**{} <= dmu({}) <= {} eV**, lower bound set by {}, upper by "
            "{}.".format(
                bound(window["dmu_min_ev"], "lower"), anion, bound(window["dmu_max_ev"], "upper"),
                window["dmu_min_set_by"], window["dmu_max_set_by"],
            ),
        ]
    lines += ["", window["note"]]
    competing = window.get("competing_stable_phases") or []
    lines += [
        "",
        "Competing phases on the hull: {}.".format(", ".join(competing) or "none"),
        "",
        "## Phases",
        "",
        "`Ef/atom` is the formation energy from the elemental references in this "
        "same set. `Above hull` is 0 for a phase that is a valid reservoir; a "
        "positive value means it would decompose.",
        "",
        "| Phase | Formula | Role | Ef/atom (eV) | Above hull (eV/atom) | On hull |",
        "|---|---|---|---:|---:|---|",
    ]
    ordered = sorted(
        payload["phases"],
        key=lambda row: (not row["stable"], row["e_above_hull_ev_per_atom"] or 0.0),
    )
    for row in ordered:
        above = row["e_above_hull_ev_per_atom"]
        lines.append("| {} | {} | {} | {:.4f} | {} | {} |".format(
            row["phase"], row["formula"], row["role"],
            row["formation_energy_per_atom_ev"],
            "-" if above is None else f"{above:.4f}",
            "yes" if row["stable"] else "NO",
        ))
    missing = payload.get("missing_known_phases") or []
    if missing:
        named = ", ".join(f"{m['formula']} ({m['mp_id']})" for m in missing)
        lines += [
            "",
            "## Incomplete hull",
            "",
            f"> **WARNING** built without {named}. An omitted stable phase can only "
            "make the window look too wide, so the bounds above are an upper limit "
            "until these are computed at the campaign's settings and the hull "
            "re-run.",
        ]
    lines += ["", "Completeness: " + payload["completeness_note"]]
    return "\n".join(lines) + "\n"


def _write_hull_figures(
    payload: Mapping[str, Any], phases: Mapping[str, Mapping[str, Any]], out: Path
) -> dict[str, str]:
    """The hull itself, plus a dmu axis showing which phase sets the bound."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.text as mtext
        from pymatgen.analysis.phase_diagram import PDPlotter
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "Phase-diagram figures require matplotlib; install interfaceforge[report]"
        ) from exc

    window = payload["chemical_potential_window"]
    anion = window["anion"]
    limit = window["method"] == "grand-potential-open-element"
    paths: dict[str, Path] = {}
    style = {
        "font.family": "sans-serif", "font.size": 8.0, "axes.titlesize": 9.0,
        "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "axes.linewidth": 0.7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }

    # 1. the hull: 2 elements -> formation-energy curve, 3 -> Gibbs triangle,
    #    4 -> tetrahedron. pymatgen picks the projection from the dimensionality.
    diagram = build_hull(phases)["diagram"]
    with plt.rc_context(style):
        axes = PDPlotter(diagram, show_unstable=0.2, backend="matplotlib").get_plot()
        figure = axes.get_figure()
        # PDPlotter hardcodes large phase labels, which collide as soon as a
        # binary edge carries more than two compounds (Ti3Si/Ti5Si3/Ti5Si4/TiSi).
        # They are not all in axes.texts -- the quaternary legend sits elsewhere
        # in the tree -- so walk every Text artist, and title only afterwards.
        for artist in (*axes.get_children(), *figure.get_children()):
            if isinstance(artist, mtext.Text) and artist.get_text():
                artist.set_fontsize(8.0)
        figure.set_size_inches(7.0, 5.6)
        # above the plot area, so it clears the apex label of a Gibbs triangle
        figure.suptitle(
            "{} convex hull".format("-".join(payload["elements"])),
            fontsize=9.0, y=1.02,
        )
        for suffix in ("png", "svg", "pdf"):
            path = out / ("phases_hull." + suffix)
            figure.savefig(path, dpi=300, bbox_inches="tight")
            paths["hull_" + suffix] = path
        plt.close(figure)

    if window.get("fixed_dmu_ev"):
        with plt.rc_context(style):
            figure, axis = plt.subplots(figsize=(6.0, 2.4))
            high = window["dmu_max_ev"]
            low = window["dmu_min_ev"]
            left = high - 3.0 if low is None else low
            axis.barh(0, high - left, left=left, height=0.4, color="#0072B2")
            if low is None:
                axis.plot(left, 0, marker="<", color="#0072B2")
                axis.text(left, -0.35, "unbounded below", fontsize=7)
            axis.set_yticks([0], ["Joint coexistence"])
            axis.set_xlabel(f"dmu({anion}) (eV/atom)")
            axis.set_title(f"Fixed dmu: {window['fixed_dmu_ev']}; upper: {window['dmu_max_set_by']}")
            axis.set_ylim(-0.7, 0.7)
            for suffix in ("png", "svg", "pdf"):
                path = out / ("phases_chempot." + suffix)
                figure.savefig(path, dpi=300, bbox_inches="tight")
                paths["chempot_" + suffix] = path
            plt.close(figure)
        return {key: str(value) for key, value in paths.items()}

    # 2. the dmu axis: one bar per compound, so the binding phase is visible
    rows = list(window["per_compound"])
    if not limit:
        rows.append({"compound": "Joint coexistence", "dmu_min_ev": window["dmu_min_ev"],
                     "dmu_max_ev": window["dmu_max_ev"]})
    with plt.rc_context(style):
        figure, axis = plt.subplots(figsize=(5.6, 0.55 * len(rows) + 1.7))
        if limit:
            left = min(row["dmu_limit_ev"] for row in rows) - 2.0
            for index, row in enumerate(rows):
                axis.barh(index, row["dmu_limit_ev"] - left, left=left, height=0.45,
                          color="#0072B2", alpha=0.55, edgecolor="#0072B2", lw=0.7)
                products = " + ".join(row["decomposition_at_limit"] or [])
                axis.annotate(
                    "  -> " + (products or "no limit"),
                    (row["dmu_limit_ev"], index), fontsize=7.0, va="center",
                    ha="left", color="#111111",
                )
                # the bar runs off the left edge: nothing bounds it from below
                axis.plot([left], [index], marker="<", markersize=4.5,
                          color="#0072B2", clip_on=False)
            axis.annotate(
                "unbounded below", (left, len(rows) - 0.75), fontsize=7.0,
                va="center", ha="left", color="#0072B2", style="italic",
            )
            axis.axvspan(left, window["dmu_max_ev"], color="#0072B2", alpha=0.10, lw=0)
            axis.axvline(window["dmu_max_ev"], color="#111111", lw=1.0)
            axis.set_title(
                "dmu({}) oxidation limit: <= {:.3f} eV, set by {}".format(
                    anion, window["dmu_max_ev"], window["dmu_max_set_by"]
                )
            )
            axis.set_xlim(left, 0.9)
        else:
            left = min(row["dmu_min_ev"] for row in rows) - 0.4
            for index, row in enumerate(rows):
                axis.barh(index, row["dmu_max_ev"] - row["dmu_min_ev"],
                          left=row["dmu_min_ev"], height=0.45, color="#0072B2",
                          alpha=0.55, edgecolor="#0072B2", lw=0.7)
            axis.axvspan(window["dmu_min_ev"], window["dmu_max_ev"],
                         color="#0072B2", alpha=0.10, lw=0)
            for bound in (window["dmu_min_ev"], window["dmu_max_ev"]):
                axis.axvline(bound, color="#111111", lw=1.0)
            axis.set_title(
                "dmu({}) window: {:.3f} to {:.3f} eV, lower bound by {}".format(
                    anion, window["dmu_min_ev"], window["dmu_max_ev"],
                    window["dmu_min_set_by"],
                )
            )
            axis.set_xlim(left, 0.4)
        axis.axvline(0.0, color="#D55E00", lw=0.9, linestyle="--")
        axis.annotate(
            "elemental " + anion + " reference", (0.0, -0.9), fontsize=7.0,
            rotation=90, va="bottom", ha="right", color="#D55E00",
        )
        axis.set_yticks(range(len(rows)))
        axis.set_yticklabels([row["compound"] for row in rows])
        axis.set_xlabel(
            f"dmu({anion}) (eV), zero at the elemental reference"
        )
        axis.set_ylim(-1.05, len(rows) - 0.45)
        axis.spines[["top", "right"]].set_visible(False)
        for suffix in ("png", "svg", "pdf"):
            path = out / ("phases_chempot." + suffix)
            figure.savefig(path, dpi=300, bbox_inches="tight")
            paths["chempot_" + suffix] = path
        plt.close(figure)
    return {key: str(value) for key, value in paths.items()}


def write_reports(
    payload: Mapping[str, Any],
    output_dir: str | Path,
    phases: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    """JSON, CSV, markdown tables and figures for one hull, into ``output_dir``."""

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "phases_hull.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    with (out / "phases_hull.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=HULL_CSV_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(payload["phases"])
    (out / "phases_hull.md").write_text(_hull_markdown(payload), encoding="utf-8")
    outputs = {
        "json": str(out / "phases_hull.json"),
        "csv": str(out / "phases_hull.csv"),
        "markdown": str(out / "phases_hull.md"),
    }
    try:
        outputs.update(_write_hull_figures(payload, phases, out))
    except (DependencyError, SafetyError) as exc:
        outputs["figures"] = "skipped: " + str(exc)
    return outputs
