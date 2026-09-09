# ruff: noqa: E501
"""Convex-hull phase diagram and rigorous chemical-potential windows (pymatgen).

The pairwise bound ``dmu_X >= max_C dH_f(C)/b_C`` in :mod:`interfaceforge.interface_mu`
only asks whether each cation's *elemental* phase precipitates. That is the right
answer for a binary, but in a real Ti-Si-N system the window can instead be cut
by a competing ternary or a silicide (Ti5Si3, TiSi2, Ti2N, ...). Building the
convex hull and intersecting the stability ranges of the compounds that actually
form the interface answers it properly, and names the phase that binds each side.

Energies must all come from **one consistent set of calculations** -- the same
functional, cutoff, dispersion and k-point density as the interface. Materials
Project entries are useful to discover *which* phases exist in the chemical
system; they must not be mixed into a hull with your own numbers, because MP's
settings and anion corrections differ. ``iface phases suggest`` does the
discovery, ``iface phases hull`` does the thermodynamics on your own runs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .errors import DependencyError, SafetyError


def _pymatgen() -> dict[str, Any]:
    try:
        from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
        from pymatgen.core import Composition, Element
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without pymatgen
        raise DependencyError(
            "Phase-diagram support needs pymatgen; install interfaceforge[phases]"
        ) from exc
    return {
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
                "energy_per_atom_ev": float(entry.energy_per_atom),
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
        "note": (
            "A reference phase that is above the hull is not a valid reservoir: it "
            "would decompose. Check the structure or the settings before using it."
            if unstable
            else "Every supplied reference phase lies on the convex hull."
        ),
    }


def hull_chempot_window(
    hull: Mapping[str, Any],
    compounds: Sequence[str],
    anion: str,
) -> dict[str, Any]:
    """``dmu_anion`` range over which every named compound stays on the hull.

    The interface is a coexistence of its constituent compounds, so the window is
    the intersection of their individual stability ranges. Values are relative to
    the elemental reference (``dmu = 0`` at the pure anion phase), matching
    :func:`interfaceforge.interface_mu.chemical_potential_window`.
    """

    api = _pymatgen()
    diagram = hull["diagram"]
    element = api["Element"](anion)
    # pymatgen returns absolute chemical potentials; shift to dmu = mu - mu0 so
    # this matches interface_mu (0 at the elemental/molecular reference).
    reference = float(diagram.el_refs[element].energy_per_atom)
    by_phase = {row["phase"]: row for row in hull["phases"]}
    per_compound: list[dict[str, Any]] = []
    low, high = float("-inf"), float("inf")
    low_by = high_by = None
    for name in compounds:
        if name not in by_phase:
            raise SafetyError(f"compound {name!r} is not among the supplied phases")
        if not by_phase[name]["stable"]:
            raise SafetyError(
                f"compound {name!r} is above the convex hull "
                f"(e_above_hull = {by_phase[name]['e_above_hull_ev_per_atom']:.4f} eV/atom); "
                "it cannot bound a chemical-potential window"
            )
        composition = api["Composition"](by_phase[name]["formula"])
        ranges = diagram.get_chempot_range_stability_phase(composition, element)
        span = ranges.get(element)
        if span is None:
            raise SafetyError(f"no {anion} chemical-potential range returned for {name!r}")
        lo, hi = float(min(span)) - reference, float(max(span)) - reference
        per_compound.append({"compound": name, "dmu_min_ev": lo, "dmu_max_ev": hi})
        if lo > low:
            low, low_by = lo, name
        if hi < high:
            high, high_by = hi, name
    if low > high:
        raise SafetyError(
            f"the supplied compounds have no common {anion} chemical potential: "
            f"{per_compound}. They cannot coexist, so this interface is not in "
            "equilibrium with both reservoirs."
        )
    elemental = {
        row["phase"] for row in hull["phases"]
        if len(api["Composition"](row["formula"]).elements) == 1
    }
    competing = [
        row["phase"] for row in hull["phases"]
        if row["stable"] and row["phase"] not in compounds and row["phase"] not in elemental
    ]
    return {
        "anion": anion,
        "method": "convex-hull",
        "mu_anion_reference_ev": reference,
        "dmu_min_ev": low,
        "dmu_max_ev": high,
        "dmu_min_set_by": low_by,
        "dmu_max_set_by": high_by,
        "per_compound": per_compound,
        "competing_stable_phases": competing,
        "note": (
            f"dmu({anion}) range over which {list(compounds)} are simultaneously on the "
            "convex hull, relative to the elemental reference. Competing phases on the "
            f"hull that could cut it further: {competing or 'none'}."
        ),
    }


def hull_report(
    phases: Mapping[str, Mapping[str, Any]],
    compounds: Sequence[str],
    anion: str,
) -> dict[str, Any]:
    """Hull + window, with the non-serialisable pymatgen object dropped."""

    hull = build_hull(phases)
    window = hull_chempot_window(hull, compounds, anion)
    return {
        "schema_version": 1,
        "elements": hull["elements"],
        "phases": hull["phases"],
        "stable_phases": hull["stable_phases"],
        "unstable_phases": hull["unstable_phases"],
        "hull_note": hull["note"],
        "chemical_potential_window": window,
    }


# ------------------------------------------------------------------ MP lookup

#: Verified Materials Project ground states for this campaign's chemical system.
#: Used to tell the user which phases to calculate when the MP API is unavailable.
KNOWN_PHASES = {
    "Ti-N": [
        ("TiN", "rocksalt B1", "Fm-3m (225)", "mp-492"),
        ("Ti", "hcp (alpha-Ti)", "P6_3/mmc (194)", "mp-46"),
    ],
    "Si-N": [
        ("Si3N4", "beta-Si3N4", "P6_3/m (176)", "mp-988"),
        ("Si", "diamond", "Fd-3m (227)", "mp-149"),
    ],
    "Ti-Si": [
        ("TiSi2", "C54", "Fddd (70)", "mp-2582"),
    ],
    "Ti-O": [
        ("TiO2", "rutile", "P4_2/mnm (136)", "mp-2657"),
        ("TiO2", "anatase", "I4_1/amd (141)", "mp-390"),
        ("TiO", "rocksalt", "Fm-3m (225)", "mp-2664"),
        ("Ti2O3", "corundum", "R-3c (167)", "mp-458"),
    ],
    "Si-O": [
        ("SiO2", "alpha-quartz", "P3_121 (152)", "mp-7000"),
    ],
}

MOLECULAR_REFERENCES = {
    "N2": "singlet; ISPIN=1 (or ISPIN=2 converging to 0 muB), >=12 A box, Gamma only",
    "O2": "triplet; ISPIN=2, MAGMOM=2*1.0, NUPDOWN=2, ISYM=0, >=12 A box, Gamma only",
}


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
            if name[0] in wanted or name[:2] in wanted
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
