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
from collections.abc import Mapping, Sequence
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np

from .errors import DependencyError, SafetyError
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


def _read_phase(name: str, directory: str | Path, *, allow_vacuum: bool) -> dict[str, Any]:
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
        "regime": regime,
        "atoms": atoms,
    }


def _classify_phases(phases: Mapping[str, dict[str, Any]], anion: str) -> dict[str, Any]:
    """Split references into compounds, elemental cations, and the anion reference."""

    compounds: dict[str, dict[str, Any]] = {}
    elemental: dict[str, dict[str, Any]] = {}
    anion_ref: dict[str, Any] | None = None
    for name, phase in phases.items():
        elements = sorted(phase["composition"])
        if elements == [anion]:
            if anion_ref is not None:
                raise SafetyError(
                    f"two elemental {anion} references given "
                    f"({anion_ref['name']!r} and {name!r})"
                )
            anion_ref = phase
        elif len(elements) == 1:
            elemental[elements[0]] = phase
        elif anion in elements:
            cations = [el for el in elements if el != anion]
            if len(cations) != 1:
                raise SafetyError(
                    f"reference compound {name!r} ({phase['formula']}) has "
                    f"{len(cations)} cation species; gamma(mu) needs one cation per "
                    "compound so its formula-unit count is determined"
                )
            compounds[name] = {**phase, "cation": cations[0]}
        else:
            raise SafetyError(
                f"reference phase {name!r} ({phase['formula']}) is neither elemental "
                f"nor a {anion} compound; it cannot act as a reference here"
            )
    if not compounds:
        raise SafetyError(f"no compound reference phase containing {anion} was given")
    if anion_ref is None:
        raise SafetyError(
            f"no elemental {anion} reference given (e.g. --phase N2=<N2 molecule run>); "
            "it sets the anion-rich limit of the chemical-potential window"
        )
    return {"compounds": compounds, "elemental": elemental, "anion_ref": anion_ref}


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
    anion: str = "N",
    mace_models: Sequence[str] = (),
    deepmd_models: Sequence[str] = (),
    n_interfaces: int = 2,
    area_axis: str | None = None,
    device: str = "cpu",
    allow_vacuum: bool = False,
) -> dict[str, Any]:
    """gamma(dmu_anion) for one or more vacuum-free periodic interface cells."""

    if not entries:
        raise SafetyError("interface-mu needs at least one (label, directory) entry")
    if n_interfaces < 1:
        raise SafetyError("n_interfaces must be a positive integer")
    labels = [label for label, _ in entries]
    if len(set(labels)) != len(labels):
        raise SafetyError(f"duplicate interface label: {labels}")

    read = {
        name: _read_phase(name, directory, allow_vacuum=allow_vacuum)
        for name, directory in phases.items()
    }
    classified = _classify_phases(read, anion)
    window = chemical_potential_window(classified, anion)

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
        regime = require_regime(
            atoms, BULK, f"interface {label!r}", axis=area_axis or "auto",
            allow_mismatch=allow_vacuum,
        )
        composition = _composition(atoms)
        area, axis = _plane_area(np.array(atoms.cell.array, dtype=float), area_axis)
        denom = n_interfaces * area
        decomposition = decompose(composition, classified, anion)
        energy = _dft_energy(run)
        row: dict[str, Any] = {
            "label": label,
            "directory": str(run),
            "composition": composition,
            "formula": _formula(composition),
            "interface_area_ang2": area,
            "area_axis": axis,
            "n_interfaces": n_interfaces,
            "regime": regime,
            "decomposition": decomposition,
            "dft": {"ready": energy is not None, "energy_ev": energy},
            "mlip": {},
        }
        if energy is not None:
            line = gamma_line(energy, decomposition, classified, window, denom, anion)
            row["dft"].update(line)
            row["dft"]["gamma_anion_rich_j_per_m2"] = gamma_at(line, 0.0)
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
    "gamma0_j_per_m2", "slope_j_per_m2_per_ev", "gamma_anion_rich_j_per_m2",
    "gamma_anion_poor_j_per_m2", "committee_spread_j_per_m2", "delta_vs_dft_j_per_m2",
)

_SOURCE_COLORS = {"dft": "#111111", "mace": "#0072B2", "deepmd": "#D55E00"}


def _rows_for_csv(payload: dict[str, Any]) -> list[dict[str, Any]]:
    low = payload["chemical_potential_window"]["dmu_min_ev"]

    def _pair(line: Mapping[str, float]) -> dict[str, Any]:
        return {
            "gamma0_j_per_m2": line["gamma0_j_per_m2"],
            "slope_j_per_m2_per_ev": line["slope_j_per_m2_per_ev"],
            "gamma_anion_rich_j_per_m2": gamma_at(line, 0.0),
            "gamma_anion_poor_j_per_m2": gamma_at(line, low) if low is not None else None,
        }

    out: list[dict[str, Any]] = []
    for row in payload["interfaces"]:
        base = {
            "interface": row["label"],
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
    grid = np.linspace(low, 0.0, 64)
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
        ax.set_xlim(low, 0.0)
        ax.set_xlabel(r"$\Delta\mu_{\mathrm{" + anion + r"}}$ (eV)")
        ax.set_ylabel(r"$\gamma_{\mathrm{int}}$ (J m$^{-2}$)")
        ax.set_title("Grand-canonical interfacial energy", loc="left", fontweight="bold")
        ax.grid(color="#D1D5DB", linewidth=0.45, alpha=0.75)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        poor = anion + "-poor (" + str(window["binding_compound"]) + " limit)"
        ax.annotate(poor, xy=(low, 0.0), xycoords=("data", "axes fraction"),
                    xytext=(5, 5), textcoords="offset points", fontsize=7,
                    color="#6B7280", ha="left", va="bottom")
        ax.annotate(anion + "-rich", xy=(0.0, 0.0), xycoords=("data", "axes fraction"),
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
        "Window: {:.3f} <= dmu({}) <= 0 eV, bounded by {}; mu0({}) = {:.4f} eV from {}.".format(
            low, anion, window["binding_compound"], anion,
            window["mu_anion_reference_ev"], window["anion_reference"],
        )
        if low is not None
        else "Window: not bounded (no elemental cation reference was supplied)."
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
            "over the whole window; where two lines cross, the preferred termination changes. "
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
