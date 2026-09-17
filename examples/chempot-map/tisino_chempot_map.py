"""Worked Ti-Si-N-O example: the 0 K window, the 2D region, and real atmospheres.

Run it with::

    python examples/chempot-map/tisino_chempot_map.py --output <DIR>

WHERE THE ENERGIES COME FROM
----------------------------

This repository holds no finished VASP runs, so this example cannot use the
campaign's own DFT numbers -- and mixing sources into one hull is exactly what
``iface phases hull`` warns against. Instead every solid here is entered at its
**experimental standard formation enthalpy**, with the elemental references set
to zero energy. That makes each number traceable to a citation rather than to a
calculation nobody can reproduce, and it makes the arithmetic visible: a
compound's "total energy" simply *is* its formation enthalpy per formula unit.

    TiN         -337.65 kJ/mol   Chase 1998 (NIST-JANAF 4th ed.), NIST WebBook
    Si3N4 (a)   -744.75 kJ/mol   Chase 1998; the campaign uses beta, which is
                                 within ~0.03 eV/f.u. of alpha
    TiO2 (rut)  -944.0  kJ/mol   CODATA (Cox, Wagman, Medvedev 1984), +/- 0.8
    SiO2 (qtz)  -910.86 kJ/mol   Chase 1998

TiSi2 is deliberately absent from that list: NIST publishes no condensed-phase
thermochemistry for it, and this example will not invent one. It is handled
instead as an explicit parameter, and :func:`silicide_handover` computes the
formation enthalpy at which TiSi2 -- rather than elemental Si -- starts to set
the N-poor edge. That threshold is the thing to check your own DFT number
against; see the report this script prints.

SUBSTITUTE YOUR OWN NUMBERS by replacing PHASES below with one entry per
finished VASP run, or by calling ``iface phases map --phase NAME=DIR ...``,
which reads the runs directly. These experimental values demonstrate the
machinery; they are not a substitute for a consistent DFT set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from interfaceforge.chempot_map import map_report, stability_polygon, write_reports
from interfaceforge.gas_reference import EV_PER_KJ_PER_MOL
from interfaceforge.phase_diagram import build_hull, hull_report


def ev(kj_per_mol: float) -> float:
    """kJ/mol -> eV per formula unit, through the exact SI conversion."""

    return kj_per_mol * EV_PER_KJ_PER_MOL


#: Experimental formation enthalpies, elemental references at zero. See the
#: module docstring for the source of every line.
SOURCES = {
    "TiN": (-337.65, "Chase 1998 (NIST-JANAF 4th ed.)"),
    "Si3N4": (-744.75, "Chase 1998, alpha phase"),
    "TiO2": (-944.0, "CODATA 1984, rutile, +/- 0.8 kJ/mol"),
    "SiO2": (-910.86, "Chase 1998, alpha-quartz"),
}

PHASES = {
    "TiN":   {"composition": {"Ti": 1, "N": 1}, "energy_ev": ev(SOURCES["TiN"][0])},
    "Si3N4": {"composition": {"Si": 3, "N": 4}, "energy_ev": ev(SOURCES["Si3N4"][0])},
    "TiO2":  {"composition": {"Ti": 1, "O": 2}, "energy_ev": ev(SOURCES["TiO2"][0])},
    "SiO2":  {"composition": {"Si": 1, "O": 2}, "energy_ev": ev(SOURCES["SiO2"][0])},
    # Elemental references: zero by construction, so a compound energy is its
    # formation enthalpy. N2 and O2 must hold exactly one molecule per cell.
    "Ti":    {"composition": {"Ti": 1}, "energy_ev": 0.0},
    "Si":    {"composition": {"Si": 1}, "energy_ev": 0.0},
    "N2":    {"composition": {"N": 2},  "energy_ev": 0.0},
    "O2":    {"composition": {"O": 2},  "energy_ev": 0.0},
}

CONSTITUENTS = ["TiN", "Si3N4"]

#: Demonstration atmospheres. These are illustrative values for a sensitivity
#: scan, NOT measurements of any real furnace -- see the "anoxic" discussion in
#: docs/gas-chempot.md before quoting any of them as an experimental condition.
TEMPERATURES = (300.0, 450.0, 600.0)
OXYGEN_PRESSURES = (1.0e-5, 1.0e-10, 1.0e-15, 1.0e-20)
NITROGEN_PRESSURE = 1.0


def conditions(p_n2: float = NITROGEN_PRESSURE) -> list[dict]:
    return [
        {
            "label": f"{temperature:.0f} K, 1e{round(__import__('math').log10(p_o2)):.0f} atm",
            "temperature_k": temperature,
            "pressures": {"N": p_n2, "O": p_o2},
            "pressure_unit": "atm",
        }
        for temperature in TEMPERATURES
        for p_o2 in OXYGEN_PRESSURES
    ]


def silicide_handover(phases=PHASES, formula=("Ti", 1, "Si", 2)) -> dict:
    """At what formation enthalpy does TiSi2 take over the N-poor edge?

    With TiN and Si3N4 held in equilibrium, dmu_Ti and dmu_Si are fixed
    functions of dmu_N, so *every* Ti-Si phase reduces to a straight lower bound
    on dmu_N. Elemental Si gives one; a silicide gives another. Which one binds
    is decided by a single inequality, and this returns the crossover rather
    than asserting either answer.

    That is the question behind the -2.068 versus -1.973 eV/N discrepancy: the
    looser value is what you get when only Si3N4 (through elemental Si) is
    allowed to cut the window, the tighter one when a silicide is included and
    happens to be the stronger constraint.
    """

    hull = build_hull(phases)
    region = stability_polygon(hull, CONSTITUENTS, ("N", "O"))
    reduction = region["reduction"]
    elements = hull["elements"]

    def line(element: str) -> tuple[float, float]:
        """dmu_element = offset + slope * dmu_N on this plane."""

        index = elements.index(element)
        return reduction["offset_ev"][element], reduction["gradient"][index][0]

    cation, n_cation, anion_metal, n_metal = formula
    off_a, slope_a = line(cation)
    off_b, slope_b = line(anion_metal)
    # n_cation*dmu_Ti + n_metal*dmu_Si <= dHf  becomes  A + B*dmu_N <= dHf
    constant = n_cation * off_a + n_metal * off_b
    slope = n_cation * slope_a + n_metal * slope_b

    silicon_edge = region["extent_ev"]["N"]["min"]
    setter = next(
        e for e in region["edges"] if e["edge"] == "N-poor"
    )["controlled_by"]
    # The silicide binds when its own lower bound is less negative than the
    # current N-poor edge: (dHf - constant)/slope > silicon_edge, slope < 0.
    threshold = constant + slope * silicon_edge
    return {
        "phase": f"{cation}{n_cation if n_cation > 1 else ''}"
                 f"{anion_metal}{n_metal if n_metal > 1 else ''}",
        "inequality": f"{constant:.4f} {slope:+.4f}*dmu_N <= dHf",
        "current_n_poor_edge_ev": silicon_edge,
        "current_n_poor_edge_set_by": setter,
        "handover_formation_enthalpy_ev_per_fu": threshold,
        "handover_formation_enthalpy_kj_per_mol": threshold / EV_PER_KJ_PER_MOL,
        "note": (
            f"With this phase set the N-poor edge sits at {silicon_edge:.4f} eV/atom, "
            f"set by {', '.join(setter)}. A TiSi2 with dHf more negative than "
            f"{threshold:.4f} eV/f.u. ({threshold / EV_PER_KJ_PER_MOL:.1f} kJ/mol) "
            "would take that edge over and tighten the window; a weaker one leaves "
            "it alone. Check your own DFT TiSi2 against that number -- it is the "
            "whole of the -2.068 versus -1.973 question."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True, help="Directory for reports and figures")
    parser.add_argument("--p-n2", type=float, default=NITROGEN_PRESSURE, metavar="ATM")
    args = parser.parse_args()
    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("1. The existing 0 K workflow, unchanged")
    print("=" * 72)
    baseline = hull_report(PHASES, CONSTITUENTS, "N")
    window = baseline["chemical_potential_window"]
    print(f"  dmu(N) window : {window['dmu_min_ev']:.4f} .. {window['dmu_max_ev']:.4f} eV/atom")
    print(f"  lower set by  : {window['dmu_min_set_by']}")
    print(f"  upper set by  : {window['dmu_max_set_by']}")
    print(f"  mu0(N)        : {window['mu_anion_reference_ev']:.4f} eV/atom  (= E(N2)/2)")
    (out / "baseline_0k.json").write_text(
        json.dumps(baseline, indent=2, default=str) + "\n", encoding="utf-8"
    )

    print()
    print("=" * 72)
    print("2. The same phases as a 2D region")
    print("=" * 72)
    payload = map_report(PHASES, CONSTITUENTS, ("N", "O"), conditions=conditions(args.p_n2))
    region = payload["stability_region"]
    print(f"  active   : {', '.join(region['active_phases'])}")
    print(f"  inactive : {', '.join(region['inactive_phases']) or 'none'}")
    print(f"  bounded  : {region['bounded']}")
    for edge in region["edges"]:
        value = edge["value_ev"]
        print("  {:8s}: {:>12s}  by {}".format(
            edge["edge"],
            "unbounded" if value is None else f"{value:.4f} eV",
            ", ".join(edge["controlled_by"]) or "-",
        ))
    print("  corners  :")
    for vertex in region["vertices"]:
        print("     ({:+.4f}, {:+.4f})  {}".format(
            vertex["point"][0], vertex["point"][1], " + ".join(vertex["phases"])
        ))

    print()
    print("  Consistency: the 2D region projected onto dmu(N) must equal the 1D window.")
    print("     1D  : {:.6f} .. {:.6f}".format(window["dmu_min_ev"], window["dmu_max_ev"]))
    print("     2D  : {:.6f} .. {:.6f}".format(
        region["extent_ev"]["N"]["min"], region["extent_ev"]["N"]["max"]
    ))

    print()
    print("=" * 72)
    print("3. Anoxic conditions (demonstration values, not measurements)")
    print("=" * 72)
    print("   T(K)   p_O2(atm)   dmu_N      dmu_O    press.O    inside   undercut by")
    for row in payload["conditions"]:
        oxygen = row["reservoirs"]["O"]
        undercut = ", ".join(v["phase"] for v in row["violated_phases"]) or "-"
        print("  {:5.0f}   {:>9.0e}   {:+.4f}   {:+.4f}   {:+.4f}    {:5s}   {}".format(
            row["temperature_k"], row["pressures"]["O"],
            row["dmu_ev_per_atom"]["N"], row["dmu_ev_per_atom"]["O"],
            oxygen["pressure_correction_atomic_ev"],
            "yes" if row["inside"] else "NO", undercut,
        ))

    print()
    print("=" * 72)
    print("4. Which phase sets the N-poor edge, and when TiSi2 would take it")
    print("=" * 72)
    handover = silicide_handover()
    print("  " + handover["note"].replace(". ", ".\n  "))
    payload["silicide_handover"] = handover

    outputs = write_reports(payload, out)
    print()
    print("Written:")
    for key, value in sorted(outputs.items()):
        print(f"  {key:16s} {value}")
    print(f"  {'baseline_0k':16s} {out / 'baseline_0k.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
