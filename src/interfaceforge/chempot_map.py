# ruff: noqa: E501
"""The 2D chemical-potential stability region, computed rather than drawn.

:mod:`interfaceforge.phase_diagram` answers "how far can dmu(N) move while the
interface's constituents all survive", as an interval. That is the right
question for a Ti-Si-N system. Add oxygen and it stops being one number: the
reservoir has two knobs, and the honest answer is a *region* in the
(dmu_N, dmu_O) plane.

How the plane is defined
------------------------

Ti-Si-N-O has four chemical potentials, so the feasible set lives in 4D and a
(dmu_N, dmu_O) picture of it needs two dimensions eliminated. Requiring the
interface's own constituents to be simultaneously in equilibrium does exactly
that, and does it without any further assumption:

    TiN   in equilibrium:   dmu_Ti + dmu_N       = dHf(TiN)
    Si3N4 in equilibrium:  3 dmu_Si + 4 dmu_N    = dHf(Si3N4)

Two equalities in four unknowns leave two free coordinates, and those two can
be taken to be dmu_N and dmu_O -- so dmu_Ti and dmu_Si are not assumed away,
they are *determined* by dmu_N. Substituting them into the stability condition
of any other phase leaves a straight line in the plane. For a phase with
composition (n_Ti, n_Si, n_N, n_O):

    (n_N - n_Ti - (4/3) n_Si) dmu_N + n_O dmu_O
        <= dHf(phase) - n_Ti dHf(TiN) - (n_Si/3) dHf(Si3N4)

That dmu_N coefficient is worth a second look. ``n_N - n_Ti - (4/3) n_Si`` is
the same excess-nitrogen combination that sets the slope of gamma(mu_N) in
:mod:`interfaceforge.interface_mu`. The interface energy and the phase
boundaries are keeping the same books, which is the check that the two halves
of this framework agree.

The reduction is done numerically and in general, not hard-coded for TiN and
Si3N4: :func:`reduce_to_plane` builds the null space of the equality system and
refuses, with a diagnosis, if the named compounds do not leave exactly two
degrees of freedom or if the requested axes are not independent coordinates on
what is left.

Active and inactive
-------------------

A line that is drawn is not a line that binds. Every constraint here is tested
for redundancy by linear programming -- maximise its own left-hand side over
all the *other* constraints and see whether it can ever be reached -- so the
report distinguishes a phase that shapes the region from one that merely
exists. Inactive phases are listed, not plotted.

This is the part that catches the failure mode that motivated the module: a
boundary plotted from one phase's individual stability while the joint
constraint is set by another. Here both come out of the same polytope, and
:func:`stability_polygon` reports which is which.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .errors import DependencyError, SafetyError
from .gas_reference import (
    ELEMENT_TO_SPECIES,
    SOLID_APPROXIMATION,
    SPECIES,
    gas_conditions,
)
from .phase_diagram import _linprog, _pymatgen, build_hull

#: Numerical slack for "this point satisfies that inequality". The constraints
#: are per-atom formation energies in eV, so 1e-9 is far below any DFT meaning
#: while staying well above double-precision noise in the LP.
FEASIBILITY_TOL = 1.0e-9

#: Two lines are treated as parallel when the magnitude of their 2x2 determinant
#: falls below this. Relative to normalised rows, so it is a genuine angle test.
PARALLEL_TOL = 1.0e-10


def _numpy() -> Any:
    import numpy as np

    return np


def plane_constraints(
    hull: Mapping[str, Any],
    compounds: Sequence[str],
    axes: Sequence[str],
    *,
    fixed_dmu: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Every phase's stability condition, rewritten as a line in the axis plane.

    Returns the reduction (how the eliminated potentials depend on the axes) and
    one row per supplied phase: ``a_n * dmu_axis0 + a_o * dmu_axis1 <= b``.
    """

    np = _numpy()
    api = _pymatgen()

    def snap(value: float) -> float:
        """Zero is zero. An elemental bound comes out as 1.7e-16 otherwise."""

        return 0.0 if abs(value) < 1.0e-12 else float(value)

    elements = list(hull["elements"])
    entries = list(hull["entries"])
    axes = [str(a) for a in axes]
    if len(axes) != 2 or axes[0] == axes[1]:
        raise SafetyError(f"exactly two distinct axes are required; got {axes}")
    unknown = [a for a in axes if a not in elements]
    if unknown:
        raise SafetyError(
            f"axis element(s) {unknown} are not in the hull ({elements}). Every "
            "axis needs an elemental reference among the supplied phases."
        )
    by_name = {entry.name: i for i, entry in enumerate(entries)}
    missing = [c for c in compounds if c not in by_name]
    if missing:
        raise SafetyError(f"compounds {missing} are not among the supplied phases")
    for name in compounds:
        row = next(r for r in hull["phases"] if r["phase"] == name)
        if not row["stable"]:
            raise SafetyError(
                f"compound {name!r} is above the convex hull; it cannot be held in "
                "equilibrium, so it cannot define the plane"
            )

    refs = np.array(
        [hull["diagram"].el_refs[api["Element"](el)].energy_per_atom for el in elements]
    )
    # Per-atom normalisation, exactly as the 1D window does it, so a supercell
    # cannot change the numbers.
    matrix = np.array(
        [
            [entry.composition[el] / entry.composition.num_atoms for el in elements]
            for entry in entries
        ]
    )
    formation = np.array([entry.energy_per_atom for entry in entries]) - matrix @ refs

    fixed = dict(fixed_dmu or {})
    bad = set(fixed) - set(elements)
    if bad:
        raise SafetyError(f"fixed_dmu names elements not in the hull: {sorted(bad)}")
    clash = set(fixed) & set(axes)
    if clash:
        raise SafetyError(
            f"cannot both fix and plot {sorted(clash)}: a fixed potential is not a "
            "free axis"
        )
    if not all(np.isfinite(v) for v in fixed.values()):
        raise SafetyError("fixed chemical potentials must be finite")

    equality = [matrix[by_name[name]] for name in compounds]
    values = [formation[by_name[name]] for name in compounds]
    for el, value in fixed.items():
        vector = np.zeros(len(elements))
        vector[elements.index(el)] = 1.0
        equality.append(vector)
        values.append(float(value))
    a_eq = np.array(equality, dtype=float)
    b_eq = np.array(values, dtype=float)

    reduction = reduce_to_plane(a_eq, b_eq, elements, axes)
    offset = reduction["offset"]  # dmu when both axes are zero
    gradient = reduction["gradient_matrix"]  # d(dmu)/d(axis), shape (n_elements, 2)

    rows = []
    for index, entry in enumerate(entries):
        coefficients = matrix[index] @ gradient  # shape (2,)
        bound = snap(float(formation[index] - matrix[index] @ offset))
        rows.append(
            {
                "phase": entry.name,
                "formula": entry.composition.reduced_formula,
                "coefficients": [snap(c) for c in coefficients],
                "bound_ev": bound,
                "is_constituent": entry.name in compounds,
                "is_elemental": bool(entry.composition.is_element),
            }
        )
    return {
        "elements": elements,
        "axes": axes,
        "fixed_dmu_ev": {k: float(v) for k, v in fixed.items()},
        "compounds": list(compounds),
        "reduction": reduction,
        "constraints": rows,
    }


def reduce_to_plane(
    a_eq: Any, b_eq: Any, elements: Sequence[str], axes: Sequence[str]
) -> dict[str, Any]:
    """Express every chemical potential as an affine function of the two axes.

    The equality system ``a_eq x = b_eq`` (the constituents held in equilibrium,
    plus anything explicitly fixed) must leave exactly two degrees of freedom,
    and the two requested axes must be genuine coordinates on that plane. Both
    are checked rather than assumed, because getting either wrong produces a
    picture that looks fine and means nothing.
    """

    np = _numpy()
    n = len(elements)
    if a_eq.size == 0:
        raise SafetyError("no equality constraints: name the interface's constituents")

    # Consistency and a particular solution, via least squares; then the null
    # space from the SVD, which is the numerically honest way to get its rank.
    solution, *_ = np.linalg.lstsq(a_eq, b_eq, rcond=None)
    residual = float(np.max(np.abs(a_eq @ solution - b_eq))) if len(b_eq) else 0.0
    if residual > 1.0e-8:
        raise SafetyError(
            "the named compounds cannot be in equilibrium simultaneously: their "
            f"equality constraints are inconsistent (residual {residual:.3e} eV/atom). "
            "Two phases of the same composition, or a fixed dmu that contradicts "
            "them, will do this."
        )
    _, singular, vt = np.linalg.svd(a_eq)
    rank = int(np.sum(singular > max(a_eq.shape) * np.finfo(float).eps * singular[0]))
    null = vt[rank:].T  # shape (n, n - rank)
    freedom = null.shape[1]
    if freedom != 2:
        raise SafetyError(
            f"the equality system leaves {freedom} degree(s) of freedom, but a 2D "
            f"map needs exactly 2. With {n} elements you need {n - 2} independent "
            "equalities: one per interface constituent, plus any --fixed-dmu. "
            + (
                "Name another constituent compound, or fix another potential."
                if freedom > 2
                else "Drop a constituent or a fixed potential -- you have over-determined the plane."
            )
        )

    index = [list(elements).index(a) for a in axes]
    projection = null[index, :]  # 2x2: how the free directions move the axes
    determinant = float(np.linalg.det(projection))
    if abs(determinant) < PARALLEL_TOL:
        raise SafetyError(
            f"({axes[0]}, {axes[1]}) are not independent coordinates on this plane "
            f"(the 2x2 projection is singular, det = {determinant:.3e}). The "
            "equalities already tie these two potentials together, so the map "
            "would collapse to a line. Choose different axes."
        )
    gradient = null @ np.linalg.inv(projection)  # (n, 2): d(dmu_i)/d(axis_j)
    offset = solution - gradient @ solution[index]  # dmu when both axes are 0
    return {
        "degrees_of_freedom": freedom,
        "offset_ev": {el: float(offset[i]) for i, el in enumerate(elements)},
        "gradient": [[float(v) for v in row] for row in gradient],
        "offset": offset,
        "gradient_matrix": gradient,
        "eliminated": [el for el in elements if el not in axes],
        "note": (
            "dmu_i = offset_i + sum_j gradient_ij * dmu_axis_j. The eliminated "
            "potentials are determined by the axes through the constituent "
            "equalities, not assumed."
        ),
    }


def _redundant(
    coefficients: Any, bound: float, others: Any, other_bounds: Any
) -> tuple[bool, float | None]:
    """Is this constraint implied by the others? Answer by LP, not by eyeball.

    Maximise the constraint's own left-hand side over everything else. If the
    maximum cannot reach its bound, the constraint never touches the region and
    is inactive. If the LP is unbounded, the constraint is doing real work --
    without it the region runs away in that direction.
    """

    np = _numpy()
    linprog = _linprog()
    if others.shape[0] == 0:
        return False, None
    result = linprog(
        -np.asarray(coefficients, dtype=float),
        A_ub=others,
        b_ub=other_bounds,
        bounds=[(None, None), (None, None)],
        method="highs",
    )
    if result.status == 3:  # unbounded: removing this constraint opens the region
        return False, None
    if result.status == 2:  # the others alone are infeasible
        return True, None
    if not result.success:
        raise SafetyError(f"redundancy test failed: {result.message}")
    reach = float(-result.fun)
    return reach <= bound - FEASIBILITY_TOL, reach


def stability_polygon(
    hull: Mapping[str, Any],
    compounds: Sequence[str],
    axes: Sequence[str] = ("N", "O"),
    *,
    fixed_dmu: Mapping[str, float] | None = None,
    clip: Mapping[str, tuple[float, float]] | None = None,
    margin_ev: float = 1.0,
) -> dict[str, Any]:
    """The feasible (dmu_a, dmu_b) region: its constraints, vertices and edges.

    Every boundary is reported with the phase that generates it, the inequality
    in the plane, and whether it actually limits the region. An unbounded region
    is reported as unbounded and then clipped for plotting -- the clip box is in
    the output, never applied silently.
    """

    np = _numpy()
    plane = plane_constraints(hull, compounds, axes, fixed_dmu=fixed_dmu)
    rows = plane["constraints"]
    axes = plane["axes"]

    # A constituent is an equality, already absorbed into the plane: its row is
    # 0 <= 0 and carries no information here. Drop it explicitly rather than
    # letting a degenerate row confuse the LP.
    live = []
    for row in rows:
        coefficients = np.array(row["coefficients"], dtype=float)
        if np.max(np.abs(coefficients)) < PARALLEL_TOL:
            row["degenerate"] = True
            row["satisfied_identically"] = bool(row["bound_ev"] >= -1.0e-8)
            if not row["satisfied_identically"]:
                raise SafetyError(
                    f"phase {row['phase']!r} reduces to the impossible condition "
                    f"0 <= {row['bound_ev']:.6f} eV/atom on this plane: it is more "
                    "stable than the constituents everywhere, so they can never "
                    "coexist. The interface as specified is not thermodynamically "
                    "reachable in this phase set."
                )
            continue
        row["degenerate"] = False
        live.append(row)
    if not live:
        raise SafetyError("no phase constrains this plane; add competing phases")

    matrix = np.array([r["coefficients"] for r in live], dtype=float)
    bounds = np.array([r["bound_ev"] for r in live], dtype=float)

    # Is the region non-empty at all?
    linprog = _linprog()
    probe = linprog(
        np.zeros(2), A_ub=matrix, b_ub=bounds,
        bounds=[(None, None), (None, None)], method="highs",
    )
    if probe.status == 2:
        raise SafetyError(
            f"the stability region is empty: no (dmu_{axes[0]}, dmu_{axes[1]}) satisfies every "
            f"phase while the constituents {list(compounds)} are in equilibrium. Either a "
            "competing phase is more stable than the interface's own components "
            "at the campaign's settings, or one of the energies is inconsistent "
            "with the rest."
        )
    if not probe.success:
        raise SafetyError(f"feasibility probe failed: {probe.message}")

    for index, row in enumerate(live):
        keep = [i for i in range(len(live)) if i != index]
        redundant, reach = _redundant(
            matrix[index], bounds[index], matrix[keep], bounds[keep]
        )
        row["active"] = not redundant
        row["slack_at_best_ev"] = None if reach is None else float(bounds[index] - reach)
        row["inequality"] = _inequality_text(row, axes)

    active_rows = [r for r in live if r["active"]]
    active = np.array([r["coefficients"] for r in active_rows], dtype=float)
    active_bounds = np.array([r["bound_ev"] for r in active_rows], dtype=float)

    extent = _extent(active, active_bounds, axes)
    vertices = _vertices(active_rows, active, active_bounds, axes)
    bounded = all(v is not None for pair in extent.values() for v in (pair["min"], pair["max"]))

    box = _clip_box(vertices, extent, clip, margin_ev, axes)
    ring = _clipped_ring(active, active_bounds, box)

    return {
        "schema_version": 1,
        "axes": axes,
        "axis_labels": {a: f"dmu({a}) (eV/atom)" for a in axes},
        "units": "eV per atom, zero at the elemental reference of each axis",
        "compounds": list(compounds),
        "fixed_dmu_ev": plane["fixed_dmu_ev"],
        "reduction": {
            k: v for k, v in plane["reduction"].items()
            if k not in {"offset", "gradient_matrix"}
        },
        "constraints": [
            {k: v for k, v in row.items() if k != "degenerate"} for row in rows
        ],
        "active_phases": [r["phase"] for r in active_rows],
        "inactive_phases": [r["phase"] for r in live if not r["active"]],
        "vertices": vertices,
        "extent_ev": extent,
        "bounded": bounded,
        "clip_box_ev": box,
        "polygon_ev": ring,
        "edges": _edges(active_rows, active, active_bounds, axes),
        "note": (
            "Every boundary was tested for redundancy by linear programming: an "
            "'active' phase limits the region, an 'inactive' one is implied by the "
            "others and is listed but not plotted. "
            + (
                "The region is bounded."
                if bounded
                else "The region is UNBOUNDED -- see extent_ev for which direction. "
                "The polygon returned for plotting is clipped to clip_box_ev; the "
                "clip is a drawing choice, not a thermodynamic bound."
            )
        ),
        "solid_approximation": SOLID_APPROXIMATION,
    }


def _inequality_text(row: Mapping[str, Any], axes: Sequence[str]) -> str:
    """The constraint as a human-checkable line, e.g. '-1.000 dmu(N) + 2.000 dmu(O) <= -10.500'."""

    a, b = row["coefficients"]
    parts = []
    for coefficient, axis in ((a, axes[0]), (b, axes[1])):
        if abs(coefficient) < PARALLEL_TOL:
            continue
        sign = "-" if coefficient < 0 else ("+" if parts else "")
        parts.append(f"{sign}{'' if not parts and coefficient > 0 else ' '}{abs(coefficient):.6g} dmu({axis})")
    left = " ".join(parts) if parts else "0"
    return f"{left} <= {row['bound_ev']:.6g}"


def _extent(matrix: Any, bounds: Any, axes: Sequence[str]) -> dict[str, Any]:
    """Min and max of each axis over the region, with the phases that set them."""

    np = _numpy()
    linprog = _linprog()
    out: dict[str, Any] = {}
    for position, axis in enumerate(axes):
        objective = np.zeros(2)
        objective[position] = 1.0
        pair: dict[str, Any] = {}
        for side, sign in (("min", 1.0), ("max", -1.0)):
            result = linprog(
                sign * objective, A_ub=matrix, b_ub=bounds,
                bounds=[(None, None), (None, None)], method="highs",
            )
            if result.status == 3:
                pair[side] = None
                pair[side + "_set_by"] = None
                continue
            if not result.success:
                raise SafetyError(f"extent optimisation failed: {result.message}")
            value = float(result.x[position])
            pair[side] = 0.0 if abs(value) < 1.0e-9 else value
            pair[side + "_set_by"] = [
                i for i in range(matrix.shape[0])
                if abs(result.ineqlin.marginals[i]) > 1.0e-8
            ]
        out[axis] = pair
    return out


def _vertices(
    rows: Sequence[Mapping[str, Any]], matrix: Any, bounds: Any, axes: Sequence[str]
) -> list[dict[str, Any]]:
    """Corners of the region: every pair of active constraints that meets inside it."""

    np = _numpy()
    found: list[dict[str, Any]] = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            pair = matrix[[i, j], :]
            determinant = float(np.linalg.det(pair))
            if abs(determinant) < PARALLEL_TOL:
                continue  # parallel boundaries never meet
            point = np.linalg.solve(pair, bounds[[i, j]])
            if np.any(matrix @ point > bounds + 1.0e-7):
                continue  # the crossing lies outside some other constraint
            if any(
                abs(v["point"][0] - point[0]) < 1.0e-7 and abs(v["point"][1] - point[1]) < 1.0e-7
                for v in found
            ):
                continue
            found.append(
                {
                    "point": [float(point[0]), float(point[1])],
                    axes[0]: float(point[0]),
                    axes[1]: float(point[1]),
                    "phases": sorted({rows[i]["phase"], rows[j]["phase"]}),
                }
            )
    return sorted(found, key=lambda v: (v["point"][0], v["point"][1]))


def _edges(
    rows: Sequence[Mapping[str, Any]], matrix: Any, bounds: Any, axes: Sequence[str]
) -> list[dict[str, Any]]:
    """Which phase controls the poor and rich end of each axis."""

    np = _numpy()
    linprog = _linprog()
    out = []
    for position, axis in enumerate(axes):
        for side, sign, label in (("min", 1.0, "poor"), ("max", -1.0, "rich")):
            objective = np.zeros(2)
            objective[position] = sign
            result = linprog(
                objective, A_ub=matrix, b_ub=bounds,
                bounds=[(None, None), (None, None)], method="highs",
            )
            if result.status == 3:
                out.append({
                    "edge": f"{axis}-{label}", "axis": axis, "side": side,
                    "value_ev": None, "controlled_by": [],
                    "note": f"unbounded: nothing in this phase set limits how {axis}-{label} the reservoir can get",
                })
                continue
            if not result.success:
                raise SafetyError(f"edge optimisation failed: {result.message}")
            binding = [
                rows[i]["phase"] for i in range(len(rows))
                if abs(result.ineqlin.marginals[i]) > 1.0e-8
            ]
            value = float(result.x[position])
            out.append({
                "edge": f"{axis}-{label}", "axis": axis, "side": side,
                "value_ev": 0.0 if abs(value) < 1.0e-9 else value,
                "controlled_by": sorted(binding),
                "note": f"the {axis}-{label} limit is set by {', '.join(sorted(binding)) or 'nothing'}",
            })
    return out


def _clip_box(
    vertices: Sequence[Mapping[str, Any]],
    extent: Mapping[str, Any],
    clip: Mapping[str, tuple[float, float]] | None,
    margin: float,
    axes: Sequence[str],
) -> dict[str, list[float]]:
    """The drawing window. Explicit if given, otherwise derived from the corners."""

    if clip:
        missing = [a for a in axes if a not in clip]
        if missing:
            raise SafetyError(f"clip must cover every axis; missing {missing}")
        return {a: [float(clip[a][0]), float(clip[a][1])] for a in axes}
    box = {}
    for position, axis in enumerate(axes):
        values = [v["point"][position] for v in vertices]
        low, high = extent[axis]["min"], extent[axis]["max"]
        if values:
            lo = min(values) if low is None else min(min(values), low)
            hi = max(values) if high is None else max(max(values), high)
        else:
            lo = -1.0 if low is None else low
            hi = 0.0 if high is None else high
        if hi - lo < 1.0e-6:
            lo, hi = lo - 1.0, hi + 1.0
        pad = margin if (low is None or high is None) else 0.15 * (hi - lo)
        box[axis] = [float(lo - (pad if low is None else 0.15 * (hi - lo))),
                     float(hi + (pad if high is None else 0.15 * (hi - lo)))]
    return box


def _clipped_ring(matrix: Any, bounds: Any, box: Mapping[str, list[float]]) -> list[list[float]]:
    """The region intersected with the drawing box, as an ordered ring of points."""

    np = _numpy()
    (x0, x1), (y0, y1) = box.values()
    walls = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
    wall_bounds = np.array([-x0, x1, -y0, y1])
    full = np.vstack([matrix, walls])
    full_bounds = np.concatenate([bounds, wall_bounds])
    points = []
    for i in range(len(full)):
        for j in range(i + 1, len(full)):
            pair = full[[i, j], :]
            if abs(float(np.linalg.det(pair))) < PARALLEL_TOL:
                continue
            point = np.linalg.solve(pair, full_bounds[[i, j]])
            if np.any(full @ point > full_bounds + 1.0e-7):
                continue
            points.append(point)
    if not points:
        return []
    array = np.array(points)
    centre = array.mean(axis=0)
    order = np.argsort(np.arctan2(array[:, 1] - centre[1], array[:, 0] - centre[0]))
    ring = array[order]
    out: list[list[float]] = []
    for point in ring:
        if out and abs(out[-1][0] - point[0]) < 1.0e-9 and abs(out[-1][1] - point[1]) < 1.0e-9:
            continue
        out.append([float(point[0]), float(point[1])])
    return out


# ------------------------------------------- putting a real atmosphere on it

def place_conditions(
    polygon: Mapping[str, Any],
    conditions: Sequence[Mapping[str, Any]],
    *,
    energies_ev: Mapping[str, float] | None = None,
    standard_pressure_bar: float = 1.0,
    include_zero_point_energy: bool = False,
    reference_scheme: str = "raw-dft",
) -> list[dict[str, Any]]:
    """Map (T, p) conditions onto the plane and say whether each one is inside.

    A condition that falls outside is the interesting case, so the violated
    phases are named with the margin by which they are violated -- that is the
    statement "at this atmosphere TiN oxidises", quantified.
    """

    np = _numpy()
    axes = list(polygon["axes"])
    rows = [
        row for row in polygon["constraints"]
        if row.get("active") and not row.get("degenerate")
    ]
    matrix = np.array([r["coefficients"] for r in rows], dtype=float) if rows else np.zeros((0, 2))
    bounds = np.array([r["bound_ev"] for r in rows], dtype=float) if rows else np.zeros(0)

    out = []
    for condition in conditions:
        pressures = {
            element: condition[key]
            for element, key in ((axes[0], "p_" + axes[0].lower()), (axes[1], "p_" + axes[1].lower()))
            if key in condition
        }
        pressures.update(condition.get("pressures", {}))
        if set(pressures) != set(axes):
            raise SafetyError(
                f"condition {condition!r} must give a partial pressure for every "
                f"axis {axes}; got {sorted(pressures)}"
            )
        gas = gas_conditions(
            condition["temperature_k"], pressures,
            unit=condition.get("pressure_unit", "bar"),
            energies_ev=dict(energies_ev or {}),
            standard_pressure_bar=standard_pressure_bar,
            include_zero_point_energy=include_zero_point_energy,
            reference_scheme=reference_scheme,
        )
        point = np.array([gas["dmu_ev_per_atom"][a] for a in axes])
        residual = matrix @ point - bounds if len(bounds) else np.zeros(0)
        violated = [
            {
                "phase": rows[i]["phase"],
                "violation_ev": float(residual[i]),
                "inequality": rows[i]["inequality"],
            }
            for i in range(len(rows))
            if residual[i] > FEASIBILITY_TOL
        ]
        limiting = _limiting_pressure(
            matrix, bounds, point, gas["reservoirs"][axes[1]], axes,
            standard_pressure_bar=standard_pressure_bar,
        )
        out.append({
            "label": condition.get("label"),
            "temperature_k": gas["temperature_k"],
            "pressure_unit": gas["pressure_unit"],
            "pressures": {a: float(pressures[a]) for a in axes},
            "dmu_ev_per_atom": {a: float(gas["dmu_ev_per_atom"][a]) for a in axes},
            "point": [float(v) for v in point],
            "inside": not violated,
            "violated_phases": sorted(violated, key=lambda r: -r["violation_ev"]),
            "closest_boundary_ev": (
                float(np.min(bounds - matrix @ point)) if len(bounds) else None
            ),
            "limiting_pressure": limiting,
            "reservoirs": gas["reservoirs"],
            "interpretation": (
                "inside the region: every supplied phase is stable against this reservoir"
                if not violated
                else "outside: " + ", ".join(
                    f"{v['phase']} is undercut by {v['violation_ev']:.3f} eV/atom"
                    for v in sorted(violated, key=lambda r: -r["violation_ev"])[:3]
                )
            ),
        })
    return out


def _limiting_pressure(
    matrix: Any,
    bounds: Any,
    point: Any,
    reservoir: Mapping[str, Any],
    axes: Sequence[str],
    *,
    standard_pressure_bar: float,
) -> dict[str, Any]:
    """How far the second axis's pressure would have to fall to reach the region.

    "Outside by 1.6 eV/atom" is hard to act on; "you would need p_O2 below 1e-54
    bar" is not. Holding the first axis fixed, the largest admissible dmu on the
    second axis is set by whichever constraint binds first, and inverting
    dmu = thermal + kB*T*ln(p/p0)/n gives the pressure that lands exactly on it.
    """

    np = _numpy()
    from .gas_reference import KB_EV_PER_K

    if not len(bounds):
        return {"feasible": None, "note": "no constraints"}
    column = matrix[:, 1]
    # Only constraints that push dmu(axis 1) down can cap it.
    capping = column > PARALLEL_TOL
    if not np.any(capping):
        return {
            "feasible": True,
            "pressure_bar": None,
            "note": f"nothing in this phase set caps dmu({axes[1]}) from above",
        }
    headroom = (bounds[capping] - matrix[capping, 0] * point[0]) / column[capping]
    ceiling = float(np.min(headroom))

    # Is the first axis itself already out of range? Then no pressure helps.
    fixed_axis_rows = np.abs(column) <= PARALLEL_TOL
    if np.any(fixed_axis_rows):
        residual = matrix[fixed_axis_rows, 0] * point[0] - bounds[fixed_axis_rows]
        if float(np.max(residual)) > FEASIBILITY_TOL:
            return {
                "feasible": False,
                "pressure_bar": None,
                "note": (
                    f"dmu({axes[0]}) = {point[0]:.4f} eV/atom is itself outside the "
                    f"region, so no p({axes[1]}) can rescue this condition -- change "
                    f"T or p({axes[0]}) instead"
                ),
            }

    temperature = float(reservoir["temperature_k"])
    n = int(reservoir["atoms_per_molecule"])
    # dmu_atomic = (thermal + zpe + empirical + kT ln(p/p0)) / n
    non_pressure = (
        reservoir["thermal_correction_ev"]
        + reservoir["zero_point_energy_ev"]
        + reservoir["empirical_correction_ev"]
    )
    exponent = (n * ceiling - non_pressure) / (KB_EV_PER_K * temperature)
    # exp() overflows long before the answer stops being "unreachably small".
    if exponent < -700.0:
        pressure_bar = 0.0
        reachable = False
    elif exponent > 700.0:
        pressure_bar = float("inf")
        reachable = True
    else:
        pressure_bar = float(math.exp(exponent)) * float(standard_pressure_bar)
        reachable = True
    return {
        "feasible": True,
        "axis": axes[1],
        "dmu_ceiling_ev": ceiling,
        "pressure_bar": pressure_bar,
        "pressure_atm": None if pressure_bar is None else pressure_bar / 1.01325,
        "log10_pressure_bar": (
            None if not reachable else
            (exponent + math.log(float(standard_pressure_bar))) / math.log(10.0)
        ),
        "note": (
            f"at T = {temperature:g} K and this p({axes[0]}), the region needs "
            f"dmu({axes[1]}) <= {ceiling:.4f} eV/atom, i.e. p({axes[1]}) below "
            + (
                f"10^{(exponent + math.log(float(standard_pressure_bar))) / math.log(10.0):.1f} bar"
                if reachable
                else "any pressure representable as a float (below 10^-304 bar)"
            )
            + ". Reaching that is a kinetic question, not a thermodynamic one."
        ),
    }


def map_report(
    phases: Mapping[str, Mapping[str, Any]],
    compounds: Sequence[str],
    axes: Sequence[str] = ("N", "O"),
    *,
    fixed_dmu: Mapping[str, float] | None = None,
    conditions: Sequence[Mapping[str, Any]] | None = None,
    clip: Mapping[str, tuple[float, float]] | None = None,
    include_zero_point_energy: bool = False,
    standard_pressure_bar: float = 1.0,
    reference_scheme: str = "raw-dft",
) -> dict[str, Any]:
    """Hull, 2D region, and any (T, p) conditions projected onto it."""

    hull = build_hull(phases)
    polygon = stability_polygon(hull, compounds, axes, fixed_dmu=fixed_dmu, clip=clip)
    # Only the diatomic gas references matter here. A 2-atom elemental Ti cell
    # is not a molecule, so filtering on "one element, two atoms" alone would
    # quietly label it one.
    energies = {}
    for phase in phases.values():
        composition = phase["composition"]
        if len(composition) != 1:
            continue
        element = str(next(iter(composition)))
        if element not in ELEMENT_TO_SPECIES:
            continue
        count = sum(composition.values())
        species = SPECIES[ELEMENT_TO_SPECIES[element]]
        if count != species.atoms_per_molecule:
            raise SafetyError(
                f"the {element} reference cell holds {count} atoms, but the gas "
                f"reference for {element} is {species.name}. Supply a single "
                "molecule per cell so the per-molecule energy is unambiguous."
            )
        energies[element] = float(phase["energy_ev"])
    placed = (
        place_conditions(
            polygon, conditions, energies_ev=energies,
            standard_pressure_bar=standard_pressure_bar,
            include_zero_point_energy=include_zero_point_energy,
            reference_scheme=reference_scheme,
        )
        if conditions
        else []
    )
    if placed and not clip:
        # A condition plotted outside the auto view is a marker nobody can see.
        # Widen the drawing window to hold them, and re-cut the ring to match.
        _widen_clip_for(polygon, placed)
    return {
        "schema_version": 1,
        "elements": hull["elements"],
        "phases": hull["phases"],
        "stable_phases": hull["stable_phases"],
        "unstable_phases": hull["unstable_phases"],
        "molecular_reference_energies_ev": energies,
        "stability_region": polygon,
        "conditions": placed,
        "note": (
            "The convex hull says which phases are stable and can therefore act as "
            "reservoirs; the (dmu, dmu) region says which reservoir conditions keep "
            "the interface's own constituents alive. They are different questions "
            "and this report keeps them separate on purpose."
        ),
    }


# --------------------------------------------------------------- reports

MAP_CSV_FIELDS = (
    "phase", "formula", "inequality", "active", "is_constituent", "is_elemental",
    "coefficient_axis0", "coefficient_axis1", "bound_ev", "slack_at_best_ev",
)

CONDITION_CSV_FIELDS = (
    "label", "temperature_k", "pressure_unit", "p_axis0", "p_axis1",
    "dmu_axis0_ev", "dmu_axis1_ev", "thermal_axis0_ev", "thermal_axis1_ev",
    "pressure_axis0_ev", "pressure_axis1_ev", "inside", "violated_phases",
)


def _map_markdown(payload: Mapping[str, Any]) -> str:
    """The inequalities, which of them bind, the corners, and what an atmosphere does."""

    region = payload["stability_region"]
    a, b = region["axes"]
    lines = [
        f"# Stability region in dmu({a})-dmu({b})",
        "",
        "Elements: {}. Constituents held in equilibrium: {}. Units: {}.".format(
            ", ".join(payload["elements"]), ", ".join(region["compounds"]), region["units"]
        ),
        "",
        "## How the plane is defined",
        "",
        "With {} elements and {} equality constraint(s) the feasible set has "
        "exactly 2 degrees of freedom, taken here to be dmu({}) and dmu({}). The "
        "eliminated potentials are *determined*, not assumed:".format(
            len(payload["elements"]),
            len(region["compounds"]) + len(region["fixed_dmu_ev"]), a, b,
        ),
        "",
        "| Potential | Value on the plane (eV/atom) |",
        "|---|---|",
    ]
    reduction = region["reduction"]
    for element in reduction["eliminated"]:
        index = payload["elements"].index(element)
        offset = reduction["offset_ev"][element]
        grad = reduction["gradient"][index]
        terms = "".join(
            f" {value:+.6g}*dmu({axis})"
            for value, axis in zip(grad, region["axes"], strict=True)
            if abs(value) > 1.0e-12
        )
        lines.append(f"| dmu({element}) | {offset:.6g}{terms} |")

    lines += [
        "",
        "## Phase boundaries",
        "",
        "Each inequality is that phase's stability condition after the eliminated "
        "potentials were substituted out. **Active** means the constraint really "
        "does limit the region: it was tested by maximising its own left-hand "
        "side over all the others, not by inspection.",
        "",
        "| Phase | Inequality (eV/atom) | Active | Slack at its best (eV/atom) |",
        "|---|---|---|---:|",
    ]
    for row in region["constraints"]:
        if row.get("is_constituent"):
            lines.append(f"| {row['phase']} | (equality: defines the plane) | - | - |")
            continue
        slack = row.get("slack_at_best_ev")
        lines.append("| {} | `{}` | {} | {} |".format(
            row["phase"], row.get("inequality", "0 <= 0"),
            "**yes**" if row.get("active") else "no",
            "-" if slack is None else f"{slack:.4f}",
        ))
    inactive = region["inactive_phases"]
    if inactive:
        lines += [
            "",
            "Inactive: {}. These are implied by the others everywhere in the "
            "region, so they are listed rather than plotted -- drawing them would "
            "suggest a boundary that never binds.".format(", ".join(inactive)),
        ]

    lines += [
        "", "## Corners", "",
        f"| dmu({a}) | dmu({b}) | Phases meeting there |", "|---:|---:|---|",
    ]
    for vertex in region["vertices"]:
        lines.append("| {:.4f} | {:.4f} | {} |".format(
            vertex["point"][0], vertex["point"][1], " + ".join(vertex["phases"])
        ))

    lines += [
        "", "## Which phase controls each edge", "",
        "| Edge | Limit (eV/atom) | Set by |", "|---|---:|---|",
    ]
    for edge in region["edges"]:
        value = edge["value_ev"]
        lines.append("| {} | {} | {} |".format(
            edge["edge"],
            "unbounded" if value is None else f"{value:.4f}",
            ", ".join(edge["controlled_by"]) or "-",
        ))
    if not region["bounded"]:
        lines += [
            "",
            "> The region is **unbounded**. The polygon drawn in the figure is "
            "clipped to {}; that clip is a drawing choice and carries no "
            "thermodynamic meaning.".format(region["clip_box_ev"]),
        ]

    conditions = payload.get("conditions") or []
    if conditions:
        lines += [
            "",
            "## Gas conditions",
            "",
            "One row per (T, p) atmosphere mapped onto the plane. The thermal and "
            "pressure columns are **per atom**, i.e. half the per-molecule value, "
            "which is the convention the solver uses.",
            "",
            f"| Condition | T (K) | p({a}) | p({b}) | thermal({a}) | press({a}) | "
            f"thermal({b}) | press({b}) | dmu({a}) | dmu({b}) | Inside |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in conditions:
            reservoirs = row["reservoirs"]
            lines.append(
                "| {} | {:.0f} | {:.3g} | {:.3g} | {:+.4f} | {:+.4f} | {:+.4f} | "
                "{:+.4f} | {:+.4f} | {:+.4f} | {} |".format(
                    row["label"] or "-", row["temperature_k"],
                    row["pressures"][a], row["pressures"][b],
                    reservoirs[a]["thermal_correction_atomic_ev"],
                    reservoirs[a]["pressure_correction_atomic_ev"],
                    reservoirs[b]["thermal_correction_atomic_ev"],
                    reservoirs[b]["pressure_correction_atomic_ev"],
                    row["dmu_ev_per_atom"][a], row["dmu_ev_per_atom"][b],
                    "yes" if row["inside"] else "**no**",
                )
            )
        outside = [row for row in conditions if not row["inside"]]
        if outside:
            lines += ["", "Conditions outside the region, and what undercuts them:", ""]
            for row in outside:
                lines.append(f"- {row['label'] or 'condition'}: {row['interpretation']}")

    lines += [
        "", "## Approximations", "", region["solid_approximation"], "",
        payload["note"], "",
    ]
    return "\n".join(lines) + "\n"


def _write_map_figure(payload: Mapping[str, Any], out: Path) -> dict[str, str]:
    """The feasible region, its active boundaries, and any atmospheres on it."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "Chemical-potential map figures require matplotlib; install "
            "interfaceforge[report]"
        ) from exc

    region = payload["stability_region"]
    a, b = region["axes"]
    x0, x1 = region["clip_box_ev"][a]
    y0, y1 = region["clip_box_ev"][b]
    style = {
        "font.family": "sans-serif", "font.size": 8.0, "axes.titlesize": 9.0,
        "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0, "axes.linewidth": 0.7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    # Okabe-Ito: readable in greyscale and to colour-blind readers.
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

    paths: dict[str, Path] = {}
    with plt.rc_context(style):
        figure, axis = plt.subplots(figsize=(5.4, 4.4))
        ring = region["polygon_ev"]
        if ring:
            axis.fill(
                [p[0] for p in ring], [p[1] for p in ring],
                color="#0072B2", alpha=0.13, lw=0, zorder=1,
                label="stable" if region["bounded"] else "stable (clipped)",
            )
        grid = np.linspace(x0, x1, 200)
        active = [r for r in region["constraints"] if r.get("active")]
        for index, row in enumerate(active):
            ca, cb = row["coefficients"]
            colour = palette[index % len(palette)]
            if abs(cb) > PARALLEL_TOL:  # a genuine sloping line in the plane
                axis.plot(
                    grid, (row["bound_ev"] - ca * grid) / cb,
                    color=colour, lw=1.2, zorder=3, label=row["phase"],
                )
            else:  # vertical: this one constrains the first axis alone
                axis.axvline(
                    row["bound_ev"] / ca, color=colour, lw=1.2, zorder=3,
                    label=row["phase"],
                )
        for vertex in region["vertices"]:
            axis.plot(*vertex["point"], marker="o", ms=3.6, color="#111111", zorder=5)

        # Conditions usually arrive as a (T x p) scan. Annotating all of them
        # stacks a dozen labels in one corner and reads as noise, so group by
        # temperature, give each group one legend entry, and label only the end
        # of each series. The full numbers are in the CSV and the markdown.
        rows = payload.get("conditions") or []
        groups: dict[float, list] = {}
        for row in rows:
            groups.setdefault(round(float(row["temperature_k"]), 6), []).append(row)
        markers = ["o", "s", "^", "D", "v", "P"]
        for index, temperature in enumerate(sorted(groups)):
            series = sorted(groups[temperature], key=lambda r: r["point"][1])
            colour = palette[(index + len(active)) % len(palette)]
            axis.plot(
                [r["point"][0] for r in series], [r["point"][1] for r in series],
                marker=markers[index % len(markers)], ms=4.2, lw=0.8, ls=":",
                color=colour, markeredgecolor="#FFFFFF", markeredgewidth=0.5,
                zorder=6, label=f"{temperature:g} K",
            )
            if len(groups) > 1 or len(series) > 3:
                extreme = series[0]  # the most negative dmu on the second axis
                if extreme.get("label"):
                    axis.annotate(
                        extreme["label"], extreme["point"],
                        textcoords="offset points", xytext=(5, -8),
                        fontsize=6.0, color=colour, zorder=7,
                    )
            else:
                for row in series:
                    if row.get("label"):
                        axis.annotate(
                            row["label"], row["point"], textcoords="offset points",
                            xytext=(5, 4), fontsize=6.2, color="#333333", zorder=7,
                        )
        # An open marker for any condition that does land inside the region.
        for row in rows:
            if row["inside"]:
                axis.plot(
                    row["point"][0], row["point"][1], marker="*", ms=9.0,
                    color="#111111", markeredgecolor="#FFFFFF",
                    markeredgewidth=0.6, zorder=8,
                )

        axis.set_xlim(x0, x1)
        axis.set_ylim(y0, y1)
        axis.set_xlabel(region["axis_labels"][a])
        axis.set_ylabel(region["axis_labels"][b])
        axis.set_title("Stability of {} in the ({}, {}) reservoir".format(
            " + ".join(region["compounds"]), a, b
        ))
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="best", frameon=False, ncols=2)
        caption = "0 K solids; gas reservoirs at T, p"
        if not region["bounded"]:
            caption += "; region unbounded, view clipped"
        axis.annotate(
            caption, (0.0, -0.16), xycoords="axes fraction",
            fontsize=6.5, color="#555555",
        )
        for suffix in ("png", "svg", "pdf"):
            path = out / ("chempot_map." + suffix)
            figure.savefig(path, dpi=300, bbox_inches="tight")
            paths["map_" + suffix] = path
        plt.close(figure)
    return {key: str(value) for key, value in paths.items()}


def write_reports(payload: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    """JSON, CSVs, markdown and the figure for one 2D map, into ``output_dir``."""

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "chempot_map.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    region = payload["stability_region"]
    with (out / "chempot_map_constraints.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MAP_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in region["constraints"]:
            writer.writerow({
                **row,
                "coefficient_axis0": row["coefficients"][0],
                "coefficient_axis1": row["coefficients"][1],
            })
    conditions = payload.get("conditions") or []
    if conditions:
        a, b = region["axes"]
        with (out / "chempot_map_conditions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CONDITION_CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in conditions:
                writer.writerow({
                    "label": row["label"],
                    "temperature_k": row["temperature_k"],
                    "pressure_unit": row["pressure_unit"],
                    "p_axis0": row["pressures"][a], "p_axis1": row["pressures"][b],
                    "dmu_axis0_ev": row["dmu_ev_per_atom"][a],
                    "dmu_axis1_ev": row["dmu_ev_per_atom"][b],
                    "thermal_axis0_ev": row["reservoirs"][a]["thermal_correction_atomic_ev"],
                    "thermal_axis1_ev": row["reservoirs"][b]["thermal_correction_atomic_ev"],
                    "pressure_axis0_ev": row["reservoirs"][a]["pressure_correction_atomic_ev"],
                    "pressure_axis1_ev": row["reservoirs"][b]["pressure_correction_atomic_ev"],
                    "inside": row["inside"],
                    "violated_phases": "; ".join(v["phase"] for v in row["violated_phases"]),
                })
    (out / "chempot_map.md").write_text(_map_markdown(payload), encoding="utf-8")
    outputs = {
        "json": str(out / "chempot_map.json"),
        "constraints_csv": str(out / "chempot_map_constraints.csv"),
        "markdown": str(out / "chempot_map.md"),
    }
    if conditions:
        outputs["conditions_csv"] = str(out / "chempot_map_conditions.csv")
    try:
        outputs.update(_write_map_figure(payload, out))
    except (DependencyError, SafetyError) as exc:
        outputs["figures"] = "skipped: " + str(exc)
    return outputs


def _widen_clip_for(
    polygon: dict[str, Any], conditions: Sequence[Mapping[str, Any]]
) -> None:
    """Grow the drawing window so every plotted condition is actually visible.

    Conditions routinely sit far outside the stable region -- that is usually the
    finding, not an error -- so a view fitted to the polygon alone would show a
    tidy diagram with all the markers off the edge.

    Only the view changes. The region, its vertices and its boundedness are
    untouched; the clip carries no thermodynamic meaning either way.
    """

    np = _numpy()
    axes = polygon["axes"]
    box = dict(polygon["clip_box_ev"])
    for position, axis in enumerate(axes):
        low, high = box[axis]
        values = [row["point"][position] for row in conditions]
        span = max(high - low, 1.0e-6)
        low = min(low, min(values) - 0.05 * span)
        high = max(high, max(values) + 0.05 * span)
        box[axis] = [float(low), float(high)]
    polygon["clip_box_ev"] = box
    rows = [
        row for row in polygon["constraints"]
        if row.get("active") and not row.get("degenerate")
    ]
    if rows:
        matrix = np.array([r["coefficients"] for r in rows], dtype=float)
        bounds = np.array([r["bound_ev"] for r in rows], dtype=float)
        polygon["polygon_ev"] = _clipped_ring(matrix, bounds, box)
