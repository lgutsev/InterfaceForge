"""Which physical regime a property calculation belongs to, and its MLIP domain.

InterfaceForge property tools split into two families that must not be mixed:

``BULK`` (vacuum-free)
    Periodic cells with no free surface: bulk phases and coherent interface
    stacks. Metrics: grand-canonical interfacial energy ``gamma(mu)``,
    bulk-referenced excess. An MLIP trained on bulk and bonded-interface MD is
    inside its training distribution here.

``FREE_SURFACE`` (vacuum)
    Anything carrying a free surface: slabs, cleaved half-slabs, works of
    adhesion and separation, surface energies. A model that never saw an
    isolated surface extrapolates here, and for a polar termination it can be
    wrong by J/m^2 while its committee spread stays small -- exactly the
    failure this split exists to prevent.

The regime is measured from the structure (the periodic vacuum gap), recorded
in every payload, and enforced: a bulk-regime metric refuses a cell with
vacuum, and merges/overlays refuse to mix regimes.
"""

from __future__ import annotations

from typing import Any

from .errors import SafetyError

BULK = "bulk"
FREE_SURFACE = "free-surface"
REGIMES = (BULK, FREE_SURFACE)

#: A periodic gap at or above this (Angstrom) is a real vacuum, not a lattice
#: void. Interlayer spacings in these nitrides are ~1-3 A; deliberate slab
#: vacuum is >=10 A. 5 A separates them with room on both sides.
VACUUM_THRESHOLD_A = 5.0

MLIP_DOMAIN = {
    BULK: "bulk + bonded interface (in-distribution for a bulk/interface-trained committee)",
    FREE_SURFACE: "isolated free surfaces (requires surface configurations in training)",
}


def measure_regime(atoms: Any, *, axis: int | str = "auto") -> dict[str, Any]:
    """Classify one structure as ``bulk`` or ``free-surface`` from its vacuum gap."""

    from .geometry import slab_vacuum

    report = slab_vacuum(atoms, axis=axis)
    vacuum = float(report["vacuum_a"])
    regime = FREE_SURFACE if vacuum >= VACUUM_THRESHOLD_A else BULK
    return {
        "regime": regime,
        "vacuum_a": vacuum,
        "vacuum_axis": report["axis"],
        "slab_span_a": float(report["slab_span_a"]),
        "vacuum_threshold_a": VACUUM_THRESHOLD_A,
    }


def require_regime(
    atoms: Any,
    expected: str,
    label: str,
    *,
    axis: int | str = "auto",
    allow_mismatch: bool = False,
) -> dict[str, Any]:
    """Measure the regime and refuse a structure that belongs to the other one."""

    if expected not in REGIMES:
        raise SafetyError(f"regime must be one of {REGIMES}; got {expected!r}")
    found = measure_regime(atoms, axis=axis)
    if found["regime"] == expected or allow_mismatch:
        found["expected_regime"] = expected
        found["regime_override"] = found["regime"] != expected
        return found
    if expected == BULK:
        raise SafetyError(
            f"{label}: this is a {FREE_SURFACE} structure -- {found['vacuum_a']:.1f} A of "
            f"vacuum along {found['vacuum_axis']} (threshold {VACUUM_THRESHOLD_A:.0f} A). "
            "Vacuum-free metrics (gamma(mu), bulk-referenced excess) are only defined for a "
            "periodic cell with no free surface; a slab's two outer surfaces would be folded "
            "into the number. Use a periodic interface cell, or run a free-surface metric "
            "(iface validate separation-energy / adhesion) instead."
        )
    raise SafetyError(
        f"{label}: this is a {BULK} structure -- only {found['vacuum_a']:.1f} A of vacuum "
        f"along {found['vacuum_axis']}. Free-surface metrics (work of adhesion/separation) "
        "need a slab with real vacuum. Use a slab, or run a vacuum-free metric "
        "(iface validate interface-mu / interface-energy) instead."
    )


def merge_regimes(regimes: list[str], context: str) -> str:
    """One regime for a merged/overlaid payload; refuse a mixed set."""

    unique = sorted(set(regimes))
    if not unique:
        raise SafetyError(f"{context}: no regime recorded")
    if len(unique) > 1:
        raise SafetyError(
            f"{context}: refusing to combine {unique} -- vacuum-free and free-surface "
            "results are not comparable, and an MLIP validated in one is not validated "
            "in the other. Overlay them in separate figures."
        )
    return unique[0]
