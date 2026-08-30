"""Reactive, magnetism-aware surface campaign tools.

This package deliberately sits above the generic geometry helpers.  It models
surface *states* (coverage, reaction motif, spin ordering, and provenance)
rather than providing another collection of POSCAR editing commands.
"""

from .campaign import build_surface_campaign, load_surface_campaign, plan_surface_campaign
from .cell import optimize_surface_cell
from .classify import audit_surface_runs
from .geometry import analyze_surface
from .selection import select_surface_candidates

__all__ = [
    "analyze_surface",
    "audit_surface_runs",
    "build_surface_campaign",
    "load_surface_campaign",
    "optimize_surface_cell",
    "plan_surface_campaign",
    "select_surface_candidates",
]
