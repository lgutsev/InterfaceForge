"""Mechanism-stratified selection of reactive-surface labeling candidates."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..selection import select_indices

DEFAULT_STATE_COLUMNS = (
    "coverage",
    "motif",
    "initial_motif",
    "binding_mode",
    "initial_binding",
    "spin_state",
    "spin_status",
)


def select_surface_candidates(
    candidates: str | Path,
    output: str | Path,
    *,
    count: int,
    uncertainty_column: str = "uncertainty",
    feature_columns: tuple[str, ...] = (),
    state_columns: tuple[str, ...] = (),
    max_per_state: int | None = 2,
    uncertainty_weight: float = 0.65,
) -> dict[str, Any]:
    """Select uncertain/diverse frames without collapsing onto one mechanism.

    State columns are combined into one mechanism key.  When not supplied,
    every recognized chemical/magnetic state column present in the CSV is
    used.  This preserves labeling budget for rare reaction and spin classes.
    """
    source = Path(candidates).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no surface candidates in {source}")
    columns = set(rows[0])
    if uncertainty_column not in columns:
        raise ValueError(f"missing uncertainty column: {uncertainty_column}")
    selected_state_columns = tuple(state_columns) or tuple(
        column for column in DEFAULT_STATE_COLUMNS if column in columns
    )
    if not selected_state_columns:
        raise ValueError("no reactive-state columns found; provide --state-column (for example coverage and motif)")
    missing_features = [column for column in feature_columns if column not in columns]
    if missing_features:
        raise ValueError(f"missing feature columns: {missing_features}")
    groups = ["|".join(f"{column}={row.get(column, '')}" for column in selected_state_columns) for row in rows]
    uncertainty = [float(row[uncertainty_column]) for row in rows]
    features = (
        np.asarray([[float(row[column]) for column in feature_columns] for row in rows], dtype=float)
        if feature_columns
        else None
    )
    indices, diagnostics = select_indices(
        uncertainty,
        count,
        features=features,
        uncertainty_weight=uncertainty_weight,
        groups=groups,
        max_per_group=max_per_state,
    )
    selected: list[dict[str, Any]] = []
    for rank, (index, diagnostic) in enumerate(zip(indices, diagnostics, strict=True), start=1):
        selected.append(
            {
                "selection_rank": rank,
                "source_row": index,
                "surface_state": groups[index],
                **diagnostic,
                **rows[index],
            }
        )
    if not selected:
        raise ValueError("state quotas prevented every candidate from being selected")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    summary = {
        "source": str(source),
        "output": str(destination),
        "available": len(rows),
        "selected": len(selected),
        "uncertainty_column": uncertainty_column,
        "feature_columns": list(feature_columns),
        "state_columns": list(selected_state_columns),
        "state_count": len(set(groups)),
        "max_per_state": max_per_state,
        "uncertainty_weight": uncertainty_weight,
    }
    destination.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
