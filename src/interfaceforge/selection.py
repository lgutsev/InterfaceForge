"""Uncertainty- and diversity-aware selection for DFT relabeling queues."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def _scaled(values: np.ndarray) -> np.ndarray:
    minimum = np.nanmin(values, axis=0)
    maximum = np.nanmax(values, axis=0)
    span = maximum - minimum
    span[span == 0] = 1.0
    return (values - minimum) / span


def select_indices(
    uncertainty: Sequence[float],
    count: int,
    *,
    features: np.ndarray | None = None,
    uncertainty_weight: float = 0.65,
    groups: Sequence[str] | None = None,
    max_per_group: int | None = None,
) -> tuple[list[int], list[dict[str, float]]]:
    """Select high-uncertainty candidates while spreading descriptor coverage."""

    values = np.asarray(uncertainty, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("uncertainty must be a non-empty one-dimensional array")
    if count < 1:
        raise ValueError("count must be positive")
    count = min(count, len(values))
    if not 0 <= uncertainty_weight <= 1:
        raise ValueError("uncertainty_weight must be between 0 and 1")
    uncertainty_scaled = _scaled(values[:, None]).ravel()
    descriptor = None
    if features is not None:
        descriptor = np.asarray(features, dtype=float)
        if descriptor.ndim != 2 or descriptor.shape[0] != len(values):
            raise ValueError("features must have one row per candidate")
        descriptor = _scaled(descriptor)
    selected: list[int] = []
    diagnostics: list[dict[str, float]] = []
    group_counts: Counter[str] = Counter()
    group_values = list(groups) if groups is not None else [""] * len(values)

    while len(selected) < count:
        best: tuple[float, int, float] | None = None
        for index in range(len(values)):
            if index in selected:
                continue
            group = group_values[index]
            if max_per_group is not None and group_counts[group] >= max_per_group:
                continue
            if descriptor is None or not selected:
                diversity = 1.0 if not selected else 0.0
            else:
                diversity = float(
                    min(np.linalg.norm(descriptor[index] - descriptor[chosen]) for chosen in selected)
                )
                diversity /= max(np.sqrt(descriptor.shape[1]), 1.0)
            score = uncertainty_weight * uncertainty_scaled[index] + (1 - uncertainty_weight) * diversity
            candidate = (score, index, diversity)
            if best is None or candidate[0] > best[0] or (
                candidate[0] == best[0] and values[index] > values[best[1]]
            ):
                best = candidate
        if best is None:
            break
        score, index, diversity = best
        selected.append(index)
        group_counts[group_values[index]] += 1
        diagnostics.append(
            {
                "selection_score": float(score),
                "uncertainty_scaled": float(uncertainty_scaled[index]),
                "diversity_score": float(diversity),
            }
        )
    return selected, diagnostics


def select_from_csv(
    candidates: str | Path,
    output: str | Path,
    *,
    count: int,
    uncertainty_column: str = "uncertainty",
    feature_columns: Sequence[str] = (),
    group_column: str | None = None,
    max_per_group: int | None = None,
    uncertainty_weight: float = 0.65,
) -> dict[str, Any]:
    """Rank a model-deviation table and write an auditable labeling queue."""

    source = Path(candidates).resolve()
    destination = Path(output).resolve()
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No candidates in {source}")
    if uncertainty_column not in rows[0]:
        raise ValueError(f"Missing uncertainty column: {uncertainty_column}")
    uncertainty = [float(row[uncertainty_column]) for row in rows]
    features = (
        np.asarray([[float(row[column]) for column in feature_columns] for row in rows])
        if feature_columns
        else None
    )
    groups = [row[group_column] for row in rows] if group_column else None
    indices, diagnostics = select_indices(
        uncertainty,
        count,
        features=features,
        uncertainty_weight=uncertainty_weight,
        groups=groups,
        max_per_group=max_per_group,
    )
    selected: list[dict[str, Any]] = []
    for rank, (index, diagnostic) in enumerate(
        zip(indices, diagnostics, strict=True), start=1
    ):
        selected.append(
            {
                "selection_rank": rank,
                "source_row": index,
                **diagnostic,
                **rows[index],
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
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
        "group_column": group_column,
        "max_per_group": max_per_group,
        "uncertainty_weight": uncertainty_weight,
    }
    destination.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
