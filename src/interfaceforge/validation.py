"""Property-aware validation shared by VASP, VASP-MLFF, MACE, and DeePMD."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

EV_A2_TO_J_M2 = 16.02176634


def _json_clean(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: _json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_clean(item) for item in value]
    return value


def scalar_metrics(reference: Sequence[float], predicted: Sequence[float]) -> dict[str, float]:
    ref = np.asarray(reference, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    if ref.shape != pred.shape or not ref.size:
        raise ValueError("Reference and prediction arrays must have the same nonzero shape")
    error = pred - ref
    return {
        "count": int(error.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_abs": float(np.max(np.abs(error))),
        "bias": float(np.mean(error)),
        "r2": float(1 - np.sum(error**2) / np.sum((ref - ref.mean()) ** 2))
        if np.sum((ref - ref.mean()) ** 2) > 0
        else float("nan"),
    }


def parity_from_csv(
    source: str | Path,
    output: str | Path,
    *,
    reference_column: str = "reference",
    predicted_column: str = "predicted",
    group_columns: Sequence[str] = ("model",),
) -> dict[str, Any]:
    input_path = Path(source).resolve()
    output_path = Path(output).resolve()
    with input_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(column, "") for column in group_columns)].append(row)
    results: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        metrics = scalar_metrics(
            [float(row[reference_column]) for row in group],
            [float(row[predicted_column]) for row in group],
        )
        results.append({**dict(zip(group_columns, key, strict=True)), **metrics})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    payload = {"source": str(input_path), "output": str(output_path), "groups": results}
    output_path.with_suffix(".json").write_text(
        json.dumps(_json_clean(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def work_of_adhesion(
    interface_energy_ev: float,
    slab_a_energy_ev: float,
    slab_b_energy_ev: float,
    area_a2: float,
) -> tuple[float, float]:
    if area_a2 <= 0:
        raise ValueError("Interface area must be positive")
    value = (slab_a_energy_ev + slab_b_energy_ev - interface_energy_ev) / area_a2
    return value, value * EV_A2_TO_J_M2


def adhesion_from_csv(source: str | Path, output: str | Path) -> dict[str, Any]:
    """Calculate adhesion and propagate independent energy uncertainties."""

    input_path = Path(source).resolve()
    output_path = Path(output).resolve()
    with input_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    results: list[dict[str, Any]] = []
    for row in rows:
        area = float(row["area_a2"])
        ev_a2, j_m2 = work_of_adhesion(
            float(row["interface_energy_ev"]),
            float(row["slab_a_energy_ev"]),
            float(row["slab_b_energy_ev"]),
            area,
        )
        sigma_values = [
            float(row.get(name, 0) or 0)
            for name in (
                "interface_sigma_ev",
                "slab_a_sigma_ev",
                "slab_b_sigma_ev",
            )
        ]
        sigma_ev_a2 = math.sqrt(sum(value**2 for value in sigma_values)) / area
        results.append(
            {
                **row,
                "work_of_adhesion_ev_a2": ev_a2,
                "work_of_adhesion_j_m2": j_m2,
                "work_of_adhesion_sigma_ev_a2": sigma_ev_a2,
                "work_of_adhesion_sigma_j_m2": sigma_ev_a2 * EV_A2_TO_J_M2,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    payload = {
        "source": str(input_path),
        "output": str(output_path),
        "conversion_ev_a2_to_j_m2": EV_A2_TO_J_M2,
        "rows": results,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def separation_curve_from_csv(source: str | Path, output: str | Path) -> dict[str, Any]:
    """Normalize rigid-separation energies to the largest separation per model."""

    input_path = Path(source).resolve()
    output_path = Path(output).resolve()
    with input_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get("model", "model")].append(row)
    results: list[dict[str, Any]] = []
    for model, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: float(row["distance_a"]))
        reference = float(ordered[-1]["energy_ev"])
        for row in ordered:
            delta = float(row["energy_ev"]) - reference
            area = float(row["area_a2"]) if row.get("area_a2") else None
            results.append(
                {
                    **row,
                    "model": model,
                    "delta_energy_ev": delta,
                    "traction_energy_j_m2": -delta / area * EV_A2_TO_J_M2 if area else "",
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    return {"source": str(input_path), "output": str(output_path), "rows": len(results)}
