"""Aggregate DeePMD ``dp test -d`` outputs without averaging RMSE values."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

QUANTITIES = {
    "energy": (".e_peratom.out", 1, True),
    "force": (".f.out", 3, True),
    "virial": (".v_peratom.out", 9, False),
}


def _errors(path: Path, split: int) -> tuple[int, float, float]:
    count = 0
    absolute = 0.0
    squared = 0.0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            values = [float(value) for value in line.split()]
            reference, predicted = values[:split], values[split:]
            if len(reference) != len(predicted):
                raise RuntimeError(
                    f"Unbalanced reference/prediction columns in {path}"
                )
            for observed, estimate in zip(reference, predicted, strict=True):
                delta = estimate - observed
                count += 1
                absolute += abs(delta)
                squared += delta * delta
    if not count:
        raise RuntimeError(f"No numeric predictions in {path}")
    return count, absolute, squared


def _metrics(count: int, absolute: float, squared: float) -> dict[str, float | int | None]:
    if not count:
        return {"count": 0, "mae": None, "rmse": None}
    return {
        "count": count,
        "mae": absolute / count,
        "rmse": math.sqrt(squared / count),
    }


def _flatten(row: dict[str, Any]) -> dict[str, Any]:
    flat = {key: value for key, value in row.items() if key != "metrics"}
    for quantity, metrics in row["metrics"].items():
        for metric, value in metrics.items():
            flat[f"{quantity}_{metric}"] = value
    energy_rmse = row["metrics"]["energy"]["rmse"]
    force_rmse = row["metrics"]["force"]["rmse"]
    virial_rmse = row["metrics"]["virial"]["rmse"]
    flat["energy_rmse_mev_per_atom"] = energy_rmse * 1000 if energy_rmse is not None else None
    flat["force_rmse_mev_per_angstrom"] = force_rmse * 1000 if force_rmse is not None else None
    flat["virial_rmse_ev_per_atom"] = virial_rmse
    return flat


def summarize(
    eval_root: Path,
    systems_path: Path,
    architecture: str,
    seeds: list[int],
) -> dict[str, Any]:
    """Write exact component-weighted RMSE summaries for a committee."""

    systems = systems_path.read_text(encoding="utf-8").splitlines()
    if not systems:
        raise RuntimeError(f"No test systems listed in {systems_path}")
    rows: list[dict[str, Any]] = []
    totals: dict[str, dict[str, list[float]]] = {}
    for model_index, seed in enumerate(seeds):
        model = f"model_{model_index:03d}"
        totals[model] = {name: [0.0, 0.0, 0.0] for name in QUANTITIES}
        for system_index, system in enumerate(systems):
            label = f"system_{system_index:03d}"
            prefix = eval_root / "by_system" / label / f"{model}_detail"
            metrics: dict[str, dict[str, float | int | None]] = {}
            for name, (suffix, split, required) in QUANTITIES.items():
                path = Path(str(prefix) + suffix)
                if not path.is_file() and not required:
                    metrics[name] = _metrics(0, 0.0, 0.0)
                    continue
                if not path.is_file():
                    raise RuntimeError(f"Missing required DeePMD detail file: {path}")
                count, absolute, squared = _errors(path, split)
                metrics[name] = _metrics(count, absolute, squared)
                total = totals[model][name]
                total[0] += count
                total[1] += absolute
                total[2] += squared
            rows.append(
                {
                    "architecture": architecture,
                    "model": model,
                    "seed": seed,
                    "system_index": system_index,
                    "system": system,
                    "metrics": metrics,
                }
            )

    overall = []
    for model_index, seed in enumerate(seeds):
        model = f"model_{model_index:03d}"
        metrics = {
            name: _metrics(int(values[0]), values[1], values[2])
            for name, values in totals[model].items()
        }
        overall.append(
            {
                "architecture": architecture,
                "model": model,
                "seed": seed,
                "metrics": metrics,
            }
        )

    flat_rows = [_flatten(row) for row in rows]
    flat_overall = [_flatten(row) for row in overall]
    eval_root.mkdir(parents=True, exist_ok=True)
    with (eval_root / "rmse_by_system.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    with (eval_root / "rmse_overall.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_overall[0]))
        writer.writeheader()
        writer.writerows(flat_overall)

    payload = {
        "schema_version": 1,
        "architecture": architecture,
        "models": overall,
        "systems": rows,
    }
    (eval_root / "rmse_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--systems", type=Path, required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args(argv)
    summarize(args.eval_root, args.systems, args.architecture, args.seeds)
    print(f"Wrote {args.eval_root / 'rmse_overall.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
