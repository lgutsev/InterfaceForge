# ruff: noqa: E501
"""Bulk-referenced interfacial energy from the synchronized canonical dataset.

For a coherent periodic A/B stack with ``n_interfaces`` equivalent interfaces
per cell::

    gamma_int(T) = ( <E_interface>_T
                     - x * <E/fu(A)>_T
                     - y * <E/fu(B)>_T ) / (n_interfaces * area)

``x`` and ``y`` are the formula-unit counts obtained by matching the interface
composition against each bulk reference's own reduced formula, so a 1:1 nitride
and Si3N4 are both handled. Energies are MD averages of the DFT ``REF_energy``
over post-equilibration frames, with block-average standard errors. This is an
approximation to the interface free energy: the vibrational-entropy term is
dropped and largely cancels in the excess quantity.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np

from .errors import SafetyError

EV_A2_TO_J_M2 = 16.02176634
_INTERFACE_T = re.compile(r"(?:^|/)interface/(\d+)k/", re.IGNORECASE)
_BULK_T = re.compile(r"(\d+)k", re.IGNORECASE)


def _read_leaf(frame_map: Path) -> str:
    with frame_map.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle), None)
    if not row or "relative_leaf" not in row:
        raise SafetyError(f"{frame_map} lacks a relative_leaf column")
    return row["relative_leaf"]


def _leaf_systems(deepmd_root: Path) -> dict[str, list[Path]]:
    leaves: dict[str, list[Path]] = {}
    for split in ("train", "valid", "test"):
        split_dir = deepmd_root / split
        if not split_dir.is_dir():
            continue
        for type_raw in split_dir.rglob("type.raw"):
            system = type_raw.parent
            frame_map = system / "frame_map.csv"
            set_dir = system / "set.000"
            if not frame_map.is_file() or not (set_dir / "energy.npy").is_file():
                continue
            leaves.setdefault(_read_leaf(frame_map), []).append(system)
    if not leaves:
        raise SafetyError(f"No DeePMD leaf systems with frame_map.csv below {deepmd_root}")
    return leaves


def _composition(system: Path) -> dict[str, int]:
    type_map = (system / "type_map.raw").read_text(encoding="utf-8").split()
    counts: dict[str, int] = {}
    for token in (system / "type.raw").read_text(encoding="utf-8").split():
        symbol = type_map[int(token)]
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def _formula_units(composition: dict[str, int]) -> int:
    values = [v for v in composition.values() if v]
    divisor = 0
    for value in values:
        divisor = gcd(divisor, value)
    return divisor or 1


def _energies_by_source_frame(systems: list[Path]) -> list[tuple[int, float]]:
    pairs: list[tuple[int, float]] = []
    for system in systems:
        energy = np.load(system / "set.000" / "energy.npy").reshape(-1)
        with (system / "frame_map.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != energy.size:
            raise SafetyError(f"frame_map/energy length mismatch in {system}")
        for row, value in zip(rows, energy, strict=True):
            pairs.append((int(row["source_frame"]), float(value)))
    pairs.sort()
    return pairs


def _block_stats(values: np.ndarray, blocks: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if values.size < 2:
        return mean, 0.0
    use = min(blocks, values.size)
    block_means = np.asarray([chunk.mean() for chunk in np.array_split(values, use)])
    sem = float(block_means.std(ddof=1) / math.sqrt(use)) if use > 1 else float(values.std(ddof=1) / math.sqrt(values.size))
    return mean, sem


def _plane_area(system: Path, stacking_axis: str | None) -> tuple[float, str]:
    cell = np.load(system / "set.000" / "box.npy")[0].reshape(3, 3)
    axes = "abc"
    if stacking_axis and stacking_axis in axes:
        stack = axes.index(stacking_axis)
    else:
        stack = int(np.argmax([np.linalg.norm(v) for v in cell]))
    keep = [i for i in range(3) if i != stack]
    area = float(np.linalg.norm(np.cross(cell[keep[0]], cell[keep[1]])))
    return area, axes[stack]


def _test_systems_by_leaf(deepmd_root: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    test_dir = deepmd_root / "test"
    if not test_dir.is_dir():
        return mapping
    for frame_map in test_dir.rglob("frame_map.csv"):
        mapping[_read_leaf(frame_map)] = frame_map.parent
    return mapping


def _npz_energy(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return np.asarray(data["energy"], dtype=float).reshape(-1)


def _mlip_gamma(
    predictions_root: Path,
    test_systems: dict[str, Path],
    interface_leaf: str,
    tin_leaf: str,
    sin_leaf: str,
    x: float,
    y: float,
    denom: float,
    blocks: int,
) -> dict[str, Any]:
    manifest_path = predictions_root / "comparison_manifest.json"
    if not manifest_path.is_file():
        return {"status": f"no comparison_manifest.json in {predictions_root}"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    system_id = {s["relative_leaf"]: s["system_id"] for s in manifest["systems"]}
    models = [m["model"] for m in manifest["models"]]
    for leaf in (interface_leaf, tin_leaf, sin_leaf):
        if leaf not in system_id or leaf not in test_systems:
            return {"status": f"{leaf} absent from predictions or test split"}

    def series(leaf: str) -> dict[str, np.ndarray]:
        sid = system_id[leaf]
        out: dict[str, np.ndarray] = {}
        for model in models:
            npz = predictions_root / "predictions" / "mace" / model / f"{sid}.npz"
            if not npz.is_file():
                return {}
            out[model] = _npz_energy(npz)
        return out

    e_int, e_tin, e_sin = series(interface_leaf), series(tin_leaf), series(sin_leaf)
    if not (e_int and e_tin and e_sin):
        return {"status": "MACE prediction files incomplete"}

    tin_units = _formula_units(_composition(test_systems[tin_leaf]))
    sin_units = _formula_units(_composition(test_systems[sin_leaf]))

    def gamma(e_i: np.ndarray, e_t: np.ndarray, e_s: np.ndarray) -> tuple[float, float]:
        mi, si = _block_stats(e_i, blocks)
        mt, st = _block_stats(e_t, blocks)
        ms, ss = _block_stats(e_s, blocks)
        excess = mi - x * mt / tin_units - y * ms / sin_units
        sem = math.sqrt(
            (si / denom) ** 2
            + (x * st / tin_units / denom) ** 2
            + (y * ss / sin_units / denom) ** 2
        )
        return excess / denom * EV_A2_TO_J_M2, sem * EV_A2_TO_J_M2

    members = {
        model: round(gamma(e_int[model], e_tin[model], e_sin[model])[0], 4) for model in models
    }
    ensemble, ensemble_sem = gamma(
        np.mean([e_int[m] for m in models], axis=0),
        np.mean([e_tin[m] for m in models], axis=0),
        np.mean([e_sin[m] for m in models], axis=0),
    )
    dft, _ = gamma(
        np.load(test_systems[interface_leaf] / "set.000" / "energy.npy").reshape(-1),
        np.load(test_systems[tin_leaf] / "set.000" / "energy.npy").reshape(-1),
        np.load(test_systems[sin_leaf] / "set.000" / "energy.npy").reshape(-1),
    )
    spread = list(members.values())
    return {
        "status": "OK",
        "frames": int(next(iter(e_int.values())).size),
        "gamma_dft_same_frames_j_per_m2": dft,
        "gamma_ensemble_j_per_m2": ensemble,
        "gamma_ensemble_sem_j_per_m2": ensemble_sem,
        "gamma_members_j_per_m2": members,
        "member_spread_j_per_m2": float(np.std(spread, ddof=1)) if len(spread) > 1 else 0.0,
        "delta_mlip_minus_dft_j_per_m2": ensemble - dft,
    }


def _bulk_reference(systems: list[Path], equilibration: int, blocks: int) -> dict[str, Any]:
    composition = _composition(systems[0])
    units = _formula_units(composition)
    per_fu = {symbol: count // units for symbol, count in composition.items()}
    kept = np.asarray([e for frame, e in _energies_by_source_frame(systems) if frame >= equilibration])
    if kept.size == 0:
        raise SafetyError("No post-equilibration frames for a bulk reference")
    mean, sem = _block_stats(kept, blocks)
    return {
        "formula_units_per_cell": units,
        "atoms_per_formula_unit": per_fu,
        "energy_per_fu_ev": mean / units,
        "energy_per_fu_sem_ev": sem / units,
        "frames_used": int(kept.size),
    }


def interface_energy(
    campaign_root: str | Path,
    *,
    dataset_root: str | Path | None = None,
    predictions_root: str | Path | None = None,
    equilibration_frames: int = 100,
    n_interfaces: int = 2,
    blocks: int = 10,
    stacking_axis: str | None = None,
) -> dict[str, Any]:
    campaign = Path(campaign_root).expanduser().resolve()
    deepmd_root = (
        Path(dataset_root).expanduser().resolve()
        if dataset_root
        else campaign / "datasets" / "canonical" / "deepmd"
    )
    if n_interfaces < 1:
        raise SafetyError("n_interfaces must be a positive integer")
    leaves = _leaf_systems(deepmd_root)
    predictions = Path(predictions_root).expanduser().resolve() if predictions_root else None
    test_systems = _test_systems_by_leaf(deepmd_root) if predictions else {}

    bulk_refs: dict[str, dict[str, Any]] = {}
    for leaf, systems in sorted(leaves.items()):
        if leaf.startswith("bulk/"):
            bulk_refs[leaf] = _bulk_reference(systems, equilibration_frames, blocks)

    def _match_bulk(element: str, temperature: int) -> str | None:
        for leaf, reference in bulk_refs.items():
            if element in reference["atoms_per_formula_unit"] and "O" not in reference["atoms_per_formula_unit"]:
                match = _BULK_T.search(leaf)
                if match and int(match.group(1)) == temperature:
                    return leaf
        return None

    rows: list[dict[str, Any]] = []
    for leaf, systems in sorted(leaves.items()):
        if not leaf.startswith("interface/"):
            continue
        composition = _composition(systems[0])
        if composition.get("O", 0):
            continue  # v1: bulk-referenced, unoxidized interfaces only
        match = _INTERFACE_T.search(leaf)
        temperature = int(match.group(1)) if match else 0
        n_ti, n_si, n_n = composition.get("Ti", 0), composition.get("Si", 0), composition.get("N", 0)

        tin_leaf = _match_bulk("Ti", temperature)
        sin_leaf = _match_bulk("Si", temperature)
        row: dict[str, Any] = {
            "leaf": leaf,
            "temperature_K": temperature,
            "family": "Ideal" if "ideal" in leaf.lower() else ("Real" if "real" in leaf.lower() else "NA"),
            "termination": "Ti_Term" if "ti_term" in leaf.lower() else ("N_Term" if "n_term" in leaf.lower() else "NA"),
            "composition": composition,
        }
        if not tin_leaf or not sin_leaf:
            row["status"] = f"missing bulk reference at {temperature} K"
            rows.append(row)
            continue

        tin, sin = bulk_refs[tin_leaf], bulk_refs[sin_leaf]
        x = n_ti / tin["atoms_per_formula_unit"]["Ti"]
        y = n_si / sin["atoms_per_formula_unit"]["Si"]
        predicted_n = x * tin["atoms_per_formula_unit"].get("N", 0) + y * sin["atoms_per_formula_unit"].get("N", 0)

        kept = np.asarray([e for frame, e in _energies_by_source_frame(systems) if frame >= equilibration_frames])
        e_int, e_int_sem = _block_stats(kept, blocks)
        area, stack = _plane_area(systems[0], stacking_axis)
        denom = n_interfaces * area

        excess = e_int - x * tin["energy_per_fu_ev"] - y * sin["energy_per_fu_ev"]
        gamma_ev_a2 = excess / denom
        sem_ev_a2 = math.sqrt(
            (e_int_sem / denom) ** 2
            + (x * tin["energy_per_fu_sem_ev"] / denom) ** 2
            + (y * sin["energy_per_fu_sem_ev"] / denom) ** 2
        )

        row.update(
            {
                "tin_reference": tin_leaf,
                "sin_reference": sin_leaf,
                "tin_formula_units": round(x, 4),
                "sin_formula_units": round(y, 4),
                "nitrogen_expected": round(predicted_n, 3),
                "nitrogen_actual": n_n,
                "nitrogen_balanced": abs(predicted_n - n_n) < 0.5,
                "stacking_axis": stack,
                "interface_area_ang2": area,
                "n_interfaces": n_interfaces,
                "frames_used": int(kept.size),
                "interface_energy_ev": e_int,
                "gamma_int_ev_per_ang2": gamma_ev_a2,
                "gamma_int_j_per_m2": gamma_ev_a2 * EV_A2_TO_J_M2,
                "gamma_int_sem_j_per_m2": sem_ev_a2 * EV_A2_TO_J_M2,
                "status": "OK" if abs(predicted_n - n_n) < 0.5 else "nitrogen imbalance; check the clean split",
            }
        )
        if predictions is not None:
            row["mlip"] = _mlip_gamma(
                predictions, test_systems, leaf, tin_leaf, sin_leaf, x, y, denom, blocks
            )
        rows.append(row)

    return {
        "schema_version": 1,
        "campaign_root": str(campaign),
        "dataset_root": str(deepmd_root),
        "predictions_root": str(predictions) if predictions else None,
        "equilibration_frames": equilibration_frames,
        "n_interfaces": n_interfaces,
        "block_count": blocks,
        "conversion_ev_a2_to_j_m2": EV_A2_TO_J_M2,
        "reference_state": "MD-averaged bulk DFT energy per formula unit (potential energy, no vibrational entropy)",
        "mlip_note": (
            "MLIP columns use the MACE committee's predicted energies on the test-split "
            "frames only; gamma_dft_same_frames is the DFT value on those same frames."
            if predictions
            else None
        ),
        "bulk_references": bulk_refs,
        "interfaces": rows,
    }


_CSV_FIELDS = (
    "leaf",
    "temperature_K",
    "family",
    "termination",
    "tin_reference",
    "sin_reference",
    "tin_formula_units",
    "sin_formula_units",
    "nitrogen_balanced",
    "interface_area_ang2",
    "stacking_axis",
    "frames_used",
    "gamma_int_j_per_m2",
    "gamma_int_sem_j_per_m2",
    "status",
    "mlip_gamma_ensemble_j_per_m2",
    "mlip_gamma_dft_same_frames_j_per_m2",
    "mlip_delta_j_per_m2",
    "mlip_member_spread_j_per_m2",
)


def write_reports(payload: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "interface_energy.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with (out / "interface_energy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in payload["interfaces"]:
            flat = dict(row)
            mlip = row.get("mlip") or {}
            if mlip.get("status") == "OK":
                flat["mlip_gamma_ensemble_j_per_m2"] = mlip["gamma_ensemble_j_per_m2"]
                flat["mlip_gamma_dft_same_frames_j_per_m2"] = mlip["gamma_dft_same_frames_j_per_m2"]
                flat["mlip_delta_j_per_m2"] = mlip["delta_mlip_minus_dft_j_per_m2"]
                flat["mlip_member_spread_j_per_m2"] = mlip["member_spread_j_per_m2"]
            writer.writerow(flat)

    has_mlip = any((row.get("mlip") or {}).get("status") == "OK" for row in payload["interfaces"])
    lines = [
        "# Bulk-referenced interfacial energy",
        "",
        f"Reference: {payload['reference_state']}.",
        f"Equilibration frames dropped: {payload['equilibration_frames']}; interfaces per cell: {payload['n_interfaces']}.",
        "",
    ]
    if has_mlip:
        lines += [
            f"MLIP columns: {payload['mlip_note']}",
            "",
            "| Interface | T (K) | γ_DFT (J/m²) | γ_MLIP (J/m²) | Δ | member σ | γ_DFT same frames |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    else:
        lines += [
            "| Interface | T (K) | γ_int (J/m²) | ± | N-balanced | area (Å²) |",
            "|---|---:|---:|---:|:--:|---:|",
        ]
    for row in payload["interfaces"]:
        if "gamma_int_j_per_m2" not in row:
            lines.append(f"| {row['leaf']} | {row['temperature_K']} | — | — | — | — | — ({row['status']}) |")
            continue
        if has_mlip:
            mlip = row.get("mlip") or {}
            if mlip.get("status") == "OK":
                lines.append(
                    f"| {row['leaf']} | {row['temperature_K']} | {row['gamma_int_j_per_m2']:.3f} | "
                    f"{mlip['gamma_ensemble_j_per_m2']:.3f} | {mlip['delta_mlip_minus_dft_j_per_m2']:+.3f} | "
                    f"{mlip['member_spread_j_per_m2']:.3f} | {mlip['gamma_dft_same_frames_j_per_m2']:.3f} |"
                )
            else:
                lines.append(
                    f"| {row['leaf']} | {row['temperature_K']} | {row['gamma_int_j_per_m2']:.3f} | "
                    f"— | — | — | — ({mlip.get('status', 'no predictions')}) |"
                )
            continue
        lines.append(
            f"| {row['leaf']} | {row['temperature_K']} | {row['gamma_int_j_per_m2']:.3f} | "
            f"{row['gamma_int_sem_j_per_m2']:.3f} | {'yes' if row['nitrogen_balanced'] else 'NO'} | "
            f"{row['interface_area_ang2']:.1f} |"
        )
    lines += [
        "",
        "γ_int here is an approximation to the interface free energy: it is the MD-averaged",
        "potential-energy excess over the bulk phases, without the vibrational-entropy term",
        "(which largely cancels in an excess quantity). Oxidized interfaces are excluded;",
        "they need an oxygen chemical-potential treatment.",
    ]
    (out / "interface_energy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "json": str(out / "interface_energy.json"),
        "csv": str(out / "interface_energy.csv"),
        "markdown": str(out / "interface_energy.md"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_root", nargs="?", default=".")
    parser.add_argument("output_dir")
    parser.add_argument("--dataset-root")
    parser.add_argument("--predictions")
    parser.add_argument("--equilibration-frames", type=int, default=100)
    parser.add_argument("--n-interfaces", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--stacking-axis", choices=("a", "b", "c"))
    args = parser.parse_args(argv)
    payload = interface_energy(
        args.campaign_root,
        dataset_root=args.dataset_root,
        predictions_root=args.predictions,
        equilibration_frames=args.equilibration_frames,
        n_interfaces=args.n_interfaces,
        blocks=args.blocks,
        stacking_axis=args.stacking_axis,
    )
    payload["outputs"] = write_reports(payload, args.output_dir)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
