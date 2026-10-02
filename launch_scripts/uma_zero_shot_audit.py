#!/usr/bin/env python3
"""Zero-shot UMA audit for the periodic SiN/TiN/TiO campaign.

The audit deliberately uses only the held-out canonical InterfaceForge frames.
It compares stock UMA/OMat against the existing DFT labels for:
  * forces
  * stress/virial
  * raw total energy
  * composition-centered relative energy

Composition centering is important because the campaign DFT setup and OMat's
training reference are not guaranteed to share identical elemental energy
zeros, pseudopotentials, or dispersion treatment.

This file has no InterfaceForge runtime dependency; it is intended to run in a
dedicated fairchem environment on LONI.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

EV_A3_TO_GPA = 160.21766208
ROW_FIELDS = (
    "frame_index",
    "source_frame",
    "leaf",
    "heritage",
    "temperature",
    "family",
    "termination",
    "oxidation",
    "composition",
    "natoms",
    "ref_energy_ev",
    "uma_energy_ev",
    "raw_energy_error_mev_per_atom",
    "centered_energy_error_mev_per_atom",
    "force_rmse_ev_per_A",
    "force_mae_ev_per_A",
    "force_max_ev_per_A",
    "stress_rmse_GPa",
    "stress_mae_GPa",
    "stress_max_GPa",
    "_force_sse",
    "_force_abs_sum",
    "_force_n",
    "_stress_sse",
    "_stress_abs_sum",
    "_stress_n",
)


def _safe_model_name(value: str) -> str:
    stem = Path(value).stem if ("/" in value or "\\" in value or value.endswith(".pt")) else value
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or "uma"


def _dataset_path(campaign: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"Dataset does not exist: {path}")
        return path
    candidates = (
        campaign / "datasets" / "canonical" / "test.extxyz",
        campaign / "models" / "mace_committee_520eV" / "test.extxyz",
    )
    for path in candidates:
        if path.is_file() and path.stat().st_size:
            return path.resolve()
    raise SystemExit(
        "Could not find held-out test.extxyz. Expected one of:\n  "
        + "\n  ".join(str(path) for path in candidates)
    )


def _composition(atoms: Any) -> str:
    counts = Counter(atoms.get_chemical_symbols())
    return "_".join(f"{symbol}{counts[symbol]}" for symbol in sorted(counts))


def _groups(atoms: Any) -> dict[str, str]:
    leaf = str(atoms.info.get("IF_leaf", "")).strip()
    lower = leaf.lower()
    heritage = "bulk" if leaf.startswith("bulk/") else "interface"
    temperature = re.search(r"(?<!\d)(300|450|600)k", lower)
    oxidation_match = re.search(r"o[_-]?x[_=-]?([01](?:\.\d+)?)", lower)
    oxidation = (
        "NA"
        if heritage == "bulk"
        else (f"{float(oxidation_match.group(1)):g}" if oxidation_match else "0")
    )
    return {
        "leaf": leaf or "NA",
        "heritage": heritage,
        "temperature": f"{temperature.group(1)}K" if temperature else "NA",
        "family": "Ideal" if "ideal" in lower else ("Real" if "real" in lower else "NA"),
        "termination": (
            "Ti_Term"
            if "ti_term" in lower
            else ("N_Term" if "n_term" in lower else "NA")
        ),
        "oxidation": oxidation,
    }


def _reference_stress_gpa(atoms: Any) -> np.ndarray | None:
    raw = atoms.info.get("REF_virial")
    if raw is None:
        return None
    virial = np.asarray(raw, dtype=np.float64).reshape(3, 3)
    volume = float(atoms.get_volume())
    if not math.isfinite(volume) or volume <= 1.0e-12:
        raise ValueError(f"Invalid cell volume {volume}")
    # InterfaceForge writes REF_virial = -V * ASE_stress.
    return (-virial / volume) * EV_A3_TO_GPA


def _load_calculator(model: str, task: str, device: str, inference_settings: str):
    try:
        from fairchem.core import FAIRChemCalculator
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "fairchem-core is not installed in this Python environment. "
            "Activate the UMA environment before running the audit."
        ) from exc
    return FAIRChemCalculator.from_model_checkpoint(
        name_or_path=model,
        task_name=task,
        inference_settings=inference_settings,
        device=device,
    )


def _load_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or not path.stat().st_size:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    int_fields = {"frame_index", "source_frame", "natoms", "_force_n", "_stress_n"}
    float_fields = {
        "ref_energy_ev",
        "uma_energy_ev",
        "raw_energy_error_mev_per_atom",
        "centered_energy_error_mev_per_atom",
        "force_rmse_ev_per_A",
        "force_mae_ev_per_A",
        "force_max_ev_per_A",
        "stress_rmse_GPa",
        "stress_mae_GPa",
        "stress_max_GPa",
        "_force_sse",
        "_force_abs_sum",
        "_stress_sse",
        "_stress_abs_sum",
    }
    for raw in raw_rows:
        row: dict[str, Any] = dict(raw)
        for key in int_fields:
            value = row.get(key, "")
            row[key] = int(value) if value not in ("", None) else 0
        for key in float_fields:
            value = row.get(key, "")
            row[key] = float(value) if value not in ("", None) else math.nan
        rows.append(row)
    return rows


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in ROW_FIELDS})
    temporary.replace(path)


def _apply_composition_centering(rows: list[dict[str, Any]]) -> int:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["composition"])].append(row)
    usable = 0
    for members in groups.values():
        if len(members) < 2:
            for row in members:
                row["centered_energy_error_mev_per_atom"] = math.nan
            continue
        # Remove the constant total-energy offset for this fixed composition.
        offsets = [
            (float(row["uma_energy_ev"]) - float(row["ref_energy_ev"]))
            for row in members
        ]
        mean_offset = float(np.mean(offsets))
        for row, offset in zip(members, offsets, strict=True):
            natoms = int(row["natoms"])
            row["centered_energy_error_mev_per_atom"] = (
                (offset - mean_offset) * 1000.0 / natoms
            )
            usable += 1
    return usable


def _rmse(values: list[float]) -> float:
    finite = np.asarray(
        [value for value in values if math.isfinite(value)], dtype=np.float64
    )
    if not len(finite):
        return math.nan
    return float(np.sqrt(np.mean(finite**2)))


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "frames": 0,
            "atoms": 0,
            "energy_raw_rmse_mev_per_atom": math.nan,
            "energy_centered_rmse_mev_per_atom": math.nan,
            "energy_centered_frames": 0,
            "force_rmse_ev_per_A": math.nan,
            "force_mae_ev_per_A": math.nan,
            "force_max_ev_per_A": math.nan,
            "stress_rmse_GPa": math.nan,
            "stress_mae_GPa": math.nan,
            "stress_max_GPa": math.nan,
            "stress_frames": 0,
        }
    force_n = sum(int(row["_force_n"]) for row in rows)
    stress_rows = [row for row in rows if int(row.get("_stress_n", 0)) > 0]
    stress_n = sum(int(row["_stress_n"]) for row in stress_rows)
    centered = [
        float(row["centered_energy_error_mev_per_atom"])
        for row in rows
        if math.isfinite(float(row["centered_energy_error_mev_per_atom"]))
    ]
    return {
        "frames": len(rows),
        "atoms": sum(int(row["natoms"]) for row in rows),
        "energy_raw_rmse_mev_per_atom": _rmse(
            [float(row["raw_energy_error_mev_per_atom"]) for row in rows]
        ),
        "energy_centered_rmse_mev_per_atom": _rmse(centered),
        "energy_centered_frames": len(centered),
        "force_rmse_ev_per_A": (
            math.sqrt(sum(float(row["_force_sse"]) for row in rows) / force_n)
            if force_n
            else math.nan
        ),
        "force_mae_ev_per_A": (
            sum(float(row["_force_abs_sum"]) for row in rows) / force_n
            if force_n
            else math.nan
        ),
        "force_max_ev_per_A": max(float(row["force_max_ev_per_A"]) for row in rows),
        "stress_rmse_GPa": (
            math.sqrt(sum(float(row["_stress_sse"]) for row in stress_rows) / stress_n)
            if stress_n
            else math.nan
        ),
        "stress_mae_GPa": (
            sum(float(row["_stress_abs_sum"]) for row in stress_rows) / stress_n
            if stress_n
            else math.nan
        ),
        "stress_max_GPa": (
            max(float(row["stress_max_GPa"]) for row in stress_rows)
            if stress_rows
            else math.nan
        ),
        "stress_frames": len(stress_rows),
    }


def _summaries(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    overall = _aggregate(rows)
    grouped: list[dict[str, Any]] = []
    for key in (
        "heritage",
        "family",
        "termination",
        "oxidation",
        "temperature",
        "composition",
        "leaf",
    ):
        values: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            values[str(row[key])].append(row)
        for value, members in sorted(values.items()):
            grouped.append(
                {"group_type": key, "group": value, **_aggregate(members)}
            )
    return overall, grouped


def _gate(value: float, threshold: float) -> dict[str, Any]:
    if not math.isfinite(value):
        return {"status": "NA", "value": value, "threshold": threshold}
    return {
        "status": "PASS" if value <= threshold else "FAIL",
        "value": value,
        "threshold": threshold,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _write_group_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "group_type",
        "group",
        "frames",
        "atoms",
        "energy_raw_rmse_mev_per_atom",
        "energy_centered_rmse_mev_per_atom",
        "energy_centered_frames",
        "force_rmse_ev_per_A",
        "force_mae_ev_per_A",
        "force_max_ev_per_A",
        "stress_rmse_GPa",
        "stress_mae_GPa",
        "stress_max_GPa",
        "stress_frames",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{value:.{digits}f}" if math.isfinite(value) else "NA"


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    overall = payload["overall"]
    gates = payload["diagnostic_gates"]
    lines = [
        "# Stock UMA SiN/TiN zero-shot audit",
        "",
        f"- Model: `{payload['model']}`",
        f"- Task: `{payload['task']}`",
        f"- Dataset: `{payload['dataset']}`",
        (
            f"- Frames: {overall['frames']} "
            f"({overall['stress_frames']} with virial/stress labels)"
        ),
        "",
        "## Overall",
        "",
        "| Metric | Value | Diagnostic gate |",
        "|---|---:|---:|",
        (
            "| composition-centered energy RMSE | "
            f"{_fmt(overall['energy_centered_rmse_mev_per_atom'])} meV/atom | "
            f"{gates['centered_energy']['status']} ≤ "
            f"{_fmt(gates['centered_energy']['threshold'])} meV/atom |"
        ),
        (
            "| force RMSE | "
            f"{_fmt(overall['force_rmse_ev_per_A'])} eV/Å | "
            f"{gates['force']['status']} ≤ "
            f"{_fmt(gates['force']['threshold'])} eV/Å |"
        ),
        (
            "| stress RMSE | "
            f"{_fmt(overall['stress_rmse_GPa'])} GPa | "
            f"{gates['stress']['status']} ≤ "
            f"{_fmt(gates['stress']['threshold'])} GPa |"
        ),
        (
            "| raw energy RMSE | "
            f"{_fmt(overall['energy_raw_rmse_mev_per_atom'])} meV/atom | "
            "diagnostic only |"
        ),
        "",
        "The PASS/FAIL gates are smoke-test thresholds, not publication acceptance criteria.",
        "Raw total-energy error is secondary because the campaign VASP setup and OMat",
        "do not necessarily share identical elemental reference energies, pseudopotentials,",
        "or dispersion treatment. The composition-centered metric removes one constant",
        "energy offset per fixed composition and therefore tests relative energies.",
        "",
        "## Files",
        "",
        "- `frames.csv`: per-frame energy/force/stress errors.",
        (
            "- `summary_by_group.csv`: bulk/interface, termination, oxidation, "
            "temperature,"
        ),
        "  composition, and leaf summaries.",
        "- `metrics.json`: machine-readable full report and diagnostic gates.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from ase.io import iread

    campaign = args.campaign.expanduser().resolve()
    if not campaign.is_dir():
        raise SystemExit(f"Campaign root does not exist: {campaign}")
    dataset = _dataset_path(campaign, args.dataset)
    output = (
        args.output.expanduser().resolve()
        if args.output
        else campaign / "audit" / "uma_zero_shot" / _safe_model_name(args.model)
    )
    if args.force and output.exists():
        try:
            output.relative_to(campaign)
        except ValueError as exc:
            raise SystemExit(
                f"Refusing --force outside campaign root: {output}"
            ) from exc
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "frames.csv"
    if rows_path.exists() and not args.resume and not args.force:
        raise SystemExit(
            f"{rows_path} already exists. Use --resume to continue or --force to replace it."
        )
    rows = _load_existing_rows(rows_path) if args.resume else []
    done = {int(row["frame_index"]) for row in rows}
    calculator = _load_calculator(
        args.model, args.task, args.device, args.inference_settings
    )
    start = time.time()
    evaluated_now = 0
    for frame_index, atoms in enumerate(iread(str(dataset), index=":")):
        if frame_index % args.stride:
            continue
        if args.max_frames and (len(done) + evaluated_now) >= args.max_frames:
            break
        if frame_index in done:
            continue
        if "REF_energy" not in atoms.info or "REF_forces" not in atoms.arrays:
            raise SystemExit(
                f"Frame {frame_index} lacks REF_energy/REF_forces canonical labels"
            )
        ref_energy = float(atoms.info["REF_energy"])
        ref_forces = np.asarray(atoms.arrays["REF_forces"], dtype=np.float64)
        if ref_forces.shape != (len(atoms), 3):
            raise SystemExit(
                f"Frame {frame_index}: REF_forces shape {ref_forces.shape}, "
                f"expected {(len(atoms), 3)}"
            )
        ref_stress = _reference_stress_gpa(atoms)
        clean = atoms.copy()
        clean.calc = calculator
        uma_energy = float(clean.get_potential_energy())
        uma_forces = np.asarray(clean.get_forces(), dtype=np.float64)
        force_error = uma_forces - ref_forces

        stress_error: np.ndarray | None = None
        if ref_stress is not None:
            try:
                uma_stress_gpa = (
                    np.asarray(clean.get_stress(voigt=False), dtype=np.float64)
                    * EV_A3_TO_GPA
                )
            except Exception as exc:
                raise SystemExit(
                    f"Frame {frame_index}: UMA/{args.task} did not return stress: {exc}"
                ) from exc
            stress_error = uma_stress_gpa - ref_stress

        groups = _groups(atoms)
        natoms = len(atoms)
        raw_energy_error = (uma_energy - ref_energy) * 1000.0 / natoms
        row = {
            "frame_index": frame_index,
            "source_frame": int(atoms.info.get("source_frame", -1)),
            **groups,
            "composition": _composition(atoms),
            "natoms": natoms,
            "ref_energy_ev": ref_energy,
            "uma_energy_ev": uma_energy,
            "raw_energy_error_mev_per_atom": raw_energy_error,
            "centered_energy_error_mev_per_atom": math.nan,
            "force_rmse_ev_per_A": float(np.sqrt(np.mean(force_error**2))),
            "force_mae_ev_per_A": float(np.mean(np.abs(force_error))),
            "force_max_ev_per_A": float(np.max(np.abs(force_error))),
            "stress_rmse_GPa": (
                float(np.sqrt(np.mean(stress_error**2)))
                if stress_error is not None
                else math.nan
            ),
            "stress_mae_GPa": (
                float(np.mean(np.abs(stress_error)))
                if stress_error is not None
                else math.nan
            ),
            "stress_max_GPa": (
                float(np.max(np.abs(stress_error)))
                if stress_error is not None
                else math.nan
            ),
            "_force_sse": float(np.sum(force_error**2)),
            "_force_abs_sum": float(np.sum(np.abs(force_error))),
            "_force_n": int(force_error.size),
            "_stress_sse": (
                float(np.sum(stress_error**2)) if stress_error is not None else 0.0
            ),
            "_stress_abs_sum": (
                float(np.sum(np.abs(stress_error)))
                if stress_error is not None
                else 0.0
            ),
            "_stress_n": int(stress_error.size) if stress_error is not None else 0,
        }
        rows.append(row)
        evaluated_now += 1
        # Re-write atomically after every frame. This is cheap relative to a UMA
        # force/stress call and makes preemption/resume safe.
        _apply_composition_centering(rows)
        _write_rows(rows_path, rows)
        if evaluated_now == 1 or evaluated_now % args.progress_every == 0:
            elapsed = time.time() - start
            print(
                f"UMA {args.model}: {len(rows)} total frames "
                f"({evaluated_now} this run), {elapsed:.1f} s",
                flush=True,
            )

    if not rows:
        raise SystemExit("No frames were evaluated")
    centered_frames = _apply_composition_centering(rows)
    _write_rows(rows_path, rows)
    overall, grouped = _summaries(rows)
    _write_group_csv(output / "summary_by_group.csv", grouped)
    gates = {
        "centered_energy": _gate(
            float(overall["energy_centered_rmse_mev_per_atom"]),
            args.max_centered_energy_rmse,
        ),
        "force": _gate(
            float(overall["force_rmse_ev_per_A"]), args.max_force_rmse
        ),
        "stress": _gate(
            float(overall["stress_rmse_GPa"]), args.max_stress_rmse_gpa
        ),
    }
    payload = {
        "schema_version": 1,
        "model": args.model,
        "task": args.task,
        "device": args.device,
        "inference_settings": args.inference_settings,
        "campaign": str(campaign),
        "dataset": str(dataset),
        "output": str(output),
        "stride": args.stride,
        "max_frames": args.max_frames,
        "frames_evaluated_this_run": evaluated_now,
        "composition_centered_frames": centered_frames,
        "overall": overall,
        "diagnostic_gates": gates,
        "notes": [
            (
                "Diagnostic gates are smoke-test thresholds, not publication "
                "acceptance criteria."
            ),
            (
                "REF_virial convention is -V * ASE stress; the audit converts it "
                "back to stress before comparison."
            ),
            (
                "Composition-centered energy removes one constant model-reference "
                "total-energy offset per fixed composition."
            ),
            "Singleton compositions are excluded from centered-energy RMSE.",
        ],
    }
    (output / "metrics.json").write_text(
        json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8"
    )
    _write_markdown(output / "summary.md", payload)
    print(json.dumps(_jsonable(payload["overall"]), indent=2))
    print("Diagnostic gates:", json.dumps(_jsonable(gates), indent=2))
    print(f"Report: {output / 'summary.md'}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Zero-shot UMA/OMat audit on InterfaceForge canonical SiN/TiN/TiO frames"
        )
    )
    parser.add_argument("campaign", type=Path, help="Periodic_MLIPs campaign root")
    parser.add_argument("--dataset", type=Path, help="Override held-out test.extxyz")
    parser.add_argument(
        "--model",
        default="uma-s-1p2p1",
        help="FAIRChem model name or local checkpoint .pt path",
    )
    parser.add_argument("--task", default="omat")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument(
        "--inference-settings",
        default="batch",
        help="FAIRChem inference settings; batch is safe for mixed compositions",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="0 means all selected frames; use 32 for a smoke",
    )
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--max-centered-energy-rmse",
        type=float,
        default=50.0,
        metavar="MEV_PER_ATOM",
    )
    parser.add_argument(
        "--max-force-rmse",
        type=float,
        default=0.15,
        metavar="EV_PER_A",
    )
    parser.add_argument(
        "--max-stress-rmse-gpa",
        type=float,
        default=3.0,
        metavar="GPA",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")
    if args.max_frames < 0:
        raise SystemExit("--max-frames must be >= 0")
    if args.progress_every < 1:
        raise SystemExit("--progress-every must be >= 1")
    for name in (
        "max_centered_energy_rmse",
        "max_force_rmse",
        "max_stress_rmse_gpa",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            raise SystemExit(
                f"--{name.replace('_', '-')} must be finite and > 0"
            )
    evaluate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
