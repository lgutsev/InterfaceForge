# ruff: noqa: E501
"""Filesystem-level progress across MLIP training, evaluation, and comparison runs.

Reads only generated artifacts (``lcurve.out``, MACE logs, evaluation detail
files, comparison manifests), so it works whether or not ``campaign.yaml`` is
still in sync and never touches a running job.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_FROZEN_NAMES = ("frozen_model.pth", "frozen_model.pt", "frozen_model.pt2", "frozen_model.pb")
_MACE_EPOCH = re.compile(r"Epoch\s+(\d+):.*?RMSE_F=\s*([0-9.]+)")
_MACE_INITIAL = re.compile(r"Initial:.*?RMSE_F=\s*([0-9.]+)")


def _mtime(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
    except OSError:
        return None


def _newest(directory: Path, pattern: str) -> Path | None:
    try:
        matches = [p for p in directory.glob(pattern) if p.is_file()]
    except OSError:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime, default=None)


def _job_sort_key(path: Path) -> tuple[int, float]:
    suffix = path.name.removeprefix("job_")
    return (int(suffix) if suffix.isdigit() else -1, path.stat().st_mtime)


def _lcurve_tail(path: Path) -> dict[str, Any] | None:
    """Return the last row of a DeePMD ``lcurve.out`` keyed by its header names."""

    if not path.is_file():
        return None
    header: list[str] = []
    last: list[str] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("#"):
                    header = stripped.lstrip("#").split()
                elif not stripped.startswith("//"):
                    last = stripped.split()
    except OSError:
        return None
    if not last:
        return None
    row = dict(zip(header or ["step"], last, strict=False))

    def _num(*names: str) -> float | None:
        for name in names:
            if name in row:
                try:
                    return float(row[name])
                except ValueError:
                    return None
        return None

    step = _num("step")
    return {
        "step": int(step) if step is not None else None,
        "rmse_f_val_ev_ang": _num("rmse_f_val", "rmse_f_trn", "rmse_f"),
        "rmse_e_val_ev_atom": _num("rmse_e_val", "rmse_e_trn", "rmse_e"),
    }


def _deepmd_target_steps(model_dir: Path) -> int | None:
    try:
        payload = json.loads((model_dir / "input.json").read_text(encoding="utf-8"))
        return int(payload["training"]["numb_steps"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _deepmd_training(deepmd_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arch_dir in sorted(p for p in deepmd_root.glob("*") if p.is_dir()):
        if arch_dir.name in {"evaluation", "smoke"}:
            continue
        model_dirs = sorted(arch_dir.glob("model_*"))
        if not model_dirs:
            continue
        target = _deepmd_target_steps(model_dirs[0])
        members: list[dict[str, Any]] = []
        for model_dir in model_dirs:
            tail = _lcurve_tail(model_dir / "lcurve.out") or {}
            checkpoint = model_dir / "model.ckpt.pt"
            members.append(
                {
                    "model": model_dir.name,
                    "step": tail.get("step"),
                    "rmse_f_val_ev_ang": tail.get("rmse_f_val_ev_ang"),
                    "checkpoint": checkpoint.is_file() and checkpoint.stat().st_size > 0,
                    "frozen": any((model_dir / name).is_file() for name in _FROZEN_NAMES),
                    "updated": _mtime(model_dir / "lcurve.out"),
                }
            )
        complete = target is not None and all(
            member["checkpoint"] and (member["step"] or 0) >= target for member in members
        )
        rows.append(
            {
                "architecture": arch_dir.name,
                "target_steps": target,
                "committee": len(members),
                "complete": complete,
                "members": members,
            }
        )
    return rows


def _deepmd_evaluation(deepmd_root: Path, committees: dict[str, int]) -> list[dict[str, Any]]:
    eval_root = deepmd_root / "evaluation"
    rows: list[dict[str, Any]] = []
    for arch_dir in sorted(p for p in eval_root.glob("*") if p.is_dir()):
        jobs = [p for p in arch_dir.glob("job_*") if p.is_dir()]
        if not jobs:
            continue
        latest = max(jobs, key=_job_sort_key)
        committee = committees.get(arch_dir.name, 4)
        by_system = latest / "by_system"
        systems = sorted(by_system.glob("system_*")) if by_system.is_dir() else []
        done = 0
        for system in systems:
            if all(
                (system / f"model_{index:03d}_detail.e_peratom.out").is_file()
                and (system / f"model_{index:03d}_detail.f.out").is_file()
                for index in range(committee)
            ):
                done += 1
        rows.append(
            {
                "architecture": arch_dir.name,
                "job": latest.name,
                "systems_complete": done,
                "systems_seen": len(systems),
                "rmse_overall": (latest / "rmse_overall.csv").is_file(),
                "updated": _mtime(latest),
            }
        )
    return rows


def _mace_log_tail(log: Path | None) -> tuple[int | None, float | None]:
    if log is None or not log.is_file():
        return None, None
    epoch: int | None = None
    rmse_f: float | None = None
    try:
        with log.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = _MACE_EPOCH.search(line)
                if match:
                    epoch, rmse_f = int(match.group(1)), float(match.group(2))
                    continue
                initial = _MACE_INITIAL.search(line)
                if initial and rmse_f is None:
                    rmse_f = float(initial.group(1))
    except OSError:
        return None, None
    return epoch, rmse_f


def _mace_committees(mace_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base in ("mace_committee", "mace_finetune_committee"):
        base_dir = mace_root / base
        seeds = sorted(p for p in base_dir.glob("seed_*") if p.is_dir())
        if not seeds:
            continue
        members: list[dict[str, Any]] = []
        for seed_dir in seeds:
            log = _newest(seed_dir / "logs", "*.log")
            epoch, rmse_f = _mace_log_tail(log)
            final = next(
                iter(sorted((seed_dir / "mace_model").glob("*_stagetwo.model")))
                or sorted(p for p in (seed_dir / "mace_model").glob("*.model") if "_compiled" not in p.name),
                None,
            )
            members.append(
                {
                    "seed": seed_dir.name,
                    "epoch": epoch,
                    "rmse_f_mev_ang": rmse_f,
                    "final_model": final.name if final else None,
                    "updated": _mtime(log) if log else _mtime(seed_dir),
                }
            )
        rows.append(
            {
                "committee": base,
                "complete": all(member["final_model"] for member in members),
                "members": members,
            }
        )
    return rows


def _comparisons(campaign: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for output in sorted((campaign / "audit").glob("mlip_compare*")):
        manifest = output / "comparison_manifest.json"
        if not manifest.is_file():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        systems = data.get("systems", [])
        models = data.get("models", [])
        expected = len(systems) * len(models)
        have = sum(
            (output / "predictions" / "mace" / model["model"] / f"{system['system_id']}.npz").is_file()
            for system in systems
            for model in models
        )
        rows.append(
            {
                "output_root": output.name,
                "deepmd_architecture": data.get("deepmd_architecture", "dpa2"),
                "systems": len(systems),
                "mace_predictions": f"{have}/{expected}",
                "finalized": (output / "comparison.json").is_file(),
            }
        )
    return rows


def mlip_progress(
    campaign_root: str | Path,
    *,
    mace_committee_root: str | Path | None = None,
) -> dict[str, Any]:
    campaign = Path(campaign_root).expanduser().resolve()
    deepmd_root = campaign / "models" / "deepmd"
    mace_root = (
        Path(mace_committee_root).expanduser().resolve()
        if mace_committee_root
        else campaign / "models" / "mace_committee_520eV"
    )
    training = _deepmd_training(deepmd_root)
    committees = {row["architecture"]: row["committee"] for row in training}
    return {
        "schema_version": 1,
        "campaign_root": str(campaign),
        "deepmd_training": training,
        "deepmd_evaluation": _deepmd_evaluation(deepmd_root, committees),
        "mace_committees": _mace_committees(mace_root),
        "comparisons": _comparisons(campaign),
    }


def _bar(done: int, total: int | None, width: int = 12) -> str:
    if not total:
        return "?" * width
    filled = round(width * min(done, total) / total)
    return "#" * filled + "-" * (width - filled)


def render(payload: dict[str, Any]) -> str:
    lines: list[str] = [f"campaign: {payload['campaign_root']}", ""]

    lines.append("DeePMD training")
    if not payload["deepmd_training"]:
        lines.append("  (no models/deepmd/<arch>/ trees yet)")
    for arch in payload["deepmd_training"]:
        flag = "OK" if arch["complete"] else ".."
        lines.append(f"  [{flag}] {arch['architecture']}  target={arch['target_steps'] or '?'} steps")
        for member in arch["members"]:
            step = member["step"] or 0
            frac = _bar(step, arch["target_steps"])
            fval = member["rmse_f_val_ev_ang"]
            ftxt = f"{fval * 1000:6.1f} meV/A" if fval is not None else f"{'-':>6}      "
            marks = ("C" if member["checkpoint"] else "-") + ("F" if member["frozen"] else "-")
            lines.append(f"      {member['model']}  {frac} {step:>7}  rmse_f_val {ftxt}  [{marks}]  {member['updated'] or ''}")

    lines += ["", "DeePMD evaluation"]
    if not payload["deepmd_evaluation"]:
        lines.append("  (no models/deepmd/evaluation/<arch>/job_* yet)")
    for row in payload["deepmd_evaluation"]:
        flag = "OK" if row["rmse_overall"] else ".."
        lines.append(
            f"  [{flag}] {row['architecture']}  {row['job']}  systems {row['systems_complete']}/{row['systems_seen']}  rmse_overall.csv={'yes' if row['rmse_overall'] else 'no'}"
        )

    lines += ["", "MACE committees"]
    if not payload["mace_committees"]:
        lines.append("  (no mace_committee/ or mace_finetune_committee/ seed dirs)")
    for row in payload["mace_committees"]:
        flag = "OK" if row["complete"] else ".."
        lines.append(f"  [{flag}] {row['committee']}")
        for member in row["members"]:
            epoch = f"epoch {member['epoch']}" if member["epoch"] is not None else "epoch  ?"
            rmse = member["rmse_f_mev_ang"]
            rtxt = f"{rmse:6.1f} meV/A" if rmse is not None else f"{'-':>6}      "
            model = member["final_model"] or "-"
            lines.append(f"      {member['seed']}  {epoch:>9}  rmse_f {rtxt}  model={model}  {member['updated'] or ''}")

    lines += ["", "Comparisons"]
    if not payload["comparisons"]:
        lines.append("  (no audit/mlip_compare* runs prepared)")
    for row in payload["comparisons"]:
        flag = "OK" if row["finalized"] else ".."
        lines.append(
            f"  [{flag}] {row['output_root']}  vs {row['deepmd_architecture']}  MACE preds {row['mace_predictions']}  finalized={'yes' if row['finalized'] else 'no'}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_root", nargs="?", default=".")
    parser.add_argument("--mace-committee-root")
    parser.add_argument("--json", action="store_true", help="Emit the raw payload instead of a table")
    args = parser.parse_args(argv)
    payload = mlip_progress(args.campaign_root, mace_committee_root=args.mace_committee_root)
    print(json.dumps(payload, indent=2) if args.json else render(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
