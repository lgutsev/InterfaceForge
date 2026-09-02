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
_MACE_UPDATES = re.compile(r"Number of gradient updates:\s*(\d+)")
_MACE_BATCH = re.compile(r"Batch size:\s*(\d+)")
_MACE_TRAIN = re.compile(r"(?:training dataset size|configurations: train)[=:]\s*(\d+)")


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


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _lcurve_tail(path: Path) -> dict[str, Any] | None:
    """Parse the last data row of a DeePMD ``lcurve.out``.

    Column 0 is the step in every DeePMD backend and version; the force and
    energy columns are located by name from the ``#`` header when present and
    left ``None`` otherwise.
    """

    if not path.is_file():
        return None
    header: list[str] = []
    last: list[str] = []
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for raw in content.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            header = line.lstrip("# ").split()
            continue
        fields = line.split()
        if fields and _is_number(fields[0]):
            last = fields
    if not last:
        return None

    def quantity(letter: str) -> float | None:
        """Column for the per-atom energy (``e``) or force (``f``) error.

        DeePMD names these ``rmse_f_val`` / ``rmse_f_trn`` / ``rmse_f`` across
        backends and versions; match any header token that carries an error
        prefix and a lone ``f``/``e`` token, preferring the validation column.
        """

        error = re.compile(r"rmse|l2|mae")
        token_letter = re.compile(rf"(?:^|_){letter}(?:$|_)")
        best: tuple[int, int] | None = None
        for index, name in enumerate(header):
            low = name.lower()
            if index >= len(last) or not _is_number(last[index]):
                continue
            if error.search(low) and token_letter.search(low):
                rank = 0 if "val" in low else 1
                if best is None or rank < best[0]:
                    best = (rank, index)
        return float(last[best[1]]) if best else None

    return {
        "step": int(float(last[0])),
        "rmse_f_val_ev_ang": quantity("f"),
        "rmse_e_val_ev_atom": quantity("e"),
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
        # A member is finished once it has a checkpoint and either a frozen model
        # (the ensemble script freezes only after training completes) or a step
        # count that reached the target. DPA-4 often has no frozen model because
        # its freeze is a separate, sometimes-failing gate.
        complete = all(
            member["checkpoint"]
            and (member["frozen"] or (target is not None and (member["step"] or 0) >= target))
            for member in members
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


def _mace_log_tail(log: Path | None) -> dict[str, Any]:
    """Latest epoch, RMSE_F (meV/A), and the planned epoch budget from a MACE log."""

    result: dict[str, Any] = {"epoch": None, "rmse_f_mev_ang": None, "target_epochs": None}
    if log is None or not log.is_file():
        return result
    updates = batch = train = None
    try:
        content = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return result
    for line in content.splitlines():
        match = _MACE_EPOCH.search(line)
        if match:
            result["epoch"], result["rmse_f_mev_ang"] = int(match.group(1)), float(match.group(2))
            continue
        initial = _MACE_INITIAL.search(line)
        if initial and result["rmse_f_mev_ang"] is None:
            result["rmse_f_mev_ang"] = float(initial.group(1))
        for pattern, key in ((_MACE_UPDATES, "u"), (_MACE_BATCH, "b"), (_MACE_TRAIN, "t")):
            hit = pattern.search(line)
            if hit:
                value = int(hit.group(1))
                if key == "u":
                    updates = value
                elif key == "b":
                    batch = value
                else:
                    train = value
    if updates and batch and train:
        per_epoch = max(train // batch, 1)
        result["target_epochs"] = max(round(updates / per_epoch), 1)
    return result


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
            tail = _mace_log_tail(log)
            final = next(
                iter(sorted((seed_dir / "mace_model").glob("*_stagetwo.model")))
                or sorted(p for p in (seed_dir / "mace_model").glob("*.model") if "_compiled" not in p.name),
                None,
            )
            members.append(
                {
                    "seed": seed_dir.name,
                    "epoch": tail["epoch"],
                    "target_epochs": tail["target_epochs"],
                    "rmse_f_mev_ang": tail["rmse_f_mev_ang"],
                    "checkpoint": any((seed_dir / "checkpoints").glob("*.pt")),
                    "final_model": final.name if final else None,
                    "updated": _mtime(log) if log else _mtime(seed_dir),
                }
            )
        target = next((m["target_epochs"] for m in members if m["target_epochs"]), None)
        rows.append(
            {
                "committee": base,
                "target_epochs": target,
                "complete": bool(members) and all(member["final_model"] for member in members),
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


def _member_line(name: str, done: int | None, total: int | None, rmse_f_mev: float | None, flags: str, updated: str | None) -> str:
    done = done or 0
    bar = _bar(done, total)
    count = f"{done:>7}/{total}" if total else f"{done:>7}/?"
    ftxt = f"{rmse_f_mev:7.1f} meV/A" if rmse_f_mev is not None else f"{'-':>7}      "
    return f"      {name:<11} {bar} {count:>15}  rmse_f {ftxt}  [{flags}]  {updated or ''}"


def render(payload: dict[str, Any]) -> str:
    lines: list[str] = [f"campaign: {payload['campaign_root']}", ""]

    lines.append("DeePMD training")
    if not payload["deepmd_training"]:
        lines.append("  (no models/deepmd/<arch>/ trees yet)")
    for arch in payload["deepmd_training"]:
        flag = "OK" if arch["complete"] else ".."
        lines.append(f"  [{flag}] {arch['architecture']:<22}  target {arch['target_steps'] or '?'} steps")
        for member in arch["members"]:
            fval = member["rmse_f_val_ev_ang"]
            lines.append(
                _member_line(
                    member["model"],
                    member["step"],
                    arch["target_steps"],
                    fval * 1000 if fval is not None else None,
                    ("C" if member["checkpoint"] else "-") + ("F" if member["frozen"] else "-"),
                    member["updated"],
                )
            )

    lines += ["", "DeePMD evaluation"]
    if not payload["deepmd_evaluation"]:
        lines.append("  (no models/deepmd/evaluation/<arch>/job_* yet)")
    for row in payload["deepmd_evaluation"]:
        flag = "OK" if row["rmse_overall"] else ".."
        lines.append(
            f"  [{flag}] {row['architecture']}  {row['job']}  systems {row['systems_complete']}/{row['systems_seen']}  rmse_overall.csv={'yes' if row['rmse_overall'] else 'no'}"
        )

    lines += ["", "MACE training"]
    if not payload["mace_committees"]:
        lines.append("  (no mace_committee/ or mace_finetune_committee/ seed dirs)")
    for row in payload["mace_committees"]:
        flag = "OK" if row["complete"] else ".."
        target = row["target_epochs"]
        lines.append(f"  [{flag}] {row['committee']:<22}  target {target or '?'} epochs")
        for member in row["members"]:
            lines.append(
                _member_line(
                    member["seed"],
                    member["epoch"],
                    target,
                    member["rmse_f_mev_ang"],
                    ("C" if member["checkpoint"] else "-") + ("M" if member["final_model"] else "-"),
                    member["updated"],
                )
            )

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
