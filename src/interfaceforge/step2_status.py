# ruff: noqa: E501
"""Runtime status of a prepared Step2 DFT-MD temperature series.

The Step2 counterpart of :mod:`interfaceforge.step1_status`. A Step2
campaign is a set of sibling ``Step2_<T>K/`` trees; this walks every run in
each tree and reports, per run:

* **frames produced** -- MD steps written so far (from ``OSZICAR``, cross
  checked against ``XDATCAR``) versus the ``NSW`` target, as a count, a
  percentage, and ps of trajectory;
* **the INCAR** -- the electronic-quality knobs (``ENCUT``, ``PREC``,
  ``EDIFF``, ``ALGO``, ``LREAL``, smearing) and the physics inherited from
  Step1 (``ISPIN``, the Hubbard ``U`` values, ``IBRION``/``NSW``/``POTIM``,
  the thermostat), plus ``ENCUT/ENMAX`` when a ``POTCAR`` is present;
* **which job is done** -- ``not-started`` / ``running`` / ``stalled?`` /
  ``done`` / ``done-early`` / ``error``, plus the mean/std MD temperature.

Read-only; never touches a running job.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .aimd import _first_float, _first_int, preheat_ps
from .audit import parse_oszicar, read_tail
from .step1_status import (
    _EXCLUDED,
    _STALE_HOURS_DEFAULT,
    _classify,
    _iso,
    _mtime,
    _nonempty,
    _potcar_enmax,
    _xdatcar_frames,
)
from .vasp import parse_incar


def _discover_runs(tree: Path) -> list[Path]:
    if (tree / "INCAR").is_file():
        return [tree]
    runs: list[Path] = []
    for incar in sorted(tree.rglob("INCAR")):
        parts = {part.lower() for part in incar.parent.relative_to(tree).parts}
        if parts & set(_EXCLUDED) or any(p.startswith("x") for p in parts):
            continue
        runs.append(incar.parent)
    return runs


def _discover_trees(root: Path) -> list[Path]:
    """A single run, a single ``Step2_<T>K`` tree, or a parent of several."""

    if (root / "INCAR").is_file():
        return [root]
    trees = sorted(p for p in root.glob("Step2_*K") if p.is_dir())
    if trees:
        return trees
    return [root]


def _manifest(tree: Path) -> dict[str, Any]:
    path = tree / "step2_manifest.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _incar_summary(incar: dict[str, str], potcar_enmax: float | None) -> dict[str, Any]:
    encut = _first_float(incar.get("ENCUT"))
    summary: dict[str, Any] = {
        "encut_ev": encut,
        "prec": incar.get("PREC"),
        "ediff": incar.get("EDIFF"),  # None -> VASP default 1E-4
        "algo": incar.get("ALGO"),
        "lreal": incar.get("LREAL"),
        "ismear": _first_int(incar.get("ISMEAR")),
        "sigma": _first_float(incar.get("SIGMA")),
        "ispin": _first_int(incar.get("ISPIN")),
        "ldau": incar.get("LDAU"),
        "ldauu": incar.get("LDAUU"),
        "lmaxmix": _first_int(incar.get("LMAXMIX")),
        "ibrion": _first_int(incar.get("IBRION")),
        "nsw": _first_int(incar.get("NSW")),
        "potim_fs": _first_float(incar.get("POTIM")),
        "mdalgo": incar.get("MDALGO"),
        "smass": incar.get("SMASS"),
        "nblock": _first_int(incar.get("NBLOCK")),
        "tebeg": _first_float(incar.get("TEBEG")),
        "teend": _first_float(incar.get("TEEND")),
        "ivdw": _first_int(incar.get("IVDW")),
        "isym": _first_int(incar.get("ISYM")),
        "lwave": incar.get("LWAVE"),
        "encut_over_enmax": None,
    }
    if encut and potcar_enmax:
        summary["encut_over_enmax"] = round(encut / potcar_enmax, 3)
    return summary


def _run_status(run: Path, *, nsw_target: int | None, stale_hours: float) -> dict[str, Any]:
    incar_path = run / "INCAR"
    incar = parse_incar(incar_path)
    potcar_enmax = _potcar_enmax(run / "POTCAR")
    summary = _incar_summary(incar, potcar_enmax)
    nsw = nsw_target or summary["nsw"]
    potim = summary["potim_fs"]

    oszicar = parse_oszicar(run / "OSZICAR")
    frames_oszicar = oszicar["md_steps"] or 0
    last_step = oszicar["last_oszicar_step"]
    frames_xdatcar = _xdatcar_frames(run / "XDATCAR")

    outcar = run / "OUTCAR"
    outcar_tail = read_tail(outcar) if _nonempty(outcar) else ""
    started = _nonempty(outcar) or _nonempty(run / "OSZICAR")
    updated = _mtime(run / "OSZICAR") or _mtime(outcar)

    state, stale = _classify(
        has_incar=incar_path.is_file(),
        outcar_tail=outcar_tail,
        started=started,
        last_step=last_step,
        nsw=nsw,
        updated=updated,
        stale_hours=stale_hours,
    )

    done_step = last_step if last_step is not None else frames_oszicar
    return {
        "run": run.name,
        "path": str(run),
        "state": state,
        "stale": stale,
        "frames_oszicar": frames_oszicar,
        "frames_xdatcar": frames_xdatcar,
        "last_step": last_step,
        "nsw_target": nsw,
        "percent_complete": (
            round(100.0 * done_step / nsw, 1) if nsw and done_step is not None else None
        ),
        "produced_ps": preheat_ps(done_step, potim),
        "target_ps": preheat_ps(nsw, potim),
        "temperature_mean_k": oszicar["temperature_mean_k"],
        "temperature_std_k": oszicar["temperature_std_k"],
        "temperature_last_k": oszicar["temperature_last_k"],
        "contcar_present": _nonempty(run / "CONTCAR"),
        "potcar_present": _nonempty(run / "POTCAR"),
        "potcar_enmax_ev": potcar_enmax,
        "updated": _iso(updated),
        "incar": summary,
    }


def _tree_status(tree: Path, *, stale_hours: float) -> dict[str, Any]:
    manifest = _manifest(tree)
    nsw_target = manifest.get("nsw")
    runs = [
        _run_status(run, nsw_target=nsw_target, stale_hours=stale_hours)
        for run in _discover_runs(tree)
    ]
    tally: dict[str, int] = {}
    for row in runs:
        tally[row["state"]] = tally.get(row["state"], 0) + 1
    return {
        "tree": tree.name,
        "path": str(tree),
        "protocol": manifest.get("protocol"),
        "temperature_k": manifest.get("temperature_k"),
        "nsw_target": nsw_target,
        "state_tally": tally,
        "runs": runs,
    }


def step2_status(root: str | Path, *, stale_hours: float = _STALE_HOURS_DEFAULT) -> dict[str, Any]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(root_path)

    trees = [_tree_status(tree, stale_hours=stale_hours) for tree in _discover_trees(root_path)]
    combined: dict[str, int] = {}
    for tree in trees:
        for state, count in tree["state_tally"].items():
            combined[state] = combined.get(state, 0) + count

    return {
        "schema_version": 1,
        "root": str(root_path),
        "stale_hours": stale_hours,
        "state_tally": combined,
        "trees": trees,
    }


def _fmt(value: Any, spec: str = "", dash: str = "-") -> str:
    if value is None:
        return dash
    try:
        return format(value, spec) if spec else str(value)
    except (TypeError, ValueError):
        return str(value)


def _render_run(row: dict[str, Any], lines: list[str]) -> None:
    inc = row["incar"]
    frames = row["frames_oszicar"]
    target = row["nsw_target"]
    count = f"{frames}/{target}" if target else f"{frames}/?"
    pct = row["percent_complete"]
    pct_txt = f"{pct:>5.1f}%" if pct is not None else "   -  "
    xdat = row["frames_xdatcar"]
    xdat_txt = f" (XDATCAR {xdat})" if xdat is not None and xdat != frames else ""
    ps_txt = ""
    if row["produced_ps"] is not None and row["target_ps"] is not None:
        ps_txt = f"  {row['produced_ps']:.2f}/{row['target_ps']:.2f} ps"
    temp_txt = ""
    if row["temperature_mean_k"] is not None:
        mean_k = row["temperature_mean_k"]
        std = row["temperature_std_k"]
        temp_txt = f"  T={mean_k:.0f} K" if std is None else f"  T={mean_k:.0f}+/-{std:.0f} K"

    lines.append(f"  [{row['state']:<11}] {row['run']}")
    lines.append(
        f"      frames {count}{xdat_txt}  {pct_txt}{ps_txt}{temp_txt}  updated {row['updated'] or '-'}"
    )
    ratio = inc["encut_over_enmax"]
    ratio_txt = f" ({ratio:g}x ENMAX)" if ratio is not None else ""
    lines.append(
        "      INCAR: "
        + f"ENCUT={_fmt(inc['encut_ev'], '.0f')}{ratio_txt}  "
        + f"PREC={_fmt(inc['prec'])}  EDIFF={_fmt(inc['ediff'], dash='default')}  "
        + f"ALGO={_fmt(inc['algo'])}  LREAL={_fmt(inc['lreal'])}  "
        + f"ISMEAR={_fmt(inc['ismear'])}/{_fmt(inc['sigma'], '.2g')}"
    )
    lines.append(
        "             "
        + f"ISPIN={_fmt(inc['ispin'])}  U(eV)={_fmt(inc['ldauu'])}  LMAXMIX={_fmt(inc['lmaxmix'])}  "
        + f"IBRION={_fmt(inc['ibrion'])}  NSW={_fmt(inc['nsw'])}  POTIM={_fmt(inc['potim_fs'])}  "
        + f"MDALGO={_fmt(inc['mdalgo'])}  SMASS={_fmt(inc['smass'])}  TEBEG={_fmt(inc['tebeg'], '.0f')}"
    )


def render(payload: dict[str, Any]) -> str:
    lines: list[str] = [f"Step2 status: {payload['root']}", ""]
    total_runs = 0
    for tree in payload["trees"]:
        total_runs += len(tree["runs"])
        bits = []
        if tree.get("protocol"):
            bits.append(f"protocol {tree['protocol']}")
        if tree.get("nsw_target"):
            bits.append(f"NSW target {tree['nsw_target']}")
        if tree.get("temperature_k"):
            bits.append(f"{tree['temperature_k']:g} K")
        suffix = f"   ({'  ·  '.join(bits)})" if bits else ""
        lines.append(f"{tree['tree']}{suffix}")
        if not tree["runs"]:
            lines.append("  (no runs with an INCAR found)")
            lines.append("")
            continue
        for row in tree["runs"]:
            _render_run(row, lines)
        tally = "  ".join(f"{s}: {n}" for s, n in sorted(tree["state_tally"].items()))
        lines += [f"  {len(tree['runs'])} runs  ({tally})", ""]

    combined = "  ".join(f"{s}: {n}" for s, n in sorted(payload["state_tally"].items()))
    lines.append(f"{total_runs} runs total  ({combined})" if combined else f"{total_runs} runs total")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Parent of the Step2_<T>K trees, a single tree, or one run directory",
    )
    parser.add_argument(
        "--stale-hours",
        type=float,
        default=_STALE_HOURS_DEFAULT,
        help=f"Flag a running job as 'stalled?' if OSZICAR is older than this (default {_STALE_HOURS_DEFAULT})",
    )
    parser.add_argument("--json", action="store_true", help="Emit the raw payload instead of a table")
    args = parser.parse_args(argv)
    payload = step2_status(args.root, stale_hours=args.stale_hours)
    print(json.dumps(payload, indent=2, default=str) if args.json else render(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
