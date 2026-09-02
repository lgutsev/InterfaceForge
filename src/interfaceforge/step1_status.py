# ruff: noqa: E501
"""Runtime status of a prepared Step1 preheat tree.

Read-only. For every run under a ``Step1/`` root (or a single run directory)
this reports three things a human checks by hand:

* **frames produced** -- MD steps written so far (from ``OSZICAR``, cross
  checked against ``XDATCAR``), versus the ``NSW`` target;
* **the INCAR** -- the electronic-quality knobs (``ENCUT``, ``PREC``,
  ``EDIFF``, ``ALGO``, ``LREAL``, smearing) and the physics inherited from
  OPT (``ISPIN``, the Hubbard ``U`` values, ``IBRION``/``NSW``/``POTIM``),
  plus ``ENCUT/ENMAX`` when a ``POTCAR`` is present;
* **which job is done** -- ``not-started`` / ``running`` / ``stalled?`` /
  ``done`` / ``done-early`` / ``error``, from ``OUTCAR`` completion markers
  and the last written step.

Never touches a running job; only reads generated files.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .aimd import _first_float, _first_int, preheat_ps
from .audit import parse_oszicar, read_tail
from .vasp import parse_incar

_EXCLUDED = ("archive", "backup", ".interfaceforge")
_TIMING_MARKER = "General timing and accounting informations"
_XDATCAR_FRAME = re.compile(r"^\s*Direct configuration=", re.MULTILINE)
_ENMAX = re.compile(r"ENMAX\s*=\s*([0-9.]+)")
_ERROR_MARKERS = (
    "VERY BAD NEWS",
    "ZBRENT: fatal error",
    "internal error",
    "Error EDDDAV",
    "The distance between some ions is very small",
    "WARNING: DENTET",
    "PRICEL",
)
_STALE_HOURS_DEFAULT = 6.0


def _mtime(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None


def _iso(moment: datetime | None) -> str | None:
    return moment.isoformat(timespec="seconds") if moment else None


def _discover_runs(root: Path) -> list[Path]:
    if (root / "INCAR").is_file():
        return [root]
    runs: list[Path] = []
    for incar in sorted(root.rglob("INCAR")):
        parts = {part.lower() for part in incar.parent.relative_to(root).parts}
        if parts & set(_EXCLUDED) or any(p.startswith("x") for p in parts):
            continue
        runs.append(incar.parent)
    return runs


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _xdatcar_frames(path: Path) -> int | None:
    if not _nonempty(path):
        return None
    return len(_XDATCAR_FRAME.findall(read_tail(path)))


def _potcar_enmax(path: Path) -> float | None:
    if not _nonempty(path):
        return None
    values = [float(v) for v in _ENMAX.findall(read_tail(path))]
    return max(values) if values else None


def _incar_summary(incar: dict[str, str], potcar_enmax: float | None) -> dict[str, Any]:
    encut = _first_float(incar.get("ENCUT"))
    ldauu = incar.get("LDAUU")
    summary: dict[str, Any] = {
        "istart": _first_int(incar.get("ISTART")),
        "encut_ev": encut,
        "prec": incar.get("PREC"),
        "ediff": incar.get("EDIFF"),
        "algo": incar.get("ALGO"),
        "lreal": incar.get("LREAL"),
        "ismear": _first_int(incar.get("ISMEAR")),
        "sigma": _first_float(incar.get("SIGMA")),
        "ispin": _first_int(incar.get("ISPIN")),
        "ldau": incar.get("LDAU"),
        "ldauu": ldauu,
        "lmaxmix": _first_int(incar.get("LMAXMIX")),
        "ibrion": _first_int(incar.get("IBRION")),
        "nsw": _first_int(incar.get("NSW")),
        "potim_fs": _first_float(incar.get("POTIM")),
        "smass": incar.get("SMASS"),
        "nblock": _first_int(incar.get("NBLOCK")),
        "tebeg": _first_float(incar.get("TEBEG")),
        "teend": _first_float(incar.get("TEEND")),
        "ivdw": _first_int(incar.get("IVDW")),
        "encut_over_enmax": None,
    }
    if encut and potcar_enmax:
        summary["encut_over_enmax"] = round(encut / potcar_enmax, 3)
    return summary


def _classify(
    *,
    has_incar: bool,
    outcar_tail: str,
    started: bool,
    last_step: int | None,
    nsw: int | None,
    updated: datetime | None,
    stale_hours: float,
) -> tuple[str, bool]:
    """Return ``(state, stale)``."""

    if not has_incar:
        return "no-incar", False
    if not started:
        return "not-started", False
    finished = _TIMING_MARKER in outcar_tail
    has_error = any(marker in outcar_tail for marker in _ERROR_MARKERS)
    if finished:
        if nsw and last_step is not None and last_step >= nsw:
            return "done", False
        return ("error" if has_error else "done-early"), False
    if has_error:
        return "error", False
    stale = False
    if updated is not None:
        age_h = (datetime.now(tz=timezone.utc) - updated).total_seconds() / 3600.0
        stale = age_h > stale_hours
    return ("stalled?" if stale else "running"), stale


def _run_status(run: Path, *, stale_hours: float) -> dict[str, Any]:
    incar_path = run / "INCAR"
    incar = parse_incar(incar_path)
    potcar_enmax = _potcar_enmax(run / "POTCAR")
    summary = _incar_summary(incar, potcar_enmax)
    nsw = summary["nsw"]
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
        "wavecar_present": _nonempty(run / "WAVECAR"),
        "contcar_present": _nonempty(run / "CONTCAR"),
        "potcar_present": _nonempty(run / "POTCAR"),
        "potcar_enmax_ev": potcar_enmax,
        "updated": _iso(updated),
        "incar": summary,
    }


def step1_status(root: str | Path, *, stale_hours: float = _STALE_HOURS_DEFAULT) -> dict[str, Any]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(root_path)

    manifest: dict[str, Any] | None = None
    manifest_path = root_path / "step1_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = None

    runs = [_run_status(run, stale_hours=stale_hours) for run in _discover_runs(root_path)]
    tally: dict[str, int] = {}
    for row in runs:
        tally[row["state"]] = tally.get(row["state"], 0) + 1

    return {
        "schema_version": 1,
        "root": str(root_path),
        "stale_hours": stale_hours,
        "protocol": (manifest or {}).get("protocol"),
        "manifest_temperature_k": (manifest or {}).get("temperature_k"),
        "manifest_nsw": (manifest or {}).get("nsw"),
        "state_tally": tally,
        "runs": runs,
    }


def _fmt(value: Any, spec: str = "", dash: str = "-") -> str:
    if value is None:
        return dash
    try:
        return format(value, spec) if spec else str(value)
    except (TypeError, ValueError):
        return str(value)


def render(payload: dict[str, Any]) -> str:
    lines: list[str] = [f"Step1 status: {payload['root']}"]
    header_bits = []
    if payload.get("protocol"):
        header_bits.append(f"protocol {payload['protocol']}")
    if payload.get("manifest_nsw"):
        header_bits.append(f"NSW target {payload['manifest_nsw']}")
    if payload.get("manifest_temperature_k"):
        header_bits.append(f"{payload['manifest_temperature_k']:g} K")
    if header_bits:
        lines.append("  " + "  ·  ".join(header_bits))
    lines.append("")

    if not payload["runs"]:
        lines.append("  (no runs with an INCAR found)")
        return "\n".join(lines)

    for row in payload["runs"]:
        inc = row["incar"]
        frames = row["frames_oszicar"]
        target = row["nsw_target"]
        pct = row["percent_complete"]
        count = f"{frames}/{target}" if target else f"{frames}/?"
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
            + f"ISTART={_fmt(inc['istart'])}  "
            + f"ENCUT={_fmt(inc['encut_ev'], '.0f')}{ratio_txt}  "
            + f"PREC={_fmt(inc['prec'])}  EDIFF={_fmt(inc['ediff'])}  "
            + f"ALGO={_fmt(inc['algo'])}  LREAL={_fmt(inc['lreal'])}  "
            + f"ISMEAR={_fmt(inc['ismear'])}/{_fmt(inc['sigma'], '.2g')}"
        )
        lines.append(
            "             "
            + f"ISPIN={_fmt(inc['ispin'])}  U(eV)={_fmt(inc['ldauu'])}  LMAXMIX={_fmt(inc['lmaxmix'])}  "
            + f"IBRION={_fmt(inc['ibrion'])}  NSW={_fmt(inc['nsw'])}  POTIM={_fmt(inc['potim_fs'])}  "
            + f"SMASS={_fmt(inc['smass'])}  TEBEG={_fmt(inc['tebeg'], '.0f')}"
        )

    summary = "  ".join(f"{state}: {n}" for state, n in sorted(payload["state_tally"].items()))
    lines += ["", f"{len(payload['runs'])} runs  ({summary})"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", nargs="?", default=".", help="Step1 tree root or a single run directory")
    parser.add_argument(
        "--stale-hours",
        type=float,
        default=_STALE_HOURS_DEFAULT,
        help=f"Flag a running job as 'stalled?' if OSZICAR is older than this (default {_STALE_HOURS_DEFAULT})",
    )
    parser.add_argument("--json", action="store_true", help="Emit the raw payload instead of a table")
    args = parser.parse_args(argv)
    payload = step1_status(args.root, stale_hours=args.stale_hours)
    print(json.dumps(payload, indent=2, default=str) if args.json else render(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
