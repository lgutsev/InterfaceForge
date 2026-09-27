"""CLI entry point with named, scientifically reviewed workflow profiles.

The core :mod:`interfaceforge.cli` remains the generic command surface.  This
entry point augments selected commands with named profiles that expand to an
explicit, auditable set of existing options rather than relying on directory
name heuristics.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .cli import build_parser, step1_density_init_kwargs
from .errors import InterfaceForgeError, SafetyError
from .vasp import prepare_step1_series

STEP1_PROFILES: dict[str, dict[str, Any]] = {
    "nio": {
        "description": (
            "Conservative magnetic DFT+U NiO surface AIMD: POTIM=0.5 fs, "
            "ALGO=Normal, EDIFF=1E-5, NELM=120, NELMIN=6, one static "
            "preconditioning SCF, and a 100 K -> target-temperature ramp."
        ),
        "conservative": True,
        "algo": "Normal",
        "ramp_from_k": 100.0,
        "precondition": True,
        "langevin": False,
    }
}


def _subparser(parser: argparse.ArgumentParser, name: str) -> argparse.ArgumentParser:
    """Return a named argparse child parser without duplicating CLI construction."""

    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            try:
                return action.choices[name]
            except KeyError:
                continue
    raise RuntimeError(f"InterfaceForge CLI has no subparser {name!r}")


def _resolve_step1_profile(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve a named Step1 profile, preserving deliberate user overrides."""

    name = getattr(args, "profile", None)
    if name is None:
        return {
            "name": None,
            "conservative": bool(args.conservative),
            "algo": args.algo,
            "ramp_from_k": args.ramp_from,
            "precondition": bool(args.precondition),
            "langevin_gamma": args.langevin_gamma if args.langevin else None,
        }
    if name not in STEP1_PROFILES:
        raise SafetyError(
            f"Unknown Step1 profile {name!r}; choose from {', '.join(sorted(STEP1_PROFILES))}"
        )

    profile = STEP1_PROFILES[name]
    # A profile is a safe baseline, not a lock. Explicit command-line tuning
    # remains available for scientifically justified cases.  The NiO baseline
    # itself forces conservative mode + preconditioning; ramp/algo can be
    # overridden explicitly and Langevin remains opt-in.
    return {
        "name": name,
        "conservative": bool(args.conservative or profile["conservative"]),
        "algo": args.algo if args.algo != "Normal" else profile["algo"],
        "ramp_from_k": (
            args.ramp_from if args.ramp_from is not None else profile["ramp_from_k"]
        ),
        "precondition": bool(args.precondition or profile["precondition"]),
        "langevin_gamma": args.langevin_gamma if args.langevin else None,
    }


def _stamp_step1_profile(output_root: str | Path, resolved: dict[str, Any]) -> None:
    """Record the profile in existing machine-readable Step1 provenance files."""

    profile = resolved.get("name")
    if profile is None:
        return
    root = Path(output_root)
    settings = {
        "conservative": resolved["conservative"],
        "algo": resolved["algo"],
        "ramp_from_k": resolved["ramp_from_k"],
        "precondition": resolved["precondition"],
        "langevin_gamma": resolved["langevin_gamma"],
    }
    for name in ("step1_manifest.json", "step1_audit.json"):
        path = root / name
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["profile"] = profile
        payload["profile_settings"] = settings
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_vasp_step1_prepare_profiled(args: argparse.Namespace) -> int:
    resolved = _resolve_step1_profile(args)
    payload = prepare_step1_series(
        args.source,
        temperature=args.temperature,
        output_root=args.output_root,
        template=args.template,
        source_structure=args.source_structure,
        protocol=args.protocol,
        dry_run=args.dry_run,
        audit_only=args.audit_only,
        fresh_start=args.fresh_start,
        require_wavecar=args.require_wavecar,
        conservative=resolved["conservative"],
        algo=resolved["algo"],
        langevin_gamma=resolved["langevin_gamma"],
        ramp_from=resolved["ramp_from_k"],
        keep_velocities=args.keep_velocities,
        precondition=resolved["precondition"],
        **step1_density_init_kwargs(args),
    )
    if resolved["name"] is not None:
        profile_settings = {
            "conservative": resolved["conservative"],
            "algo": resolved["algo"],
            "ramp_from_k": resolved["ramp_from_k"],
            "precondition": resolved["precondition"],
            "langevin_gamma": resolved["langevin_gamma"],
        }
        payload["profile"] = resolved["name"]
        payload["profile_settings"] = profile_settings
        if not args.dry_run and not args.audit_only:
            _stamp_step1_profile(payload["output_root"], resolved)
    print(json.dumps(payload, indent=2, default=str))
    return 0


def build_profile_parser() -> argparse.ArgumentParser:
    """Build the normal InterfaceForge parser plus named workflow profiles."""

    parser = build_parser()
    vasp = _subparser(parser, "vasp")
    step1 = _subparser(vasp, "step1-prepare")
    step1.add_argument(
        "--profile",
        choices=sorted(STEP1_PROFILES),
        help=(
            "Named Step1 preparation policy. 'nio' applies the conservative NiO "
            "magnetic DFT+U surface baseline (0.5 fs, Normal/tight SCF, static "
            "preconditioner, 100 K ramp); explicit tuning flags may override "
            "ramp/algo and Langevin remains opt-in."
        ),
    )
    step1.set_defaults(func=_cmd_vasp_step1_prepare_profiled)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_profile_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (InterfaceForgeError, FileNotFoundError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
