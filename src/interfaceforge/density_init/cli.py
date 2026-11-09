"""``iface vasp`` subcommands for optional neural density initialization."""

from __future__ import annotations

import argparse
import json
from typing import Any

from .base import MAGMOM_SOURCES, SPIN_CHANNEL_MODES
from .benchmark import DEFAULT_TOLERANCES, compare_benchmark, prepare_benchmark
from .outputs import audit_initialized_run
from .workflow import BACKENDS, initialize_density, make_backend


def _json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def add_backend_options(parser: argparse.ArgumentParser, *, prefix: str = "") -> None:
    """neural-paw adapter options (shared by every density-init entry point)."""

    group = parser.add_argument_group("neural-paw backend")
    group.add_argument(
        f"--{prefix}ndi-python",
        dest="ndi_python",
        help="Python interpreter with neural_paw_dft installed (default: $IFACE_NDI_PYTHON, else this one)",
    )
    group.add_argument(f"--{prefix}device", dest="ndi_device", help="auto | cpu | cuda | cuda:N (default auto)")
    group.add_argument(
        f"--{prefix}weights-dir", dest="ndi_weights_dir", help="Model weights directory (default $NDI_WEIGHTS_DIR)"
    )
    group.add_argument(f"--{prefix}ndi-config", dest="ndi_config", help="neural_paw_dft pipeline YAML")
    group.add_argument(f"--{prefix}electrafi-checkpoint", dest="ndi_electrafi", help="ELECTRAFI registry name/path")
    group.add_argument(f"--{prefix}augnet-total-checkpoint", dest="ndi_augnet_total", help="AugNet total checkpoint")
    group.add_argument(f"--{prefix}augnet-mag-checkpoint", dest="ndi_augnet_mag", help="AugNet spin checkpoint")
    group.add_argument(
        f"--{prefix}inference-timeout", dest="ndi_timeout", type=float, help="Kill the worker after N seconds"
    )
    group.add_argument(
        f"--{prefix}allow-potcar-variant",
        dest="ndi_allow_potcar_variant",
        action="store_true",
        help=(
            "Proceed when a POTCAR dataset name differs from the models' Materials Project set but "
            "its projector l-channels agree (out of distribution; a projector mismatch is always refused)"
        ),
    )


def backend_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "python": args.ndi_python,
        "device": args.ndi_device,
        "weights_dir": args.ndi_weights_dir,
        "config": args.ndi_config,
        "electrafi_checkpoint": args.ndi_electrafi,
        "augnet_total_checkpoint": args.ndi_augnet_total,
        "augnet_mag_checkpoint": args.ndi_augnet_mag,
        "timeout": args.ndi_timeout,
        "allow_potcar_variant": bool(args.ndi_allow_potcar_variant) or None,
    }


def cmd_initialize_density(args: argparse.Namespace) -> int:
    _json(
        initialize_density(
            args.run_dir,
            backend=args.backend,
            magmom_source=args.magmom_source,
            spin_channel=args.spin_channel,
            grid=args.grid,
            grid_from=args.grid_from,
            grid_dry_run_command=args.grid_dry_run_command,
            dry_run=args.dry_run,
            stage_only=args.stage_only,
            overwrite=args.overwrite,
            force=args.force,
            set_icharg=not args.no_set_icharg,
            backend_options=backend_options(args) if args.backend != "standard" else None,
        )
    )
    return 0


def cmd_density_init_probe(args: argparse.Namespace) -> int:
    backend = make_backend("neural-paw", **backend_options(args))
    payload = {"probe": backend.probe()}
    if args.prefetch and payload["probe"].get("available"):
        payload["prefetch"] = backend.prefetch(spin=not args.no_spin)
    _json(payload)
    return 0 if payload["probe"].get("available") else 3


def cmd_density_init_audit(args: argparse.Namespace) -> int:
    payload = audit_initialized_run(
        args.run_dir,
        reference_threshold=args.reference_threshold,
        min_moment=args.min_moment,
        write=not args.no_write,
    )
    _json(payload)
    return 0 if payload["status"] in {"PASS", "WARN", "INCOMPLETE"} else 1


def cmd_density_init_bench_prepare(args: argparse.Namespace) -> int:
    _json(
        prepare_benchmark(
            args.pilot,
            args.output,
            launcher_template=args.launcher_template,
            neural_init=args.neural_init,
            interfaceforge_command=args.interfaceforge_command,
            backend_options={k: v for k, v in backend_options(args).items() if v is not None},
            dry_run=args.dry_run,
        )
    )
    return 0


def cmd_density_init_bench_compare(args: argparse.Namespace) -> int:
    tolerances = {
        "energy_ev_per_atom": args.energy_tol,
        "max_force_ev_per_a": args.force_tol,
        "stress_kb": args.stress_tol,
        "moment_mu_b": args.moment_tol,
    }
    payload = compare_benchmark(args.root, tolerances=tolerances, write=not args.no_write)
    if args.markdown:
        from .benchmark import render_case_table

        for case in payload["cases"]:
            print(render_case_table(case))
    else:
        _json(payload)
    return 0


def register_vasp_commands(vasp_commands: argparse._SubParsersAction) -> None:
    init = vasp_commands.add_parser(
        "initialize-density",
        help="Seed a fresh VASP run with an opt-in neural initial CHGCAR (never changes the DFT method)",
        description=(
            "Generate an initial density for RUN_DIR (INCAR/POSCAR/POTCAR/KPOINTS required). The "
            "density is written to CHGCAR.neural_init and promoted to CHGCAR only when no foreign "
            "CHGCAR exists (or --overwrite, which backs it up); INCAR gains only ICHARG = 1. "
            "Writes density_init.json provenance. See docs/density-init.md."
        ),
    )
    init.add_argument("run_dir", help="VASP calculation directory (not yet started)")
    init.add_argument("--backend", choices=sorted(BACKENDS), default="neural-paw")
    init.add_argument(
        "--magmom-source",
        choices=MAGMOM_SOURCES,
        default="incar",
        help=(
            "incar (default): the signed INCAR MAGMOM is authoritative; the initializer's own "
            "(unsigned CHGNet) moments are never used. initializer: explicit opt-in to CHGNet "
            "moments, refused for mixed-sign (AFM/ferri) INCARs. none: non-spin-polarized only"
        ),
    )
    init.add_argument(
        "--spin-channel",
        choices=SPIN_CHANNEL_MODES,
        default="auto",
        help=(
            "auto: charge-only seed unless the moment source is sign-uniform (mixed-sign MAGMOM -> "
            "charge-only, so VASP initializes the spin from MAGMOM); off: always charge-only; "
            "model: require the model spin channel (refused for mixed-sign MAGMOM)"
        ),
    )
    grid = init.add_mutually_exclusive_group()
    grid.add_argument("--grid", nargs=3, type=int, metavar=("NGXF", "NGYF", "NGZF"), help="Target fine FFT grid")
    grid.add_argument("--grid-from", help="OUTCAR/CHGCAR of a run with identical ENCUT/PREC/ENAUG and lattice")
    grid.add_argument(
        "--grid-dry-run-command",
        help="VASP command for a throw-away NELM=1 step on copies to read VASP's own grid (counted as overhead)",
    )
    init.add_argument("--dry-run", action="store_true", help="Validate and print the plan; write nothing")
    init.add_argument("--stage-only", action="store_true", help="Write CHGCAR.neural_init only; never promote")
    init.add_argument(
        "--overwrite",
        action="store_true",
        help="Promote over an existing foreign CHGCAR (it is renamed CHGCAR.pre_density_init.<time>, never deleted)",
    )
    init.add_argument("--force", action="store_true", help="Regenerate even when already initialized")
    init.add_argument(
        "--no-set-icharg",
        action="store_true",
        help="Leave INCAR untouched (VASP then ignores the CHGCAR unless ICHARG = 1 is already set)",
    )
    add_backend_options(init)
    init.set_defaults(func=cmd_initialize_density)

    probe = vasp_commands.add_parser(
        "density-init-probe",
        help="Check the neural-paw backend environment (and optionally pre-download weights)",
    )
    probe.add_argument("--prefetch", action="store_true", help="Download registry weights now (needs network)")
    probe.add_argument("--no-spin", action="store_true", help="With --prefetch: charge-only weights only")
    add_backend_options(probe)
    probe.set_defaults(func=cmd_density_init_probe)

    audit = vasp_commands.add_parser(
        "density-init-audit",
        help="After VASP: check electronic convergence and that the signed MAGMOM pattern survived",
    )
    audit.add_argument("run_dir")
    audit.add_argument("--reference-threshold", type=float, default=0.5, help="|MAGMOM| defining pattern sites")
    audit.add_argument("--min-moment", type=float, default=0.5, help="Converged |m| below this counts as quenched")
    audit.add_argument("--no-write", action="store_true", help="Do not write density_init_audit.json")
    audit.set_defaults(func=cmd_density_init_audit)

    bench = vasp_commands.add_parser(
        "density-init-bench",
        help="Paired standard-vs-neural start benchmark on completed structures",
    )
    bench_commands = bench.add_subparsers(dest="density_init_bench_command", required=True)
    prepare = bench_commands.add_parser("prepare", help="Create <case>/{standard,neural} arms from a pilot YAML")
    prepare.add_argument("pilot", help="Pilot YAML (see examples/density-init/pilot.yaml)")
    prepare.add_argument("output", help="New benchmark root")
    prepare.add_argument("--launcher-template", help="Launcher for both arms (default: each source's launcher)")
    prepare.add_argument(
        "--neural-init",
        choices=("launch", "now"),
        default="launch",
        help="launch (default): infer inside the neural job before VASP; now: infer during prepare",
    )
    prepare.add_argument(
        "--interfaceforge-command",
        help="Command the job uses to run InterfaceForge (default: '<this python> -m interfaceforge')",
    )
    prepare.add_argument("--dry-run", action="store_true")
    add_backend_options(prepare)
    prepare.set_defaults(func=cmd_density_init_bench_prepare)

    compare = bench_commands.add_parser("compare", help="Compare finished arms; write JSON/Markdown/TSV reports")
    compare.add_argument("root", help="Benchmark root written by 'prepare'")
    compare.add_argument("--energy-tol", type=float, default=DEFAULT_TOLERANCES["energy_ev_per_atom"])
    compare.add_argument("--force-tol", type=float, default=DEFAULT_TOLERANCES["max_force_ev_per_a"])
    compare.add_argument("--stress-tol", type=float, default=DEFAULT_TOLERANCES["stress_kb"])
    compare.add_argument("--moment-tol", type=float, default=DEFAULT_TOLERANCES["moment_mu_b"])
    compare.add_argument("--markdown", action="store_true", help="Print the per-case tables instead of JSON")
    compare.add_argument("--no-write", action="store_true")
    compare.set_defaults(func=cmd_density_init_bench_compare)
