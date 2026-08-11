"""Command-line interface for InterfaceForge."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import Any

from . import __version__
from .ai2kit import (
    adapter_status,
    approve_round,
    export_adapter,
    preflight_adapter,
    run_adapter,
    stage_import,
)
from .audit import run_audit
from .beef import plot_beef_campaign
from .campaign import build_plan, prepare_campaign, submit_campaign
from .config import load_campaign
from .data import collect_dataset
from .errors import InterfaceForgeError, SafetyError
from .exploration import generate_exploration
from .geometry import (
    build_slab,
    build_supercell,
    clean_duplicates,
    convert_structure,
    freeze_structure,
    structure_summary,
    write_summary,
)
from .mace_roi import evaluate_mace_roi_predictions, prepare_mace_roi_dataset
from .report import build_report
from .selection import select_from_csv
from .training import generate_deepmd_training, generate_mace_training
from .validation import adhesion_from_csv, parity_from_csv, separation_curve_from_csv
from .vasp import (
    INCAR_PRESETS,
    apply_incar_preset,
    assemble_potcar,
    package_outputs,
    prepare_band_run,
    prepare_recovery,
    prepare_standard_restart,
    submit_run,
)
from .workfunction import analyze_workfunction


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, default=str))


def _campaign(args: argparse.Namespace):
    return load_campaign(args.campaign)


def _copy_resource(name: str, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    resource = resources.files("interfaceforge").joinpath(f"templates/{name}")
    destination.write_text(resource.read_text(encoding="utf-8"), encoding="utf-8")


def cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.directory).expanduser().resolve()
    if root.exists() and any(root.iterdir()) and not args.force:
        allowed = {"profiles", "inputs", "structures"}
        unexpected = [path for path in root.iterdir() if path.name not in allowed]
        if unexpected:
            raise SafetyError(
                f"Initialization directory is not empty: {root}. Use --force intentionally."
            )
    root.mkdir(parents=True, exist_ok=True)
    _copy_resource("campaign.yaml", root / "campaign.yaml", args.force)
    _copy_resource("profile_loni.yaml", root / "profiles" / "loni.yaml", args.force)
    _copy_resource("profile_local.yaml", root / "profiles" / "local.yaml", args.force)
    _copy_resource(
        "potcar_pbe_54.yaml", root / "profiles" / "potcar_pbe_54.yaml", args.force
    )
    for name in ("inputs", "structures"):
        (root / name).mkdir(exist_ok=True)
    _json({"campaign_root": str(root), "campaign": str(root / "campaign.yaml")})
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    _json(build_plan(_campaign(args)))
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    _json(prepare_campaign(_campaign(args), force=args.force))
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    _json(
        submit_campaign(
            _campaign(args),
            system=args.system,
            stage=args.stage,
            execute=args.execute,
        )
    )
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    payload = run_audit(
        args.root,
        output_dir=args.output,
        recursive=not args.shallow,
        include_archives=args.include_archives,
    )
    _json({key: payload[key] for key in ("run_count", "health_counts", "outputs")})
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    payload = run_audit(
        args.root,
        output_dir=args.output,
        recursive=True,
        include_archives=args.include_archives,
    )
    for row in payload["runs"]:
        progress = "" if row["progress_pct"] is None else f"{row['progress_pct']:.1f}%"
        print(f"{row['relative_path']:<52} {row['ml_mode']:<7} {progress:<8} {row['health']}")
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    _json(
        collect_dataset(
            _campaign(args),
            source_root=args.source,
            output_root=args.output,
            force=args.force,
            seed=args.seed,
        )
    )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    campaign = _campaign(args)
    payload = (
        generate_mace_training(campaign, force=args.force)
        if args.engine == "mace"
        else generate_deepmd_training(campaign, force=args.force)
    )
    _json(payload)
    return 0


def cmd_mace_roi_prepare(args: argparse.Namespace) -> int:
    _json(
        prepare_mace_roi_dataset(
            _campaign(args),
            source_root=args.source,
            output_root=args.output,
            cycle_manifest=args.cycles,
            force=args.force,
        )
    )
    return 0


def cmd_mace_roi_evaluate(args: argparse.Namespace) -> int:
    _json(
        evaluate_mace_roi_predictions(
            args.source,
            args.output,
            reference_energy_key=args.reference_energy_key,
            predicted_energy_key=args.predicted_energy_key,
            reference_forces_key=args.reference_forces_key,
            predicted_forces_key=args.predicted_forces_key,
        )
    )
    return 0


def cmd_explore(args: argparse.Namespace) -> int:
    _json(generate_exploration(_campaign(args), output=args.output))
    return 0


def cmd_select(args: argparse.Namespace) -> int:
    _json(
        select_from_csv(
            args.candidates,
            args.output,
            count=args.count,
            uncertainty_column=args.uncertainty_column,
            feature_columns=args.feature,
            group_column=args.group_column,
            max_per_group=args.max_per_group,
            uncertainty_weight=args.uncertainty_weight,
        )
    )
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    if args.validation == "parity":
        payload = parity_from_csv(
            args.source,
            args.output,
            reference_column=args.reference_column,
            predicted_column=args.predicted_column,
            group_columns=args.group,
        )
    elif args.validation == "adhesion":
        payload = adhesion_from_csv(args.source, args.output)
    else:
        payload = separation_curve_from_csv(args.source, args.output)
    _json(payload)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    _json(build_report(_campaign(args), output=args.output))
    return 0


def cmd_ai2kit(args: argparse.Namespace) -> int:
    campaign = _campaign(args)
    output_root = getattr(args, "adapter_output", None)
    if args.ai2kit_command == "export":
        payload = export_adapter(
            campaign,
            output=args.output,
            force=args.force,
        )
    elif args.ai2kit_command == "preflight":
        payload = preflight_adapter(
            campaign,
            output_root=output_root,
            remote=args.remote,
            report_output=args.output,
        )
        _json(payload)
        return 0 if payload["passed"] else 2
    elif args.ai2kit_command == "run":
        payload = run_adapter(
            campaign,
            output_root=output_root,
            execute=args.execute,
            resume=args.resume,
            allow_multiple_iterations=args.allow_multiple_iterations,
        )
    elif args.ai2kit_command == "status":
        payload = adapter_status(campaign, output_root=output_root)
    elif args.ai2kit_command == "import":
        payload = stage_import(
            campaign,
            round_number=args.round_number,
            result_root=args.result_root,
            output_root=output_root,
        )
    else:
        payload = approve_round(
            campaign,
            round_number=args.round_number,
            output_root=output_root,
        )
    _json(payload)
    return 0


def cmd_vasp_recover(args: argparse.Namespace) -> int:
    _json(
        prepare_recovery(
            args.folder,
            args.operation,
            temperature=args.temperature,
            nsw=args.nsw,
            ml_mb=args.ml_mb,
            ml_mconf=args.ml_mconf,
            force_expand=args.force_expand,
            increase_eps_low=args.increase_eps_low,
        )
    )
    return 0


def cmd_vasp_restart(args: argparse.Namespace) -> int:
    _json(
        prepare_standard_restart(
            args.folder,
            from_contcar=not args.keep_poscar,
            clean_electronic=args.clean_electronic,
        )
    )
    return 0


def cmd_vasp_band(args: argparse.Namespace) -> int:
    _json(
        prepare_band_run(
            args.source,
            args.destination,
            line_kpoints=args.kpoints,
            lmaxmix=args.lmaxmix,
            force=args.force,
        )
    )
    return 0


def cmd_vasp_potcar(args: argparse.Namespace) -> int:
    _json(
        assemble_potcar(
            args.poscar,
            args.output,
            pseudopotential_root=args.root,
            mapping_file=args.map,
            force=args.force,
        )
    )
    return 0


def cmd_vasp_incar(args: argparse.Namespace) -> int:
    _json(
        apply_incar_preset(
            args.incar,
            args.preset,
            temperature=args.temperature,
            nsw=args.nsw,
            potim=args.potim,
            create=args.create,
        )
    )
    return 0


def cmd_vasp_pack(args: argparse.Namespace) -> int:
    _json(
        package_outputs(
            args.root,
            args.output,
            include_large=args.include_large,
            force=args.force,
        )
    )
    return 0


def cmd_vasp_submit(args: argparse.Namespace) -> int:
    if (args.ml_mb is not None or args.ml_mconf is not None) and not args.recover_capacity:
        raise SafetyError("--ml-mb/--ml-mconf require --recover-capacity")
    if args.ml_mconf is not None and args.ml_mb is None:
        raise SafetyError("--ml-mconf is supported only with the explicit --ml-mb expansion path")
    if args.increase_eps_low and args.ml_mb is not None:
        raise SafetyError("--increase-eps-low cannot be combined with --ml-mb expansion")
    if args.increase_eps_low and not args.recover_capacity:
        raise SafetyError("--increase-eps-low requires --recover-capacity")
    prepared = None
    if args.recover_capacity:
        operation = "expand" if args.ml_mb is not None else "discard"
        prepared = prepare_recovery(
            args.folder,
            operation,
            ml_mb=args.ml_mb,
            ml_mconf=args.ml_mconf,
            increase_eps_low=args.increase_eps_low,
        )
    _json(
        {
            "folder": str(Path(args.folder).resolve()),
            "capacity_recovery": prepared,
            "job_id": submit_run(args.folder, args.launcher),
        }
    )
    return 0


def cmd_vasp_beef_plot(args: argparse.Namespace) -> int:
    _json(
        plot_beef_campaign(
            args.root,
            output=args.output,
            data_output=args.data_output,
            include_archives=args.include_archives,
            individual=args.individual,
            dpi=args.dpi,
        )
    )
    return 0


def cmd_geom(args: argparse.Namespace) -> int:
    if args.geometry == "convert":
        payload = convert_structure(
            args.source,
            args.output,
            cell_from=args.cell_from,
            cell=args.cell,
            center=args.center,
            sort_atoms=not args.no_sort,
            force=args.force,
        )
    elif args.geometry == "supercell":
        payload = build_supercell(
            args.source,
            args.output,
            args.repeat,
            vacuum=args.vacuum,
            axis=args.axis,
            sort_atoms=not args.no_sort,
            force=args.force,
        )
    elif args.geometry == "slab":
        payload = build_slab(
            args.source,
            args.output,
            args.miller,
            args.layers,
            repeat=args.repeat,
            vacuum=args.vacuum,
            force=args.force,
        )
    elif args.geometry == "freeze":
        payload = freeze_structure(
            args.source,
            args.output,
            axis=args.axis,
            lower=args.lower,
            upper=args.upper,
            region=args.region,
            elements=args.element,
            append=args.append,
            force=args.force,
        )
    elif args.geometry == "clean":
        payload = clean_duplicates(
            args.source, args.output, cutoff=args.cutoff, force=args.force
        )
    else:
        payload = structure_summary(args.source)
    write_summary(payload, getattr(args, "summary", None))
    _json(payload)
    return 0


def cmd_workfunction(args: argparse.Namespace) -> int:
    _json(
        analyze_workfunction(
            args.locpot,
            args.outcar,
            axis=args.axis,
            data_output=args.data_output,
            plot_output=args.plot_output,
            summary_output=args.summary_output,
        )
    )
    return 0


def add_campaign_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-c", "--campaign", default="campaign.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iface",
        description="Reproducible interface campaigns across VASP, MACE, and DeePMD.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="Create a portable campaign skeleton")
    init.add_argument("directory", nargs="?", default=".")
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=cmd_init)

    plan = commands.add_parser("plan", help="Validate and print the campaign DAG")
    add_campaign_option(plan)
    plan.set_defaults(func=cmd_plan)

    prepare = commands.add_parser("prepare", help="Scaffold VASP-MLFF run directories")
    add_campaign_option(prepare)
    prepare.add_argument("--force", action="store_true")
    prepare.set_defaults(func=cmd_prepare)

    submit = commands.add_parser("submit", help="List jobs; submit only with --execute")
    add_campaign_option(submit)
    submit.add_argument("--system")
    submit.add_argument("--stage", choices=("train", "refit", "stability"))
    submit.add_argument("--execute", action="store_true")
    submit.set_defaults(func=cmd_submit)

    status = commands.add_parser("status", help="Compact VASP campaign status")
    status.add_argument("root", nargs="?", default=".")
    status.add_argument("--output")
    status.add_argument(
        "--include-archives",
        action="store_true",
        help="Include run folders whose relative path contains 'archive'",
    )
    status.set_defaults(func=cmd_status)

    audit = commands.add_parser("audit", help="Mode-aware VASP-MLFF audit")
    audit.add_argument("root", nargs="?", default=".")
    audit.add_argument("-o", "--output")
    audit.add_argument("--shallow", action="store_true")
    audit.add_argument(
        "--include-archives",
        action="store_true",
        help="Include run folders whose relative path contains 'archive'",
    )
    audit.set_defaults(func=cmd_audit)

    collect = commands.add_parser("collect", help="Create synchronized extxyz and DeePMD datasets")
    add_campaign_option(collect)
    collect.add_argument("--source")
    collect.add_argument("--output")
    collect.add_argument("--seed", type=int, default=20260730)
    collect.add_argument("--force", action="store_true")
    collect.set_defaults(func=cmd_collect)

    mace_roi = commands.add_parser(
        "mace-roi", help="Prepare region- and thermodynamic-cycle-aware MACE data"
    )
    mace_roi_commands = mace_roi.add_subparsers(dest="mace_roi_command", required=True)
    mace_roi_prepare = mace_roi_commands.add_parser(
        "prepare", help="Create a derived extxyz dataset with MACE-ROI metadata"
    )
    add_campaign_option(mace_roi_prepare)
    mace_roi_prepare.add_argument("--source")
    mace_roi_prepare.add_argument("--output")
    mace_roi_prepare.add_argument("--cycles")
    mace_roi_prepare.add_argument("--force", action="store_true")
    mace_roi_prepare.set_defaults(func=cmd_mace_roi_prepare)
    mace_roi_evaluate = mace_roi_commands.add_parser(
        "evaluate", help="Report global, interface-local and cycle prediction errors"
    )
    mace_roi_evaluate.add_argument("source")
    mace_roi_evaluate.add_argument("output")
    mace_roi_evaluate.add_argument("--reference-energy-key", default="REF_energy")
    mace_roi_evaluate.add_argument("--predicted-energy-key", default="MACE_energy")
    mace_roi_evaluate.add_argument("--reference-forces-key", default="REF_forces")
    mace_roi_evaluate.add_argument("--predicted-forces-key", default="MACE_forces")
    mace_roi_evaluate.set_defaults(func=cmd_mace_roi_evaluate)

    train = commands.add_parser("train", help="Generate model training campaigns")
    train.add_argument("engine", choices=("mace", "deepmd"))
    add_campaign_option(train)
    train.add_argument("--force", action="store_true")
    train.set_defaults(func=cmd_train)

    explore = commands.add_parser("explore", help="Expand active-learning conditions")
    add_campaign_option(explore)
    explore.add_argument("--output")
    explore.set_defaults(func=cmd_explore)

    select = commands.add_parser("select", help="Select uncertain and diverse labeling candidates")
    select.add_argument("candidates")
    select.add_argument("output")
    select.add_argument("-n", "--count", type=int, required=True)
    select.add_argument("--uncertainty-column", default="uncertainty")
    select.add_argument("--feature", action="append", default=[])
    select.add_argument("--group-column")
    select.add_argument("--max-per-group", type=int)
    select.add_argument("--uncertainty-weight", type=float, default=0.65)
    select.set_defaults(func=cmd_select)

    validate = commands.add_parser("validate", help="Interface and parity validation")
    validation = validate.add_subparsers(dest="validation", required=True)
    parity = validation.add_parser("parity")
    parity.add_argument("source")
    parity.add_argument("output")
    parity.add_argument("--reference-column", default="reference")
    parity.add_argument("--predicted-column", default="predicted")
    parity.add_argument("--group", action="append", default=["model"])
    parity.set_defaults(func=cmd_validate)
    adhesion = validation.add_parser("adhesion")
    adhesion.add_argument("source")
    adhesion.add_argument("output")
    adhesion.set_defaults(func=cmd_validate)
    separation = validation.add_parser("separation")
    separation.add_argument("source")
    separation.add_argument("output")
    separation.set_defaults(func=cmd_validate)

    report = commands.add_parser("report", help="Build a self-contained HTML dashboard")
    add_campaign_option(report)
    report.add_argument("--output")
    report.set_defaults(func=cmd_report)

    active_learning = commands.add_parser(
        "active-learning", help="Optional supervised active-learning adapters"
    )
    active_backends = active_learning.add_subparsers(dest="active_backend", required=True)
    ai2kit = active_backends.add_parser(
        "ai2kit", help="AI2-Kit 1.0.9 DeepMD/LAMMPS/VASP closed-loop adapter"
    )
    ai2kit_commands = ai2kit.add_subparsers(dest="ai2kit_command", required=True)
    ai2kit_export = ai2kit_commands.add_parser("export", help="Generate deterministic CLL configuration")
    add_campaign_option(ai2kit_export)
    ai2kit_export.add_argument("--output")
    ai2kit_export.add_argument("--force", action="store_true")
    ai2kit_export.set_defaults(func=cmd_ai2kit)
    ai2kit_preflight = ai2kit_commands.add_parser("preflight", help="Check adapter and engine compatibility")
    add_campaign_option(ai2kit_preflight)
    ai2kit_preflight.add_argument("--adapter-output")
    ai2kit_preflight.add_argument("--remote", action="store_true")
    ai2kit_preflight.add_argument("--output", help="Optional preflight JSON path")
    ai2kit_preflight.set_defaults(func=cmd_ai2kit)
    ai2kit_run = ai2kit_commands.add_parser("run", help="Dry-run by default; execute only explicitly")
    add_campaign_option(ai2kit_run)
    ai2kit_run.add_argument("--adapter-output")
    ai2kit_run.add_argument("--execute", action="store_true")
    ai2kit_run.add_argument("--resume", action="store_true")
    ai2kit_run.add_argument("--allow-multiple-iterations", action="store_true")
    ai2kit_run.set_defaults(func=cmd_ai2kit)
    ai2kit_status = ai2kit_commands.add_parser("status", help="Read safe manifest and log status")
    add_campaign_option(ai2kit_status)
    ai2kit_status.add_argument("--adapter-output")
    ai2kit_status.set_defaults(func=cmd_ai2kit)
    ai2kit_import = ai2kit_commands.add_parser("import", help="Stage labeled results without dataset mutation")
    add_campaign_option(ai2kit_import)
    ai2kit_import.add_argument("--adapter-output")
    ai2kit_import.add_argument("--round", dest="round_number", type=int, required=True)
    ai2kit_import.add_argument("--result-root", required=True)
    ai2kit_import.set_defaults(func=cmd_ai2kit)
    ai2kit_approve = ai2kit_commands.add_parser("approve", help="Approve one reviewed staged round")
    add_campaign_option(ai2kit_approve)
    ai2kit_approve.add_argument("--adapter-output")
    ai2kit_approve.add_argument("--round", dest="round_number", type=int, required=True)
    ai2kit_approve.set_defaults(func=cmd_ai2kit)

    vasp = commands.add_parser("vasp", help="Safe VASP utilities")
    vasp_commands = vasp.add_subparsers(dest="vasp_command", required=True)
    recover = vasp_commands.add_parser("recover", help="Continue/refit/stability/capacity recovery")
    recover.add_argument(
        "operation", choices=("continue", "discard", "expand", "refit", "stability")
    )
    recover.add_argument("folder")
    recover.add_argument("--temperature", type=float)
    recover.add_argument("--nsw", type=int)
    recover.add_argument("--ml-mb", type=int)
    recover.add_argument("--ml-mconf", type=int)
    recover.add_argument("--force-expand", action="store_true")
    recover.add_argument(
        "--increase-eps-low",
        action="store_true",
        help="For discard recovery, multiply ML_EPS_LOW by 10 while enforcing <1E-7",
    )
    recover.set_defaults(func=cmd_vasp_recover)
    restart = vasp_commands.add_parser("restart", help="Prepare an ordinary VASP restart")
    restart.add_argument("folder")
    restart.add_argument("--keep-poscar", action="store_true")
    restart.add_argument("--clean-electronic", action="store_true")
    restart.set_defaults(func=cmd_vasp_restart)
    band = vasp_commands.add_parser("band", help="Prepare a line-mode band run")
    band.add_argument("source")
    band.add_argument("destination")
    band.add_argument("--kpoints", required=True)
    band.add_argument("--lmaxmix", type=int, default=4)
    band.add_argument("--force", action="store_true")
    band.set_defaults(func=cmd_vasp_band)
    potcar = vasp_commands.add_parser("potcar", help="Assemble POTCAR from a licensed local tree")
    potcar.add_argument("poscar")
    potcar.add_argument("--root", required=True)
    potcar.add_argument("--map", required=True)
    potcar.add_argument("-o", "--output", default="POTCAR")
    potcar.add_argument("--force", action="store_true")
    potcar.set_defaults(func=cmd_vasp_potcar)
    incar = vasp_commands.add_parser(
        "incar", help="Apply a conservative static/relax/MD/DOS preset"
    )
    incar.add_argument("preset", choices=INCAR_PRESETS)
    incar.add_argument("incar", nargs="?", default="INCAR")
    incar.add_argument("--temperature", type=float, default=300.0)
    incar.add_argument("--nsw", type=int, default=3000)
    incar.add_argument("--potim", type=float, default=1.0)
    incar.add_argument("--create", action="store_true")
    incar.set_defaults(func=cmd_vasp_incar)
    pack = vasp_commands.add_parser("pack", help="Package lightweight reproducibility outputs")
    pack.add_argument("output")
    pack.add_argument("--root", default=".")
    pack.add_argument("--include-large", action="store_true")
    pack.add_argument("--force", action="store_true")
    pack.set_defaults(func=cmd_vasp_pack)
    vsubmit = vasp_commands.add_parser(
        "submit", help="Submit one VASP run, optionally recovering MLFF capacity first"
    )
    vsubmit.add_argument("folder")
    vsubmit.add_argument(
        "--launcher",
        help="Batch script (default: prefer runvasp.sh, then run.slurm)",
    )
    vsubmit.add_argument(
        "--recover-capacity",
        action="store_true",
        help="Archive a capacity stop and resubmit; defaults to bounded-memory basis discarding",
    )
    vsubmit.add_argument(
        "--ml-mb",
        type=int,
        help="Opt into expansion by setting a larger ML_MB; omit for safer basis discarding",
    )
    vsubmit.add_argument(
        "--ml-mconf",
        type=int,
        help="Optional new ML_MCONF allocation for capacity recovery",
    )
    vsubmit.add_argument(
        "--increase-eps-low",
        action="store_true",
        help="Also multiply ML_EPS_LOW by 10; guarded to remain strictly below 1E-7",
    )
    vsubmit.set_defaults(func=cmd_vasp_submit)
    beef_plot = vasp_commands.add_parser(
        "beef-plot", help="Plot campaign Bayesian force errors and ML_CTIFOR"
    )
    beef_plot.add_argument("root", nargs="?", default=".")
    beef_plot.add_argument("-o", "--output")
    beef_plot.add_argument("--data-output")
    beef_plot.add_argument("--include-archives", action="store_true")
    beef_plot.add_argument("--individual", action="store_true")
    beef_plot.add_argument("--dpi", type=int, default=160)
    beef_plot.set_defaults(func=cmd_vasp_beef_plot)
    workfunction = vasp_commands.add_parser("workfunction", help="Analyze planar-averaged LOCPOT")
    workfunction.add_argument("locpot")
    workfunction.add_argument("outcar")
    workfunction.add_argument("--axis", choices=("x", "y", "z"), default="z")
    workfunction.add_argument("--data-output", default="locpot.dat")
    workfunction.add_argument("--plot-output")
    workfunction.add_argument("--summary-output")
    workfunction.set_defaults(func=cmd_workfunction)

    geom = vasp_commands.add_parser("geom", help="ASE-backed geometry preparation")
    geom_commands = geom.add_subparsers(dest="geometry", required=True)
    convert = geom_commands.add_parser("convert")
    convert.add_argument("source")
    convert.add_argument("output")
    convert.add_argument("--cell-from")
    convert.add_argument("--cell", nargs="+", type=float)
    convert.add_argument("--center", action="store_true")
    convert.add_argument("--no-sort", action="store_true")
    convert.add_argument("--force", action="store_true")
    convert.add_argument("--summary")
    convert.set_defaults(func=cmd_geom)
    supercell = geom_commands.add_parser("supercell")
    supercell.add_argument("source")
    supercell.add_argument("output")
    supercell.add_argument("--repeat", nargs=3, type=int, required=True)
    supercell.add_argument("--vacuum", type=float)
    supercell.add_argument("--axis", type=int, choices=(0, 1, 2), default=2)
    supercell.add_argument("--no-sort", action="store_true")
    supercell.add_argument("--force", action="store_true")
    supercell.add_argument("--summary")
    supercell.set_defaults(func=cmd_geom)
    slab = geom_commands.add_parser("slab")
    slab.add_argument("source")
    slab.add_argument("output")
    slab.add_argument("--miller", nargs=3, type=int, required=True)
    slab.add_argument("--layers", type=int, required=True)
    slab.add_argument("--repeat", nargs=3, type=int, default=(1, 1, 1))
    slab.add_argument("--vacuum", type=float, default=15.0)
    slab.add_argument("--force", action="store_true")
    slab.add_argument("--summary")
    slab.set_defaults(func=cmd_geom)
    freeze = geom_commands.add_parser("freeze")
    freeze.add_argument("source")
    freeze.add_argument("output")
    freeze.add_argument("--axis", choices=("x", "y", "z"), default="z")
    freeze.add_argument("--lower", type=float)
    freeze.add_argument("--upper", type=float)
    freeze.add_argument("--region", choices=("inside", "outside"), default="outside")
    freeze.add_argument("--element", action="append", default=[])
    freeze.add_argument("--append", action="store_true")
    freeze.add_argument("--force", action="store_true")
    freeze.add_argument("--summary")
    freeze.set_defaults(func=cmd_geom)
    clean = geom_commands.add_parser("clean")
    clean.add_argument("source")
    clean.add_argument("output")
    clean.add_argument("--cutoff", type=float, default=0.5)
    clean.add_argument("--force", action="store_true")
    clean.add_argument("--summary")
    clean.set_defaults(func=cmd_geom)
    inspect = geom_commands.add_parser("inspect")
    inspect.add_argument("source")
    inspect.add_argument("--summary")
    inspect.set_defaults(func=cmd_geom)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (InterfaceForgeError, FileNotFoundError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
