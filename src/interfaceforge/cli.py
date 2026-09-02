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
from .adhesion import METHODS as ADHESION_METHODS
from .adhesion import SLAB_MODES as ADHESION_SLAB_MODES
from .adhesion import audit_adhesion, prepare_adhesion, summarize_adhesion
from .ai2kit import (
    adapter_status,
    approve_round,
    export_adapter,
    preflight_adapter,
    run_adapter,
    stage_import,
)
from .aimd import AIMD_PROTOCOLS, sample_step2_runs, switch_step1_protocol
from .audit import READINESS_PROFILES, run_audit
from .beef import plot_beef_campaign
from .campaign import build_plan, prepare_campaign, submit_campaign
from .committee import collect_committee, verify_committee_bundle
from .config import load_campaign, merge_interface_metadata, references_for
from .data import collect_dataset
from .errors import InterfaceForgeError, SafetyError
from .exploration import generate_exploration
from .geometry import (
    batch_slab_vacuum,
    build_slab,
    build_supercell,
    clean_duplicates,
    convert_structure,
    extend_slab_vacuum,
    freeze_structure,
    slab_vacuum,
    structure_summary,
    write_summary,
)
from .interface_energy import interface_energy
from .interface_energy import write_reports as write_interface_energy_reports
from .intermat import generate_intermat_interfaces, intermat_status
from .mace_roi import evaluate_mace_roi_predictions, prepare_mace_roi_dataset
from .mlff_interfaces import (
    discover_mlff_interface_sources,
    generate_mlff_interfaces_campaign,
    mass_audit_mlff_interfaces,
    write_throttled_array_launcher,
)
from .mlip_compare import comparison_status, finalize_comparison, prepare_comparison
from .progress import mlip_progress
from .progress import render as render_progress
from .reference_import import (
    activate_reference_profile,
    expand_reference_profile,
    list_reference_profiles,
    load_reference_profile,
)
from .regfgw import compare_registry_selection, regfgw_status, run_regfgw_optimize
from .report import build_report
from .selection import select_from_csv
from .separation_energy import separation_energy
from .separation_energy import write_reports as write_separation_energy_reports
from .slab_alignment import analyze_slab_alignment
from .slab_publication import plot_slab_publication
from .step1_status import render as render_step1_status
from .step1_status import step1_status
from .step2_status import render as render_step2_status
from .step2_status import step2_status
from .surface import (
    analyze_surface,
    audit_surface_runs,
    build_surface_campaign,
    optimize_surface_cell,
    plan_surface_campaign,
    select_surface_candidates,
)
from .training import generate_deepmd_training, generate_mace_training
from .validation import (
    adhesion_from_csv,
    parity_from_csv,
    separation_curve_from_csv,
    stratified_parity_from_csv,
)
from .vasp import (
    INCAR_PRESETS,
    apply_incar_preset,
    archive_mlff_models,
    assemble_potcar,
    ensure_run_potcar,
    launch_opt_runs,
    launch_step2_runs,
    package_outputs,
    prepare_band_run,
    prepare_opt_tree,
    prepare_recovery,
    prepare_standard_restart,
    prepare_step1_series,
    prepare_step2_series,
    resolve_potcar_root,
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
        readiness_profile=args.readiness_profile,
    )
    _json(
        {
            key: payload[key]
            for key in ("readiness_profile", "run_count", "health_counts", "outputs")
        }
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    payload = run_audit(
        args.root,
        output_dir=args.output,
        recursive=True,
        include_archives=args.include_archives,
        readiness_profile=args.readiness_profile,
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


def cmd_committee(args: argparse.Namespace) -> int:
    if args.committee_command == "verify":
        payload = verify_committee_bundle(args.bundle)
    else:
        payload = collect_committee(
            args.source,
            args.output,
            engine=args.engine,
            expected_members=args.expected_members,
            model_pattern=args.model_pattern,
            training_data=args.training_data,
            training_data_output=args.training_data_output,
            training_data_compression=args.training_data_compression,
            label=args.label,
            notes=args.notes,
        )
    _json(payload)
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


def cmd_mlip_compare(args: argparse.Namespace) -> int:
    campaign = _campaign(args)
    if args.mlip_compare_command == "prepare":
        payload = prepare_comparison(
            campaign.root,
            output_root=args.output_root,
            mace_models_root=args.mace_models_root,
            seeds=tuple(args.seeds),
            deepmd_arch=args.deepmd_arch,
            force=args.force,
        )
    elif args.mlip_compare_command == "status":
        payload = comparison_status(
            campaign.root,
            output_root=args.output_root,
            deepmd_eval_root=args.deepmd_eval_root,
        )
    else:
        payload = finalize_comparison(
            campaign.root,
            output_root=args.output_root,
            deepmd_eval_root=args.deepmd_eval_root,
        )
    _json(payload)
    return 0 if payload.get("status") != "INCOMPLETE" else 1


def cmd_mlip_progress(args: argparse.Namespace) -> int:
    payload = mlip_progress(
        _campaign(args).root, mace_committee_root=args.mace_committee_root
    )
    if args.json:
        _json(payload)
    else:
        print(render_progress(payload))
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
        references = None
        if getattr(args, "campaign_file", None) and Path(args.campaign_file).is_file():
            references = references_for(
                load_campaign(args.campaign_file).validation.get("references"),
                "work_of_adhesion",
            )
        payload = adhesion_from_csv(args.source, args.output, references=references)
    elif args.validation == "separation":
        payload = separation_curve_from_csv(args.source, args.output)
    elif args.validation == "separation-energy":
        entries: list[tuple[str, str]] = []
        for item in args.entries:
            spec, sep, directory = item.partition("=")
            if not sep:
                spec, directory = Path(item).name, item
            entries.append((spec, directory))
        validation = None
        if args.campaign and Path(args.campaign).is_file():
            validation = load_campaign(args.campaign).validation
        payload = separation_energy(
            entries,
            mace_models=args.mace_models,
            deepmd_models=args.deepmd_models,
            reference=args.reference,
            n_interfaces=args.n_interfaces,
            area_axis=args.area_axis,
            device=args.device,
            campaign_validation=validation,
        )
        payload["outputs"] = write_separation_energy_reports(payload, args.output)
    elif args.validation == "interface-energy":
        campaign = _campaign(args)
        payload = interface_energy(
            campaign.root,
            dataset_root=args.dataset_root,
            predictions_root=args.predictions,
            equilibration_frames=args.equilibration_frames,
            n_interfaces=args.n_interfaces,
            blocks=args.blocks,
            stacking_axis=args.stacking_axis,
            interface_metadata=campaign.validation.get("interfaces"),
        )
        payload["outputs"] = write_interface_energy_reports(payload, args.output)
    else:
        payload = stratified_parity_from_csv(
            args.source,
            args.output,
            reference_column=args.reference_column,
            predicted_column=args.predicted_column,
            kind_column=args.kind_column,
            high_temperature_column=args.high_temperature_column,
            min_coordination_column=args.min_coordination_column,
            low_coordination_percentile=args.low_coordination_percentile,
        )
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
            ml_outblock=args.ml_outblock,
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


def cmd_vasp_step2_series(args: argparse.Namespace) -> int:
    _json(
        prepare_step2_series(
            args.source,
            temperatures=args.temperatures,
            output_root=args.output_root,
            template=args.template,
            source_structure=args.source_structure,
            protocol=args.protocol,
            dry_run=args.dry_run,
            audit_only=args.audit_only,
            reprotocol=args.set_protocol,
        )
    )
    return 0


def cmd_vasp_step1_prepare(args: argparse.Namespace) -> int:
    _json(
        prepare_step1_series(
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
        )
    )
    return 0


def cmd_vasp_step1_status(args: argparse.Namespace) -> int:
    payload = step1_status(args.root, stale_hours=args.stale_hours)
    if args.json:
        _json(payload)
    else:
        print(render_step1_status(payload))
    return 0


def cmd_vasp_step2_status(args: argparse.Namespace) -> int:
    payload = step2_status(args.root, stale_hours=args.stale_hours)
    if args.json:
        _json(payload)
    else:
        print(render_step2_status(payload))
    return 0


def cmd_vasp_step1_protocol(args: argparse.Namespace) -> int:
    _json(
        switch_step1_protocol(
            args.target,
            args.protocol,
            nsw=args.nsw,
            audit_only=args.audit_only,
            create=args.create,
        )
    )
    return 0


def cmd_vasp_step2_sample(args: argparse.Namespace) -> int:
    _json(sample_step2_runs(args.roots, dry_run=args.dry_run))
    return 0


def cmd_vasp_step2_launch(args: argparse.Namespace) -> int:
    _json(
        launch_step2_runs(
            args.roots,
            execute=args.execute,
            launcher=args.launcher,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
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
            pseudopotential_root=resolve_potcar_root(args.root),
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
            workfunction=args.workfunction,
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


def cmd_vasp_archive_models(args: argparse.Namespace) -> int:
    _json(
        archive_mlff_models(
            args.root,
            args.output,
            include_large=args.include_large,
            exclude_folders=args.exclude_folders,
            recursive=args.recursive,
            force=args.force,
        )
    )
    return 0


def cmd_vasp_submit(args: argparse.Namespace) -> int:
    recovery_requested = args.recover_continue or args.recover_capacity
    if (args.ml_mb is not None or args.ml_mconf is not None) and not args.recover_capacity:
        raise SafetyError("--ml-mb/--ml-mconf require --ml-capacity-recovery")
    if args.ml_mconf is not None and args.ml_mb is None:
        raise SafetyError("--ml-mconf is supported only with the explicit --ml-mb expansion path")
    if args.increase_eps_low and args.ml_mb is not None:
        raise SafetyError("--increase-eps-low cannot be combined with --ml-mb expansion")
    if args.increase_eps_low and not args.recover_capacity:
        raise SafetyError("--increase-eps-low requires --ml-capacity-recovery")
    if (args.temperature is not None or args.nsw is not None) and not recovery_requested:
        raise SafetyError("--temperature/--nsw require --ml-continue or --ml-capacity-recovery")
    potcar = ensure_run_potcar(
        args.folder,
        pseudopotential_root=args.potcar_root,
        mapping_file=args.potcar_map,
    )
    prepared = None
    if args.recover_continue:
        prepared = prepare_recovery(
            args.folder,
            "continue",
            temperature=args.temperature,
            nsw=args.nsw,
        )
    elif args.recover_capacity:
        operation = "expand" if args.ml_mb is not None else "discard"
        prepared = prepare_recovery(
            args.folder,
            operation,
            temperature=args.temperature,
            nsw=args.nsw,
            ml_mb=args.ml_mb,
            ml_mconf=args.ml_mconf,
            increase_eps_low=args.increase_eps_low,
        )
    _json(
        {
            "folder": str(Path(args.folder).resolve()),
            "potcar": potcar,
            "continuation_recovery": prepared if args.recover_continue else None,
            "capacity_recovery": prepared if args.recover_capacity else None,
            "ml_continuation_recovery": prepared if args.recover_continue else None,
            "ml_capacity_recovery": prepared if args.recover_capacity else None,
            "job_id": submit_run(
                args.folder,
                args.launcher,
                potcar_root=args.potcar_root,
                potcar_mapping=args.potcar_map,
            ),
        }
    )
    return 0


def cmd_vasp_opt_prepare(args: argparse.Namespace) -> int:
    _json(
        prepare_opt_tree(
            args.root,
            manifest=args.manifest,
            launcher_template=args.launcher_template,
            launcher=args.launcher,
            potcar_command=args.potcar_command,
            required_module=args.require_module,
            dry_run=args.dry_run,
            audit_only=args.audit_only,
            force_launcher=args.force_launcher,
            exclude_prefixes=args.exclude_prefix,
            progress=print,
        )
    )
    return 0


def cmd_vasp_opt_launch(args: argparse.Namespace) -> int:
    _json(
        launch_opt_runs(
            args.roots,
            execute=args.execute,
            launcher=args.launcher,
            progress=print,
        )
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
    elif args.geometry == "vacuum":
        if args.output and len(args.sources) > 1:
            raise SafetyError("-o/--output takes a single source; use --execute for many")
        results = [_vacuum_one(source, args) for source in args.sources]
        payload = results[0] if len(results) == 1 else {"sources": args.sources, "results": results}
    else:
        payload = structure_summary(args.source)
    write_summary(payload, getattr(args, "summary", None))
    _json(payload)
    if args.geometry == "vacuum":
        batches = [r for r in (results if len(results) > 1 else [payload]) if "rows" in r]
        if batches:
            _print_vacuum_summary(batches, args)
    return 0


def cmd_surface_analyze(args: argparse.Namespace) -> int:
    _json(
        analyze_surface(
            args.structure,
            metal=args.metal,
            anion=args.anion,
            coordination_cutoff=args.coordination_cutoff,
            bulk_coordination=args.bulk_coordination,
            top_tolerance=args.top_tolerance,
        ).to_dict()
    )
    return 0


def cmd_surface_cell_optimize(args: argparse.Namespace) -> int:
    _json(
        optimize_surface_cell(
            args.slab,
            adsorbate_path=args.adsorbate,
            min_multiplier=args.min_multiplier,
            max_multiplier=args.max_multiplier,
            max_atoms=args.max_atoms,
            min_translation=args.min_translation,
            min_image_gap=args.min_image_gap,
            max_aspect=args.max_aspect,
            translation_parity=tuple(args.translation_parity) if args.translation_parity else None,
            orientation_samples=args.orientation_samples,
            frozen_bottom_layers=args.freeze_bottom_layers,
            output=args.output,
            force=args.force,
            top=args.top,
        )
    )
    return 0


def cmd_surface_plan(args: argparse.Namespace) -> int:
    _json(plan_surface_campaign(args.campaign))
    return 0


def cmd_surface_build(args: argparse.Namespace) -> int:
    _json(build_surface_campaign(args.campaign, force=args.force))
    return 0


def cmd_surface_audit(args: argparse.Namespace) -> int:
    _json(audit_surface_runs(args.root, output=args.output))
    return 0


def cmd_surface_select(args: argparse.Namespace) -> int:
    _json(
        select_surface_candidates(
            args.candidates,
            args.output,
            count=args.count,
            uncertainty_column=args.uncertainty_column,
            feature_columns=tuple(args.feature_column),
            state_columns=tuple(args.state_column),
            max_per_state=args.max_per_state,
            uncertainty_weight=args.uncertainty_weight,
        )
    )
    return 0


def cmd_surface_init(args: argparse.Namespace) -> int:
    destination = Path(args.output).expanduser().resolve()
    _copy_resource("surface_nio110.yaml", destination, args.force)
    _json({"template": str(destination)})
    return 0


def _vacuum_one(source: str, args: argparse.Namespace) -> dict[str, Any]:
    if Path(source).is_dir():
        return batch_slab_vacuum(
            source,
            axis=args.axis,
            min_vacuum=args.min_vacuum,
            extend=args.extend,
            execute=args.execute,
        )
    if args.extend is not None:
        if args.execute or args.output:
            output = source if args.execute else (args.output or f"{source}.extended")
            return extend_slab_vacuum(
                source,
                output,
                axis=args.axis,
                vacuum=args.extend,
                recenter=not args.no_recenter,
                sort_atoms=not args.no_sort,
                force=args.force or args.execute,
            )
        report = slab_vacuum(source, axis=args.axis)
        return {
            **report,
            "mode": "dry-run",
            "would_be_a": round(max(report["vacuum_a"], args.extend), 2),
            "hint": "pass -o OUT (or --execute to overwrite the source) to write it",
        }
    report = slab_vacuum(source, axis=args.axis)
    report["status"] = "PASS" if report["vacuum_a"] >= args.min_vacuum else "THIN"
    report["min_vacuum_a"] = args.min_vacuum
    if report["status"] == "THIN":
        report["fix"] = f"iface vasp geom vacuum {source} --extend 18 -o {source}.extended"
    return report


def _print_vacuum_summary(batches: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """Human-readable recap of one or more directory `geom vacuum` runs (stderr)."""

    def _run(path: str) -> str:
        return path.replace("\\", "/").split("/")[0]

    all_rows = [(b, r) for b in batches for r in b["rows"]]
    total = len(all_rows)
    thin = [(b, r) for b, r in all_rows if r["status"] == "THIN"]
    mode = batches[0]["mode"]
    lines: list[str] = []

    for b in batches:
        rows = b["rows"]
        name_w = max((len(r["path"]) for r in rows), default=4)
        axis = rows[0]["axis"] if rows else "?"
        lines += [
            "",
            f"Slab vacuum  {b['root']}   (min {b['min_vacuum_a']:.0f} A, axis {axis})",
            f"  {'run'.ljust(name_w)}   vacuum_A   would_be",
        ]
        for r in rows:
            if r.get("extended"):
                would = "-> done"
            elif "would_be_a" in r:
                would = f"-> {r['would_be_a']:.1f}"
            else:
                would = ""
            flag = "  THIN" if (r["status"] == "THIN" and not r.get("extended")) else ""
            lines.append(f"  {r['path'].ljust(name_w)}   {r['vacuum_a']:>7.1f}   {would}{flag}")
    lines.append("")

    who = ", ".join(dict.fromkeys(_run(r["path"]) for _, r in thin))
    scope = " ".join(args.sources)
    if not thin:
        lines.append(f"All {total} clear {batches[0]['min_vacuum_a']:.0f} A - nothing to do.")
    elif mode == "audit":
        lines.append(
            f"{len(thin)} of {total} thin: {who}. "
            f"Preview a fix:  iface vasp geom vacuum {scope} --extend 18"
        )
    elif mode == "dry-run":
        lines.append(
            f"{len(thin)} of {total} thin: {who} -> would stretch to {args.extend:.0f} A. "
            f"Dry run, nothing written. Re-run with --execute to apply."
        )
    else:  # extended
        done = sum(b["extended"] for b in batches)
        lines.append(f"Stretched {done} structure(s) to {args.extend:.0f} A in place: {who}.")
    print("\n".join(lines), file=sys.stderr)


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


def cmd_slab_alignment(args: argparse.Namespace) -> int:
    payload = analyze_slab_alignment(
        args.root,
        config=args.config,
        run_sumo=args.run_sumo,
        write_dipole_fixes=args.write_dipole_fixes,
        only=args.only,
    )
    _json(payload)
    return 1 if payload["failures"] else 0


def cmd_slab_publication(args: argparse.Namespace) -> int:
    _json(
        plot_slab_publication(
            args.root,
            config=args.config,
            output_dir=args.output_dir,
            run_sumo=args.run_sumo,
        )
    )
    return 0


def cmd_adhesion_prepare(args: argparse.Namespace) -> int:
    _json(
        prepare_adhesion(
            args.interface_dir,
            method=args.method,
            structure=args.structure,
            incar=args.incar,
            curve_incar=args.curve_incar,
            kpoints=args.kpoints,
            potcar=args.potcar,
            z_plane=args.z_plane,
            guard=args.guard,
            min_side_fraction=args.min_side_fraction,
            lower_name=args.lower_name,
            upper_name=args.upper_name,
            distances=args.distances,
            output_dir=args.output_dir,
            launcher=args.launcher,
            propagate_launcher=not args.no_launcher,
            slab_mode=args.slab_mode,
            interface_static=args.interface_sp,
        )
    )
    return 0


def cmd_adhesion_audit(args: argparse.Namespace) -> int:
    references = None
    attrs = None
    if args.campaign and Path(args.campaign).is_file():
        validation = load_campaign(args.campaign).validation
        references = references_for(validation.get("references"), "work_of_adhesion")
        if args.interface:
            attrs = merge_interface_metadata(validation.get("interfaces"), args.interface)
    _json(audit_adhesion(args.output_dir, references=references, attrs=attrs))
    return 0


def cmd_adhesion_summary(args: argparse.Namespace) -> int:
    entries: list[tuple[str, str]] = []
    for item in args.entries:
        spec, sep, directory = item.partition("=")
        if not sep:
            spec, directory = Path(item).name, item
        entries.append((spec, directory))
    validation = None
    if args.campaign and Path(args.campaign).is_file():
        validation = load_campaign(args.campaign).validation
    _json(
        summarize_adhesion(
            entries, args.output, campaign_validation=validation, title=args.title
        )
    )
    return 0


def cmd_reference(args: argparse.Namespace) -> int:
    if args.reference_command == "list":
        _json({"bundled": list_reference_profiles()})
        return 0
    if args.reference_command == "activate":
        result = activate_reference_profile(args.campaign, args.name, write=args.write)
        _json(result)
        if not args.write and result["changed"]:
            print(
                "\n# dry run -- pass --write to apply the above to "
                f"{result['campaign']}",
                file=sys.stderr,
            )
        return 0
    profile = load_reference_profile(args.name)
    _json(
        {
            "profile": profile,
            "validation_references": expand_reference_profile(profile),
        }
    )
    return 0


def cmd_intermat(args: argparse.Namespace) -> int:
    if args.intermat_command == "status":
        payload = intermat_status()
    else:
        payload = generate_intermat_interfaces(
            args.film,
            args.substrate,
            args.output,
            film_miller=args.film_miller,
            substrate_miller=args.substrate_miller,
            film_thickness=args.film_thickness,
            substrate_thickness=args.substrate_thickness,
            separations=args.separation or [2.5],
            vacuum=args.vacuum,
            displacement_interval=args.displacement_interval,
            max_area=args.max_area,
            length_tolerance=args.length_tolerance,
            angle_tolerance=args.angle_tolerance,
            apply_strain=args.apply_strain,
            use_conventional_film=not args.primitive_film,
            use_conventional_substrate=not args.primitive_substrate,
            max_candidates=args.max_candidates,
            force=args.force,
        )
    _json(payload)
    return 0


def cmd_regfgw(args: argparse.Namespace) -> int:
    if args.regfgw_command == "status":
        payload = regfgw_status()
    elif args.regfgw_command == "optimize":
        payload = run_regfgw_optimize(
            args.substrate,
            args.film,
            args.output_dir,
            embedding=args.embedding,
            budget=args.budget,
            max_miller_idx=args.max_miller_idx,
            substrate_layers=args.substrate_layers,
            film_layers=args.film_layers,
            gap=args.gap,
            vacuum=args.vacuum,
        )
    else:
        payload = compare_registry_selection(
            args.topk,
            args.exhaustive,
            args.output,
            id_column=args.id_column,
            energy_column=args.energy_column,
            lower_energy_is_better=args.lower_energy_is_better,
            k_values=tuple(args.k) if args.k else (1, 3, 5),
        )
    _json(payload)
    return 0


def cmd_mlff_interfaces(args: argparse.Namespace) -> int:
    if args.mlff_interfaces_command == "discover":
        payload = discover_mlff_interface_sources(
            args.source_root,
            args.output_manifest,
            families=tuple(args.families),
            terms=tuple(args.terms),
            x_values=tuple(args.x_values),
            structure_name=args.structure_name,
        )
    elif args.mlff_interfaces_command == "build":
        payload = generate_mlff_interfaces_campaign(
            args.manifest,
            args.campaign_root,
            profile_path=args.profile,
            profile_name=args.profile_name,
            encut=args.encut,
            ivdw=args.ivdw,
            tebeg=args.tebeg,
            teend=args.teend,
            train_nsw=args.train_nsw,
            refit_nsw=args.refit_nsw,
            stability_nsw=args.stability_nsw,
            potim=args.potim,
            force=args.force,
        )
    elif args.mlff_interfaces_command == "array-launch":
        payload = write_throttled_array_launcher(
            load_campaign(args.campaign),
            stage=args.stage,
            concurrency=args.concurrency,
            array_profile_name=args.array_profile_name,
            output=args.output,
            force=args.force,
        )
    else:
        payload = mass_audit_mlff_interfaces(
            load_campaign(args.campaign), readiness_profile=args.readiness_profile
        )
    _json(payload)
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
        "--readiness-profile",
        choices=READINESS_PROFILES,
        default="general",
        help="Use 'perovskite' for fluxional perovskite sampling plateaus and capacity checkpoints",
    )
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
        "--readiness-profile",
        choices=READINESS_PROFILES,
        default="general",
        help="Use 'perovskite' for fluxional perovskite sampling plateaus and capacity checkpoints",
    )
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

    committee = commands.add_parser(
        "committee", help="Collect and verify immutable MLIP committee bundles"
    )
    committee_commands = committee.add_subparsers(dest="committee_command", required=True)
    committee_collect = committee_commands.add_parser(
        "collect", help="Copy completed seed models into a checksummed deployment bundle"
    )
    committee_collect.add_argument("source", help="Directory containing seed_* training runs")
    committee_collect.add_argument(
        "output", help="New bundle directory or .zip name; both directory and ZIP are created"
    )
    committee_collect.add_argument("--engine", choices=("mace",), default="mace")
    committee_collect.add_argument("--expected-members", type=int, default=4)
    committee_collect.add_argument(
        "--model-pattern",
        default="seed_*/mace_model/*_stagetwo.model",
        help="Source-relative glob identifying one final model per seed run",
    )
    committee_collect.add_argument(
        "--training-data",
        action="append",
        default=[],
        help="Training-data file to hash as provenance; repeat as needed",
    )
    committee_collect.add_argument(
        "--training-data-output",
        help="Optional separate ZIP for the training-data files; never added to the model ZIP",
    )
    committee_collect.add_argument(
        "--training-data-compression",
        choices=("deflated", "stored"),
        default="deflated",
        help="Compression for the separate training-data ZIP (default: deflated)",
    )
    committee_collect.add_argument("--label")
    committee_collect.add_argument("--notes")
    committee_collect.set_defaults(func=cmd_committee)
    committee_verify = committee_commands.add_parser(
        "verify", help="Validate every checksum in a collected directory or ZIP archive"
    )
    committee_verify.add_argument("bundle")
    committee_verify.set_defaults(func=cmd_committee)

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

    mlip_compare = commands.add_parser(
        "mlip-compare",
        help="Compare MACE and DeePMD committees on exactly matched canonical frames",
    )
    mlip_compare_commands = mlip_compare.add_subparsers(
        dest="mlip_compare_command", required=True
    )
    compare_prepare = mlip_compare_commands.add_parser(
        "prepare", help="Validate frame identity and generate MACE inference"
    )
    add_campaign_option(compare_prepare)
    compare_prepare.add_argument("--output-root")
    compare_prepare.add_argument("--mace-models-root")
    compare_prepare.add_argument(
        "--seeds", nargs="+", type=int, default=[11, 23, 37, 53]
    )
    compare_prepare.add_argument(
        "--deepmd-arch",
        default="dpa2",
        choices=("dpa2", "dpa2_ft", "dpa3", "dpa4"),
        help="Which trained DeePMD committee to compare against (default dpa2)",
    )
    compare_prepare.add_argument("--force", action="store_true")
    compare_prepare.set_defaults(func=cmd_mlip_compare)
    compare_status = mlip_compare_commands.add_parser(
        "status", help="Count complete model/system predictions for both backends"
    )
    add_campaign_option(compare_status)
    compare_status.add_argument("--output-root")
    compare_status.add_argument("--deepmd-eval-root")
    compare_status.set_defaults(func=cmd_mlip_compare)
    compare_finalize = mlip_compare_commands.add_parser(
        "finalize", help="Write matched micro, macro, grouped, and uncertainty metrics"
    )
    add_campaign_option(compare_finalize)
    compare_finalize.add_argument("--output-root")
    compare_finalize.add_argument("--deepmd-eval-root")
    compare_finalize.set_defaults(func=cmd_mlip_compare)

    mlip_progress_parser = commands.add_parser(
        "mlip-progress",
        help="One-glance training/evaluation/comparison progress for every committee",
    )
    add_campaign_option(mlip_progress_parser)
    mlip_progress_parser.add_argument(
        "--mace-committee-root",
        help="Directory holding mace_committee/ and mace_finetune_committee/ "
        "(default <campaign>/models/mace_committee_520eV)",
    )
    mlip_progress_parser.add_argument(
        "--json", action="store_true", help="Emit the raw payload instead of a table"
    )
    mlip_progress_parser.set_defaults(func=cmd_mlip_progress)

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
    adhesion.add_argument(
        "-c",
        "--campaign",
        dest="campaign_file",
        default=None,
        help="campaign.yaml whose validation.references (quantity: work_of_adhesion) "
        "are matched against each row's own columns (e.g. orientation, termination)",
    )
    adhesion.set_defaults(func=cmd_validate)
    separation = validation.add_parser("separation")
    separation.add_argument("source")
    separation.add_argument("output")
    separation.set_defaults(func=cmd_validate)

    separation_energy_parser = validation.add_parser(
        "separation-energy",
        help="Slab-referenced separation energy (J/m2) of hand-built interfaces, DFT vs MLIP",
    )
    separation_energy_parser.add_argument(
        "output", help="Directory for separation_energy.{json,csv,md,png,svg,pdf}"
    )
    separation_energy_parser.add_argument(
        "entries",
        nargs="+",
        metavar="[LABEL=]SET_DIR",
        help="Each SET_DIR holds interface/ slab_a/ slab_b/ run directories; an "
        "optional 'LABEL=' prefix is fnmatched against validation.interfaces for "
        "the literature overlay",
    )
    separation_energy_parser.add_argument(
        "--mace-model", action="append", default=[], dest="mace_models",
        help="Path to a MACE committee member (repeat for the committee)",
    )
    separation_energy_parser.add_argument(
        "--deepmd-model", action="append", default=[], dest="deepmd_models",
        help="Path to a DeePMD committee member (repeat for the committee)",
    )
    separation_energy_parser.add_argument(
        "--reference", choices=("free-surface", "bulk"), default="free-surface",
        help="free-surface (relaxed half-slabs; equals the work of adhesion) or bulk",
    )
    separation_energy_parser.add_argument("--n-interfaces", type=int, default=1)
    separation_energy_parser.add_argument("--area-axis", choices=("a", "b", "c"))
    separation_energy_parser.add_argument("--device", default="cpu")
    add_campaign_option(separation_energy_parser)
    separation_energy_parser.set_defaults(func=cmd_validate)

    interface_energy_parser = validation.add_parser(
        "interface-energy",
        help="Bulk-referenced interfacial energy (J/m2) from the canonical dataset",
    )
    add_campaign_option(interface_energy_parser)
    interface_energy_parser.add_argument("output", help="Directory for interface_energy.{json,csv,md}")
    interface_energy_parser.add_argument("--dataset-root")
    interface_energy_parser.add_argument(
        "--predictions",
        help="An audit/mlip_compare* directory; adds MACE-committee gamma_int on the "
        "test-split frames alongside the DFT value",
    )
    interface_energy_parser.add_argument("--equilibration-frames", type=int, default=100)
    interface_energy_parser.add_argument(
        "--n-interfaces",
        type=int,
        default=None,
        help="override; the default per interface comes from validation.interfaces "
        "in the campaign (else 2)",
    )
    interface_energy_parser.add_argument("--blocks", type=int, default=10)
    interface_energy_parser.add_argument(
        "--stacking-axis",
        choices=("a", "b", "c"),
        help="override; the default per interface comes from validation.interfaces",
    )
    interface_energy_parser.set_defaults(func=cmd_validate)
    stratified = validation.add_parser(
        "stratified",
        help="Report parity errors per geometry class (kind/high-temperature/low-coordination) "
        "instead of one pooled number",
    )
    stratified.add_argument("source")
    stratified.add_argument("output")
    stratified.add_argument("--reference-column", default="reference")
    stratified.add_argument("--predicted-column", default="predicted")
    stratified.add_argument("--kind-column", default="kind")
    stratified.add_argument("--high-temperature-column", default="high_temperature")
    stratified.add_argument("--min-coordination-column", default="min_coordination_number")
    stratified.add_argument(
        "--low-coordination-percentile",
        type=float,
        default=10.0,
        help="Percentile of this CSV's own coordination-number distribution used as the "
        "low-coordination cutoff (default: 10)",
    )
    stratified.set_defaults(func=cmd_validate)

    reference = commands.add_parser(
        "reference",
        help="Bundled literature reference profiles (validation.reference_profiles)",
    )
    reference_commands = reference.add_subparsers(dest="reference_command", required=True)
    reference_commands.add_parser("list", help="List the bundled reference profiles")
    reference_show = reference_commands.add_parser(
        "show",
        help="Print a profile and the validation.references entries it expands to",
    )
    reference_show.add_argument(
        "name", help="Bundled profile name, or a path to a profile YAML file"
    )
    reference_activate = reference_commands.add_parser(
        "activate",
        help="Add a profile to validation.reference_profiles in a campaign file "
        "(dry run unless --write)",
    )
    reference_activate.add_argument(
        "name", help="Bundled profile name, or a path to a profile YAML file"
    )
    add_campaign_option(reference_activate)
    reference_activate.add_argument(
        "--write",
        action="store_true",
        help="Apply the edit; without it the resulting file is only printed",
    )
    reference.set_defaults(func=cmd_reference)

    report = commands.add_parser("report", help="Build a self-contained HTML dashboard")
    add_campaign_option(report)
    report.add_argument("--output")
    report.set_defaults(func=cmd_report)

    active_learning = commands.add_parser(
        "active-learning", help="Optional supervised active-learning adapters"
    )
    active_backends = active_learning.add_subparsers(dest="active_backend", required=True)
    ai2kit = active_backends.add_parser(
        "ai2kit",
        help="AI2-Kit active learning: TESLA MACE/OpenMM/VASP or legacy DeepMD/LAMMPS/VASP",
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

    intermat = commands.add_parser(
        "intermat", help="Optional InterMat crystalline-interface geometry adapter"
    )
    intermat_commands = intermat.add_subparsers(dest="intermat_command", required=True)
    intermat_status_parser = intermat_commands.add_parser(
        "status", help="Report dependency availability and adapter boundaries"
    )
    intermat_status_parser.set_defaults(func=cmd_intermat)
    intermat_generate = intermat_commands.add_parser(
        "generate", help="Generate commensurate film/substrate interface candidates"
    )
    intermat_generate.add_argument("film", help="Bulk film POSCAR")
    intermat_generate.add_argument("substrate", help="Bulk substrate POSCAR")
    intermat_generate.add_argument("output", help="Dedicated adapter output directory")
    intermat_generate.add_argument("--film-miller", nargs=3, type=int, default=(0, 0, 1))
    intermat_generate.add_argument("--substrate-miller", nargs=3, type=int, default=(0, 0, 1))
    intermat_generate.add_argument("--film-thickness", type=float, default=16.0)
    intermat_generate.add_argument("--substrate-thickness", type=float, default=16.0)
    intermat_generate.add_argument(
        "--separation",
        action="append",
        type=float,
        default=None,
        help="Interface separation in angstrom; repeat for a scan (default: 2.5)",
    )
    intermat_generate.add_argument("--vacuum", type=float, default=12.0)
    intermat_generate.add_argument(
        "--displacement-interval",
        type=float,
        default=0.0,
        help="Fractional xy registry interval; zero emits one registry",
    )
    intermat_generate.add_argument("--max-area", type=float, default=300.0)
    intermat_generate.add_argument("--length-tolerance", type=float, default=0.08)
    intermat_generate.add_argument("--angle-tolerance", type=float, default=1.0)
    intermat_generate.add_argument("--apply-strain", action="store_true")
    intermat_generate.add_argument("--primitive-film", action="store_true")
    intermat_generate.add_argument("--primitive-substrate", action="store_true")
    intermat_generate.add_argument("--max-candidates", type=int, default=500)
    intermat_generate.add_argument("--force", action="store_true")
    intermat_generate.set_defaults(func=cmd_intermat)

    regfgw = commands.add_parser(
        "regfgw",
        help="Optional RegFGW registry pre-screening adapter and a trial comparison report",
    )
    regfgw_commands = regfgw.add_subparsers(dest="regfgw_command", required=True)
    regfgw_status_parser = regfgw_commands.add_parser(
        "status", help="Report dependency availability; output schema is not verified"
    )
    regfgw_status_parser.set_defaults(func=cmd_regfgw)
    regfgw_optimize = regfgw_commands.add_parser(
        "optimize",
        help="Run regfgw_coherent --mode optimize (does not parse its output)",
    )
    regfgw_optimize.add_argument("substrate", help="Substrate structure (CIF)")
    regfgw_optimize.add_argument("film", help="Film structure (CIF)")
    regfgw_optimize.add_argument("output_dir")
    regfgw_optimize.add_argument("--embedding", help="Optional embedding config (JSON)")
    regfgw_optimize.add_argument("--budget", type=int, default=3)
    regfgw_optimize.add_argument("--max-miller-idx", type=int)
    regfgw_optimize.add_argument("--substrate-layers", type=int)
    regfgw_optimize.add_argument("--film-layers", type=int)
    regfgw_optimize.add_argument("--gap", type=float)
    regfgw_optimize.add_argument("--vacuum", type=float)
    regfgw_optimize.set_defaults(func=cmd_regfgw)
    regfgw_compare = regfgw_commands.add_parser(
        "compare",
        help="Check whether a top-k registry selection preserved the true low-energy ones",
    )
    regfgw_compare.add_argument("topk", help="Ranked top-k registry CSV")
    regfgw_compare.add_argument("exhaustive", help="Exhaustive-grid CSV with relaxed energies")
    regfgw_compare.add_argument("output")
    regfgw_compare.add_argument("--id-column", default="registry_id")
    regfgw_compare.add_argument("--energy-column", default="work_of_adhesion_ev_a2")
    regfgw_compare.add_argument(
        "--lower-energy-is-better",
        action="store_true",
        help="Set if energy_column is lower-is-better (default: higher, matching work of adhesion)",
    )
    regfgw_compare.add_argument(
        "--k", action="append", type=int, default=None, help="Repeat for multiple k values (default: 1 3 5)"
    )
    regfgw_compare.set_defaults(func=cmd_regfgw)

    mlff_interfaces = commands.add_parser(
        "mlff-interfaces",
        help="Bulk MLFF training campaign for a family x termination x composition grid",
    )
    mlff_interfaces_commands = mlff_interfaces.add_subparsers(
        dest="mlff_interfaces_command", required=True
    )
    mi_discover = mlff_interfaces_commands.add_parser(
        "discover", help="Best-effort discovery of one structure per grid cell; review before build"
    )
    mi_discover.add_argument("source_root")
    mi_discover.add_argument("output_manifest")
    mi_discover.add_argument("--families", nargs="+", default=["Real", "Ideal"])
    mi_discover.add_argument("--terms", nargs="+", default=["N_Term", "Ti_Term"])
    mi_discover.add_argument(
        "--x-values", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0]
    )
    mi_discover.add_argument("--structure-name", default="CONTCAR")
    mi_discover.set_defaults(func=cmd_mlff_interfaces)
    mi_build = mlff_interfaces_commands.add_parser(
        "build", help="Generate campaign.yaml from a reviewed (fully 'matched') manifest"
    )
    mi_build.add_argument("manifest")
    mi_build.add_argument("campaign_root")
    mi_build.add_argument("--profile", required=True, help="Scheduler profile YAML path")
    mi_build.add_argument("--profile-name", default="vasp_train")
    mi_build.add_argument("--encut", type=float, default=520.0)
    mi_build.add_argument("--ivdw", type=int, default=11)
    mi_build.add_argument("--tebeg", type=float, default=300.0)
    mi_build.add_argument("--teend", type=float, default=600.0)
    mi_build.add_argument("--train-nsw", type=int, default=3000)
    mi_build.add_argument("--refit-nsw", type=int, default=0)
    mi_build.add_argument("--stability-nsw", type=int, default=3000)
    mi_build.add_argument("--potim", type=float, default=1.0)
    mi_build.add_argument("--force", action="store_true")
    mi_build.set_defaults(func=cmd_mlff_interfaces)
    mi_array = mlff_interfaces_commands.add_parser(
        "array-launch", help="Write one throttled Slurm array job over every prepared leaf"
    )
    add_campaign_option(mi_array)
    mi_array.add_argument("--stage", default="train")
    mi_array.add_argument("--concurrency", type=int, default=4)
    mi_array.add_argument(
        "--array-profile-name",
        help="Optional resource profile; defaults to the selected stage's own profile",
    )
    mi_array.add_argument("--output")
    mi_array.add_argument("--force", action="store_true")
    mi_array.set_defaults(func=cmd_mlff_interfaces)
    mi_audit = mlff_interfaces_commands.add_parser(
        "audit", help="Audit the grid and roll it up by family/term/x instead of a flat run list"
    )
    add_campaign_option(mi_audit)
    mi_audit.add_argument("--readiness-profile", default="general", choices=READINESS_PROFILES)
    mi_audit.set_defaults(func=cmd_mlff_interfaces)

    surface = commands.add_parser(
        "surface",
        help="Reaction- and magnetism-aware surface campaign generation",
    )
    surface_commands = surface.add_subparsers(dest="surface_command", required=True)

    surface_init = surface_commands.add_parser(
        "init", help="Write the documented NiO(110) reactive-campaign template"
    )
    surface_init.add_argument("-o", "--output", default="surface_campaign.yaml")
    surface_init.add_argument("--force", action="store_true")
    surface_init.set_defaults(func=cmd_surface_init)

    surface_analyze = surface_commands.add_parser(
        "analyze", help="Identify exposed under-coordinated surface metal sites"
    )
    surface_analyze.add_argument("structure")
    surface_analyze.add_argument("--metal", required=True)
    surface_analyze.add_argument("--anion", default="O")
    surface_analyze.add_argument("--coordination-cutoff", type=float, default=2.7)
    surface_analyze.add_argument("--bulk-coordination", type=int, default=6)
    surface_analyze.add_argument("--top-tolerance", type=float, default=0.8)
    surface_analyze.set_defaults(func=cmd_surface_analyze)

    surface_cell = surface_commands.add_parser(
        "cell-optimize",
        help="Choose a tractable surface cell satisfying image-gap and AFM constraints",
    )
    surface_cell.add_argument("slab", help="Primitive or small periodic slab")
    surface_cell.add_argument("--adsorbate", help="Molecule used for periodic-image clearance")
    surface_cell.add_argument("--min-multiplier", type=int, default=1)
    surface_cell.add_argument("--max-multiplier", type=int, default=30)
    surface_cell.add_argument("--max-atoms", type=int)
    surface_cell.add_argument("--min-translation", type=float, default=0.0)
    surface_cell.add_argument("--min-image-gap", type=float, default=3.5)
    surface_cell.add_argument("--max-aspect", type=float, default=2.0)
    surface_cell.add_argument(
        "--translation-parity",
        nargs=2,
        type=int,
        metavar=("P1", "P2"),
        help="AFM phase vector modulo 2; NiO(110) uses 1 0",
    )
    surface_cell.add_argument("--orientation-samples", type=int, default=12)
    surface_cell.add_argument("--freeze-bottom-layers", type=int)
    surface_cell.add_argument("--top", type=int, default=20)
    surface_cell.add_argument("-o", "--output", help="Write the best cell as a POSCAR")
    surface_cell.add_argument("--force", action="store_true")
    surface_cell.set_defaults(func=cmd_surface_cell_optimize)

    surface_plan = surface_commands.add_parser(
        "plan", help="Validate and enumerate a reactive surface campaign without writing runs"
    )
    surface_plan.add_argument("campaign")
    surface_plan.set_defaults(func=cmd_surface_plan)

    surface_build = surface_commands.add_parser(
        "build", help="Generate provenance-stamped reactive surface run directories"
    )
    surface_build.add_argument("campaign")
    surface_build.add_argument("--force", action="store_true")
    surface_build.set_defaults(func=cmd_surface_build)

    surface_audit = surface_commands.add_parser(
        "audit", help="Classify relaxed chemistry and audit converged surface spins"
    )
    surface_audit.add_argument("root")
    surface_audit.add_argument("-o", "--output")
    surface_audit.set_defaults(func=cmd_surface_audit)

    surface_select = surface_commands.add_parser(
        "select",
        help="Select uncertain/diverse frames while preserving reaction- and spin-state coverage",
    )
    surface_select.add_argument("candidates", help="CSV containing uncertainty and surface-state labels")
    surface_select.add_argument("output", help="Labeling-queue CSV")
    surface_select.add_argument("--count", type=int, required=True)
    surface_select.add_argument("--uncertainty-column", default="uncertainty")
    surface_select.add_argument("--feature-column", action="append", default=[])
    surface_select.add_argument("--state-column", action="append", default=[])
    surface_select.add_argument("--max-per-state", type=int, default=2)
    surface_select.add_argument("--uncertainty-weight", type=float, default=0.65)
    surface_select.set_defaults(func=cmd_surface_select)

    vasp = commands.add_parser("vasp", help="Safe VASP utilities")
    vasp_commands = vasp.add_subparsers(dest="vasp_command", required=True)
    recover = vasp_commands.add_parser(
        "ml-recover",
        aliases=["recover"],
        help="Continue/refit/stability/heat/capacity recovery for VASP MLFF",
    )
    recover.add_argument(
        "operation",
        choices=("continue", "discard", "expand", "refit", "stability", "heat"),
        help="'heat' promotes ML_FFN and sets ML_LHEAT=.TRUE. for Green-Kubo heat-flux production",
    )
    recover.add_argument("folder")
    recover.add_argument("--temperature", type=float)
    recover.add_argument("--nsw", type=int)
    recover.add_argument("--ml-mb", type=int)
    recover.add_argument("--ml-mconf", type=int)
    recover.add_argument("--force-expand", action="store_true")
    recover.add_argument(
        "--ml-outblock",
        type=int,
        default=1,
        help="For heat recovery, write ML_HEAT every N steps (default: 1)",
    )
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
    opt_prepare = vasp_commands.add_parser(
        "opt-prepare",
        help="Prepare and audit a notebook-generated VASP slab-optimization batch",
    )
    opt_prepare.add_argument("root", help="Generated OPT root containing leaf run directories")
    opt_prepare.add_argument(
        "--manifest",
        help="CSV with a path column (for example generated/manifest_batch.csv); "
        "without it, discover POSCAR leaves recursively",
    )
    opt_prepare.add_argument(
        "--launcher-template",
        help="Launcher copied into every selected leaf (existing differing launchers require --force-launcher)",
    )
    opt_prepare.add_argument(
        "--exclude-prefix",
        action="append",
        default=[],
        help="Exclude a manifest path prefix such as OH0; may be repeated and is remembered by --audit-only",
    )
    opt_prepare.add_argument(
        "--launcher",
        help="Launcher name to audit (default: prefer runvasp.sh, then run.slurm)",
    )
    opt_prepare.add_argument(
        "--potcar-command",
        default="POTCAR_gen",
        help="Command run inside leaves with a missing/empty POTCAR (default: POTCAR_gen)",
    )
    opt_prepare.add_argument(
        "--require-module",
        default="vasp6/6.5.1-cpu",
        help="Exact VASP module string required in every launcher; pass an empty string to disable",
    )
    opt_prepare.add_argument("--dry-run", action="store_true", help="Preview without writing or running POTCAR_gen")
    opt_prepare.add_argument(
        "--audit-only",
        action="store_true",
        help="Re-audit existing inputs without copying launchers or running POTCAR_gen",
    )
    opt_prepare.add_argument(
        "--force-launcher",
        action="store_true",
        help="Replace a differing leaf launcher with --launcher-template",
    )
    opt_prepare.set_defaults(func=cmd_vasp_opt_prepare)
    opt_launch = vasp_commands.add_parser(
        "opt-launch",
        help="Launch all unchanged PASS-audited OPT leaves (dry-run by default)",
    )
    opt_launch.add_argument("roots", nargs="+", help="Prepared OPT root(s)")
    opt_launch.add_argument(
        "--launcher",
        help="Audited launcher name (default: use the launcher recorded by opt-prepare)",
    )
    opt_launch.add_argument(
        "--execute",
        action="store_true",
        help="Actually call sbatch; without this flag only print the fully verified plan",
    )
    opt_launch.set_defaults(func=cmd_vasp_opt_launch)
    step2 = vasp_commands.add_parser(
        "step2-prepare",
        aliases=["step2-series"],
        help="Prepare and audit fixed-temperature Step2 DFT-MD trees from Step1",
    )
    step2.add_argument("source", help="Step1 root containing per-run INCAR and CONTCAR files")
    step2.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        default=[300.0, 450.0, 600.0],
        help="Fixed-temperature output series (default: 300 450 600)",
    )
    step2.add_argument(
        "--output-root",
        help="Parent for Step2_<T>K trees (default: parent of the Step1 root)",
    )
    step2.add_argument(
        "--template",
        help="Step2 INCAR template (default: packaged INCAR.step2_dft_md)",
    )
    step2.add_argument(
        "--source-structure",
        default="CONTCAR",
        help="Finished Step1 structure promoted to POSCAR (default: CONTCAR)",
    )
    step2.add_argument(
        "--protocol",
        choices=sorted(AIMD_PROTOCOLS),
        default="academic",
        help=(
            "AIMD profile. 'academic' (default): NSW=5000 (5 ps), dense NBLOCK "
            "stride. 'training': NSW=1000 (1 ps), OUTCAR-trimmed INCAR, frames "
            "thinned by decorrelation via 'iface vasp step2-sample'"
        ),
    )
    step2.add_argument("--dry-run", action="store_true", help="Validate and print the exact plan only")
    step2.add_argument(
        "--audit-only",
        action="store_true",
        help="Re-audit existing Step2_<T>K trees without preparing or submitting anything",
    )
    step2.add_argument(
        "--set-protocol",
        action="store_true",
        help=(
            "Rewrite the INCAR of every run in existing Step2_<T>K trees to the "
            "given --protocol (re-inherits LDAU*/spin from Step1, refreshes the "
            "manifest + audit); refuses once a run has started. Combine with "
            "--dry-run to preview"
        ),
    )
    step2.set_defaults(func=cmd_vasp_step2_series)

    step1_prepare = vasp_commands.add_parser(
        "step1-prepare",
        help="Promote a recursive OPT tree into a sibling Step1 preheat tree",
    )
    step1_prepare.add_argument(
        "source", help="OPT root with per-run INCAR, CONTCAR (WAVECAR optional)"
    )
    step1_prepare.add_argument("--temperature", type=float, default=300.0)
    step1_prepare.add_argument("--output-root", help="Parent for Step1/ (default: parent of OPT root)")
    step1_prepare.add_argument("--template", help="Step1 INCAR template (default: packaged)")
    step1_prepare.add_argument("--source-structure", default="CONTCAR")
    step1_prepare.add_argument(
        "--protocol",
        choices=sorted(AIMD_PROTOCOLS),
        default="academic",
        help="academic: NSW=2000 (~2 ps); training: NSW=400 (~0.4 ps)",
    )
    step1_prepare.add_argument(
        "--fresh-start",
        action="store_true",
        help="Force ISTART=0 (fresh electronic start) for every run and never "
        "require a WAVECAR; without it, each run uses ISTART=1 when the OPT "
        "produced a nonempty WAVECAR and falls back to ISTART=0 with a warning",
    )
    step1_prepare.add_argument(
        "--require-wavecar",
        action="store_true",
        help="Restore the old strict behaviour: refuse any run without a nonempty "
        "OPT WAVECAR instead of falling back to a fresh start",
    )
    step1_prepare.add_argument("--dry-run", action="store_true")
    step1_prepare.add_argument("--audit-only", action="store_true")
    step1_prepare.set_defaults(func=cmd_vasp_step1_prepare)

    step1_status_parser = vasp_commands.add_parser(
        "step1-status",
        help="Runtime status of a Step1 preheat tree: frames produced, INCAR quality, which jobs are done",
    )
    step1_status_parser.add_argument(
        "root", nargs="?", default=".", help="Step1 tree root or a single run directory"
    )
    step1_status_parser.add_argument(
        "--stale-hours",
        type=float,
        default=6.0,
        help="Flag a running job as 'stalled?' when its OSZICAR is older than this (default 6)",
    )
    step1_status_parser.add_argument(
        "--json", action="store_true", help="Emit the raw payload instead of a table"
    )
    step1_status_parser.set_defaults(func=cmd_vasp_step1_status)

    step2_status_parser = vasp_commands.add_parser(
        "step2-status",
        help="Runtime status of a Step2 DFT-MD series: frames produced, INCAR quality, which jobs are done",
    )
    step2_status_parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Parent of the Step2_<T>K trees, a single tree, or one run directory",
    )
    step2_status_parser.add_argument(
        "--stale-hours",
        type=float,
        default=6.0,
        help="Flag a running job as 'stalled?' when its OSZICAR is older than this (default 6)",
    )
    step2_status_parser.add_argument(
        "--json", action="store_true", help="Emit the raw payload instead of a table"
    )
    step2_status_parser.set_defaults(func=cmd_vasp_step2_status)

    step1_protocol = vasp_commands.add_parser(
        "step1-protocol",
        help="Switch a Step1 preheat INCAR between academic/training length and audit it",
    )
    step1_protocol.add_argument(
        "target", help="A Step1 INCAR, a run directory, or a Step1 tree root"
    )
    step1_protocol.add_argument(
        "--protocol",
        choices=sorted(AIMD_PROTOCOLS),
        default="academic",
        help=(
            "academic (~2 ps preheat, default, matches step2-prepare) or "
            "training (~0.4 ps preheat)"
        ),
    )
    step1_protocol.add_argument(
        "--nsw",
        type=int,
        help=(
            "Override the profile preheat length in steps; by default the "
            "protocol's own value is used (academic 2000, training 400)"
        ),
    )
    step1_protocol.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit against the protocol without rewriting NSW",
    )
    step1_protocol.add_argument(
        "--create",
        action="store_true",
        help="Create the INCAR if the target file does not exist",
    )
    step1_protocol.set_defaults(func=cmd_vasp_step1_protocol)

    step2_sample = vasp_commands.add_parser(
        "step2-sample",
        help="Select Step2 training frames at the measured energy decorrelation time",
    )
    step2_sample.add_argument("roots", nargs="+", help="Prepared Step2_<T>K root(s)")
    step2_sample.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print the per-run frame plan without writing step2_sample.*",
    )
    step2_sample.set_defaults(func=cmd_vasp_step2_sample)

    step2_launch = vasp_commands.add_parser(
        "step2-launch",
        help="Launch all unchanged PASS-audited daughter runs in one or more Step2 roots",
    )
    step2_launch.add_argument("roots", nargs="+", help="Step2_<T>K root(s) to launch")
    step2_launch.add_argument(
        "--launcher",
        help="Audited launcher name (default: prefer runvasp.sh, then run.slurm)",
    )
    step2_launch.add_argument(
        "--execute",
        action="store_true",
        help="Actually call sbatch; without this flag only print the verified launch plan",
    )
    step2_launch.set_defaults(func=cmd_vasp_step2_launch)
    band = vasp_commands.add_parser("band", help="Prepare a line-mode band run")
    band.add_argument("source")
    band.add_argument("destination")
    band.add_argument("--kpoints", required=True)
    band.add_argument("--lmaxmix", type=int, default=4)
    band.add_argument("--force", action="store_true")
    band.set_defaults(func=cmd_vasp_band)
    potcar = vasp_commands.add_parser("potcar", help="Assemble POTCAR from a licensed local tree")
    potcar.add_argument("poscar")
    potcar.add_argument(
        "--root",
        help="Licensed PBE PAW tree; otherwise use IFACE_POTCAR_ROOT/VASP_PP_PATH/~/pot/potpaw_PBE",
    )
    potcar.add_argument(
        "--map",
        help="Optional mapping override (default: built-in supplied POTCAR_DEFS dictionary)",
    )
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
    incar.add_argument(
        "--workfunction",
        action="store_true",
        help="Also set LVHAR=True so LOCPOT contains the electrostatic potential "
        "needed for work-function analysis of a surface calculation",
    )
    incar.add_argument("--create", action="store_true")
    incar.set_defaults(func=cmd_vasp_incar)
    pack = vasp_commands.add_parser("pack", help="Package lightweight reproducibility outputs")
    pack.add_argument("output")
    pack.add_argument("--root", default=".")
    pack.add_argument("--include-large", action="store_true")
    pack.add_argument("--force", action="store_true")
    pack.set_defaults(func=cmd_vasp_pack)
    archive_models = vasp_commands.add_parser(
        "archive-models",
        help="Archive accepted VASP-MLFF runs with ML_AB and checksum provenance",
    )
    archive_models.add_argument(
        "output",
        nargs="?",
        help="ZIP path (default: timestamped archive in the current directory)",
    )
    archive_models.add_argument("--root", default=".")
    archive_models.add_argument(
        "--exclude-folder",
        "--exclude-folders",
        dest="exclude_folders",
        action="extend",
        nargs="+",
        default=[],
        metavar="NAME",
        help="Exact folder names to skip; accepts a list and may be repeated",
    )
    archive_models.add_argument(
        "--recursive",
        action="store_true",
        help="Also scan below immediate child folders (off by default)",
    )
    archive_models.add_argument("--include-large", action="store_true")
    archive_models.add_argument("--force", action="store_true")
    archive_models.set_defaults(func=cmd_vasp_archive_models)
    vsubmit = vasp_commands.add_parser(
        "submit", help="Submit one VASP run, optionally preparing an MLFF recovery first"
    )
    vsubmit.add_argument("folder")
    vsubmit.add_argument(
        "--launcher",
        help="Batch script (default: prefer runvasp.sh, then run.slurm)",
    )
    vsubmit.add_argument(
        "--potcar-root",
        help="Licensed PBE PAW tree used only when POTCAR is missing",
    )
    vsubmit.add_argument(
        "--potcar-map",
        help="Optional mapping override; default is the built-in supplied dictionary",
    )
    recovery = vsubmit.add_mutually_exclusive_group()
    recovery.add_argument(
        "--ml-continue",
        "--recover-continue",
        dest="recover_continue",
        action="store_true",
        help="Archive and prepare an interrupted MLFF continuation before submission",
    )
    recovery.add_argument(
        "--ml-capacity-recovery",
        "--recover-capacity",
        dest="recover_capacity",
        action="store_true",
        help="Archive a capacity stop and resubmit; defaults to bounded-memory basis discarding",
    )
    vsubmit.add_argument(
        "--temperature",
        type=float,
        help="Continuation temperature; otherwise preserve TEBEG or use 300 K fallback",
    )
    vsubmit.add_argument(
        "--nsw",
        type=int,
        help="Continuation NSW; otherwise preserve the existing INCAR value",
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

    slab_alignment = vasp_commands.add_parser(
        "slab-align",
        help="Vacuum-align slab band edges across a calculation family",
    )
    slab_alignment.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root whose immediate children are VASP calculations (default: .)",
    )
    slab_alignment.add_argument(
        "--config",
        default="slab_alignment.json",
        help="JSON configuration path, relative to root by default",
    )
    slab_alignment.add_argument(
        "--run-sumo",
        action="store_true",
        help="Run sumo-dosplot in each analyzed calculation directory",
    )
    slab_alignment.add_argument(
        "--write-dipole-fixes",
        dest="write_dipole_fixes",
        action="store_true",
        help="Write non-destructive INCAR.dipole_fix previews for flagged cases (default)",
    )
    slab_alignment.add_argument(
        "--no-write-dipole-fixes",
        dest="write_dipole_fixes",
        action="store_false",
        help="Audit and flag non-flat cases without writing proposed INCAR files",
    )
    slab_alignment.add_argument(
        "--only",
        help="Analyze only this immediate child directory",
    )
    slab_alignment.set_defaults(write_dipole_fixes=True)
    slab_alignment.set_defaults(func=cmd_slab_alignment)

    slab_publication = vasp_commands.add_parser(
        "slab-publish",
        help="Create publication vacuum, PDOS, and band-alignment figures",
    )
    slab_publication.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root containing the selected VASP calculation directories (default: .)",
    )
    slab_publication.add_argument(
        "--config",
        default="slab_publication.json",
        help="Publication JSON configuration path, relative to root by default",
    )
    slab_publication.add_argument(
        "--output-dir",
        default="publication_figures",
        help="Output directory, relative to root by default",
    )
    slab_publication.add_argument(
        "--run-sumo",
        action="store_true",
        help="Run sumo-dosplot before plotting (recommended through sbatch)",
    )
    slab_publication.set_defaults(func=cmd_slab_publication)

    adhesion = vasp_commands.add_parser(
        "adhesion", help="Prepare work-of-adhesion calculations (MLFF or DFT)"
    )
    adhesion_commands = adhesion.add_subparsers(dest="adhesion_command", required=True)
    adhesion_prepare = adhesion_commands.add_parser(
        "prepare",
        help="Create isolated-slab and rigid-separation-curve inputs from a reference interface run",
    )
    adhesion_prepare.add_argument(
        "interface_dir", nargs="?", default=".", help="Reference run directory (default: .)"
    )
    adhesion_prepare.add_argument("--method", choices=ADHESION_METHODS, default="mlff")
    adhesion_prepare.add_argument(
        "--structure", help="Structure file relative to interface_dir (default: CONTCAR, else POSCAR)"
    )
    adhesion_prepare.add_argument(
        "--incar", help="Slab-relaxation INCAR (default: INCAR_MLFF_RELAX for mlff, else INCAR)"
    )
    adhesion_prepare.add_argument(
        "--curve-incar", help="Optional fixed INCAR for every rigid-curve point (default: generated)"
    )
    adhesion_prepare.add_argument("--kpoints", help="default: KPOINTS")
    adhesion_prepare.add_argument("--potcar", help="default: POTCAR")
    adhesion_prepare.add_argument(
        "--z-plane", type=float, help="Cartesian z split plane in Angstrom (default: auto-detected)"
    )
    adhesion_prepare.add_argument(
        "--guard", type=float, default=0.20, help="Minimum atom-to-plane distance in Angstrom"
    )
    adhesion_prepare.add_argument("--min-side-fraction", type=float, default=0.10)
    adhesion_prepare.add_argument("--lower-name", default="lower")
    adhesion_prepare.add_argument("--upper-name", default="upper")
    adhesion_prepare.add_argument(
        "--slab-mode",
        choices=ADHESION_SLAB_MODES,
        default="relax",
        help="relax (default) lets each isolated slab relax; static evaluates it at the "
        "as-cut geometry with no ionic motion -- use this when the driving model "
        "extrapolates poorly for an isolated fragment (e.g. it collapses on relaxation)",
    )
    adhesion_prepare.add_argument(
        "--interface-sp",
        action="store_true",
        help="also generate interface_static/: a fresh single-point of the whole "
        "interface at the slab INCAR settings, so all three energies share one "
        "electronic setup (recommended with --slab-mode static -> ideal work of "
        "separation; avoids reusing a loose-EDIFF MD energy)",
    )
    adhesion_prepare.add_argument(
        "--distances",
        nargs="+",
        type=float,
        default=[0.5, 1, 2, 3, 4, 6, 8],
        help="Positive rigid-separation distances in Angstrom",
    )
    adhesion_prepare.add_argument(
        "--output-dir", help="default: sibling INTERFACE_DIR_adhesion_METHOD"
    )
    adhesion_prepare.add_argument(
        "--launcher",
        help="Launcher name in interface_dir to propagate (default: auto-detect "
        "runvasp.sh, then run.slurm)",
    )
    adhesion_prepare.add_argument(
        "--no-launcher",
        action="store_true",
        help="Do not copy a launcher into the generated slab/rigid-curve directories",
    )
    adhesion_prepare.set_defaults(func=cmd_adhesion_prepare)
    adhesion_audit = adhesion_commands.add_parser(
        "audit",
        help="Read back finished slab/rigid-curve runs and compute work of adhesion + curve",
    )
    adhesion_audit.add_argument(
        "output_dir", help="Directory previously created by 'adhesion prepare'"
    )
    adhesion_audit.add_argument(
        "-c",
        "--campaign",
        default=None,
        help="campaign.yaml providing validation.references (quantity: work_of_adhesion) "
        "for a literature comparison of the computed work of adhesion",
    )
    adhesion_audit.add_argument(
        "--interface",
        help="interface leaf / id, fnmatched against validation.interfaces; its "
        "orientation/termination select which reference values apply",
    )
    adhesion_audit.set_defaults(func=cmd_adhesion_audit)

    adhesion_summary = adhesion_commands.add_parser(
        "summary",
        help="Roll several audited adhesion trees into one W_ad table + publication figure",
    )
    adhesion_summary.add_argument(
        "entries",
        nargs="+",
        metavar="[LEAF=]AUDIT_DIR",
        help="Each is an 'adhesion prepare' output directory, optionally prefixed "
        "'<interface leaf>=' so its validation.interfaces metadata selects which "
        "literature values to overlay",
    )
    adhesion_summary.add_argument(
        "-o", "--output", required=True, help="Directory for adhesion_summary.{json,csv,md,png,svg,pdf}"
    )
    adhesion_summary.add_argument(
        "-c",
        "--campaign",
        default=None,
        help="campaign.yaml providing validation.references / validation.interfaces",
    )
    adhesion_summary.add_argument("--title", help="Optional figure title")
    adhesion_summary.set_defaults(func=cmd_adhesion_summary)

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
    vacuum = geom_commands.add_parser(
        "vacuum", help="Audit slab vacuum per face; --extend stretches the cell"
    )
    vacuum.add_argument(
        "sources",
        nargs="+",
        help="One or more slab structures, or directories to audit every POSCAR/CONTCAR below",
    )
    vacuum.add_argument(
        "--axis",
        default="auto",
        help="Surface normal: a/b/c or auto (default: the axis with the most vacuum)",
    )
    vacuum.add_argument(
        "--min-vacuum",
        type=float,
        default=12.0,
        help="Audit threshold on the slab-to-image gap (default: 12 A)",
    )
    vacuum.add_argument(
        "--extend",
        nargs="?",
        type=float,
        const=18.0,
        metavar="VACUUM",
        help=(
            "Plan a cell stretch to this slab-to-image gap (default 18 A). Without "
            "--execute this only reports what would change"
        ),
    )
    vacuum.add_argument(
        "--execute",
        action="store_true",
        help="Actually apply --extend, overwriting each thin structure in place",
    )
    vacuum.add_argument(
        "-o", "--output", help="Single-file --extend: write here instead of overwriting"
    )
    vacuum.add_argument(
        "--no-recenter",
        action="store_true",
        help="With --extend, leave the slab where it is instead of re-centring it in the box",
    )
    vacuum.add_argument("--no-sort", action="store_true")
    vacuum.add_argument("--force", action="store_true")
    vacuum.add_argument("--summary")
    vacuum.set_defaults(func=cmd_geom)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (InterfaceForgeError, FileNotFoundError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
