"""Post-training bulk-property validation of trained MLIPs (``iface properties``).

The runner is engine- and chemistry-independent: it takes an ASE ``Atoms``,
a factory that builds one ASE calculator per committee member, and a
:class:`PropertyConfig`. Every property is computed independently for every
member (never on a committee-mean PES, since a nonlinear property of the mean
PES is not the mean of the members' properties); members are reported first
and summarized second. MatCalc does the property calculations through
:mod:`interfaceforge.matcalc_adapter`, the only module that imports it.

This is bulk validation: every property requires a fully 3D-periodic cell
without a vacuum gap. Interface energetics, adhesion and N/O ordering stay
with ``interface_energy``, ``adhesion`` and ``swap_mc``.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
import traceback
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from .errors import ConfigurationError, DependencyError, SafetyError
from .state import sha256_file, utc_now

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "interfaceforge_property_validation"
RESULT_FILE = "properties.json"
PROPERTIES = ("relax", "eos", "elasticity", "phonon")
# Every property here relaxes (or strains) the cell, so all of them need stress.
STRESS_PROPERTIES = frozenset(PROPERTIES)
MAX_BULK_GAP_A = 6.0
EOS_MAX_STRAIN_LIMIT = 0.2
# Pressure derivative of the bulk modulus for ordinary solids is ~3-6; outside this band the fit is suspect.
BPRIME_PLAUSIBLE = (2.0, 8.0)

_SCALARS = {
    "relax": ("energy_per_atom", "volume_per_atom", "a", "b", "c", "alpha", "beta", "gamma", "max_abs_final_stress"),
    "eos": (
        "equilibrium_energy",
        "equilibrium_energy_per_atom",
        "equilibrium_volume",
        "equilibrium_volume_per_atom",
        "bulk_modulus",
        "bulk_modulus_derivative",
        "r2",
    ),
    "elasticity": ("bulk_modulus_vrh", "shear_modulus_vrh", "youngs_modulus"),
    "phonon": ("min_frequency",),
}


@dataclass(frozen=True)
class PropertyConfig:
    properties: tuple[str, ...]
    fmax: float = 0.01
    max_steps: int = 500
    optimizer: str = "FIRE"
    phonon_min_length: float = 20.0
    # MatCalc EOSCalc defaults (linear strain, evenly spaced points), kept for parity with MatCalc benchmarks.
    eos_max_strain: float = 0.1
    eos_points: int = 11
    # Fractional volume window around the scan centre for the near-equilibrium Birch-Murnaghan refit.
    eos_refit_window: float = 0.05

    def __post_init__(self) -> None:
        if not self.properties:
            raise ConfigurationError(f"Request at least one --property ({', '.join(PROPERTIES)})")
        unknown = [name for name in self.properties if name not in PROPERTIES]
        if unknown:
            raise ConfigurationError(f"Unknown property {unknown[0]!r}; choose from {', '.join(PROPERTIES)}")
        if len(set(self.properties)) != len(self.properties):
            raise ConfigurationError("Each --property may be given once")
        if not (self.fmax > 0 and math.isfinite(self.fmax)):
            raise ConfigurationError("--fmax must be a positive number of eV/A")
        if self.max_steps < 1:
            raise ConfigurationError("--max-steps must be positive")
        if self.phonon_min_length <= 0:
            raise ConfigurationError("--phonon-min-length must be positive")
        if not (0 < self.eos_max_strain <= EOS_MAX_STRAIN_LIMIT):
            raise ConfigurationError(
                f"--eos-max-strain must be in (0, {EOS_MAX_STRAIN_LIMIT}] (linear strain; 0.1 = V/V0 0.73-1.33)"
            )
        if isinstance(self.eos_points, bool) or not isinstance(self.eos_points, int) or self.eos_points < 5 \
                or self.eos_points % 2 == 0:
            raise ConfigurationError("--eos-points must be an odd integer >= 5 (the scan is centred on V0)")
        if not (0 < self.eos_refit_window <= EOS_MAX_STRAIN_LIMIT):
            raise ConfigurationError(f"--eos-refit-window must be in (0, {EOS_MAX_STRAIN_LIMIT}] (volume fraction)")

    @property
    def ordered(self) -> tuple[str, ...]:
        return tuple(name for name in PROPERTIES if name in self.properties)


@dataclass
class Member:
    """One committee member as the runner sees it; ``make_calculator`` is called once."""

    label: str
    make_calculator: Callable[[], Any]
    record: dict[str, Any] = field(default_factory=dict)
    hashed_path: Path | None = None
    # Optional extra provenance gathered once the member is being run (e.g. native model dtype).
    inspect: Callable[[], dict[str, Any]] | None = None


# ------------------------------------------------------------------ JSON


def to_jsonable(value: Any) -> Any:
    """Plain-JSON view of numpy / pymatgen / path values; non-finite floats become ``None``."""

    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int)):
        return value
    if hasattr(value, "as_dict"):
        return to_jsonable(value.as_dict())
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def _dumps(payload: dict[str, Any]) -> str:
    return json.dumps(to_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"


# ------------------------------------------------------------- provenance


def package_versions(engine: str, *, phonon: bool) -> dict[str, str | None]:
    from .property_models import engine_packages

    names = ["interfaceforge", "matcalc", "ase", "pymatgen", "numpy", *engine_packages(engine)]
    if phonon:
        names.append("phonopy")
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def structure_record(atoms: Any, source: Path | None) -> dict[str, Any]:
    return {
        "source_path": str(source) if source is not None else None,
        "sha256": sha256_file(source) if source is not None else None,
        "formula": atoms.get_chemical_formula(mode="hill"),
        "n_atoms": len(atoms),
        "cell": np.asarray(atoms.cell.array, dtype=float),
        "pbc": [bool(flag) for flag in atoms.pbc],
        "volume": float(atoms.get_volume()) if all(atoms.pbc) else None,
    }


# ------------------------------------------------------------------ guards


def require_bulk(atoms: Any) -> dict[str, Any]:
    """Refuse anything that is not a fully 3D-periodic, gap-free bulk cell."""

    from .geometry import slab_vacuum

    if len(atoms) == 0:
        raise SafetyError("Structure has no atoms")
    if not all(bool(flag) for flag in atoms.pbc):
        raise SafetyError(
            f"Bulk-property validation needs a fully 3D-periodic structure; pbc={list(map(bool, atoms.pbc))}. "
            "EOS, elasticity and phonons are undefined for slabs and molecules."
        )
    if abs(float(np.linalg.det(np.asarray(atoms.cell.array)))) < 1e-8:
        raise SafetyError("Structure cell is degenerate (zero volume); it is not a bulk cell")
    gap = slab_vacuum(atoms)
    if gap["vacuum_a"] > MAX_BULK_GAP_A:
        raise SafetyError(
            f"Structure has a {gap['vacuum_a']:.2f} A empty gap along lattice vector {gap['axis']} "
            f"(> {MAX_BULK_GAP_A} A): this looks like a slab with vacuum, not bulk. "
            "A bulk modulus or elastic tensor of a vacuum-padded cell is meaningless."
        )
    return {"fully_periodic": True, "largest_gap_a": float(gap["vacuum_a"]), "gap_axis": gap["axis"],
            "max_allowed_gap_a": MAX_BULK_GAP_A}


def require_stress(atoms: Any, calculator: Any) -> None:
    """Fail unless the calculator really returns a finite 3x3/Voigt stress."""

    implemented = getattr(calculator, "implemented_properties", None)
    if implemented is not None and "stress" not in implemented:
        raise SafetyError(
            f"{type(calculator).__name__} does not implement stress; relaxation, EOS, elasticity and phonons "
            "here all need a stress-capable calculator"
        )
    probe = atoms.copy()
    probe.calc = calculator
    try:
        stress = np.asarray(probe.get_stress(), dtype=float)
    except Exception as exc:  # noqa: BLE001 - any failure means "no usable stress"
        raise SafetyError(f"{type(calculator).__name__} failed to return stress: {exc}") from exc
    if stress.size not in (6, 9) or not np.all(np.isfinite(stress)):
        raise SafetyError(f"{type(calculator).__name__} returned an unusable stress: {stress!r}")


# --------------------------------------------------------------- execution


def check_eos(result: dict[str, Any]) -> list[str]:
    """Flag an implausible B' in the full-scan and near-equilibrium fits; also tags ``result`` in place."""

    low, high = BPRIME_PLAUSIBLE
    messages = []
    fits = [("full scan", result)]
    refit = result.get("near_equilibrium_refit")
    if isinstance(refit, dict) and refit.get("status") == "ok":
        fits.append(("near-equilibrium refit", refit))
    for what, fit in fits:
        value = fit.get("bulk_modulus_derivative")
        plausible = value is not None and math.isfinite(value) and low <= value <= high
        fit["bulk_modulus_derivative_plausible"] = plausible
        if plausible:
            continue
        if value is None:
            messages.append(f"eos: {what} B' is missing")
            continue
        window = result.get("scan_window", {})
        span = (
            f" over V/V_ref {window['min_volume_ratio']:.3f}-{window['max_volume_ratio']:.3f}"
            if what == "full scan" and "min_volume_ratio" in window
            else ""
        )
        messages.append(
            f"eos: {what} B'={value:.3g} is outside [{low:g}, {high:g}]{span}; "
            "the Birch-Murnaghan fit (and its B) is unreliable - narrow --eos-max-strain"
        )
    result.setdefault("warnings", []).extend(messages)
    return messages


def _run_member(atoms: Any, member: Member, config: PropertyConfig, member_dir: Path, backend: Any) -> dict[str, Any]:
    record = dict(member.record)
    record.update({"label": member.label, "status": "failed", "error": None, "failed_property": None, "results": {},
                   "warnings": []})
    member_dir.mkdir(parents=True, exist_ok=True)
    stage = "load"
    try:
        calculator = member.make_calculator()
        record["calculator_class"] = f"{type(calculator).__module__}.{type(calculator).__name__}"
        if member.inspect is not None:
            record.update(member.inspect())
        stage = "stress_check"
        require_stress(atoms, calculator)
        stage = "relax"
        relaxed = backend.relax(
            atoms, calculator, fmax=config.fmax, max_steps=config.max_steps,
            optimizer=config.optimizer, member_dir=member_dir,
        )
        record["results"]["relax"] = relaxed["record"]
        if not relaxed["record"]["converged"]:
            info = relaxed["record"]
            raise SafetyError(
                f"relaxation did not converge (max|F|={info['max_force']:.4g} eV/A, fmax={config.fmax}, "
                f"steps={info['n_steps']}/{config.max_steps}, hit_max_steps={info['hit_max_steps']}); "
                "no EOS/elastic/phonon result is computed from an unrelaxed structure"
            )
        equilibrium = relaxed["atoms"]
        from ase.io import write

        write(str(member_dir / "relaxed.vasp"), equilibrium, format="vasp", direct=True)
        record["relaxed_structure"] = {"path": "relaxed.vasp", "sha256": sha256_file(member_dir / "relaxed.vasp")}
        for name in config.ordered:
            if name == "relax":
                continue
            stage = name
            if name == "eos":
                result = backend.eos(equilibrium, calculator, fmax=config.fmax, max_steps=config.max_steps,
                                     optimizer=config.optimizer, max_abs_strain=config.eos_max_strain,
                                     n_points=config.eos_points, refit_volume_window=config.eos_refit_window)
                record["warnings"].extend(check_eos(result))
            elif name == "elasticity":
                result = backend.elasticity(equilibrium, calculator, fmax=config.fmax)
            else:
                result = backend.phonon(equilibrium, calculator, min_length=config.phonon_min_length,
                                        member_dir=member_dir)
            record["results"][name] = result
        stage = "done"
        record["status"] = "ok"
    except DependencyError:
        raise
    except Exception as exc:  # noqa: BLE001 - one member's failure must not hide the others
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["failed_property"] = stage
        record["traceback_tail"] = traceback.format_exc().strip().splitlines()[-6:]
    return record


# ------------------------------------------------------------------ summary


def _stats(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else None,
        "min": float(array.min()),
        "max": float(array.max()),
        "n": int(len(array)),
    }


def _array_stats(arrays: list[Any], what: str) -> dict[str, Any]:
    shapes = {np.asarray(item).shape for item in arrays}
    if len(shapes) != 1:
        return {"aggregated": False, "reason": f"{what} shapes differ across members: {sorted(shapes)}"}
    stack = np.stack([np.asarray(item, dtype=float) for item in arrays])
    return {
        "aggregated": True,
        "mean": stack.mean(axis=0),
        "std": stack.std(axis=0, ddof=1) if len(stack) > 1 else None,
        "n": int(len(stack)),
    }


def summarize(models: Sequence[dict[str, Any]], properties: Sequence[str]) -> dict[str, Any]:
    ok = [model for model in models if model["status"] == "ok"]
    failed = [model for model in models if model["status"] != "ok"]
    summary: dict[str, Any] = {
        "n_members": len(models),
        "n_succeeded": len(ok),
        "n_failed": len(failed),
        "success_fraction": f"{len(ok)}/{len(models)}",
        "failed_members": [
            {"label": model["label"], "failed_property": model.get("failed_property"), "error": model.get("error")}
            for model in failed
        ],
        "aggregation": "statistics over independently computed member properties (never a committee-mean PES)",
        "std_convention": "sample standard deviation (ddof=1); null for a single member",
        "properties": {},
    }
    for name in properties:
        results = [model["results"][name] for model in ok if name in model["results"]]
        entry: dict[str, Any] = {"n_members": len(results)}
        if results:
            entry["units"] = results[0].get("units", {})
            entry["scalars"] = {key: _stats([item[key] for item in results]) for key in _SCALARS[name]}
        if name == "eos" and results:
            entry["implausible_bulk_modulus_derivative_members"] = sum(
                item.get("bulk_modulus_derivative_plausible") is False for item in results
            )
            refits = [item["near_equilibrium_refit"] for item in results
                      if item.get("near_equilibrium_refit", {}).get("status") == "ok"]
            entry["near_equilibrium_refit"] = {
                "n_members": len(refits),
                "scalars": {key: _stats([item[key] for item in refits]) for key in _SCALARS["eos"]} if refits else {},
                "implausible_bulk_modulus_derivative_members": sum(
                    item.get("bulk_modulus_derivative_plausible") is False for item in refits
                ),
            }
        if name == "elasticity" and results:
            conventions = {item["tensor_convention"] for item in results}
            entry["elastic_tensor"] = (
                _array_stats([item["elastic_tensor"] for item in results], "elastic tensor")
                if len(conventions) == 1
                else {"aggregated": False, "reason": f"tensor conventions differ: {sorted(conventions)}"}
            )
        if name == "phonon" and results:
            entry["dynamically_stable_members"] = sum(bool(item["dynamically_stable"]) for item in results)
            entry["frequencies"] = _array_stats([item["frequencies"] for item in results], "phonon frequency")
            temps = [np.asarray(item["thermal_properties"].get("temperatures", [])) for item in results]
            if all(t.shape == temps[0].shape and np.allclose(t, temps[0]) for t in temps):
                entry["thermal_properties"] = {
                    key: _array_stats([item["thermal_properties"][key] for item in results], key)
                    for key in results[0]["thermal_properties"]
                    if key != "temperatures"
                }
                entry["thermal_properties"]["temperatures"] = temps[0]
            else:
                entry["thermal_properties"] = {"aggregated": False, "reason": "temperature grids differ"}
            entry["note"] = "no scalar phonon score is generated; min_frequency is reported for stability only"
        summary["properties"][name] = entry
    return summary


# -------------------------------------------------------------- orchestration


def _prepare_output(output: Path, force: bool) -> None:
    if output.exists() and not output.is_dir():
        raise SafetyError(f"Output path exists and is not a directory: {output}")
    if output.is_dir() and any(output.iterdir()):
        if not force:
            raise SafetyError(f"Output directory is not empty: {output}. Use --force to replace a prior result.")
        if not (output / RESULT_FILE).is_file():
            raise SafetyError(
                f"Refusing --force on {output}: it holds no {RESULT_FILE}, so it is not a prior property result"
            )


def _publish(staging: Path, output: Path) -> None:
    if output.exists():
        retired = output.with_name(f".{output.name}.replaced-{utc_now().replace(':', '')}")
        output.rename(retired)
        staging.rename(output)
        shutil.rmtree(retired, ignore_errors=True)
    else:
        staging.rename(output)


def run_properties(
    atoms: Any,
    members: Sequence[Member],
    config: PropertyConfig,
    output: str | Path,
    *,
    engine: str,
    structure_source: str | Path | None = None,
    settings: dict[str, Any] | None = None,
    force: bool = False,
    backend: Any | None = None,
) -> dict[str, Any]:
    """Run every requested property for every member and write ``<output>/properties.json``."""

    if not members:
        raise ConfigurationError("No models given; pass --model (repeatable) or --committee")
    labels = [member.label for member in members]
    if len(set(labels)) != len(labels):
        raise ConfigurationError("Committee member labels must be unique")
    source = Path(structure_source).resolve() if structure_source is not None else None
    bulk = require_bulk(atoms)
    if backend is None:
        from . import matcalc_adapter

        matcalc_adapter.require_matcalc()
        backend = matcalc_adapter
    target = Path(output).expanduser().resolve()
    _prepare_output(target, force)
    target.parent.mkdir(parents=True, exist_ok=True)

    tracked = {"structure": source} if source is not None else {}
    tracked.update({member.label: member.hashed_path for member in members if member.hashed_path is not None})
    before = {key: sha256_file(path) for key, path in tracked.items()}
    source_atoms = atoms.copy()

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        models = [
            _run_member(atoms.copy(), member, config, staging / "members" / member.label, backend)
            for member in members
        ]
        after = {key: sha256_file(path) for key, path in tracked.items()}
        changed = sorted(key for key in tracked if before[key] != after[key])
        atoms_unchanged = (
            np.array_equal(source_atoms.positions, atoms.positions) and np.array_equal(source_atoms.cell, atoms.cell)
        )
        for model in models:
            model["member_dir"] = f"members/{model['label']}"
        payload = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "created_at": utc_now(),
            "structure": {**structure_record(atoms, source), "bulk_guard": bulk},
            "engine": engine,
            "properties_requested": list(config.properties),
            "settings": {**asdict(config), **(settings or {})},
            "packages": package_versions(engine, phonon="phonon" in config.properties),
            "models": models,
            "summary": summarize(models, config.ordered),
            "inputs_unchanged": not changed and atoms_unchanged,
            "changed_inputs": changed,
        }
        for model in models:
            model["package_versions"] = payload["packages"]
        (staging / RESULT_FILE).write_text(_dumps(payload), encoding="utf-8")
        _publish(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if changed or not atoms_unchanged:
        raise SafetyError(f"Inputs changed during the property run: {changed or ['in-memory structure']}")
    payload["output"] = str(target)
    return payload


# ------------------------------------------------------------------ CLI


def _read_structure(path: str) -> tuple[Any, Path]:
    try:
        from ase.io import read
    except ImportError as exc:
        raise DependencyError("Reading structures needs ASE: pip install -e '.[properties]'") from exc
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Structure not found: {source}")
    return read(str(source)), source


def cmd_run(args: argparse.Namespace) -> int:
    from . import property_models

    config = PropertyConfig(
        properties=tuple(args.property or ()),
        fmax=args.fmax,
        max_steps=args.max_steps,
        optimizer=args.optimizer,
        phonon_min_length=args.phonon_min_length,
        eos_max_strain=args.eos_max_strain,
        eos_points=args.eos_points,
        eos_refit_window=args.eos_refit_window,
    )
    if bool(args.model) == bool(args.committee):
        raise ConfigurationError("Pass either --model (repeatable) or --committee, not both / neither")
    dtype = property_models.resolve_dtype(args.engine, args.dtype)
    specs = (
        property_models.models_from_paths(args.model, args.engine)
        if args.model
        else property_models.models_from_committee(args.committee, args.engine)
    )
    atoms, source = _read_structure(args.structure)
    from . import matcalc_adapter

    matcalc_adapter.require_matcalc()
    note = property_models.device_note(args.engine)
    members = [
        Member(
            label=spec.label,
            make_calculator=lambda spec=spec: property_models.build_calculator(spec, device=args.device, dtype=dtype),
            hashed_path=spec.path,
            inspect=lambda spec=spec: property_models.precision_record(spec, dtype),
            record={
                "model_path": str(spec.path),
                "model_sha256": sha256_file(spec.path),
                "model_source": spec.source,
                "engine": spec.engine,
                "seed": spec.seed,
                "device": args.device,
                "device_note": note,
                "dtype": dtype,
                "dtype_note": None if dtype else "fixed by the frozen model",
            },
        )
        for spec in specs
    ]
    payload = run_properties(
        atoms,
        members,
        config,
        args.output,
        engine=args.engine,
        structure_source=source,
        settings={"device": args.device, "dtype": dtype, "committee": args.committee},
        force=args.force,
    )
    summary = payload["summary"]
    print(f"Wrote {Path(payload['output']) / RESULT_FILE}")
    print(f"members succeeded: {summary['success_fraction']}")
    for failure in summary["failed_members"]:
        print(f"  FAILED {failure['label']} at {failure['failed_property']}: {failure['error']}")
    for name, entry in summary["properties"].items():
        for key, stats in entry.get("scalars", {}).items():
            if key in ("bulk_modulus", "bulk_modulus_vrh", "shear_modulus_vrh", "equilibrium_volume_per_atom",
                       "min_frequency"):
                unit = entry["units"].get(key, "")
                spread = f" +/- {stats['std']:.4g}" if stats["std"] is not None else ""
                print(f"  {name}.{key}: {stats['mean']:.4g}{spread} {unit} (n={stats['n']})")
        refit = entry.get("near_equilibrium_refit", {}).get("scalars", {}).get("bulk_modulus")
        if refit:
            spread = f" +/- {refit['std']:.4g}" if refit["std"] is not None else ""
            print(f"  {name}.near_equilibrium_refit.bulk_modulus: {refit['mean']:.4g}{spread} GPa (n={refit['n']})")
    for model in payload["models"]:
        for message in model.get("warnings", []):
            print(f"  WARNING {model['label']}: {message}")
    return 0 if summary["n_failed"] == 0 else 1


def register_commands(commands: Any) -> None:
    run = commands.add_parser(
        "run",
        help="Relaxation / EOS / elasticity / phonons for each trained model via MatCalc (needs Python>=3.11)",
    )
    run.add_argument("structure", help="Fully 3D-periodic bulk structure readable by ASE (POSCAR, extxyz, cif, ...)")
    run.add_argument("--engine", required=True, choices=("mace", "deepmd"))
    run.add_argument("--model", action="append", default=[], help="Trained model file; repeat for committee members")
    run.add_argument("--committee", help="Collected committee bundle directory (iface committee collect)")
    run.add_argument("--property", action="append", default=[], help=f"Repeatable; one of {', '.join(PROPERTIES)}")
    run.add_argument("--output", required=True, help="Output directory (must be empty unless --force)")
    run.add_argument("--device", default="cpu", help="MACE torch device (default cpu)")
    run.add_argument("--dtype", choices=("float32", "float64"),
                     help="MACE precision (default float64); rejected for deepmd")
    run.add_argument("--fmax", type=float, default=0.01, help="Relaxation force/cell criterion, eV/A (default 0.01)")
    run.add_argument("--max-steps", type=int, default=500, help="Relaxation step cap (default 500)")
    run.add_argument("--optimizer", default="FIRE", choices=("FIRE", "BFGS", "LBFGS"))
    run.add_argument("--phonon-min-length", type=float, default=20.0,
                     help="Minimum phonon supercell edge, A (MatCalc default 20)")
    run.add_argument("--eos-max-strain", type=float, default=0.1,
                     help="EOS scan half-width as LINEAR strain, 0 < x <= 0.2 (MatCalc default 0.1 = V/V0 "
                          "0.73-1.33; ~0.05 is advisable for stiff covalent solids)")
    run.add_argument("--eos-points", type=int, default=11,
                     help="Number of EOS scan points, odd >= 5 (MatCalc default 11)")
    run.add_argument("--eos-refit-window", type=float, default=0.05,
                     help="Volume fraction around the relaxed volume for the near-equilibrium EOS refit "
                          "(default 0.05 = +/-5%%; needs >= 5 scan points inside it)")
    run.add_argument("--force", action="store_true", help="Replace a prior property result in --output")
    run.set_defaults(func=cmd_run)
