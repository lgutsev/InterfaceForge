"""Controlled standard-vs-neural start benchmark (``iface vasp density-init-bench``).

``prepare`` copies *completed* calculations (no rerun of the original job) into
small ``<case>/standard`` and ``<case>/neural`` directories with identical
INCAR/POSCAR/POTCAR/KPOINTS apart from the start itself; ``compare`` reads both
finished arms and reports, per case and in aggregate:

* SCF acceleration (electronic steps of the seeded SCF) **separately from**
* end-to-end acceleration (VASP wall time + all initializer overhead:
  model loading, inference, CHGCAR writing, and any FFT-grid dry run).

It also checks that both arms reached the same electronic solution (energy,
forces, stress within tight tolerances) and, for magnetic cases, that the
signed INCAR pattern (e.g. NiO AFM-II) survived in both.  No performance claim
is made until every case has finished.
"""

from __future__ import annotations

import csv
import json
import math
import shlex
import shutil
import statistics
from pathlib import Path
from typing import Any

import yaml

from ..errors import SafetyError
from ..vasp import _write_poscar_without_velocities, parse_incar, update_incar
from .inputs import magnetic_settings, nonempty, poscar_species, reusable_grid, sha256_file
from .launch import (
    FALLBACK_NAME,
    default_interfaceforge_command,
    vasp_command_from_launcher,
    wrap_launcher_with_density_init,
)
from .outputs import magnetic_pattern, summarize_run
from .potcar import potcar_provenance
from .workflow import backend_option_args, hook_command, initialize_density, potcar_declaration_args

MANIFEST_NAME = "density_init_bench.json"
REPORT_JSON = "density_init_bench_report.json"
REPORT_MD = "density_init_bench_report.md"
REPORT_TSV = "density_init_bench_report.tsv"
MODES = ("static", "as-is")
ARMS = ("standard", "neural")
#: Inputs that must be byte-identical between the arms (INCAR differs only in the start).
SHARED_INPUTS = ("POSCAR", "POTCAR", "KPOINTS")
ORDERS = ("none", "afm-ii", "mixed-sign", "sign-uniform")
LAUNCHERS = ("runvasp.sh", "run.slurm")

# The same edits are applied to both arms; the only intended difference
# between them is the initial density (ICHARG=2 vs a promoted CHGCAR).
STATIC_TAGS = {"NSW": "0", "IBRION": "-1", "ISTART": "0", "LWAVE": ".FALSE.", "LCHARG": ".FALSE."}

DEFAULT_TOLERANCES = {
    "energy_ev_per_atom": 1e-4,
    "max_force_ev_per_a": 5e-3,
    "stress_kb": 0.5,
    "moment_mu_b": 0.05,
}


def load_pilot(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    cases = payload.get("cases") or []
    if not cases:
        raise SafetyError(f"{source}: no 'cases'")
    defaults = payload.get("defaults") or {}
    names = set()
    resolved = []
    for raw in cases:
        case = {**defaults, **raw}
        if not case.get("name") or not case.get("source"):
            raise SafetyError(f"{source}: every case needs 'name' and 'source'")
        if case["name"] in names:
            raise SafetyError(f"{source}: duplicate case name {case['name']!r}")
        names.add(case["name"])
        src = Path(str(case["source"])).expanduser()
        case["source"] = str((source.parent / src).resolve() if not src.is_absolute() else src)
        if case.get("potcar_definitions"):
            defs = Path(str(case["potcar_definitions"])).expanduser()
            case["potcar_definitions"] = str((source.parent / defs).resolve() if not defs.is_absolute() else defs)
        case.setdefault("structure", "CONTCAR")
        case.setdefault("mode", "static")
        case.setdefault("expect_magnetic_order", "none")
        case.setdefault("category", "")
        if case["mode"] not in MODES:
            raise SafetyError(f"case {case['name']}: mode must be one of {', '.join(MODES)}")
        if case["expect_magnetic_order"] not in ORDERS:
            raise SafetyError(f"case {case['name']}: expect_magnetic_order must be one of {', '.join(ORDERS)}")
        resolved.append(case)
    return {"path": str(source), "cases": resolved, "backend_options": payload.get("backend_options") or {}}


def _find_input(run: Path, name: str) -> Path | None:
    for directory in (run, *run.parents[:2]):
        path = directory / name
        if nonempty(path):
            return path
    return None


def _prepare_arm(
    case: dict[str, Any],
    arm: str,
    destination: Path,
    *,
    launcher_template: Path | None,
) -> dict[str, Any]:
    source = Path(case["source"])
    structure = source / case["structure"]
    if not nonempty(structure):
        structure = source / "POSCAR"
    destination.mkdir(parents=True, exist_ok=False)
    _write_poscar_without_velocities(structure, destination / "POSCAR")
    copied = {}
    for name in ("INCAR", "KPOINTS", "POTCAR"):
        path = source / name if name == "INCAR" else _find_input(source, name)
        if path is None or not nonempty(path):
            if name == "KPOINTS" and "KSPACING" in parse_incar(source / "INCAR"):
                continue
            raise SafetyError(f"case {case['name']}: no nonempty {name} for {source}")
        shutil.copy2(path, destination / name)
        copied[name] = str(path)
    launcher_source = launcher_template
    if launcher_source is None:
        for name in LAUNCHERS:
            found = _find_input(source, name)
            if found is not None:
                launcher_source = found
                break
    if launcher_source is None:
        raise SafetyError(f"case {case['name']}: no runvasp.sh/run.slurm (pass --launcher-template)")
    launcher_name = launcher_source.name if launcher_source.name in LAUNCHERS else "runvasp.sh"
    shutil.copy2(launcher_source, destination / launcher_name)
    (destination / launcher_name).chmod((destination / launcher_name).stat().st_mode | 0o111)

    edits: dict[str, str] = {}
    source_incar = parse_incar(destination / "INCAR")
    if case["mode"] == "static":
        edits.update(STATIC_TAGS)
        if int(float(source_incar.get("ISIF", "2"))) == 0:
            edits["ISIF"] = "2"  # stress is only computed for ISIF >= 1
    else:
        edits["ISTART"] = "0"
    if arm == "standard":
        edits["ICHARG"] = "2"
    delete = ("ICHARG",) if arm == "neural" else ()
    update_incar(destination / "INCAR", edits, delete=delete)
    return {
        "directory": str(destination),
        "structure_source": str(structure),
        "inputs_from": copied,
        "launcher": launcher_name,
        "launcher_source": str(launcher_source),
        "incar_edits": edits,
        "incar_deleted": list(delete),
    }


def prepare_benchmark(
    pilot: str | Path,
    output_root: str | Path,
    *,
    launcher_template: str | Path | None = None,
    neural_init: str = "launch",
    interfaceforge_command: str | None = None,
    backend_options: dict[str, Any] | None = None,
    potcar_definitions: str | Path | None = None,
    potcar_generator: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Create ``<output_root>/<case>/{standard,neural}`` benchmark arms.

    Both arms get the *same* POTCAR (copied once from the source and hash
    checked).  Its provenance is recorded per case; ``potcar_definitions``
    (or a case's own ``potcar_definitions`` key) declares the mapping it was
    generated with and must agree with the POTCAR actually copied.
    """

    if neural_init not in {"launch", "now"}:
        raise SafetyError("--neural-init must be 'launch' or 'now'")
    config = load_pilot(pilot)
    root = Path(output_root).expanduser().resolve()
    template = Path(launcher_template).expanduser().resolve() if launcher_template else None
    options = {**config["backend_options"], **(backend_options or {})}
    plan = []
    for case in config["cases"]:
        source = Path(case["source"])
        if not (source.is_dir() and nonempty(source / "INCAR")):
            raise SafetyError(f"case {case['name']}: {source} has no INCAR")
        incar = parse_incar(source / "INCAR")
        structure = source / case["structure"]
        if not nonempty(structure):
            structure = source / "POSCAR"
        species = poscar_species(structure)
        magnetism = magnetic_settings(incar, len(species))
        expected = case["expect_magnetic_order"]
        if expected == "none" and magnetism["ispin"] == 2 and magnetism["order"] not in {"zero"}:
            raise SafetyError(
                f"case {case['name']}: INCAR is spin-polarized ({magnetism['order']}); declare "
                "expect_magnetic_order (afm-ii / mixed-sign / sign-uniform)"
            )
        if expected in {"afm-ii", "mixed-sign"} and not magnetism["mixed_sign"]:
            raise SafetyError(
                f"case {case['name']}: expect_magnetic_order={expected} but the INCAR MAGMOM is "
                f"{magnetism['order']}; supply the signed AFM-II MAGMOM"
            )
        if expected == "sign-uniform" and magnetism["order"] != "sign-uniform":
            raise SafetyError(f"case {case['name']}: MAGMOM is {magnetism['order']}, not sign-uniform")
        plan.append({"case": case, "magnetism": magnetism, "n_ions": len(species)})
    if dry_run:
        return {
            "mode": "dry-run",
            "output_root": str(root),
            "pilot": config["path"],
            "cases": [
                {
                    "name": item["case"]["name"],
                    "source": item["case"]["source"],
                    "category": item["case"]["category"],
                    "mode": item["case"]["mode"],
                    "magnetic_order": item["magnetism"]["order"],
                    "n_ions": item["n_ions"],
                }
                for item in plan
            ],
        }
    if (root / MANIFEST_NAME).exists():
        raise SafetyError(f"Refusing to overwrite existing benchmark {root / MANIFEST_NAME}")
    root.mkdir(parents=True, exist_ok=True)
    interfaceforge = interfaceforge_command or default_interfaceforge_command()
    rows = []
    try:
        for item in plan:
            case = item["case"]
            case_root = root / case["name"]
            if case_root.exists():
                raise SafetyError(f"Refusing to overwrite existing benchmark case {case_root}")
            arms = {
                arm: _prepare_arm(case, arm, case_root / arm, launcher_template=template) for arm in ARMS
            }
            standard_dir = case_root / "standard"
            neural_dir = case_root / "neural"
            shared = _shared_input_hashes(standard_dir, neural_dir)
            mismatched = [name for name, same in shared["identical"].items() if not same]
            if mismatched:
                raise SafetyError(f"case {case['name']}: arms differ in {', '.join(mismatched)}")
            case_defs = case.get("potcar_definitions") or potcar_definitions
            case_generator = case.get("potcar_generator") or potcar_generator
            potcar = potcar_provenance(
                standard_dir / "POTCAR",
                poscar_species(standard_dir / "POSCAR"),
                definitions=case_defs,
                generator=case_generator,
            )
            initialize_density(
                standard_dir, backend="standard", potcar_definitions=case_defs, potcar_generator=case_generator
            )

            grid, grid_source = reusable_grid(neural_dir, Path(case["source"]))
            if case.get("grid"):
                grid, grid_source = tuple(int(x) for x in case["grid"]), "pilot 'grid'"
            extra = backend_option_args(options) + potcar_declaration_args(case_defs, case_generator)
            launcher_name = arms["neural"]["launcher"]
            launcher_path = neural_dir / launcher_name
            launcher_text = launcher_path.read_text(encoding="utf-8", errors="ignore")
            if grid is not None:
                extra += ["--grid", *(str(v) for v in grid)]
            else:
                vasp_line = vasp_command_from_launcher(launcher_text)
                if vasp_line is None:
                    raise SafetyError(
                        f"case {case['name']}: no reusable FFT grid ({grid_source}) and no VASP line in "
                        f"{launcher_name} for a grid dry run; set 'grid: [NGXF, NGYF, NGZF]' in the pilot"
                    )
                extra += ["--grid-dry-run-command", vasp_line]
                grid_source = f"launch-time NELM=1 dry run ({grid_source})"
            magmom_source = "incar"
            initialization: dict[str, Any] = {"mode": neural_init, "grid_source": grid_source}
            if neural_init == "now":
                if grid is None:
                    raise SafetyError(
                        f"case {case['name']}: --neural-init now needs a known grid ({grid_source}); "
                        "set 'grid' in the pilot or use --neural-init launch"
                    )
                report = initialize_density(
                    neural_dir,
                    backend="neural-paw",
                    magmom_source=magmom_source,
                    grid=grid,
                    backend_options={k: v for k, v in options.items() if v is not None},
                    potcar_definitions=case_defs,
                    potcar_generator=case_generator,
                )
                initialization["status"] = report["status"]
            else:
                hook = hook_command(
                    interfaceforge=interfaceforge,
                    magmom_source=magmom_source,
                    extra=extra,
                )
                launcher_path.write_text(
                    wrap_launcher_with_density_init(
                        launcher_text, launcher_name=launcher_name, hook=hook, on_failure="abort"
                    ),
                    encoding="utf-8",
                )
                initialization["hook"] = hook
            rows.append(
                {
                    "name": case["name"],
                    "category": case["category"],
                    "source": case["source"],
                    "mode": case["mode"],
                    "expect_magnetic_order": case["expect_magnetic_order"],
                    "magnetic_order": item["magnetism"]["order"],
                    "incar_magmom": item["magnetism"]["incar_magmom"],
                    "n_ions": item["n_ions"],
                    "arms": arms,
                    "shared_inputs_sha256": shared["sha256"],
                    "potcar": potcar,
                    "neural_initialization": initialization,
                }
            )
    except Exception:
        for row in rows:
            shutil.rmtree(root / row["name"], ignore_errors=True)
        raise
    manifest = {
        "format": "interfaceforge-density-init-bench",
        "schema_version": 1,
        "pilot": config["path"],
        "output_root": str(root),
        "backend_options": options,
        "static_tags": STATIC_TAGS,
        "note": (
            "Both arms share INCAR/POSCAR/POTCAR/KPOINTS except the start (standard: ICHARG=2; "
            "neural: promoted CHGCAR + ICHARG=1). Submit both arms on identical hardware/partition."
        ),
        "cases": rows,
    }
    (root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "mode": "prepared",
        "output_root": str(root),
        "manifest": str(root / MANIFEST_NAME),
        "cases": len(rows),
        "submit": [
            f"cd {shlex.quote(arm['directory'])} && sbatch {arm['launcher']}"
            for row in rows
            for arm in row["arms"].values()
        ],
    }


def _shared_input_hashes(standard: Path, neural: Path) -> dict[str, Any]:
    """SHA-256 of each shared input in both arms and whether they are identical."""

    hashes: dict[str, dict[str, str | None]] = {}
    identical: dict[str, bool] = {}
    for name in SHARED_INPUTS:
        pair = {
            arm: sha256_file(directory / name) if nonempty(directory / name) else None
            for arm, directory in (("standard", standard), ("neural", neural))
        }
        if name == "KPOINTS" and pair["standard"] is None and pair["neural"] is None:
            continue  # KSPACING runs have no KPOINTS in either arm
        hashes[name] = pair
        identical[name] = pair["standard"] is not None and pair["standard"] == pair["neural"]
    return {"sha256": hashes, "identical": identical}


# ----------------------------------------------------------------------------
# compare
# ----------------------------------------------------------------------------


def _diff(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else b - a


def _max_force_diff(a: list[list[float]] | None, b: list[list[float]] | None) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    return max(math.dist(fa, fb) for fa, fb in zip(a, b, strict=True))


def _max_abs_diff(a: list[float] | None, b: list[float] | None) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    return max(abs(x - y) for x, y in zip(a, b, strict=True))


def _case_metrics(row: dict[str, Any], tolerances: dict[str, float]) -> dict[str, Any]:
    arms = {arm: summarize_run(Path(row["arms"][arm]["directory"])) for arm in ARMS}
    std, neu = arms["standard"], arms["neural"]
    n_ions = row["n_ions"]
    reference = std["incar_magmom"] if std["ispin"] == 2 else None
    patterns = {arm: magnetic_pattern(arms[arm]["local_moments"], reference) for arm in ARMS}
    states = {arm: arms[arm]["electronic"]["state"] for arm in ARMS}
    init = neu["density_init"]
    neural_dir = Path(row["arms"]["neural"]["directory"])
    fallback = nonempty(neural_dir / FALLBACK_NAME)
    init_ok = init.get("status") == "PROMOTED" and bool(init.get("active")) and not fallback
    shared = _shared_input_hashes(Path(row["arms"]["standard"]["directory"]), neural_dir)
    same_inputs = all(shared["identical"].values())
    recorded = row.get("shared_inputs_sha256")
    if recorded is not None and recorded != shared["sha256"]:
        same_inputs = False  # an arm's input was replaced after prepare (e.g. a regenerated POTCAR)
    complete = all(state in {"CONVERGED", "UNCONVERGED"} for state in states.values())
    if states["neural"] == "NOT_RUN" and init.get("status") == "FAILED":
        complete = True  # the initializer failure itself is the neural arm's outcome
    overhead = float(init.get("overhead_s") or 0.0) if init.get("status") else 0.0
    wall = {
        "standard": std["vasp_wall_s"],
        "neural": neu["vasp_wall_s"],
    }
    total = {
        "standard": wall["standard"],
        "neural": None if wall["neural"] is None else wall["neural"] + overhead,
    }
    scf = {arm: arms[arm]["electronic"]["scf_steps_first_ionic"] for arm in ARMS}
    scf_total = {arm: arms[arm]["electronic"]["scf_steps_total"] for arm in ARMS}
    energy_diff = _diff(std["energy_sigma0_ev"], neu["energy_sigma0_ev"])
    force_diff = _max_force_diff(std["forces"], neu["forces"])
    stress_diff = _max_abs_diff(std["stress_kb"], neu["stress_kb"])
    moment_diff = _max_abs_diff(std["local_moments"], neu["local_moments"])

    checks: dict[str, bool | None] = {
        "same_inputs": same_inputs,
        "both_converged": None if not complete else states["standard"] == states["neural"] == "CONVERGED",
        "neural_init_active": init_ok if complete else None,
        "energy": None if energy_diff is None else abs(energy_diff) / n_ions <= tolerances["energy_ev_per_atom"],
        "forces": None if force_diff is None else force_diff <= tolerances["max_force_ev_per_a"],
        "stress": None if stress_diff is None else stress_diff <= tolerances["stress_kb"],
    }
    if reference is not None:
        checks["moments"] = None if moment_diff is None else moment_diff <= tolerances["moment_mu_b"]
        checks["magnetic_pattern_neural"] = (
            None
            if patterns["neural"]["status"] == "UNKNOWN"
            else patterns["neural"]["status"] in {"PRESERVED", "PRESERVED_GLOBAL_FLIP"}
        )
        checks["magnetic_pattern_standard"] = (
            None
            if patterns["standard"]["status"] == "UNKNOWN"
            else patterns["standard"]["status"] in {"PRESERVED", "PRESERVED_GLOBAL_FLIP"}
        )
    if not same_inputs:
        verdict = "INPUT_MISMATCH"
    elif not complete:
        verdict = "INCOMPLETE"
    elif states["neural"] != "CONVERGED" or not init_ok:
        verdict = "NEURAL_FAILED" if states["standard"] == "CONVERGED" else "BOTH_FAILED"
    elif states["standard"] != "CONVERGED":
        verdict = "STANDARD_FAILED"
    elif all(value is not False for value in checks.values()) and None not in (
        checks["energy"],
        checks["forces"],
    ):
        verdict = "SAME_SOLUTION"
    else:
        verdict = "DIFFERENT_SOLUTION"

    def ratio(a: float | None, b: float | None) -> float | None:
        return None if not a or not b else a / b

    return {
        "name": row["name"],
        "category": row["category"],
        "expect_magnetic_order": row["expect_magnetic_order"],
        "n_ions": n_ions,
        "verdict": verdict,
        "electronic_state": states,
        "density_init": init,
        "neural_fallback_occurred": fallback,
        "shared_inputs": shared,
        "scf_iterations": {**scf, "difference": _diff(scf["standard"], scf["neural"])},
        "scf_iterations_all_ionic": scf_total,
        "vasp_wall_s": {**wall, "difference": _diff(wall["standard"], wall["neural"])},
        "inference_overhead_s": {"standard": 0.0, "neural": overhead if init.get("status") else None},
        "total_wall_s": {**total, "difference": _diff(total["standard"], total["neural"])},
        "scf_speedup": ratio(scf["standard"], scf["neural"]),
        "end_to_end_speedup": ratio(total["standard"], total["neural"]),
        "final_energy_ev": {
            "standard": std["energy_sigma0_ev"],
            "neural": neu["energy_sigma0_ev"],
            "difference": energy_diff,
            "difference_per_atom": None if energy_diff is None else energy_diff / n_ions,
        },
        "max_force_difference_ev_per_a": force_diff,
        "max_stress_difference_kb": stress_diff,
        "max_local_moment_difference_mu_b": moment_diff,
        "final_local_moments": {arm: arms[arm]["local_moments"] for arm in ARMS},
        "magnetic_pattern": patterns,
        "checks": checks,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if value != 0 and (abs(value) < 10 ** (-digits) or abs(value) >= 1e6):
            return f"{value:.2e}"
        return f"{value:.{digits}f}"
    return str(value)


def _magnetic_cell(metrics: dict[str, Any], arm: str) -> str:
    pattern = metrics["magnetic_pattern"][arm]
    if pattern["status"] == "NOT_APPLICABLE":
        return "non-magnetic"
    rng = pattern.get("final_abs_moment_range")
    suffix = f" (moments {rng[0]}–{rng[1]} μB)" if rng else ""
    return f"{pattern['status']}{suffix}"


def render_case_table(metrics: dict[str, Any]) -> str:
    """The per-case Markdown table requested for the pilot report."""

    rows = [
        ("SCF iterations", *(metrics["scf_iterations"][k] for k in ("standard", "neural", "difference"))),
        ("VASP wall time (s)", *(metrics["vasp_wall_s"][k] for k in ("standard", "neural", "difference"))),
        ("Inference time (s)", 0, metrics["inference_overhead_s"]["neural"], metrics["inference_overhead_s"]["neural"]),
        ("Total wall time (s)", *(metrics["total_wall_s"][k] for k in ("standard", "neural", "difference"))),
        (
            "Final energy (eV)",
            metrics["final_energy_ev"]["standard"],
            metrics["final_energy_ev"]["neural"],
            metrics["final_energy_ev"]["difference"],
        ),
        ("Max force difference (eV/Å)", "ref", "", metrics["max_force_difference_ev_per_a"]),
        ("Stress difference (kB, max comp.)", "ref", "", metrics["max_stress_difference_kb"]),
        (
            "Magnetic state",
            _magnetic_cell(metrics, "standard"),
            _magnetic_cell(metrics, "neural"),
            "max Δm " + _fmt(metrics["max_local_moment_difference_mu_b"]) + " μB"
            if metrics["max_local_moment_difference_mu_b"] is not None
            else "—",
        ),
        ("Electronic convergence", metrics["electronic_state"]["standard"], metrics["electronic_state"]["neural"], ""),
    ]
    energy_digits = {"Final energy (eV)": 6}
    lines = [
        f"### {metrics['name']} — {metrics['verdict']}",
        "",
        f"Category: {metrics['category'] or '—'} · ions: {metrics['n_ions']} · "
        f"SCF speedup: {_fmt(metrics['scf_speedup'], 2)}× · end-to-end speedup: "
        f"{_fmt(metrics['end_to_end_speedup'], 2)}×",
        "",
        "| Metric | Standard start | Neural start | Difference |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, *values in rows:
        digits = energy_digits.get(label, 3)
        lines.append(f"| {label} | " + " | ".join(_fmt(v, digits) for v in values) + " |")
    return "\n".join(lines) + "\n"


def _aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    # INPUT_MISMATCH arms are not a valid pair: they block evaluation like unfinished cases.
    finished = [case for case in cases if case["verdict"] not in {"INCOMPLETE", "INPUT_MISMATCH"}]
    compared = [case for case in finished if case["verdict"] in {"SAME_SOLUTION", "DIFFERENT_SOLUTION"}]
    std_fail = sum(1 for case in finished if case["electronic_state"]["standard"] != "CONVERGED")
    neu_fail = sum(1 for case in finished if case["verdict"] in {"NEURAL_FAILED", "BOTH_FAILED"})
    scf_reductions = [
        case["scf_iterations"]["standard"] - case["scf_iterations"]["neural"]
        for case in compared
        if case["scf_iterations"]["difference"] is not None
    ]
    std_total = sum(case["total_wall_s"]["standard"] or 0.0 for case in compared)
    neu_total = sum(case["total_wall_s"]["neural"] or 0.0 for case in compared)
    magnetic = [case for case in finished if case["expect_magnetic_order"] in {"afm-ii", "mixed-sign"}]
    all_done = len(finished) == len(cases) and bool(cases)

    def criterion(value: bool) -> bool | str:
        return value if all_done else "not evaluated (benchmark incomplete)"

    return {
        "cases": len(cases),
        "finished": len(finished),
        "input_mismatch": sum(1 for case in cases if case["verdict"] == "INPUT_MISMATCH"),
        "compared": len(compared),
        "same_solution": sum(1 for case in compared if case["verdict"] == "SAME_SOLUTION"),
        "failures": {"standard": std_fail, "neural": neu_fail},
        "scf_iterations_saved": {
            "per_case": scf_reductions,
            "median": statistics.median(scf_reductions) if scf_reductions else None,
            "cases_reduced": sum(1 for value in scf_reductions if value > 0),
        },
        "total_wall_s": {
            "standard": std_total if compared else None,
            "neural_including_overhead": neu_total if compared else None,
            "saved": (std_total - neu_total) if compared else None,
            "end_to_end_speedup": (std_total / neu_total) if compared and neu_total else None,
        },
        "acceptance": {
            "1_same_electronic_solution": criterion(
                bool(compared) and all(case["checks"]["energy"] for case in compared)
            ),
            "2_forces_and_stress_unchanged": criterion(
                bool(compared)
                and all(case["checks"]["forces"] and case["checks"]["stress"] is not False for case in compared)
            ),
            "3_no_increase_in_failure_rate": criterion(neu_fail <= std_fail),
            "4_afm_ii_preserved": criterion(
                all(case["checks"].get("magnetic_pattern_neural") is True for case in magnetic)
            )
            if magnetic
            else "not applicable (no AFM case)",
            "5_scf_reduced_in_some_cases": criterion(any(value > 0 for value in scf_reductions)),
            "6_end_to_end_wall_time_saved": criterion(bool(compared) and neu_total < std_total),
        },
        "claim_policy": (
            "SCF acceleration and end-to-end acceleration are reported separately; inference, model "
            "loading, CHGCAR writing and any grid dry run count against the neural arm. No criterion "
            "is evaluated until every case has finished with identical POSCAR/POTCAR/KPOINTS in both arms."
        ),
    }


def compare_benchmark(
    bench_root: str | Path, *, tolerances: dict[str, float] | None = None, write: bool = True
) -> dict[str, Any]:
    root = Path(bench_root).expanduser().resolve()
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tol = {**DEFAULT_TOLERANCES, **(tolerances or {})}
    cases = [_case_metrics(row, tol) for row in manifest["cases"]]
    payload = {
        "format": "interfaceforge-density-init-bench-report",
        "schema_version": 1,
        "output_root": str(root),
        "tolerances": tol,
        "summary": _aggregate(cases),
        "cases": cases,
    }
    if write:
        (root / REPORT_JSON).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary = payload["summary"]
        lines = [
            "# Neural density initialization benchmark",
            "",
            f"Finished {summary['finished']}/{summary['cases']} cases; same solution in "
            f"{summary['same_solution']}/{summary['compared']} compared; failures standard "
            f"{summary['failures']['standard']}, neural {summary['failures']['neural']}.",
            "",
            "## Acceptance criteria",
            "",
            *(f"- **{key}**: {value}" for key, value in summary["acceptance"].items()),
            "",
            f"> {summary['claim_policy']}",
            "",
            "## Cases",
            "",
            *(render_case_table(case) for case in cases),
        ]
        (root / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")
        fields = (
            "name",
            "category",
            "verdict",
            "scf_standard",
            "scf_neural",
            "vasp_wall_standard_s",
            "vasp_wall_neural_s",
            "inference_overhead_s",
            "total_wall_standard_s",
            "total_wall_neural_s",
            "energy_diff_ev",
            "max_force_diff",
            "max_stress_diff_kb",
            "magnetic_neural",
        )
        with (root / REPORT_TSV).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            for case in cases:
                writer.writerow(
                    {
                        "name": case["name"],
                        "category": case["category"],
                        "verdict": case["verdict"],
                        "scf_standard": case["scf_iterations"]["standard"],
                        "scf_neural": case["scf_iterations"]["neural"],
                        "vasp_wall_standard_s": case["vasp_wall_s"]["standard"],
                        "vasp_wall_neural_s": case["vasp_wall_s"]["neural"],
                        "inference_overhead_s": case["inference_overhead_s"]["neural"],
                        "total_wall_standard_s": case["total_wall_s"]["standard"],
                        "total_wall_neural_s": case["total_wall_s"]["neural"],
                        "energy_diff_ev": case["final_energy_ev"]["difference"],
                        "max_force_diff": case["max_force_difference_ev_per_a"],
                        "max_stress_diff_kb": case["max_stress_difference_kb"],
                        "magnetic_neural": case["magnetic_pattern"]["neural"]["status"],
                    }
                )
        payload["reports"] = [str(root / name) for name in (REPORT_JSON, REPORT_MD, REPORT_TSV)]
    return payload
