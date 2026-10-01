"""Read-only VASP source qualification; content-bound admission to collectors.

Source-quality findings require scientific review. Successful export or a VASP
completion marker alone never certifies k-point/thermostat/SCF correctness.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import gzip
import json
import math
import os
import re
from collections import Counter
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .errors import ConfigurationError, SafetyError
from .vasp import parse_incar
from .vasp_provenance import sha256_file

TRACKED = (
    "ENCUT",
    "PREC",
    "LREAL",
    "ISMEAR",
    "SIGMA",
    "IVDW",
    "ISPIN",
    "NELECT",
    "EDIFF",
    "NELM",
    "IBRION",
    "POTIM",
    "TEBEG",
    "TEEND",
    "MDALGO",
    "SMASS",
    "ISIF",
)
INPUTS = ("INCAR", "POSCAR", "CONTCAR", "KPOINTS", "POTCAR", "OSZICAR", "OUTCAR", "OUTCAR.gz")


def _number(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def issue(code: str, detail: str, *, hard: bool = False) -> dict[str, Any]:
    return {"code": code, "detail": detail, "hard": hard}


def parse_oszicar(
    path: Path, *, nelm: int = 60, residual_limit: float = 1e-3, jump_factor: float = 100.0
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Inspect every ionic frame. Missing rms(c) on the last step is not zero."""
    frames, issues, electronic = [], [], []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            if re.match(r"\s*(?:DAV|RMM|CG|DMP):", line):
                tokens = line.split()
                try:
                    values = [_number(token) for token in tokens[2:5]]
                    residual = _number(tokens[7]) if len(tokens) > 7 else None
                    if not all(math.isfinite(v) for v in values + ([] if residual is None else [residual])):
                        raise ValueError("nonfinite electronic values")
                    electronic.append((int(tokens[1]), residual))
                except (ValueError, IndexError):
                    issues.append(issue("MALFORMED_SCF", f"OSZICAR line {line_number}", hard=True))
                continue
            match = re.match(r"\s*(\d+)\s+(?:T=|F=)", line)
            if not match:
                continue
            fields = {}
            for key in ("T", "F", "E0"):
                item = re.search(rf"\b{key}=\s*(\S+)", line)
                if item:
                    try:
                        value = _number(item.group(1))
                        fields[key] = value if math.isfinite(value) else None
                    except ValueError:
                        fields[key] = None
            if fields.get("F") is None or fields.get("E0") is None or ("T" in fields and fields["T"] is None):
                issues.append(issue("INVALID_IONIC", f"OSZICAR step {match.group(1)}", hard=True))
            residuals = [value for _, value in electronic if value is not None]
            jumps = sum(
                b > max(a, 1e-15) * jump_factor and b > residual_limit
                for a, b in zip(residuals, residuals[1:], strict=False)
            )
            frames.append(
                {
                    "source_frame": len(frames),
                    "ionic_step": int(match.group(1)),
                    "scf_steps": len(electronic),
                    "last_iteration": electronic[-1][0] if electronic else 0,
                    "last_rms_c": residuals[-1] if residuals else None,
                    "residual_jumps": jumps,
                    "temperature_k": fields.get("T"),
                    "free_energy_ev": fields.get("F"),
                    "energy_e0_ev": fields.get("E0"),
                }
            )
            if not electronic:
                issues.append(issue("SCF_HISTORY_MISSING", f"ionic step {match.group(1)}"))
            electronic = []
    if electronic:
        issues.append(issue("UNFINISHED_SCF", "Electronic steps without ionic summary", hard=True))
    if not frames:
        issues.append(issue("NO_IONIC_SUMMARIES", "No completed OSZICAR frames", hard=True))
    for code, selected in (
        ("SCF_LIMIT", [f["source_frame"] for f in frames if f["last_iteration"] >= nelm]),
        (
            "SCF_RESIDUAL",
            [f["source_frame"] for f in frames if f["last_rms_c"] is not None and f["last_rms_c"] > residual_limit],
        ),
        ("SCF_RESIDUAL_JUMP", [f["source_frame"] for f in frames if f["residual_jumps"]]),
        (
            "IONIC_INDEX_RESET",
            [f["source_frame"] for i, f in enumerate(frames) if i and f["ionic_step"] <= frames[i - 1]["ionic_step"]],
        ),
    ):
        if selected:
            issues.append(issue(code, f"{len(selected)} frames; first zero-based indices {selected[:20]}"))
    return frames, issues


def scan_outcar(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"frames": 0, "completed": False, "executed": {}, "potcar_titles": []}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            if "POSITION" in line and "TOTAL-FORCE" in line:
                result["frames"] += 1
            if "General timing and accounting" in line:
                result["completed"] = True
            if index >= 20000:
                continue
            if "vasp_version" not in result:
                version = re.search(r"\bvasp\.\S+", line, re.I)
                if version:
                    result["vasp_version"] = version.group(0)
            for tag in TRACKED:
                match = re.search(rf"\b{tag}\s*=\s*([^\s;]+)", line)
                if match:
                    result["executed"][tag] = match.group(1)
            for tag in ("NKPTS", "NIONS"):
                match = re.search(rf"\b{tag}\s*=\s*(\d+)", line)
                if match:
                    result[tag.lower()] = int(match.group(1))
            if "TITEL" in line and "=" in line:
                title = line.split("=", 1)[1].strip()
                if title not in result["potcar_titles"]:
                    result["potcar_titles"].append(title)
    return result


def equivalent(a: str, b: str, tag: str = "") -> bool:
    boolean = {".TRUE.": "T", "TRUE": "T", ".FALSE.": "F", "FALSE": "F"}
    a, b = a.upper(), b.upper()
    if tag == "PREC":
        aliases = {"ACCURA": "ACCURATE", "MEDIUM": "MED", "NORM": "NORMAL"}
        a, b = aliases.get(a, a), aliases.get(b, b)
    if tag == "LREAL":
        aliases = {"AUTO": "A", "AUTOMATIC": "A"}
        a, b = aliases.get(a, a), aliases.get(b, b)
    if boolean.get(a, a) == boolean.get(b, b):
        return True
    try:
        return math.isclose(_number(a), _number(b), rel_tol=1e-7, abs_tol=1e-10)
    except ValueError:
        return False


def input_path(run: Path, root: Path, name: str) -> Path:
    current = run
    while True:
        candidate = current / name
        if candidate.is_file():
            return candidate
        if current == root or name not in {"INCAR", "KPOINTS", "POTCAR"}:
            return run / name
        current = current.parent


def source_snapshot(run: Path, root: Path) -> tuple[dict[str, Any], str]:
    files = {}
    for name in INPUTS:
        path = input_path(run, root, name)
        if path.is_file():
            files[name] = {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": sha256_file(path)}
    return files, sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def scan_labels(path: Path, output: Path, thresholds: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """ASE reads all frames. Geometry statistics never silently skip bad frames."""
    from ase.io import iread

    issues, count, symbols, previous_energy = [], 0, None, None
    try:
        with output.open("w", encoding="utf-8") as handle:
            for index, atoms in enumerate(iread(str(path), index=":")):
                count += 1
                current = atoms.get_chemical_symbols()
                if symbols is not None and current != symbols:
                    issues.append(issue("ATOM_IDENTITY_CHANGED", f"frame {index}", hard=True))
                symbols = current
                energy = float(atoms.get_potential_energy())
                forces = np.asarray(atoms.get_forces(apply_constraint=False))
                stress = np.asarray(atoms.get_stress(voigt=False))
                volume = float(atoms.get_volume())
                if (
                    not len(atoms)
                    or forces.shape != (len(atoms), 3)
                    or stress.shape != (3, 3)
                    or not all(np.isfinite(x).all() for x in (energy, forces, stress, atoms.cell, atoms.positions))
                    or volume <= 0
                ):
                    issues.append(issue("INVALID_LABEL_GEOMETRY", f"frame {index}", hard=True))
                    continue
                force_max = float(np.linalg.norm(forces, axis=1).max())
                stress_gpa = stress * 160.21766208
                if force_max > thresholds.get("force_max_ev_a", 20.0):
                    issues.append(issue("LARGE_FORCE", f"frame {index}: {force_max:.3g} eV/A"))
                if float(np.abs(stress_gpa).max()) > thresholds.get("stress_max_gpa", 30.0):
                    issues.append(issue("LARGE_STRESS", f"frame {index}: {np.abs(stress_gpa).max():.3g} GPa"))
                per_atom = energy / len(atoms)
                if previous_energy is not None and abs(per_atom - previous_energy) > thresholds.get(
                    "energy_jump_ev_atom", 0.1
                ):
                    issues.append(issue("ENERGY_JUMP", f"frame {index}: {per_atom - previous_energy:.3g} eV/atom"))
                previous_energy = per_atom
                # Sample contact statistics at a declared interval to bound cost on long MD.
                minimum = None
                if index % int(thresholds.get("contact_stride", 50)) == 0:
                    distances = atoms.get_all_distances(mic=True)
                    np.fill_diagonal(distances, np.inf)
                    minimum = float(distances.min()) if len(atoms) > 1 else None
                    if minimum is not None and minimum < thresholds.get("min_distance_a", 0.6):
                        issues.append(issue("SHORT_CONTACT", f"frame {index}: {minimum:.3g} A"))
                handle.write(
                    json.dumps(
                        {
                            "source_frame": index,
                            "natoms": len(atoms),
                            "energy_ev": energy,
                            "free_energy_ev": atoms.calc.results.get("free_energy"),
                            "force_max_ev_a": force_max,
                            "stress_gpa": stress_gpa.tolist(),
                            "volume_a3": volume,
                            "min_distance_a": minimum,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
    except Exception as exc:
        issues.append(issue("LABEL_PARSE_FAILED", f"{type(exc).__name__}: {exc}", hard=True))
    if not count:
        issues.append(issue("NO_LABELS", "ASE read no labelled frames", hard=True))
    return {"frames": count, "symbols": symbols}, issues


def matches(relative: str, pattern: str) -> bool:
    return fnmatch.fnmatch(relative, pattern) or any(fnmatch.fnmatch(part, pattern) for part in Path(relative).parts)


def expand(value: str, base: Path) -> Path:
    value = os.path.expandvars(os.path.expanduser(value))
    if "$" in value:
        raise ConfigurationError(f"Unresolved variable in {value}")
    return (base / value).resolve()


def audit_sources(config_path: str | Path, output: str | Path, *, labels: bool = True) -> dict[str, Any]:
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    policy_hash = sha256_file(config_path)
    config = yaml.safe_load(config_path.read_text())
    if (
        not isinstance(config, dict)
        or config.get("schema_version") != 1
        or not (config.get("roots") or config.get("mapped_config"))
    ):
        raise ConfigurationError("Source audit requires schema_version: 1 and nonempty roots")
    specs = list(config.get("roots", []))
    if config.get("mapped_config"):
        mapped_path = expand(config["mapped_config"], config_path.parent)
        mapped_hash = sha256_file(mapped_path)
        mapped = yaml.safe_load(mapped_path.read_text())
        specs = [
            {
                "id": f"mapped/{item['target']}",
                "path": str(expand(item["source"], mapped_path.parent)),
                "role": "train",
                "required": item.get("required", True) if item.get("enabled", True) else False,
                "quarantine_reason": item.get("reason") if item.get("enabled", True) is False else None,
            }
            for item in mapped["sources"]
        ] + specs
    roots = [(spec, expand(spec["path"], config_path.parent)) for spec in specs]
    if len({spec["id"] for spec, _ in roots}) != len(roots):
        raise ConfigurationError("Root IDs must be unique")
    if any(spec.get("role") not in {"train", "reference", "audit", "holdout"} for spec, _ in roots):
        raise ConfigurationError("Every root requires a train/reference/audit/holdout role")
    if any(output == root or root in output.parents for _, root in roots):
        raise SafetyError("Audit output must be outside source roots")
    if output.exists() and any(output.iterdir()):
        raise SafetyError(f"Use a new audit output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "frames").mkdir()
    rows, root_errors = [], []
    seen_runs: set[Path] = set()
    thresholds = config.get("thresholds", {})
    for spec, root in roots:
        if not root.is_dir():
            if spec.get("required", True):
                root_errors.append(f"Missing source root: {spec['id']} {root}")
            continue
        runs = {
            p.parent for name in ("INCAR", "OUTCAR", "OUTCAR.gz", "OSZICAR") for p in root.rglob(name) if p.is_file()
        }
        if not runs:
            root_errors.append(f"No calculation folders found: {spec['id']} {root}")
        for run in sorted(runs):
            if run.resolve() in seen_runs:
                continue
            seen_runs.add(run.resolve())
            relative = run.relative_to(root).as_posix()
            rule = next((r for r in config.get("quarantine", []) if matches(str(run), r["pattern"])), None)
            if spec.get("quarantine_reason"):
                rule = {"reason": spec["quarantine_reason"]}
            row: dict[str, Any] = {
                "source_id": f"{spec['id']}/{relative}",
                "root": str(root),
                "run": str(run),
                "role": spec["role"],
                "issues": [],
            }
            files, fingerprint = source_snapshot(run, root)
            row.update(files=files, fingerprint=fingerprint)
            try:
                _audit_run(row, output, labels, thresholds, spec)
                if source_snapshot(run, root)[1] != fingerprint:
                    row["issues"].append(issue("SOURCE_CHANGED_DURING_AUDIT", "Repeat after completion", hard=True))
            except Exception as exc:
                row["issues"].append(issue("SOURCE_PARSE_FAILED", f"{type(exc).__name__}: {exc}", hard=True))
            review = config.get("reviews", {}).get(row["source_id"], {})
            codes = {i["code"] for i in row["issues"]}
            row["status"] = "FAILED" if any(i["hard"] for i in row["issues"]) else "REVIEW"
            if (
                row["status"] != "FAILED"
                and review.get("fingerprint") == fingerprint
                and review.get("decision") == "accept"
                and review.get("reviewer")
                and review.get("evidence")
                and codes <= set(review.get("acknowledged_codes", []))
            ):
                row.update(status="ACCEPTED", review=review)
            if rule:
                row.update(status="QUARANTINED", reason=rule["reason"])
            rows.append(row)
    if sha256_file(config_path) != policy_hash or (
        config.get("mapped_config") and sha256_file(mapped_path) != mapped_hash
    ):
        raise SafetyError("Audit policy or source directory map changed during scanning; repeat audit")
    counts = dict(Counter(row["status"] for row in rows))
    report = {
        "schema_version": 1,
        "config": str(config_path),
        "policy_sha256": policy_hash,
        "mapped_config": ({"path": str(mapped_path), "sha256": mapped_hash} if config.get("mapped_config") else None),
        "labels_scanned": labels,
        "root_errors": root_errors,
        "counts": counts,
        "status": "READY"
        if not root_errors and rows and all(r["status"] in {"ACCEPTED", "QUARANTINED"} for r in rows)
        else "BLOCKED",
        "rows": rows,
    }
    (output / "source_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (output / "source_audit.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_id", "role", "status", "fingerprint", "issues", "reason"])
        for row in rows:
            writer.writerow(
                [
                    row["source_id"],
                    row["role"],
                    row["status"],
                    row["fingerprint"],
                    ";".join(sorted({i["code"] for i in row["issues"]})),
                    row.get("reason", ""),
                ]
            )
    (output / "SUMMARY.md").write_text(
        f"# VASP source qualification: {report['status']}\n\nCounts: {counts}\n\nRoot errors: {root_errors}\n\n"
        "No raw sources were changed. REVIEW is not permission to train. Inspect per-frame evidence, "
        "record fingerprint-bound scientific decisions in the YAML, and rerun into a fresh directory. "
        "ACCEPTED applies only to the recorded role.\n"
    )
    return report


def _audit_run(
    row: dict[str, Any], output: Path, labels: bool, thresholds: dict[str, Any], spec: dict[str, Any]
) -> None:
    run, root, files, issues = Path(row["run"]), Path(row["root"]), row["files"], row["issues"]
    incar = input_path(run, root, "INCAR")
    tags = parse_incar(incar) if incar.is_file() else {}
    row["incar"] = tags
    for name in ("INCAR", "POSCAR", "KPOINTS", "POTCAR", "OSZICAR"):
        if name not in files:
            issues.append(issue("MISSING_INPUT", name))
    outcar = run / ("OUTCAR" if (run / "OUTCAR").is_file() else "OUTCAR.gz")
    metadata = scan_outcar(outcar) if outcar.is_file() else {}
    row["outcar"] = metadata
    if not metadata:
        issues.append(issue("MISSING_OUTCAR", "No source label output", hard=True))
    else:
        if not metadata["completed"]:
            issues.append(issue("OUTCAR_NOT_FINISHED", "No VASP completion marker", hard=True))
        for tag in TRACKED:
            if (
                tag in tags
                and tag in metadata["executed"]
                and not equivalent(tags[tag], metadata["executed"][tag], tag)
            ):
                issues.append(issue("EXECUTED_INPUT_MISMATCH", tag, hard=True))
    effective = {**tags, **metadata.get("executed", {})}
    for tag, expected in spec.get("expected_incar", {}).items():
        if not equivalent(str(expected), effective.get(tag, ""), tag):
            issues.append(issue("LABEL_SETTINGS_MISMATCH", f"{tag}: expected {expected}, got {effective.get(tag)}"))
    ionic = []
    oszicar = run / "OSZICAR"
    if not oszicar.is_file():
        issues.append(issue("MISSING_SCF_HISTORY", "OSZICAR required for electronic-step qualification", hard=True))
    name = sha256(row["source_id"].encode()).hexdigest()[:16]
    if oszicar.is_file():
        ionic, electronic_issues = parse_oszicar(
            oszicar,
            nelm=int(float(effective.get("NELM", 60))),
            residual_limit=thresholds.get("scf_residual_limit", 1e-3),
            jump_factor=thresholds.get("scf_jump_factor", 100.0),
        )
        issues.extend(electronic_issues)
        temps = [f["temperature_k"] for f in ionic if f["temperature_k"] is not None]
        if temps:
            burn = min(int(thresholds.get("temperature_burn_in", 1000)), len(temps) // 3)
            settled = np.asarray(temps[burn:])
            target = float(effective.get("TEBEG", 0))
            end = float(effective.get("TEEND", target))
            row["temperature"] = {
                "target_k": target,
                "end_k": end,
                "burn_in_frames": burn,
                "mean_k": float(settled.mean()),
                "std_k": float(settled.std()),
                "p05_p50_p95_k": np.percentile(settled, [5, 50, 95]).tolist(),
            }
            if target == end and target > 0:
                if settled.std() / target > thresholds.get("temperature_std_ratio", 0.75) or abs(
                    np.median(settled) / target - 1
                ) > thresholds.get("temperature_median_fraction", 0.5):
                    issues.append(issue("TEMPERATURE_DISTRIBUTION", "Large fluctuations or median far from target"))
            else:
                issues.append(issue("TEMPERATURE_RAMP_REVIEW", "Nonconstant target needs protocol-specific review"))
        if metadata and len(ionic) != metadata["frames"]:
            issues.append(
                issue("FRAME_COUNT_MISMATCH", f"OSZICAR {len(ionic)}, OUTCAR {metadata['frames']}", hard=True)
            )
        (output / "frames" / f"{name}.scf.json").write_text(json.dumps(ionic, indent=2, allow_nan=False) + "\n")
    row["oszicar_frames"] = len(ionic)
    issues.append(issue("KPOINT_CONVERGENCE_UNVERIFIED", "Review paired converged-k E/F/stress evidence"))
    if labels and outcar.is_file():
        label_report, label_issues = scan_labels(outcar, output / "frames" / f"{name}.labels.jsonl", thresholds)
        row["labels"] = label_report
        issues.extend(label_issues)
        if metadata and label_report["frames"] != metadata["frames"]:
            issues.append(issue("LABEL_FRAME_COUNT", "Parser omitted or added source frames", hard=True))
    else:
        issues.append(issue("LABELS_NOT_SCANNED", "Logs-only cannot qualify training data", hard=True))


def require_source_admission(report_path: str | Path, outcars: Sequence[Path]) -> dict[str, Any]:
    """Check BEFORE staging, deleting old exports or writing any training data."""
    path = Path(report_path).resolve()
    report = json.loads(path.read_text())
    if report.get("schema_version") != 1 or not report.get("labels_scanned") or report.get("root_errors"):
        raise SafetyError("Source audit is incomplete or logs-only; collection blocked")
    policy = Path(report["config"])
    if not policy.is_file() or sha256_file(policy) != report["policy_sha256"]:
        raise SafetyError("Audit policy changed or is missing; repeat audit")
    mapped = report.get("mapped_config")
    if mapped and (not Path(mapped["path"]).is_file() or sha256_file(Path(mapped["path"])) != mapped["sha256"]):
        raise SafetyError("Source directory map changed; repeat audit")
    by_path = {}
    for row in report["rows"]:
        for name in ("OUTCAR", "OUTCAR.gz"):
            if name in row.get("files", {}):
                key = row["files"][name]["path"]
                if key in by_path:
                    raise SafetyError(f"Ambiguous audit record: {key}")
                by_path[key] = row
    admitted = []
    for outcar in outcars:
        row = by_path.get(str(outcar.resolve()))
        if not row or row["status"] != "ACCEPTED" or row["role"] != "train":
            raise SafetyError(f"Source not accepted for training: {outcar}")
        if source_snapshot(Path(row["run"]), Path(row["root"]))[1] != row["fingerprint"]:
            raise SafetyError(f"Source changed since audit: {outcar}")
        admitted.append(row["source_id"])
    return {"path": str(path), "sha256": sha256_file(path), "admitted_sources": admitted}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logs-only", action="store_true", help="Diagnostic only; never qualifies exports")
    args = parser.parse_args(argv)
    try:
        report = audit_sources(args.config, args.output, labels=not args.logs_only)
    except (OSError, ValueError, ConfigurationError, SafetyError) as exc:
        parser.error(str(exc))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return 0 if report["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
