"""Temperature-consistency audit for a bulk-referenced interface excess.

A balanced interface excess

    gamma = [ sum_i c_i E_i  -  sum_j n_j e_j  -  dN mu ] / D

subtracts bulk reference energies ``e_j`` (per formula unit) from interface
frame energies ``E_i``. When the references are thermal means from MD at
``T_ref``, every frame must come from MD at the same temperature: in the
classical (equipartition) limit each atom carries (3/2) k_B T of potential
energy, so a frame at ``T_i`` biases gamma by

    d_gamma = sum_i c_i * n_atoms_i * (3/2) k_B (T_i - T_ref) / D

(converted from eV/A^2 to J/m^2). The audit performs three checks:

C1  equipartition: for every thermal-mean bulk with a matching static
    reference, (e_thermal - e_static) / ((3/2) k_B T * atoms_per_fu) must be
    within ``tol_ratio`` of 1. The default 0.10 leaves room for the
    anharmonicity of real solids at a few hundred K (measured excesses of a few
    per cent), while still catching a reference taken at the wrong
    temperature, a static energy mislabelled as a mean, or a per-atom/per-f.u.
    mix-up (all of which give ratios far from 1).
C2  temperature match: one common reference temperature (bulk refs that
    disagree are an input error, exit 2), the per-frame dT and the bias above.
C3  spread: mean, standard deviation and standard error of per-frame gamma
    per family, when at least two values exist.

Verdict: CONSISTENT when every |dT| <= tol_T and C1 passes; INCONSISTENT when
|d_gamma| > tol_gamma or C1 fails; otherwise UNDETERMINED with the reason (a
null frame temperature, a missing static reference, ...).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .errors import ConfigurationError, SafetyError
from .interface_mu import K_B_EV_PER_K
from .schema_check import schema_errors
from .state import StateStore, sha256_file, utc_now
from .validation import EV_A2_TO_J_M2

STAGE = "excess_temperature"
INPUT_SCHEMA = "excess_temperature_input.schema.json"
DEFAULT_TOL_RATIO = 0.10
DEFAULT_TOL_T_K = 1.0
DEFAULT_TOL_GAMMA_J_M2 = 0.01
CONSISTENT, INCONSISTENT, UNDETERMINED = "CONSISTENT", "INCONSISTENT", "UNDETERMINED"
_RANK = {CONSISTENT: 0, UNDETERMINED: 1, INCONSISTENT: 2}
OUTPUT_NAMES = (
    f"{STAGE}_manifest.json",
    f"{STAGE}_audit.json",
    f"{STAGE}_audit.tsv",
    f"{STAGE}_audit.md",
)


def equipartition_check(
    e_thermal: float, e_static: float, temperature_k: float, atoms_per_fu: float, *, tol_ratio: float
) -> dict[str, Any]:
    """C1 for one bulk: thermal-minus-static energy against (3/2) k_B T per atom."""

    delta = e_thermal - e_static
    expected = 1.5 * K_B_EV_PER_K * temperature_k * atoms_per_fu
    if expected == 0.0:
        return {"delta_eV_fu": delta, "expected_eV_fu": 0.0, "ratio": None, "status": "NOT_APPLICABLE"}
    ratio = delta / expected
    return {
        "delta_eV_fu": delta,
        "expected_eV_fu": expected,
        "ratio": ratio,
        "status": "PASS" if abs(ratio - 1.0) <= tol_ratio else "FAIL",
    }


def frame_bias_J_m2(coeff: float, n_atoms: int, delta_t_k: float, denominator_a2: float) -> float:
    """One frame's contribution to the gamma bias, in J/m^2."""

    return coeff * n_atoms * 1.5 * K_B_EV_PER_K * delta_t_k / denominator_a2 * EV_A2_TO_J_M2


def load_cases(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    try:
        data = json.loads(text) if source.suffix.lower() == ".json" else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"{source}: cannot parse input: {exc}") from exc
    problems = schema_errors(data, INPUT_SCHEMA)
    if problems:
        raise ConfigurationError(f"{source} does not conform to {INPUT_SCHEMA}:\n  " + "\n  ".join(problems))
    cases = data["cases"] if "cases" in data else [data]
    return data, cases


def audit_case(
    case: dict[str, Any],
    *,
    index: int = 0,
    tol_ratio: float = DEFAULT_TOL_RATIO,
    tol_t_k: float = DEFAULT_TOL_T_K,
    tol_gamma: float = DEFAULT_TOL_GAMMA_J_M2,
) -> dict[str, Any]:
    label = case.get("label") or f"case_{index}"
    frames = case["interface_frames"]
    ids = [frame["id"] for frame in frames]
    duplicates = sorted({frame_id for frame_id in ids if ids.count(frame_id) > 1})
    if duplicates:
        raise ConfigurationError(f"{label}: duplicate interface frame id(s) {duplicates}")

    temperatures = sorted({float(ref["T_K"]) for ref in case["bulk_refs"]})
    if len(temperatures) != 1:
        detail = ", ".join(f"{ref['name']}={ref['T_K']} K" for ref in case["bulk_refs"])
        raise ConfigurationError(f"{label}: bulk references disagree on temperature ({detail}); no common T_ref")
    t_ref = temperatures[0]
    denominator = float(case["denominator_A2"])

    # C1 equipartition
    statics = {ref["name"]: ref for ref in case.get("static_refs", [])}
    c1_rows = []
    for ref in case["bulk_refs"]:
        row: dict[str, Any] = {
            "name": ref["name"],
            "kind": ref["kind"],
            "T_K": float(ref["T_K"]),
            "atoms_per_fu": ref["atoms_per_fu"],
            "e_thermal_eV_fu": ref["e_per_fu_eV"] if ref["kind"] == "thermal_mean" else None,
            "e_static_eV_fu": None,
        }
        if ref["kind"] == "static":
            row.update(status="NOT_APPLICABLE", note="static reference; no thermal energy to check")
        elif ref["name"] not in statics:
            row.update(status="NO_STATIC_REF", note="no static_refs entry with this name")
        else:
            static = statics[ref["name"]]["e_per_fu_eV"]
            row["e_static_eV_fu"] = static
            row.update(
                equipartition_check(
                    ref["e_per_fu_eV"], static, float(ref["T_K"]), float(ref["atoms_per_fu"]), tol_ratio=tol_ratio
                )
            )
        c1_rows.append(row)
    statuses = {row["status"] for row in c1_rows}
    c1_status = "FAIL" if "FAIL" in statuses else "INCOMPLETE" if "NO_STATIC_REF" in statuses else "PASS"

    # C2 temperature match
    frame_rows = []
    for frame in frames:
        t_k = frame["T_K"]
        delta_t = None if t_k is None else float(t_k) - t_ref
        bias = None if delta_t is None else frame_bias_J_m2(frame["coeff"], frame["n_atoms"], delta_t, denominator)
        frame_rows.append(
            {
                "id": frame["id"],
                "family": frame["family"],
                "n_atoms": frame["n_atoms"],
                "coeff": frame["coeff"],
                "T_K": t_k,
                "T_ref_K": t_ref,
                "delta_T_K": delta_t,
                "delta_gamma_J_m2": bias,
                "provenance": frame.get("provenance"),
            }
        )
    unknown = [row["id"] for row in frame_rows if row["T_K"] is None]
    known_bias = sum(row["delta_gamma_J_m2"] for row in frame_rows if row["delta_gamma_J_m2"] is not None)
    delta_gamma = None if unknown else known_bias
    max_abs_dt = max((abs(row["delta_T_K"]) for row in frame_rows if row["delta_T_K"] is not None), default=None)
    sensitivity = sum(
        frame_bias_J_m2(frame["coeff"], frame["n_atoms"], 1.0, denominator) for frame in frames
    )

    # C3 spread
    by_family: dict[str, list[float]] = defaultdict(list)
    for value in case.get("gamma_values", []):
        by_family[value["family"]].append(float(value["gamma_J_m2"]))
    spread = {}
    for family, values in by_family.items():
        n = len(values)
        mean = sum(values) / n
        if n >= 2:
            std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
            spread[family] = {"n": n, "mean_J_m2": mean, "std_J_m2": std, "sem_J_m2": std / math.sqrt(n)}
        else:
            spread[family] = {"n": n, "mean_J_m2": mean, "std_J_m2": None, "sem_J_m2": None,
                              "note": "fewer than 2 values; spread not assessed"}

    # Verdict
    reasons: list[str] = []
    verdict = UNDETERMINED
    if c1_status == "FAIL":
        failed = [f"{row['name']} ratio {row['ratio']:.4f}" for row in c1_rows if row["status"] == "FAIL"]
        reasons.append(f"C1 equipartition fails (|ratio-1| > {tol_ratio}): {', '.join(failed)}")
        verdict = INCONSISTENT
    if delta_gamma is not None and abs(delta_gamma) > tol_gamma:
        reasons.append(f"|d_gamma| = {abs(delta_gamma):.6g} J/m^2 > tol_gamma {tol_gamma}")
        verdict = INCONSISTENT
    if verdict != INCONSISTENT:
        if unknown:
            reasons.append(f"T_K is null for frame(s) {', '.join(unknown)}; d_gamma cannot be computed")
        if c1_status == "INCOMPLETE":
            missing = [row["name"] for row in c1_rows if row["status"] == "NO_STATIC_REF"]
            reasons.append(f"C1 not evaluated for {', '.join(missing)} (no static reference)")
        if not unknown and max_abs_dt is not None and max_abs_dt > tol_t_k:
            reasons.append(
                f"max |dT| = {max_abs_dt:g} K > tol_T {tol_t_k} K although |d_gamma| = {abs(delta_gamma):.6g} "
                f"J/m^2 <= tol_gamma"
            )
        if not reasons:
            verdict = CONSISTENT
            reasons.append(f"every |dT| <= {tol_t_k} K and C1 passes")
    return {
        "label": label,
        "verdict": verdict,
        "reasons": reasons,
        "T_ref_K": t_ref,
        "denominator_A2": denominator,
        "c1_status": c1_status,
        "c1_equipartition": c1_rows,
        "frames": frame_rows,
        "frames_with_null_T": unknown,
        "max_abs_delta_T_K": max_abs_dt,
        "delta_gamma_J_m2": delta_gamma,
        "delta_gamma_known_frames_J_m2": known_bias,
        "sensitivity_J_m2_per_K": sensitivity,
        "c3_spread": spread or None,
        "source": case.get("source"),
        "notes": case.get("notes", []),
    }


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Excess-temperature audit",
        "",
        f"**Verdict:** {payload['verdict']}  ·  input `{payload['input']}` (sha256 `{payload['input_sha256'][:12]}…`)",
        f"Tolerances: ratio ±{payload['tolerances']['ratio']}, T {payload['tolerances']['T_K']} K, "
        f"gamma {payload['tolerances']['gamma_J_m2']} J/m²  ·  k_B = {K_B_EV_PER_K} eV/K, "
        f"1 eV/Å² = {EV_A2_TO_J_M2} J/m²",
    ]
    for case in payload["cases"]:
        lines += ["", f"## {case['label']} — {case['verdict']}", ""]
        lines += [f"- {reason}" for reason in case["reasons"]]
        lines += [
            f"- T_ref = {case['T_ref_K']:g} K, D = {case['denominator_A2']:.6g} Å²; "
            f"d_gamma = {_fmt(case['delta_gamma_J_m2'])} J/m²; "
            f"sensitivity to a uniform frame offset = {case['sensitivity_J_m2_per_K']:.6g} J/m² per K",
            "",
            "### C1 equipartition",
            "",
            "| Bulk | kind | T K | e_thermal eV/f.u. | e_static eV/f.u. | Δ meV/f.u. | expected meV/f.u. "
            "| ratio | status |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for row in case["c1_equipartition"]:
            delta = row.get("delta_eV_fu")
            expected = row.get("expected_eV_fu")
            lines.append(
                f"| {row['name']} | {row['kind']} | {row['T_K']:g} | {_fmt(row['e_thermal_eV_fu'], 9)} | "
                f"{_fmt(row['e_static_eV_fu'], 9)} | {_fmt(None if delta is None else delta * 1e3)} | "
                f"{_fmt(None if expected is None else expected * 1e3)} | {_fmt(row.get('ratio'), 5)} | "
                f"{row['status']} |"
            )
        lines += [
            "",
            "### C2 frames",
            "",
            "| Frame | family | n_atoms | c_i | T K | ΔT K | Δγ J/m² |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in case["frames"]:
            lines.append(
                f"| {row['id']} | {row['family']} | {row['n_atoms']} | {_fmt(row['coeff'])} | {_fmt(row['T_K'])} | "
                f"{_fmt(row['delta_T_K'])} | {_fmt(row['delta_gamma_J_m2'])} |"
            )
        lines += ["", "### C3 spread", ""]
        if case["c3_spread"]:
            for family, stats in case["c3_spread"].items():
                lines.append(
                    f"- {family}: n={stats['n']}, mean {_fmt(stats['mean_J_m2'])}, std {_fmt(stats['std_J_m2'])}, "
                    f"SEM {_fmt(stats['sem_J_m2'])} J/m²"
                )
        else:
            lines.append("- not assessed (no per-frame gamma_values)")
        if case["notes"]:
            lines += ["", "### Notes", ""] + [f"- {note}" for note in case["notes"]]
    return "\n".join(lines) + "\n"


def run_audit(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    force: bool = False,
    tol_ratio: float = DEFAULT_TOL_RATIO,
    tol_t_k: float = DEFAULT_TOL_T_K,
    tol_gamma: float = DEFAULT_TOL_GAMMA_J_M2,
) -> dict[str, Any]:
    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(f"input not found: {source}")
    output = Path(output_dir) if output_dir else source.parent
    existing = [name for name in OUTPUT_NAMES if (output / name).exists()]
    if existing and not force:
        raise SafetyError(f"{output} already has {', '.join(existing)}; pass --force to overwrite")

    _, cases = load_cases(source)
    results = [
        audit_case(case, index=index, tol_ratio=tol_ratio, tol_t_k=tol_t_k, tol_gamma=tol_gamma)
        for index, case in enumerate(cases)
    ]
    verdict = max((case["verdict"] for case in results), key=_RANK.__getitem__)
    input_sha = sha256_file(source)
    payload = {
        "stage": STAGE,
        "verdict": verdict,
        "input": str(source),
        "input_sha256": input_sha,
        "tolerances": {"ratio": tol_ratio, "T_K": tol_t_k, "gamma_J_m2": tol_gamma},
        "constants": {"k_B_eV_per_K": K_B_EV_PER_K, "eV_per_A2_to_J_per_m2": EV_A2_TO_J_M2},
        "cases": results,
    }

    output.mkdir(parents=True, exist_ok=True)
    audit_json = output / f"{STAGE}_audit.json"
    audit_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = ("case", "frame_id", "family", "n_atoms", "coeff", "T_K", "T_ref_K", "delta_T_K", "delta_gamma_J_m2")
    with (output / f"{STAGE}_audit.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(fields)
        for case in results:
            for row in case["frames"]:
                writer.writerow(
                    [case["label"], row["id"], row["family"], row["n_atoms"], row["coeff"]]
                    + ["" if row[key] is None else row[key] for key in ("T_K", "T_ref_K", "delta_T_K",
                                                                       "delta_gamma_J_m2")]
                )
    (output / f"{STAGE}_audit.md").write_text(_markdown(payload), encoding="utf-8")
    outputs = {name: sha256_file(output / name) for name in OUTPUT_NAMES[1:]}
    manifest = {
        "stage": STAGE,
        "created_at": utc_now(),
        "tool": "interfaceforge",
        "tool_version": __version__,
        "command": "iface audit excess-temperature",
        "input": str(source.resolve()),
        "input_sha256": input_sha,
        "tolerances": payload["tolerances"],
        "constants": payload["constants"],
        "verdict": verdict,
        "outputs_sha256": outputs,
    }
    manifest_path = output / OUTPUT_NAMES[0]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    StateStore(output).event(
        "excess_temperature_audit",
        input=str(source.resolve()),
        input_sha256=input_sha,
        verdict=verdict,
        manifest_sha256=sha256_file(manifest_path),
    )
    return {
        "stage": STAGE,
        "verdict": verdict,
        "input_sha256": input_sha,
        "output": str(output),
        "cases": [
            {
                "label": case["label"],
                "verdict": case["verdict"],
                "reasons": case["reasons"],
                "T_ref_K": case["T_ref_K"],
                "delta_gamma_J_m2": case["delta_gamma_J_m2"],
                "sensitivity_J_m2_per_K": case["sensitivity_J_m2_per_K"],
                "c1": {row["name"]: row.get("ratio") for row in case["c1_equipartition"]},
            }
            for case in results
        ],
        "manifest": str(manifest_path),
    }


def cmd_excess_temperature(args: argparse.Namespace) -> int:
    payload = run_audit(
        args.input,
        args.output,
        force=args.force,
        tol_ratio=args.tol_ratio,
        tol_t_k=args.tol_t,
        tol_gamma=args.tol_gamma,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def register_commands(commands: Any) -> None:
    """Attach ``excess-temperature`` to an ``iface audit`` subparsers object."""

    parser = commands.add_parser(
        "excess-temperature",
        help="Check that a bulk-referenced interface excess compares energies at one temperature",
    )
    parser.add_argument("--input", required=True, help="JSON/YAML input (schemas/excess_temperature_input.schema.json)")
    parser.add_argument("-o", "--output", help="Output directory (default: the input file's directory)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing excess_temperature_* outputs")
    parser.add_argument("--tol-ratio", type=float, default=DEFAULT_TOL_RATIO,
                        help="C1: allowed |ratio - 1| (default 0.10, anharmonic margin)")
    parser.add_argument("--tol-t", type=float, default=DEFAULT_TOL_T_K,
                        help="C2: allowed |T_i - T_ref| in K for CONSISTENT (default 1)")
    parser.add_argument("--tol-gamma", type=float, default=DEFAULT_TOL_GAMMA_J_M2,
                        help="C2: |d_gamma| in J/m^2 above which the excess is INCONSISTENT (default 0.01)")
    parser.set_defaults(func=cmd_excess_temperature)
