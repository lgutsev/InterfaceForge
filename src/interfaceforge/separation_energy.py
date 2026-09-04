# ruff: noqa: E501
"""Slab-referenced separation energy of a hand-built interface, DFT vs MLIP.

For a supplied interface slab and its two isolated half-slabs (``interface``,
``slab_a``, ``slab_b``)::

    gamma_sep = ( E(slab_a) + E(slab_b) - E(interface) ) / (n_interfaces * A)

converted to J/m^2. With ``reference: free-surface`` (each half-slab relaxed
against its own vacuum surface) and ``n_interfaces = 1`` this is exactly the
Dupre work of adhesion -- the quantity Sharifi et al. (2026) report.

Unlike ``iface validate interface-energy`` (which references the MD dataset
against bulk phases), this evaluates a small set of *hand-built* structures with
both DFT (read back from finished VASP runs) and one or more MLIP committees
(evaluated in place), so the headline number is ``gamma_sep^MLIP -
gamma_sep^DFT`` on identical geometry. The literature value, when a matching
``validation.references`` entry exists, is a secondary check.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .audit import audit_run
from .config import merge_interface_metadata, references_for
from .errors import DependencyError, SafetyError
from .validation import compare_to_references

EV_A2_TO_J_M2 = 16.02176634
_PARTS = ("interface", "slab_a", "slab_b")
_REFERENCE_KINDS = ("free-surface", "bulk")


# --------------------------------------------------------------------------- IO


def _structure_file(directory: Path) -> Path:
    for name in ("CONTCAR", "POSCAR"):
        candidate = directory / name
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    raise SafetyError(f"No CONTCAR or POSCAR in {directory}")


def _read_atoms(path: Path) -> Any:
    try:
        from ase.io import read
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without ASE
        raise DependencyError(
            "separation-energy needs ASE to read structures; install interfaceforge[vasp]"
        ) from exc
    return read(path)


def _resolve_set(directory: Path) -> tuple[dict[str, Path], str | None]:
    """Locate the ``interface`` / ``slab_a`` / ``slab_b`` run directories.

    Two layouts are accepted: a plain directory with ``interface/``, ``slab_a/``
    and ``slab_b/`` sub-directories, or an ``iface vasp adhesion prepare`` output
    tree -- ``interface_static/`` (or the reference directory) plus the two
    ``slabs/*`` fragments, read from its ``manifest.json``. Returns the mapping
    and the tree's ``slab_mode`` (``None`` for the plain layout).
    """

    manifest_path = directory / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SafetyError(f"Could not parse {manifest_path}: {exc}") from exc
        slabs = manifest.get("slabs")
        if isinstance(slabs, list) and len(slabs) == 2:
            static = manifest.get("interface_static")
            interface_dir = (
                directory / static["directory"]
                if static
                else Path(manifest["reference_directory"])
            )
            return (
                {
                    "interface": interface_dir,
                    "slab_a": directory / slabs[0]["directory"],
                    "slab_b": directory / slabs[1]["directory"],
                },
                manifest.get("slab_mode"),
            )

    out: dict[str, Path] = {}
    for part in _PARTS:
        candidate = directory / part
        if not candidate.is_dir():
            raise SafetyError(
                f"{directory} is neither an 'adhesion prepare' tree nor has a "
                f"'{part}/' sub-directory"
            )
        out[part] = candidate
    return out, None


def _plane_area(cell: np.ndarray, axis: str | None) -> tuple[float, str]:
    axes = "abc"
    if axis and axis in axes:
        stack = axes.index(axis)
    else:
        stack = int(np.argmax([np.linalg.norm(v) for v in cell]))
    keep = [i for i in range(3) if i != stack]
    area = float(np.linalg.norm(np.cross(cell[keep[0]], cell[keep[1]])))
    return area, axes[stack]


def _dft_energy(run: Path) -> float | None:
    if not (run / "OUTCAR").is_file():
        return None
    row = audit_run(run, run)
    if not row.get("finished_normally"):
        return None
    energy = row.get("sigma0_energy_ev_last")
    return float(energy) if energy is not None else None


# --------------------------------------------------------------- MLIP back-ends


def _model_member_labels(model_paths: Sequence[str]) -> list[str]:
    """Return stable, unique labels without collapsing same-named models.

    DeePMD committees normally contain paths such as
    ``model_000/frozen_model.pth`` ... ``model_003/frozen_model.pth``. Using
    only ``Path.stem`` silently overwrote all but the last member. Keep the
    shortest unique trailing path instead.
    """

    normalized = [str(Path(path).expanduser()) for path in model_paths]
    duplicates = [path for path, count in Counter(normalized).items() if count > 1]
    if duplicates:
        raise SafetyError(f"Duplicate MLIP model path: {duplicates[0]}")

    stemmed_parts = [Path(path).with_suffix("").parts for path in normalized]
    labels: list[str] = []
    for index, parts in enumerate(stemmed_parts):
        for depth in range(1, len(parts) + 1):
            candidate = "/".join(parts[-depth:])
            matches = sum("/".join(other[-depth:]) == candidate for other in stemmed_parts)
            if matches == 1:
                labels.append(candidate)
                break
        else:  # pragma: no cover - duplicate normalized paths are rejected above
            labels.append(f"member_{index:03d}")
    return labels


def _mace_energies(model_paths: Sequence[str], atoms_by_part: Mapping[str, Any], device: str) -> dict[str, dict[str, float]]:
    try:
        from mace.calculators import MACECalculator
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "MACE evaluation needs mace-torch; install interfaceforge[mace-roi] "
            "or run this where the committee's environment is available"
        ) from exc
    out: dict[str, dict[str, float]] = {}
    for path, label in zip(model_paths, _model_member_labels(model_paths), strict=True):
        calc = MACECalculator(model_paths=[path], device=device, default_dtype="float64")
        member: dict[str, float] = {}
        for part, atoms in atoms_by_part.items():
            probe = atoms.copy()
            probe.calc = calc
            member[part] = float(probe.get_potential_energy())
        out[label] = member
    return out


def _deepmd_energies(model_paths: Sequence[str], atoms_by_part: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    try:
        from deepmd.calculator import DP
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "DeePMD evaluation needs deepmd-kit; run this where the committee's "
            "environment is available"
        ) from exc
    out: dict[str, dict[str, float]] = {}
    for path, label in zip(model_paths, _model_member_labels(model_paths), strict=True):
        calc = DP(model=path)
        member: dict[str, float] = {}
        for part, atoms in atoms_by_part.items():
            probe = atoms.copy()
            probe.calc = calc
            member[part] = float(probe.get_potential_energy())
        out[label] = member
    return out


# ----------------------------------------------------------------------- maths


def _gamma(energies: Mapping[str, float], denom: float) -> float:
    excess = energies["slab_a"] + energies["slab_b"] - energies["interface"]
    return excess / denom * EV_A2_TO_J_M2


def _family_block(members: Mapping[str, Mapping[str, float]], denom: float, dft_gamma: float | None) -> dict[str, Any]:
    per_member = {name: _gamma(energy, denom) for name, energy in members.items()}
    values = np.array(list(per_member.values()), dtype=float)
    ensemble = float(values.mean())
    spread = float(values.std(ddof=1)) if values.size > 1 else 0.0
    block: dict[str, Any] = {
        "members": len(per_member),
        "gamma_sep_members_j_per_m2": {name: float(value) for name, value in per_member.items()},
        "gamma_sep_ensemble_j_per_m2": ensemble,
        "committee_spread_j_per_m2": spread,
    }
    if dft_gamma is not None:
        block["delta_vs_dft_j_per_m2"] = ensemble - dft_gamma
    return block


def _family_block_from_gammas(members: Mapping[str, float], dft_gamma: float | None) -> dict[str, Any]:
    values = np.array(list(members.values()), dtype=float)
    if values.size == 0:
        raise SafetyError("Cannot merge an empty MLIP committee")
    ensemble = float(values.mean())
    spread = float(values.std(ddof=1)) if values.size > 1 else 0.0
    block: dict[str, Any] = {
        "members": len(members),
        "gamma_sep_members_j_per_m2": {name: float(value) for name, value in members.items()},
        "gamma_sep_ensemble_j_per_m2": ensemble,
        "committee_spread_j_per_m2": spread,
    }
    if dft_gamma is not None:
        block["delta_vs_dft_j_per_m2"] = ensemble - dft_gamma
    return block


def _literature_hits(
    row: Mapping[str, Any],
    *,
    lit_refs: Sequence[Mapping[str, Any]] | None,
    interfaces_meta: Any,
) -> list[dict[str, Any]]:
    if not lit_refs:
        return []
    attrs = merge_interface_metadata(interfaces_meta, row["spec"]) if interfaces_meta else {}
    probes: list[tuple[str, float]] = []
    if row["dft"]["ready"]:
        probes.append(("dft", float(row["dft"]["gamma_sep_j_per_m2"])))
    for family, block in row["mlip"].items():
        probes.append((family, float(block["gamma_sep_ensemble_j_per_m2"])))
    hits: list[dict[str, Any]] = []
    for source, value in probes:
        for hit in compare_to_references(value, lit_refs, attrs, quantity="work_of_adhesion"):
            hits.append({"source": source, **hit})
    return hits


# --------------------------------------------------------------------- driver


def separation_energy(
    entries: Sequence[tuple[str, str | Path]],
    *,
    mace_models: Sequence[str] = (),
    deepmd_models: Sequence[str] = (),
    reference: str = "free-surface",
    n_interfaces: int = 1,
    area_axis: str | None = None,
    device: str = "cpu",
    campaign_validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if reference not in _REFERENCE_KINDS:
        raise SafetyError(f"reference must be one of {_REFERENCE_KINDS}")
    if n_interfaces < 1:
        raise SafetyError("n_interfaces must be a positive integer")
    if not entries:
        raise SafetyError("separation_energy needs at least one (label, directory) entry")

    lit_refs = (
        references_for(campaign_validation.get("references"), "work_of_adhesion")
        if campaign_validation
        else None
    )
    interfaces_meta = campaign_validation.get("interfaces") if campaign_validation else None

    slab_modes: set[str] = set()
    rows: list[dict[str, Any]] = []
    for spec, directory in entries:
        directory = Path(directory).expanduser().resolve()
        runs, slab_mode = _resolve_set(directory)
        if slab_mode:
            slab_modes.add(slab_mode)
        atoms = {part: _read_atoms(_structure_file(runs[part])) for part in _PARTS}
        cell = np.array(atoms["interface"].cell.array, dtype=float)
        area, axis = _plane_area(cell, area_axis)
        denom = n_interfaces * area

        dft_energies = {part: _dft_energy(runs[part]) for part in _PARTS}
        dft_ready = all(value is not None for value in dft_energies.values())
        dft_gamma = _gamma({k: float(v) for k, v in dft_energies.items()}, denom) if dft_ready else None

        row: dict[str, Any] = {
            "label": spec.strip("/").rsplit("/", 1)[-1] or spec,
            "spec": spec,
            "directory": str(directory),
            "slab_mode": slab_mode,
            "interface_area_ang2": area,
            "area_axis": axis,
            "n_interfaces": n_interfaces,
            "reference": reference,
            "natoms": {part: int(len(atoms[part])) for part in _PARTS},
            "dft": {
                "ready": dft_ready,
                "energies_ev": dft_energies,
                "gamma_sep_j_per_m2": dft_gamma,
            },
            "mlip": {},
            "literature": [],
        }

        if mace_models:
            row["mlip"]["mace"] = _family_block(
                _mace_energies(mace_models, atoms, device), denom, dft_gamma
            )
        if deepmd_models:
            row["mlip"]["deepmd"] = _family_block(
                _deepmd_energies(deepmd_models, atoms), denom, dft_gamma
            )

        row["literature"] = _literature_hits(
            row, lit_refs=lit_refs, interfaces_meta=interfaces_meta
        )

        rows.append(row)

    if slab_modes == {"static"}:
        interpretation = "ideal work of separation (slabs frozen at the interface geometry)"
    elif slab_modes == {"relax"}:
        interpretation = "Dupre work of adhesion (relaxed slab geometries from DFT)"
    else:
        interpretation = (
            "Dupre work of adhesion for relaxed free-surface half-slabs with n_interfaces=1"
        )

    return {
        "schema_version": 1,
        "quantity": "separation_energy",
        "definition": "gamma_sep = (E(slab_a) + E(slab_b) - E(interface)) / (n_interfaces * A); "
        + interpretation,
        "reference": reference,
        "slab_modes": sorted(slab_modes),
        "n_interfaces": n_interfaces,
        "conversion_ev_a2_to_j_m2": EV_A2_TO_J_M2,
        "mace_models": [str(p) for p in mace_models],
        "deepmd_models": [str(p) for p in deepmd_models],
        "interfaces": rows,
    }


_MERGE_ROW_FIELDS = (
    "label",
    "spec",
    "directory",
    "slab_mode",
    "area_axis",
    "n_interfaces",
    "reference",
    "natoms",
)


def _load_partial(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SafetyError(f"Separation-energy partial does not exist: {source}") from exc
    except json.JSONDecodeError as exc:
        raise SafetyError(f"Could not parse separation-energy partial {source}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("quantity") != "separation_energy":
        raise SafetyError(f"Not a separation-energy JSON payload: {source}")
    if payload.get("schema_version") != 1 or not isinstance(payload.get("interfaces"), list):
        raise SafetyError(f"Unsupported separation-energy payload schema in {source}")
    return source, payload


def _merge_dft(base: dict[str, Any], incoming: Mapping[str, Any], spec: str) -> dict[str, Any]:
    if not base.get("ready") and incoming.get("ready"):
        return copy.deepcopy(dict(incoming))
    if base.get("ready") and incoming.get("ready"):
        for part in _PARTS:
            left = float(base["energies_ev"][part])
            right = float(incoming["energies_ev"][part])
            if not np.isclose(left, right, rtol=0.0, atol=1.0e-8):
                raise SafetyError(
                    f"DFT energy mismatch for {spec} {part}: {left} vs {right}; "
                    "the partials do not describe the same completed calculation"
                )
    return base


def merge_separation_energy(
    partials: Sequence[str | Path],
    *,
    campaign_validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge backend-isolated separation-energy JSON payloads.

    Each partial may be produced in a different Python/CUDA environment. This
    function imports neither MACE nor DeePMD; it validates that every payload
    describes the same structures, combines committee members, and recomputes
    ensemble statistics and deltas against the best available DFT result.
    """

    if not partials:
        raise SafetyError("separation-energy merge needs at least one --merge-json file")
    loaded = [_load_partial(path) for path in partials]
    first_path, first = loaded[0]
    merged = copy.deepcopy(first)
    merged.pop("outputs", None)
    merged["merged_from"] = [str(first_path)]

    top_fields = ("reference", "n_interfaces", "conversion_ev_a2_to_j_m2")
    rows_by_spec = {row["spec"]: row for row in merged["interfaces"]}
    if len(rows_by_spec) != len(merged["interfaces"]):
        raise SafetyError(f"Duplicate interface spec in {first_path}")

    for source, payload in loaded[1:]:
        for field in top_fields:
            if payload.get(field) != merged.get(field):
                raise SafetyError(
                    f"Cannot merge {source}: top-level {field!r} differs "
                    f"({payload.get(field)!r} vs {merged.get(field)!r})"
                )
        incoming_by_spec = {row["spec"]: row for row in payload["interfaces"]}
        if set(incoming_by_spec) != set(rows_by_spec):
            raise SafetyError(
                f"Cannot merge {source}: interface specs differ from {first_path}"
            )
        for spec, base_row in rows_by_spec.items():
            incoming = incoming_by_spec[spec]
            for field in _MERGE_ROW_FIELDS:
                if incoming.get(field) != base_row.get(field):
                    raise SafetyError(
                        f"Cannot merge {source}: {spec} field {field!r} differs"
                    )
            if not np.isclose(
                float(incoming["interface_area_ang2"]),
                float(base_row["interface_area_ang2"]),
                rtol=0.0,
                atol=1.0e-8,
            ):
                raise SafetyError(f"Cannot merge {source}: {spec} interface area differs")
            base_row["dft"] = _merge_dft(base_row["dft"], incoming["dft"], spec)
            for family, block in incoming["mlip"].items():
                if family not in base_row["mlip"]:
                    base_row["mlip"][family] = copy.deepcopy(block)
                    continue
                members = dict(base_row["mlip"][family]["gamma_sep_members_j_per_m2"])
                for name, value in block["gamma_sep_members_j_per_m2"].items():
                    if name in members and not np.isclose(
                        float(members[name]), float(value), rtol=0.0, atol=1.0e-8
                    ):
                        raise SafetyError(
                            f"Cannot merge {source}: {spec} {family} member {name!r} differs"
                        )
                    members[name] = float(value)
                dft_gamma = (
                    float(base_row["dft"]["gamma_sep_j_per_m2"])
                    if base_row["dft"]["ready"]
                    else None
                )
                base_row["mlip"][family] = _family_block_from_gammas(members, dft_gamma)
        for key in ("mace_models", "deepmd_models"):
            merged[key] = list(dict.fromkeys([*merged.get(key, []), *payload.get(key, [])]))
        merged["merged_from"].append(str(source))

    lit_refs = (
        references_for(campaign_validation.get("references"), "work_of_adhesion")
        if campaign_validation
        else None
    )
    interfaces_meta = campaign_validation.get("interfaces") if campaign_validation else None
    for row in merged["interfaces"]:
        dft_gamma = (
            float(row["dft"]["gamma_sep_j_per_m2"])
            if row["dft"]["ready"]
            else None
        )
        for family, block in list(row["mlip"].items()):
            row["mlip"][family] = _family_block_from_gammas(
                block["gamma_sep_members_j_per_m2"], dft_gamma
            )
        if campaign_validation:
            row["literature"] = _literature_hits(
                row, lit_refs=lit_refs, interfaces_meta=interfaces_meta
            )
        else:
            combined: list[dict[str, Any]] = []
            seen: set[tuple[str, str]] = set()
            for _, payload in loaded:
                incoming = next(item for item in payload["interfaces"] if item["spec"] == row["spec"])
                for hit in incoming.get("literature", []):
                    key = (str(hit.get("source")), str(hit.get("key")))
                    if key not in seen:
                        seen.add(key)
                        combined.append(copy.deepcopy(hit))
            row["literature"] = combined
    return merged


# ---------------------------------------------------------------------- report

_CSV_FIELDS = (
    "interface",
    "reference",
    "source",
    "gamma_sep_j_per_m2",
    "committee_spread_j_per_m2",
    "delta_vs_dft_j_per_m2",
    "literature_key",
    "literature_j_per_m2",
    "literature_delta_j_per_m2",
    "literature_within_tolerance",
)


def _row_sources(row: dict[str, Any]) -> list[tuple[str, float | None, float, float | None]]:
    """(source, gamma, spread, delta_vs_dft) tuples for one interface."""

    out: list[tuple[str, float | None, float, float | None]] = []
    if row["dft"]["ready"]:
        out.append(("dft", row["dft"]["gamma_sep_j_per_m2"], 0.0, None))
    for family, block in row["mlip"].items():
        out.append(
            (
                family,
                block["gamma_sep_ensemble_j_per_m2"],
                block["committee_spread_j_per_m2"],
                block.get("delta_vs_dft_j_per_m2"),
            )
        )
    return out


def write_json_payload(payload: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write only the backend-neutral JSON used by isolated evaluation stages."""

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    target = out / "separation_energy.json"
    target.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return {"json": str(target)}


def write_reports(payload: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir).expanduser().resolve()
    outputs = write_json_payload(payload, out)

    with (out / "separation_energy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in payload["interfaces"]:
            lit_by_source: dict[str, dict[str, Any]] = {}
            for item in row["literature"]:
                lit_by_source.setdefault(item["source"], item)
            for source, gamma, spread, delta in _row_sources(row):
                record = {
                    "interface": row["label"],
                    "reference": row["reference"],
                    "source": source,
                    "gamma_sep_j_per_m2": gamma,
                    "committee_spread_j_per_m2": spread,
                    "delta_vs_dft_j_per_m2": delta,
                }
                hit = lit_by_source.get(source)
                if hit:
                    record.update(
                        {
                            "literature_key": hit["key"],
                            "literature_j_per_m2": hit["reference_j_per_m2"],
                            "literature_delta_j_per_m2": hit["delta_j_per_m2"],
                            "literature_within_tolerance": hit["within_tolerance"],
                        }
                    )
                writer.writerow(record)

    lines = [
        "# Separation energy (DFT vs MLIP)",
        "",
        f"γ_sep = (E(slab_a) + E(slab_b) − E(interface)) / ({payload['n_interfaces']} · A), "
        f"reference: {payload['reference']}.",
        "",
        f"Interpretation: {payload['definition'].split('; ', 1)[-1]}.",
        "",
        "| Interface | Source | γ_sep (J/m²) | committee σ | Δ vs DFT | vs literature |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["interfaces"]:
        lit_by_source = {item["source"]: item for item in row["literature"]}
        for source, gamma, spread, delta in _row_sources(row):
            gamma_text = "—" if gamma is None else f"{gamma:.3f}"
            spread_text = f"{spread:.3f}" if source != "dft" else "—"
            delta_text = f"{delta:+.3f}" if delta is not None else "—"
            hit = lit_by_source.get(source)
            lit_text = (
                f"{hit['delta_j_per_m2']:+.3f} ({'ok' if hit['within_tolerance'] else 'NO'})"
                if hit
                else "—"
            )
            lines.append(
                f"| {row['label']} | {source} | {gamma_text} | {spread_text} | {delta_text} | {lit_text} |"
            )
    (out / "separation_energy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    outputs.update(
        {
            "csv": str(out / "separation_energy.csv"),
            "markdown": str(out / "separation_energy.md"),
        }
    )
    try:
        outputs.update(_write_figure(payload, out))
    except (DependencyError, SafetyError) as exc:
        outputs["figure"] = f"skipped: {exc}"
    return outputs


_SOURCE_COLORS = {
    "dft": "#111111",
    "mace": "#0072B2",
    "deepmd": "#D55E00",
}


def _write_figure(payload: dict[str, Any], out: Path) -> dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "The separation-energy figure requires matplotlib; install interfaceforge[report]"
        ) from exc

    rows = [row for row in payload["interfaces"] if _row_sources(row)]
    if not rows:
        raise SafetyError("no interface has a finished energy to plot")
    order = list(reversed(rows))
    y = np.arange(len(order), dtype=float)
    has_delta = any(
        block.get("delta_vs_dft_j_per_m2") is not None
        for row in order
        for block in row["mlip"].values()
    )
    offsets = {"dft": 0.0, "mace": 0.18, "deepmd": -0.18}
    seen: set[str] = set()

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        panels = 2 if has_delta else 1
        fig, axes = plt.subplots(
            1,
            panels,
            figsize=(7.2 if panels == 2 else 4.6, 0.6 * len(order) + 1.6),
            squeeze=False,
            layout="constrained",
        )
        ax = axes[0][0]
        for index, row in enumerate(order):
            for source, gamma, spread, _ in _row_sources(row):
                if gamma is None:
                    continue
                seen.add(source)
                ax.errorbar(
                    gamma,
                    y[index] + offsets.get(source, 0.0),
                    xerr=spread or None,
                    fmt="o",
                    ms=5.0,
                    color=_SOURCE_COLORS.get(source, "#4B5563"),
                    ecolor=_SOURCE_COLORS.get(source, "#4B5563"),
                    elinewidth=1.0,
                    capsize=2.5,
                    zorder=3,
                )
            for hit in row["literature"]:
                if hit["source"] != "dft" and "dft" in {s for s, *_ in _row_sources(row)}:
                    continue  # draw the literature marker once per interface
                ax.scatter(
                    [hit["reference_j_per_m2"]],
                    [y[index]],
                    marker="D",
                    s=26,
                    facecolors="white",
                    edgecolors="#4B5563",
                    linewidths=0.9,
                    zorder=4,
                )
                break
        ax.set_yticks(y, labels=[row["label"] for row in order])
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_xlim(left=0.0)
        ax.set_xlabel(r"$\gamma_{\mathrm{sep}}$ (J m$^{-2}$)")
        ax.set_title("(a) Separation energy" if has_delta else "Separation energy", loc="left", fontweight="bold")
        ax.grid(axis="x", color="#D1D5DB", linewidth=0.45, alpha=0.75)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)

        if has_delta:
            ax2 = axes[0][1]
            ax2.axvline(0.0, color="#6B7280", lw=0.8, zorder=2)
            for index, row in enumerate(order):
                for family, block in row["mlip"].items():
                    delta = block.get("delta_vs_dft_j_per_m2")
                    if delta is None:
                        continue
                    ax2.errorbar(
                        delta,
                        y[index] + offsets.get(family, 0.0),
                        xerr=block["committee_spread_j_per_m2"] or None,
                        fmt="o",
                        ms=5.0,
                        color=_SOURCE_COLORS.get(family, "#4B5563"),
                        ecolor=_SOURCE_COLORS.get(family, "#4B5563"),
                        elinewidth=1.0,
                        capsize=2.5,
                        zorder=3,
                    )
            ax2.set_yticks(y, labels=["" for _ in order])
            ax2.set_ylim(len(order) - 0.5, -0.5)
            ax2.set_xlabel(r"$\gamma_{\mathrm{sep}}^{\mathrm{MLIP}} - \gamma_{\mathrm{sep}}^{\mathrm{DFT}}$ (J m$^{-2}$)")
            ax2.set_title("(b) MLIP − DFT", loc="left", fontweight="bold")
            ax2.grid(axis="x", color="#D1D5DB", linewidth=0.45, alpha=0.75)
            ax2.set_axisbelow(True)
            for spine in ("top", "right", "left"):
                ax2.spines[spine].set_visible(False)
            ax2.tick_params(axis="y", length=0)

        handles = [
            Line2D([0], [0], marker="o", color=_SOURCE_COLORS.get(s, "#4B5563"), lw=0, markersize=5,
                   label={"dft": "DFT", "mace": "MACE committee", "deepmd": "DeePMD committee"}.get(s, s))
            for s in sorted(seen)
        ]
        if any(row["literature"] for row in order):
            handles.append(
                Line2D([0], [0], marker="D", color="#4B5563", markerfacecolor="white", lw=0,
                       markersize=5, label="literature value")
            )
        fig.legend(handles=handles, loc="outside lower center", ncols=min(len(handles), 4),
                   frameon=False, handlelength=1.4, columnspacing=1.1)

        stem = "separation_energy"
        paths = {
            "figure_png": out / f"{stem}.png",
            "figure_svg": out / f"{stem}.svg",
            "figure_pdf": out / f"{stem}.pdf",
        }
        fig.savefig(paths["figure_png"], dpi=300, bbox_inches="tight")
        fig.savefig(paths["figure_svg"], bbox_inches="tight")
        fig.savefig(paths["figure_pdf"], bbox_inches="tight")
        plt.close(fig)
    return {key: str(value) for key, value in paths.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="Directory for separation_energy.{json,csv,md,png,svg,pdf}")
    parser.add_argument(
        "entries",
        nargs="*",
        metavar="[LABEL=]SET_DIR",
        help="Each SET_DIR holds interface/ slab_a/ slab_b/ sub-directories, or is "
        "an 'iface vasp adhesion prepare' output tree; the optional LABEL= prefix "
        "is fnmatched against validation.interfaces",
    )
    parser.add_argument("--mace-model", action="append", default=[], dest="mace_models")
    parser.add_argument("--deepmd-model", action="append", default=[], dest="deepmd_models")
    parser.add_argument("--reference", choices=_REFERENCE_KINDS, default="free-surface")
    parser.add_argument("--n-interfaces", type=int, default=1)
    parser.add_argument("--area-axis", choices=("a", "b", "c"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--merge-json",
        action="append",
        default=[],
        help="Merge a separation_energy.json partial (repeatable); imports no ML backend",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Write only separation_energy.json (recommended for isolated backend jobs)",
    )
    parser.add_argument("-c", "--campaign", default=None)
    args = parser.parse_args(argv)

    validation = None
    if args.campaign and Path(args.campaign).is_file():
        from .config import load_campaign

        validation = load_campaign(args.campaign).validation

    if args.merge_json:
        if args.entries or args.mace_models or args.deepmd_models or args.json_only:
            raise SafetyError(
                "--merge-json cannot be combined with entries, model options, or --json-only"
            )
        payload = merge_separation_energy(args.merge_json, campaign_validation=validation)
    else:
        entries: list[tuple[str, str]] = []
        for item in args.entries:
            spec, sep, directory = item.partition("=")
            if not sep:
                spec, directory = Path(item).name, item
            entries.append((spec, directory))
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
    payload["outputs"] = (
        write_json_payload(payload, args.output)
        if args.json_only
        else write_reports(payload, args.output)
    )
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
