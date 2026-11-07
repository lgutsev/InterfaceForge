"""Finite-displacement and strain probes for derivative-sensitive MLIP validation."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
import shutil
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from .errors import DependencyError, SafetyError

SCHEMA_VERSION = 1
QUANTITY = "mlip_derivative_probe"
DEFAULT_DISPLACEMENT_A = 0.03
DEFAULT_VOLUME_STRAINS = (-0.01, 0.0, 0.01)
DEFAULT_SEED = 2026

PAPER = {
    "citation": (
        "Póta, B., Ahlawat, P., Csányi, G. & Simoncelli, M. "
        "Thermal conductivity predictions with foundation atomistic models. "
        "Nature Communications (2026)."
    ),
    "doi": "10.1038/s41467-026-76391-w",
    "url": "https://doi.org/10.1038/s41467-026-76391-w",
}

CURVATURE_DEFINITIONS = {
    "quantity": "directional second derivative of the potential-energy surface",
    "units": "eV/Angstrom^2",
    "displacement_vector": "d is the full 3N Cartesian displacement of one symmetric pair",
    "unit_direction": "u = d / ||d||",
    "energy_derived": "k_E = [E(+d) - 2 E(0) + E(-d)] / ||d||^2",
    "force_derived": "k_F = -[F(+d) - F(-d)] . d / (2 ||d||^2)",
    "interpretation": (
        "Both estimators approximate u^T H u, the directional curvature of the "
        "3N x 3N Hessian H along u, in eV/Angstrom^2."
    ),
    "not_established": (
        "These are not mass-weighted phonon frequencies, a full Hessian, or a "
        "phonon dispersion; no dynamical matrix is built or diagonalized."
    ),
    "internal_delta": (
        "k_F - k_E is a finite-difference/self-consistency diagnostic for one "
        "model, not a comparison against DFT."
    ),
    "compatibility_alias": (
        "'directional_curvature_ev_a2' is retained as an alias of "
        "'force_curvature_ev_a2' for readers of schema_version 1 results."
    ),
}

_STATIC_OVERRIDES = {
    "IBRION": "-1",
    "NSW": "0",
    "ISYM": "0",
    "LWAVE": ".FALSE.",
    "LCHARG": ".FALSE.",
}
_REMOVE_INCAR = {
    "TEBEG",
    "TEEND",
    "SMASS",
    "MDALGO",
    "POTIM",
    "EDIFFG",
    "LANGEVIN_GAMMA",
    "LANGEVIN_GAMMA_L",
    "PMASS",
    "ANDERSEN_PROB",
}
_LABEL = re.compile(r"^[A-Za-z0-9_.-]+$")
_POTCAR_ELEMENT = re.compile(r"VRHFIN\s*=\s*([A-Z][a-z]?)\s*:")


def _ase_io() -> tuple[Any, Any]:
    try:
        from ase.io import read, write
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "ASE is required for derivative probes; install interfaceforge[vasp]"
        ) from exc
    return read, write


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_write(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _resolve_structure(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_dir():
        for name in ("CONTCAR", "POSCAR"):
            structure = candidate / name
            if structure.is_file() and structure.stat().st_size:
                return structure
        raise SafetyError(f"No non-empty CONTCAR or POSCAR in {candidate}")
    if not candidate.is_file() or not candidate.stat().st_size:
        raise SafetyError(f"Missing or empty structure: {candidate}")
    return candidate


def parse_probe_entry(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if separator:
        if not label or not _LABEL.fullmatch(label):
            raise ValueError(
                f"Invalid probe label {label!r}; use letters, numbers, '.', '_' or '-'"
            )
        path = _resolve_structure(raw_path)
    else:
        path = _resolve_structure(value)
        label = path.parent.name if path.name.upper() in {"POSCAR", "CONTCAR"} else path.stem
        label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-") or "structure"
    return label, path


def _prepare_output(root: Path, force: bool) -> None:
    if root.exists() and any(root.iterdir()):
        if not force:
            raise SafetyError(f"Derivative-probe output is not empty: {root}")
        marker = root / "manifest.json"
        try:
            prior = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SafetyError(
                f"Refusing --force because {root} is not a recognized derivative-probe tree"
            ) from exc
        if prior.get("quantity") != QUANTITY:
            raise SafetyError(
                f"Refusing --force because {root} is not a derivative-probe tree"
            )
        if root == Path(root.anchor) or root == Path.home().resolve():
            raise SafetyError(f"Refusing broad output replacement: {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def _incar_assignments(source: Path) -> dict[str, str]:
    """Read assignments, including semicolons, comments and continued lines."""
    result: dict[str, str] = {}
    text = source.read_text(encoding="utf-8", errors="strict")
    text = text.replace("\\\n", " ")
    for line in text.splitlines():
        active = re.split(r"[!#]", line, maxsplit=1)[0]
        for assignment in active.split(";"):
            if not assignment.strip():
                continue
            match = re.fullmatch(r"\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.*?)\s*", assignment)
            if not match:
                raise SafetyError(f"Cannot safely parse INCAR assignment: {assignment!r}")
            result[match[1].upper()] = match[2]
    return result


def _static_incar(source: Path) -> str:
    kept = [
        f"{tag} = {value}"
        for tag, value in _incar_assignments(source).items()
        if tag not in _STATIC_OVERRIDES and tag not in _REMOVE_INCAR
        and not tag.startswith("ML_")
    ]
    while kept and not kept[-1].strip():
        kept.pop()
    kept.extend(
        [
            "",
            "# Set by InterfaceForge for derivative-probe DFT single points",
            *[f"{tag:<15} = {value}" for tag, value in _STATIC_OVERRIDES.items()],
        ]
    )
    return "\n".join(kept) + "\n"


def _template_payload(template: str | Path | None) -> dict[str, Any] | None:
    if template is None:
        return None
    root = Path(template).expanduser().resolve()
    required = [root / name for name in ("INCAR", "KPOINTS", "POTCAR")]
    missing = [path.name for path in required if not path.is_file() or not path.stat().st_size]
    if missing:
        raise SafetyError(
            f"VASP template {root} is missing non-empty files: {', '.join(missing)}"
        )
    potcar = (root / "POTCAR").read_bytes()
    potcar_elements = _POTCAR_ELEMENT.findall(
        potcar.decode("utf-8", errors="ignore")
    )
    if not potcar_elements:
        raise SafetyError(f"Could not read VRHFIN species order from {root / 'POTCAR'}")
    return {
        "root": root,
        "incar": _static_incar(root / "INCAR"),
        "kpoints": (root / "KPOINTS").read_bytes(),
        "potcar": potcar,
        "potcar_elements": potcar_elements,
        "launcher": next(
            (
                root / name
                for name in ("runvasp.sh", "submit.sh", "job.sh")
                if (root / name).is_file()
            ),
            None,
        ),
    }


def _write_case_inputs(directory: Path, atoms: Any, write: Any, template: dict[str, Any] | None) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=False)
    poscar = directory / "POSCAR"
    write(str(poscar), atoms, format="vasp", direct=True, vasp5=True, sort=False)
    if template is not None:
        (directory / "INCAR").write_text(template["incar"], encoding="utf-8")
        (directory / "KPOINTS").write_bytes(template["kpoints"])
        (directory / "POTCAR").write_bytes(template["potcar"])
        launcher = template["launcher"]
        if launcher is not None:
            target = directory / launcher.name
            shutil.copy2(launcher, target)
    names = ("POSCAR", "INCAR", "KPOINTS", "POTCAR")
    return {
        name: _sha256(directory / name)
        for name in names
        if (directory / name).is_file()
    }


def _case_id(strain: float, sample: int | None, sign: int) -> str:
    strain_tag = f"{strain:+.6f}".replace("+", "p").replace("-", "m").replace(".", "p")
    if sample is None:
        return f"strain_{strain_tag}_center"
    side = "plus" if sign > 0 else "minus"
    return f"strain_{strain_tag}_rattle_{sample:03d}_{side}"


def prepare_derivative_probe(
    entries: Sequence[str],
    output: str | Path,
    *,
    displacement_a: float = DEFAULT_DISPLACEMENT_A,
    strains: Sequence[float] = DEFAULT_VOLUME_STRAINS,
    rattles: int = 1,
    seed: int = DEFAULT_SEED,
    paired: bool = True,
    strain_mode: str = "volume",
    vasp_template: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Create deterministic strained centers and paired Cartesian rattles.

    The paper motivates the 0.03 Angstrom and +/-1% volume defaults. InterfaceForge
    adds paired +/- displacement vectors so directional curvature can be estimated
    by central differences; that paired construction is not claimed to be the
    paper's exact training-set protocol.
    """

    if displacement_a <= 0:
        raise ValueError("--displacement must be positive")
    if rattles < 1:
        raise ValueError("--rattles must be at least 1")
    if strain_mode not in {"volume", "linear"}:
        raise ValueError("--strain-mode must be 'volume' or 'linear'")
    unique_strains = tuple(dict.fromkeys(float(value) for value in strains))
    if not unique_strains:
        raise ValueError("Provide at least one strain")
    if any(value <= -1.0 for value in unique_strains):
        raise ValueError("Every strain must be greater than -1")

    parsed = [parse_probe_entry(value) for value in entries]
    labels = [label for label, _ in parsed]
    if len(labels) != len(set(labels)):
        raise SafetyError(f"Probe labels must be unique: {labels}")

    read, write = _ase_io()
    root = Path(output).expanduser().resolve()
    _prepare_output(root, force)
    template = _template_payload(vasp_template)
    rows: list[dict[str, Any]] = []
    extxyz_frames: list[Any] = []

    for source_index, (label, source) in enumerate(parsed):
        base = read(str(source), index=0)
        base.set_constraint()  # Full-coordinate PES probes, including previously fixed atoms.
        cell = np.asarray(base.cell.array, dtype=float)
        if len(base) == 0 or cell.shape != (3, 3) or abs(float(np.linalg.det(cell))) < 1e-10:
            raise SafetyError(f"Derivative probes require a non-empty periodic cell: {source}")
        if template is not None:
            symbols = base.get_chemical_symbols()
            species_blocks = [
                symbol
                for index, symbol in enumerate(symbols)
                if index == 0 or symbol != symbols[index - 1]
            ]
            if species_blocks != template["potcar_elements"]:
                raise SafetyError(
                    f"POTCAR order {template['potcar_elements']} does not match "
                    f"POSCAR species blocks {species_blocks} for {label}"
                )
        source_hash = _sha256(source)

        for strain in unique_strains:
            scale = (1.0 + strain) ** (1.0 / 3.0) if strain_mode == "volume" else 1.0 + strain
            center = base.copy()
            center.set_cell(cell * scale, scale_atoms=True)
            center_id = _case_id(strain, None, 0)
            center_dir = root / label / center_id
            hashes = _write_case_inputs(center_dir, center, write, template)
            row = {
                "structure_id": f"{label}/{center_id}",
                "label": label,
                "source": str(source),
                "source_sha256": source_hash,
                "relative_directory": str(center_dir.relative_to(root)),
                "kind": "center",
                "strain_fraction": strain,
                "strain_mode": strain_mode,
                "linear_scale": scale,
                "sample": None,
                "sign": 0,
                "displacement_component_stdev_a": 0.0,
                "displacement_vector_norm_a": 0.0,
                "natoms": len(center),
                "input_sha256": hashes,
            }
            rows.append(row)
            frame = center.copy()
            frame.info.update(
                {
                    "IF_probe_id": row["structure_id"],
                    "IF_probe_kind": "center",
                    "IF_probe_strain": strain,
                    "IF_probe_sign": 0,
                }
            )
            extxyz_frames.append(frame)

            for sample in range(1, rattles + 1):
                rng = np.random.default_rng(seed + source_index * 1_000_003 + sample)
                displacement = rng.normal(0.0, displacement_a, size=(len(center), 3))
                signs = (-1, 1) if paired else (1,)
                for sign in signs:
                    atoms = center.copy()
                    atoms.positions = np.asarray(center.positions) + sign * displacement
                    case_id = _case_id(strain, sample, sign)
                    directory = root / label / case_id
                    hashes = _write_case_inputs(directory, atoms, write, template)
                    row = {
                        "structure_id": f"{label}/{case_id}",
                        "label": label,
                        "source": str(source),
                        "source_sha256": source_hash,
                        "relative_directory": str(directory.relative_to(root)),
                        "kind": "rattle",
                        "strain_fraction": strain,
                        "strain_mode": strain_mode,
                        "linear_scale": scale,
                        "sample": sample,
                        "sign": sign,
                        "displacement_component_stdev_a": displacement_a,
                        "displacement_realized_rms_a": float(np.sqrt(np.mean(displacement**2))),
                        "displacement_vector_norm_a": float(np.linalg.norm(displacement)),
                        "natoms": len(atoms),
                        "input_sha256": hashes,
                    }
                    rows.append(row)
                    frame = atoms.copy()
                    frame.info.update(
                        {
                            "IF_probe_id": row["structure_id"],
                            "IF_probe_kind": "rattle",
                            "IF_probe_strain": strain,
                            "IF_probe_sample": sample,
                            "IF_probe_sign": sign,
                        }
                    )
                    extxyz_frames.append(frame)

    extxyz = root / "derivative_probe.extxyz"
    write(str(extxyz), extxyz_frames, format="extxyz")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "quantity": QUANTITY,
        "method": "finite displacement and homogeneous strain derivative probe",
        "citation": PAPER,
        "protocol": {
            "displacement_component_stdev_a": displacement_a,
            "strains": list(unique_strains),
            "strain_mode": strain_mode,
            "rattles_per_strain": rattles,
            "paired_displacements": paired,
            "constraint_policy": "full-coordinate displacements and raw forces",
            "seed": seed,
            "paper_derived_defaults": {
                "displacement_a": DEFAULT_DISPLACEMENT_A,
                "volume_strains": list(DEFAULT_VOLUME_STRAINS),
            },
            "interfaceforge_extension": (
                "Paired plus/minus displacement vectors enable central-difference "
                "directional-curvature checks; this is an InterfaceForge extension."
            ),
        },
        "vasp_template": (
            {
                "root": str(template["root"]),
                "static_incar_sha256": hashlib.sha256(
                    template["incar"].encode("utf-8")
                ).hexdigest(),
                "kpoints_sha256": hashlib.sha256(template["kpoints"]).hexdigest(),
                "potcar_sha256": hashlib.sha256(template["potcar"]).hexdigest(),
            }
            if template is not None
            else None
        ),
        "structures": rows,
        "extxyz": str(extxyz),
    }
    _json_write(root / "manifest.json", manifest)
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "structure_id",
            "label",
            "relative_directory",
            "kind",
            "strain_fraction",
            "strain_mode",
            "linear_scale",
            "sample",
            "sign",
            "displacement_component_stdev_a",
            "displacement_vector_norm_a",
            "natoms",
            "source",
            "source_sha256",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (root / "runs.txt").write_text(
        "".join(f"{row['relative_directory']}\n" for row in rows),
        encoding="utf-8",
    )
    return {
        "status": "PREPARED",
        "output": str(root),
        "structures": len(rows),
        "sources": len(parsed),
        "manifest": str(root / "manifest.json"),
        "extxyz": str(extxyz),
        "vasp_ready": template is not None,
        "citation": PAPER,
    }


def _verify_inputs(root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        directory = root / str(row["relative_directory"])
        for name, expected in row.get("input_sha256", {}).items():
            path = directory / name
            if not path.is_file() or _sha256(path) != expected:
                raise SafetyError(
                    f"Probe input changed since prepare: {path}. "
                    "Prepare a new tree rather than mixing protocols."
                )


def _model_label(backend: str, path: Path, used: set[str]) -> str:
    base = f"{backend}:{path.parent.name}/{path.stem}"
    label = base
    counter = 2
    while label in used:
        label = f"{base}#{counter}"
        counter += 1
    used.add(label)
    return label


def _calculators(
    mace_models: Sequence[str | Path],
    deepmd_models: Sequence[str | Path],
    device: str,
    mace_dtype: str = "float64",
) -> dict[str, Any]:
    calculators: dict[str, Any] = {}
    used: set[str] = set()
    if mace_models:
        try:
            from mace.calculators import MACECalculator
        except ModuleNotFoundError as exc:
            raise DependencyError("MACE models requested but mace-torch is unavailable") from exc
        for value in mace_models:
            path = Path(value).expanduser().resolve()
            if not path.is_file():
                raise SafetyError(f"Missing MACE model: {path}")
            label = _model_label("mace", path, used)
            calculators[label] = MACECalculator(
                model_paths=str(path), device=device, default_dtype=mace_dtype
            )
    if deepmd_models:
        try:
            from deepmd.calculator import DP
        except ModuleNotFoundError as exc:
            raise DependencyError("DeePMD models requested but deepmd-kit is unavailable") from exc
        for value in deepmd_models:
            path = Path(value).expanduser().resolve()
            if not path.is_file():
                raise SafetyError(f"Missing DeePMD model: {path}")
            label = _model_label("deepmd", path, used)
            calculators[label] = DP(model=str(path))
    return calculators


def _prediction(atoms: Any, calculator: Any) -> tuple[float, np.ndarray, np.ndarray | None]:
    probe = atoms.copy()
    probe.calc = calculator
    energy = float(probe.get_potential_energy())
    forces = np.asarray(probe.get_forces(apply_constraint=False), dtype=float)
    stress: np.ndarray | None
    try:
        stress = np.asarray(probe.get_stress(voigt=True, apply_constraint=False), dtype=float)
    except (NotImplementedError, RuntimeError, ValueError):
        stress = None
    return energy, forces, stress


def _metrics(reference: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = np.asarray(predicted, dtype=float) - np.asarray(reference, dtype=float)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_abs": float(np.max(np.abs(error))),
        "bias": float(np.mean(error)),
    }


FORCE_EXPORT_SCHEMA = "interfaceforge.derivative_probe.forces/1"


def _export_predictions(
    path: Path, values: Mapping[tuple[str, str], Mapping[str, Any]]
) -> dict[str, Any]:
    """Write every per-structure force array to a pickle-free ``.npz``.

    ``np.load(path)`` reconstructs the arrays without ``allow_pickle``; the
    parallel ``model``/``structure_id`` string arrays give each ``forces_<i>``
    array an unambiguous owner, so the reported force errors can be recomputed
    without rerunning inference.
    """

    keys = sorted(values)
    arrays: dict[str, Any] = {}
    entries: list[dict[str, Any]] = []
    for index, (model, structure_id) in enumerate(keys):
        value = values[(model, structure_id)]
        forces = np.asarray(value["forces"], dtype=float)
        force_key = f"forces_{index:06d}"
        arrays[force_key] = forces
        stress = value["stress"]
        stress_key = None
        if stress is not None:
            stress_key = f"stress_{index:06d}"
            arrays[stress_key] = np.asarray(stress, dtype=float)
        entries.append(
            {
                "index": index,
                "model": model,
                "structure_id": structure_id,
                "natoms": int(value["natoms"]),
                "forces_key": force_key,
                "forces_shape": [int(size) for size in forces.shape],
                "stress_key": stress_key,
            }
        )
    np.savez_compressed(
        path,
        schema=np.asarray(FORCE_EXPORT_SCHEMA),
        model=np.asarray([model for model, _ in keys], dtype=np.str_),
        structure_id=np.asarray([sid for _, sid in keys], dtype=np.str_),
        natoms=np.asarray([entry["natoms"] for entry in entries], dtype=np.int64),
        energy_ev=np.asarray(
            [values[key]["energy_ev"] for key in keys], dtype=float
        ),
        has_stress=np.asarray(
            [entry["stress_key"] is not None for entry in entries], dtype=bool
        ),
        forces_units=np.asarray("eV/Angstrom"),
        stress_units=np.asarray("eV/Angstrom^3"),
        energy_units=np.asarray("eV"),
        **arrays,
    )
    return {
        "path": str(path),
        "schema": FORCE_EXPORT_SCHEMA,
        "format": "numpy .npz (loadable with np.load without allow_pickle)",
        "units": {
            "forces": "eV/Angstrom",
            "stress": "eV/Angstrom^3 (Voigt xx, yy, zz, yz, xz, xy)",
            "energy": "eV",
        },
        "force_array_layout": "(natoms, 3) Cartesian, POSCAR atom order, raw (unconstrained)",
        "entries": entries,
        "sha256": _sha256(path),
    }


def _response_key(row: Mapping[str, Any]) -> tuple[str, float, int]:
    return (str(row["label"]), float(row["strain_fraction"]), int(row["sample"]))


def _summarize_model(
    model: str,
    rows: Sequence[Mapping[str, Any]],
    values: Mapping[tuple[str, str], Mapping[str, Any]],
    row_by_key: Mapping[tuple[str, float, Any, int], Mapping[str, Any]],
    response_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Compare one model with DFT, metric by metric.

    Each metric depends only on its own prerequisites: forces need a matched
    DFT/MLIP pair, stress additionally needs stress from both sides,
    relative energies need that source's zero-strain center, and curvatures
    need a complete minus/center/plus triplet. A missing prerequisite omits
    only the metrics that need it and is explained in ``omitted_metrics``.
    """

    matched = [
        row
        for row in rows
        if ("DFT", str(row["structure_id"])) in values
        and (model, str(row["structure_id"])) in values
    ]
    if not matched:
        return None

    omitted: dict[str, str] = {}
    notes: dict[str, list[str]] = {}

    summary: dict[str, Any] = {
        "model": model,
        # Retained meaning: probe structures with both a DFT and an MLIP result.
        "matched_structures": len(matched),
    }

    # Forces: every matched DFT/MLIP structure.
    dft_force = np.concatenate(
        [values[("DFT", str(row["structure_id"]))]["forces"].reshape(-1) for row in matched]
    )
    model_force = np.concatenate(
        [values[(model, str(row["structure_id"]))]["forces"].reshape(-1) for row in matched]
    )
    summary["force_ev_a"] = _metrics(dft_force, model_force)
    summary["force_matched_structures"] = len(matched)
    summary["force_components"] = int(dft_force.size)

    # Stress: matched structures that carry stress on both sides.
    stress_matched = [
        row
        for row in matched
        if values[("DFT", str(row["structure_id"]))]["stress"] is not None
        and values[(model, str(row["structure_id"]))]["stress"] is not None
    ]
    if stress_matched:
        summary["stress_ev_a3"] = _metrics(
            np.concatenate(
                [values[("DFT", str(row["structure_id"]))]["stress"] for row in stress_matched]
            ),
            np.concatenate(
                [values[(model, str(row["structure_id"]))]["stress"] for row in stress_matched]
            ),
        )
        summary["stress_matched_structures"] = len(stress_matched)
        if len(stress_matched) < len(matched):
            notes.setdefault("stress_ev_a3", []).append(
                f"{len(matched) - len(stress_matched)} matched structure(s) lack stress "
                "from DFT or from the model and are excluded."
            )
    else:
        summary["stress_matched_structures"] = 0
        omitted["stress_ev_a3"] = (
            "No matched structure provides stress from both DFT and the model."
        )

    # Relative energies: need this source's zero-strain center on both sides.
    energy_matched: list[tuple[Mapping[str, Any], str]] = []
    dropped_sources: dict[str, str] = {}
    for row in matched:
        label = str(row["label"])
        center = row_by_key.get((label, 0.0, None, 0))
        if center is None:
            dropped_sources[label] = "no zero-strain center was prepared for this source"
            continue
        center_id = str(center["structure_id"])
        if ("DFT", center_id) not in values or (model, center_id) not in values:
            dropped_sources[label] = (
                "the zero-strain center has no matched DFT and model result"
            )
            continue
        energy_matched.append((row, center_id))
    if energy_matched:
        dft_delta = [
            (
                values[("DFT", str(row["structure_id"]))]["energy_ev"]
                - values[("DFT", center_id)]["energy_ev"]
            )
            / values[("DFT", str(row["structure_id"]))]["natoms"]
            for row, center_id in energy_matched
        ]
        model_delta = [
            (
                values[(model, str(row["structure_id"]))]["energy_ev"]
                - values[(model, center_id)]["energy_ev"]
            )
            / values[(model, str(row["structure_id"]))]["natoms"]
            for row, center_id in energy_matched
        ]
        summary["relative_energy_mev_atom"] = {
            key: value * 1000.0
            for key, value in _metrics(
                np.asarray(dft_delta), np.asarray(model_delta)
            ).items()
        }
        summary["relative_energy_matched_structures"] = len(energy_matched)
        if dropped_sources:
            notes.setdefault("relative_energy_mev_atom", []).extend(
                f"Source {label!r} excluded: {reason}."
                for label, reason in sorted(dropped_sources.items())
            )
    else:
        summary["relative_energy_matched_structures"] = 0
        detail = "; ".join(
            f"{label}: {reason}" for label, reason in sorted(dropped_sources.items())
        )
        omitted["relative_energy_mev_atom"] = (
            "No source has a usable zero-strain energy reference"
            + (f" ({detail})." if detail else ".")
        )

    # Curvatures: complete minus/center/plus triplets for DFT and the model.
    dft_responses = {
        _response_key(row): row for row in response_rows if row["model"] == "DFT"
    }
    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = [
        (dft_responses[_response_key(row)], row)
        for row in response_rows
        if row["model"] == model and _response_key(row) in dft_responses
    ]
    for field, count_field, name in (
        ("energy_curvature_ev_a2", "matched_energy_curvatures", "energy-derived"),
        ("force_curvature_ev_a2", "matched_force_curvatures", "force-derived"),
    ):
        summary[count_field] = len(pairs)
        if pairs:
            summary[field] = _metrics(
                np.asarray([reference[field] for reference, _ in pairs]),
                np.asarray([predicted[field] for _, predicted in pairs]),
            )
        else:
            omitted[field] = (
                "No symmetric displacement pair has a complete minus/center/plus "
                f"triplet for both DFT and the model, so no {name} curvature "
                "could be compared."
            )
    if pairs:
        # Compatibility alias for schema_version 1 readers; a copy so that a
        # consumer editing one view cannot silently change the other.
        summary["directional_curvature_ev_a2"] = dict(summary["force_curvature_ev_a2"])
        summary["matched_curvatures"] = len(pairs)
    else:
        summary["matched_curvatures"] = 0
    summary["curvature_alias"] = (
        "directional_curvature_ev_a2 == force_curvature_ev_a2"
    )

    if omitted:
        summary["omitted_metrics"] = omitted
    if notes:
        summary["metric_notes"] = notes
    return summary


_DFT_TAGS = (
    "ENCUT", "EDIFF", "ISMEAR", "SIGMA", "ISPIN", "GGA", "METAGGA",
    "LHFCALC", "AEXX", "HFSCREEN", "LDAU", "LDAUTYPE", "IVDW",
    "IBRION", "NSW", "ISYM", "ML_LMLFF",
)


def _validate_dft(directory: Path, expected: Any, actual: Any) -> dict[str, Any]:
    """Accept only converged static results on the indexed probe geometry."""
    from ase.geometry import find_mic

    from .dft_evidence import _same_setting
    from .vasp_provenance import _outcar_fingerprint

    outcar = directory / "OUTCAR"
    text = outcar.read_text(errors="replace")
    if expected.get_chemical_symbols() != actual.get_chemical_symbols():
        raise SafetyError(f"OUTCAR species/order differs from POSCAR: {directory}")
    if not np.allclose(expected.cell.array, actual.cell.array, atol=1e-5, rtol=0):
        raise SafetyError(f"OUTCAR cell differs from POSCAR: {directory}")
    _, distances = find_mic(actual.positions - expected.positions, expected.cell, pbc=expected.pbc)
    if not np.all(np.isfinite(distances)) or np.max(distances) > 1e-5:
        raise SafetyError(f"OUTCAR positions differ from POSCAR: {directory}")
    identity = _outcar_fingerprint(outcar, tracked_tags=_DFT_TAGS)
    if identity["ionic_frames_detected"] != 1:
        raise SafetyError(f"Expected exactly one static force frame: {directory}")
    if "General timing and accounting informations" not in text:
        raise SafetyError(f"OUTCAR lacks normal termination evidence: {directory}")
    if "aborting loop because EDIFF is reached" not in text:
        raise SafetyError(f"OUTCAR lacks electronic convergence evidence: {directory}")
    executed = identity["outcar_executed_tags"]
    for tag, value in (("IBRION", "-1"), ("NSW", "0"), ("ISYM", "0")):
        if tag not in executed or not _same_setting(executed[tag], value):
            raise SafetyError(f"OUTCAR must demonstrate {tag}={value}: {directory}")
    if executed.get("ML_LMLFF", "F").upper() in {"T", ".TRUE.", "TRUE"}:
        raise SafetyError(f"OUTCAR used VASP ML forces: {directory}")
    inputs = _incar_assignments(directory / "INCAR") if (directory / "INCAR").is_file() else {}
    missing = []
    for tag in _DFT_TAGS:
        if tag in inputs:
            if tag not in executed:
                missing.append(f"Executed {tag} unavailable")
            elif not _same_setting(inputs[tag], executed[tag]):
                raise SafetyError(f"INCAR/OUTCAR {tag} differs: {directory}")
    if not inputs:
        missing.append("INCAR unavailable; executed/input consistency not checked")
    if "ENCUT" not in executed:
        missing.append("Executed ENCUT unavailable")
    # The shared scalar fingerprint does not establish Hubbard arrays or spin initialization.
    if any(tag in inputs for tag in ("LDAUL", "LDAUU", "LDAUJ")):
        missing.append("Species-resolved Hubbard parameters require review")
    if not (directory / "KPOINTS").is_file():
        missing.append("KPOINTS unavailable; k-point input identity not checked")
    if not identity["nkpts"]:
        missing.append("Executed NKPTS unavailable")
    potcar = directory / "POTCAR"
    if potcar.is_file():
        titles = re.findall(r"TITEL\s*=\s*([^\n]+)", potcar.read_text(errors="replace"))
        titles = list(dict.fromkeys(title.strip() for title in titles))
        if not titles or not identity["potcar_titles"]:
            missing.append("POTCAR title identity unavailable")
        elif titles != identity["potcar_titles"]:
            raise SafetyError(f"POTCAR/OUTCAR titles differ: {directory}")
    else:
        missing.append("POTCAR unavailable; potential identity not checked")
    return {
        "outcar_sha256": identity["outcar_sha256"],
        "geometry": "PASS", "electronic_convergence": "PASS", "termination": "PASS",
        "missing_evidence": missing,
        "executed_identity": {key: identity[key] for key in (
            "outcar_executed_tags", "potcar_titles", "nkpts", "vasp_version")},
        "input_sha256": {name: _sha256(directory / name)
                         for name in ("INCAR", "KPOINTS", "POTCAR")
                         if (directory / name).is_file()},
    }


def evaluate_derivative_probe(
    root: str | Path,
    *,
    mace_models: Sequence[str | Path] = (),
    deepmd_models: Sequence[str | Path] = (),
    device: str = "cpu",
    output_stem: str = "derivative_probe",
    mace_dtype: str = "float64",
    _test_calculators: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect completed DFT points and evaluate MLIPs on the identical probe set."""

    if mace_dtype not in {"float32", "float64"}:
        raise ValueError("mace_dtype must be float32 or float64")
    if not output_stem or not _LABEL.fullmatch(output_stem):
        raise ValueError(
            "--output-stem must use only letters, numbers, '.', '_' or '-'"
        )
    probe_root = Path(root).expanduser().resolve()
    manifest_path = probe_root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SafetyError(f"Invalid derivative-probe manifest: {manifest_path}") from exc
    if manifest.get("quantity") != QUANTITY:
        raise SafetyError(f"Not a derivative-probe tree: {probe_root}")
    rows = manifest.get("structures", [])
    if not rows:
        raise SafetyError(f"No structures in {manifest_path}")
    _verify_inputs(probe_root, rows)
    read, _ = _ase_io()
    calculators = dict(_test_calculators or {})
    calculators.update(_calculators(mace_models, deepmd_models, device, mace_dtype))

    values: dict[tuple[str, str], dict[str, Any]] = {}
    prediction_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    dft_completed = 0
    dft_evidence: dict[str, Any] = {}
    executed_by_label: dict[str, dict[str, Any]] = {}
    inputs_by_label: dict[str, dict[str, str]] = {}

    for row in rows:
        structure_id = str(row["structure_id"])
        directory = probe_root / str(row["relative_directory"])
        poscar_atoms = read(str(directory / "POSCAR"), index=0)
        outcar = directory / "OUTCAR"
        sources: dict[str, tuple[float, np.ndarray, np.ndarray | None]] = {}
        if outcar.is_file() and outcar.stat().st_size:
            try:
                dft_atoms = read(str(outcar), index=-1)
                evidence = _validate_dft(directory, poscar_atoms, dft_atoms)
                dft_evidence[structure_id] = evidence
                identity = evidence["executed_identity"]
                previous = executed_by_label.setdefault(str(row["label"]), identity)
                if identity != previous:
                    raise SafetyError(f"Executed DFT settings differ across probes for {row['label']}")
                # Missing files stay warnings; the first *available* hash of each
                # file name is the reference, so a probe that lacks one input
                # cannot hide a conflict between probes that do provide it.
                reference_inputs = inputs_by_label.setdefault(str(row["label"]), {})
                for name, digest in evidence["input_sha256"].items():
                    first = reference_inputs.setdefault(name, digest)
                    if first != digest:
                        raise SafetyError(f"{name} differs across probes for {row['label']}")
                warnings.extend(f"{structure_id}: {note}" for note in evidence["missing_evidence"])
                try:
                    dft_stress = np.asarray(
                        dft_atoms.get_stress(voigt=True, apply_constraint=False), dtype=float
                    )
                except (NotImplementedError, RuntimeError, ValueError):
                    dft_stress = None
                sources["DFT"] = (
                    float(dft_atoms.get_potential_energy()),
                    np.asarray(dft_atoms.get_forces(apply_constraint=False), dtype=float),
                    dft_stress,
                )
                dft_completed += 1
            except SafetyError:
                raise
            except Exception as exc:
                warnings.append(f"Could not parse {outcar}: {exc}")
        for label, calculator in calculators.items():
            sources[label] = _prediction(poscar_atoms, calculator)

        for model, (energy, forces, stress) in sources.items():
            if not np.isfinite(energy) or not np.all(np.isfinite(forces)) or (
                stress is not None and (stress.shape != (6,) or not np.all(np.isfinite(stress)))
            ):
                raise SafetyError(f"Non-finite or invalid prediction for {model} on {structure_id}")
            if forces.shape != (int(row["natoms"]), 3):
                raise SafetyError(
                    f"Unexpected force shape for {model} on {structure_id}: {forces.shape}"
                )
            value = {
                "energy_ev": energy,
                "forces": forces,
                "stress": stress,
                "natoms": int(row["natoms"]),
            }
            values[(model, structure_id)] = value
            output = {
                "model": model,
                "structure_id": structure_id,
                "label": row["label"],
                "kind": row["kind"],
                "strain_fraction": row["strain_fraction"],
                "sample": row["sample"],
                "sign": row["sign"],
                "energy_ev": energy,
                "force_rms_ev_a": float(np.sqrt(np.mean(forces**2))),
                "force_max_ev_a": float(np.max(np.linalg.norm(forces, axis=1))),
            }
            if stress is not None:
                for index, component in enumerate(stress):
                    output[f"stress_{index}_ev_a3"] = float(component)
            prediction_rows.append(output)

    models = sorted({model for model, _ in values})
    response_rows: list[dict[str, Any]] = []
    row_by_key = {
        (
            str(row["label"]),
            float(row["strain_fraction"]),
            row.get("sample"),
            int(row["sign"]),
        ): row
        for row in rows
    }
    for model in models:
        for row in rows:
            if row["kind"] != "rattle" or int(row["sign"]) != 1:
                continue
            key = (str(row["label"]), float(row["strain_fraction"]), row["sample"])
            center = row_by_key.get((*key[:2], None, 0))
            minus = row_by_key.get((*key, -1))
            if center is None or minus is None:
                continue
            plus_value = values.get((model, str(row["structure_id"])))
            minus_value = values.get((model, str(minus["structure_id"])))
            center_value = values.get((model, str(center["structure_id"])))
            if plus_value is None or minus_value is None or center_value is None:
                continue
            plus_atoms = read(
                str(probe_root / str(row["relative_directory"]) / "POSCAR"), index=0
            )
            minus_atoms = read(
                str(probe_root / str(minus["relative_directory"]) / "POSCAR"), index=0
            )
            vector = (
                np.asarray(plus_atoms.positions) - np.asarray(minus_atoms.positions)
            ) / 2.0
            q = float(np.linalg.norm(vector))
            if q <= 0:
                raise SafetyError(f"Zero displacement vector for {row['structure_id']}")
            unit = vector.reshape(-1) / q
            energy_curvature = (
                plus_value["energy_ev"]
                + minus_value["energy_ev"]
                - 2.0 * center_value["energy_ev"]
            ) / (q * q)
            force_curvature = -float(
                np.dot(
                    plus_value["forces"].reshape(-1)
                    - minus_value["forces"].reshape(-1),
                    unit,
                )
            ) / (2.0 * q)
            response_rows.append(
                {
                    "model": model,
                    "label": row["label"],
                    "strain_fraction": row["strain_fraction"],
                    "sample": row["sample"],
                    "displacement_vector_norm_a": q,
                    "energy_curvature_ev_a2": energy_curvature,
                    "force_curvature_ev_a2": force_curvature,
                    "curvature_internal_delta_ev_a2": force_curvature
                    - energy_curvature,
                }
            )

    summaries: list[dict[str, Any]] = []
    if "DFT" in models:
        for model in [item for item in models if item != "DFT"]:
            summary = _summarize_model(model, rows, values, row_by_key, response_rows)
            if summary is not None:
                summaries.append(summary)

    def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
        if not records:
            path.write_text("", encoding="utf-8")
            return
        fields = list(dict.fromkeys(key for record in records for key in record))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, restval="")
            writer.writeheader()
            writer.writerows(records)

    predictions_path = probe_root / f"{output_stem}_predictions.csv"
    responses_path = probe_root / f"{output_stem}_responses.csv"
    results_path = probe_root / f"{output_stem}_results.json"
    arrays_path = probe_root / f"{output_stem}_arrays.npz"
    write_csv(predictions_path, prediction_rows)
    write_csv(responses_path, response_rows)
    force_export = _export_predictions(arrays_path, values)
    status = "INCOMPLETE" if dft_completed != len(rows) else "CHECK" if warnings else "COMPLETE"
    packages = {}
    for name in ("interfaceforge", "numpy", "ase", "mace-torch", "deepmd-kit", "torch"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    model_records = [
        {"backend": backend, "path": str(Path(path).expanduser().resolve()),
         "sha256": _sha256(Path(path).expanduser().resolve())}
        for backend, paths in (("mace", mace_models), ("deepmd", deepmd_models))
        for path in paths
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "quantity": QUANTITY,
        "status": status,
        "root": str(probe_root),
        "structures": len(rows),
        "dft_completed": dft_completed,
        "dft_evidence": dft_evidence,
        "provenance": {
            "manifest_sha256": _sha256(manifest_path),
            "implementation_sha256": _sha256(Path(__file__)),
            "models": model_records,
            "packages": packages,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mace_device": device,
            "mace_dtype": mace_dtype,
            "deepmd_device": "backend-managed",
            "constraint_policy": "raw forces; full-coordinate displacements",
        },
        "models": models,
        "summaries": summaries,
        "curvature_definitions": CURVATURE_DEFINITIONS,
        "force_export": force_export,
        "warnings": warnings,
        "citation": PAPER,
        "output_stem": output_stem,
        "outputs": {
            "predictions": str(predictions_path),
            "responses": str(responses_path),
            "summary": str(results_path),
            "arrays": str(arrays_path),
        },
    }
    _json_write(results_path, payload)
    return payload
