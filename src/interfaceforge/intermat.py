"""Optional geometry-only adapter for InterMat crystalline interfaces."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .errors import DependencyError, SafetyError
from .state import sha256_file

ADAPTER_ID = "interfaceforge.intermat"
MINIMUM_VERSION = "2024.3.24"
_GENERATED_PATTERN = re.compile(r"^interface_[0-9]{4}\.vasp$")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def intermat_status() -> dict[str, Any]:
    """Report optional dependency availability without importing heavy modules."""

    installed = importlib.util.find_spec("intermat") is not None
    jarvis = importlib.util.find_spec("jarvis") is not None
    spglib = importlib.util.find_spec("spglib") is not None
    return {
        "adapter": ADAPTER_ID,
        "available": installed and jarvis and spglib,
        "intermat_version": _package_version("intermat"),
        "jarvis_tools_version": _package_version("jarvis-tools"),
        "spglib_version": _package_version("spglib"),
        "minimum_tested_intermat_version": MINIMUM_VERSION,
        "capabilities": [
            "crystalline film/substrate lattice matching",
            "surface construction from bulk inputs",
            "separation and lateral-registry candidate generation",
            "POSCAR and InterfaceForge systems-fragment export",
        ],
        "excluded": [
            "calculator execution",
            "VASP input or scheduler generation",
            "molecule-on-surface adsorption",
            "automatic campaign mutation",
        ],
        "install": "pip install 'interfaceforge[intermat]'",
    }


def _runtime() -> dict[str, Any]:
    try:
        import spglib  # noqa: F401 - required indirectly by JARVIS surface generation
        from intermat.generate import InterfaceCombi
        from jarvis.core.atoms import Atoms
    except (ImportError, ModuleNotFoundError) as exc:
        raise DependencyError(
            "InterMat and JARVIS-Tools are required. Install with: "
            "pip install 'interfaceforge[intermat]'"
        ) from exc
    return {"InterfaceCombi": InterfaceCombi, "Atoms": Atoms}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_miller(value: list[int] | tuple[int, ...], label: str) -> list[int]:
    if len(value) != 3 or all(int(item) == 0 for item in value):
        raise ValueError(f"{label} must contain three integers and cannot be 0 0 0")
    return [int(item) for item in value]


def _registry_count(interval: float) -> int:
    if interval == 0:
        return 1
    if not math.isfinite(interval) or interval <= 0 or interval > 1:
        raise ValueError("displacement_interval must be 0 or a finite value in (0, 1]")
    # Match InterMat's inclusive np.mgrid convention. Periodic 1.0 endpoints
    # are later removed by structure fingerprinting.
    points = len(np.arange(0.0, 1.0 + interval, interval))
    return points * points


def _validate_output(root: Path, force: bool) -> None:
    if root == Path.cwd().resolve() or len(root.parts) < 3:
        raise SafetyError(f"Unsafe InterMat output directory: {root}")
    if not root.exists() or not any(root.iterdir()):
        return
    if not force:
        raise SafetyError(
            f"InterMat output directory is not empty: {root}; use --force only to replace a prior adapter export"
        )
    manifest_path = root / "intermat_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SafetyError(
            f"Refusing --force because {root} is not a recognized InterfaceForge InterMat export"
        ) from exc
    if manifest.get("adapter") != ADAPTER_ID:
        raise SafetyError(f"Refusing --force for unrecognized output directory: {root}")


def _prepare_output(root: Path, force: bool) -> None:
    _validate_output(root, force)
    root.mkdir(parents=True, exist_ok=True)
    if not force:
        return
    structures = root / "structures"
    if structures.is_dir():
        for path in structures.iterdir():
            if path.is_file() and _GENERATED_PATTERN.fullmatch(path.name):
                path.unlink()
    for name in ("intermat_manifest.json", "campaign_fragment.yaml"):
        path = root / name
        if path.is_file():
            path.unlink()


def _structure_fingerprint(atoms: Any) -> str:
    coordinates = np.mod(np.asarray(atoms.frac_coords, dtype=float), 1.0)
    coordinates[np.isclose(coordinates, 1.0, atol=1e-10)] = 0.0
    sites = sorted(
        (str(element), *[round(float(item), 10) for item in coordinate])
        for element, coordinate in zip(atoms.elements, coordinates, strict=True)
    )
    payload = {
        "lattice": np.round(np.asarray(atoms.lattice_mat, dtype=float), 10).tolist(),
        "sites": sites,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _candidate_coordinates(name: str) -> tuple[float | None, list[float] | None]:
    match = re.search(
        r"_seperation_([-+0-9.eE]+)_disp_([-+0-9.eE]+)_([-+0-9.eE]+)$",
        name,
    )
    if not match:
        return None, None
    return float(match.group(1)), [float(match.group(2)), float(match.group(3))]


def generate_intermat_interfaces(
    film: str | Path,
    substrate: str | Path,
    output: str | Path,
    *,
    film_miller: list[int] | tuple[int, ...] = (0, 0, 1),
    substrate_miller: list[int] | tuple[int, ...] = (0, 0, 1),
    film_thickness: float = 16.0,
    substrate_thickness: float = 16.0,
    separations: list[float] | tuple[float, ...] = (2.5,),
    vacuum: float = 12.0,
    displacement_interval: float = 0.0,
    max_area: float = 300.0,
    length_tolerance: float = 0.08,
    angle_tolerance: float = 1.0,
    apply_strain: bool = False,
    use_conventional_film: bool = True,
    use_conventional_substrate: bool = True,
    max_candidates: int = 500,
    force: bool = False,
) -> dict[str, Any]:
    """Generate crystalline interface candidates while keeping calculators in InterfaceForge."""

    film_path = Path(film).expanduser().resolve()
    substrate_path = Path(substrate).expanduser().resolve()
    for label, path in (("film", film_path), ("substrate", substrate_path)):
        if not path.is_file():
            raise FileNotFoundError(f"InterMat {label} bulk structure does not exist: {path}")
    film_index = _validate_miller(film_miller, "film_miller")
    substrate_index = _validate_miller(substrate_miller, "substrate_miller")
    if film_thickness <= 0 or substrate_thickness <= 0:
        raise ValueError("film_thickness and substrate_thickness must be positive")
    separation_values = [float(item) for item in separations]
    if not separation_values or any(not math.isfinite(item) or item <= 0 for item in separation_values):
        raise ValueError("separations must contain finite positive values")
    for name, value in (
        ("vacuum", vacuum),
        ("max_area", max_area),
        ("length_tolerance", length_tolerance),
        ("angle_tolerance", angle_tolerance),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    estimated = _registry_count(float(displacement_interval)) * len(separation_values)
    if estimated > max_candidates:
        raise SafetyError(
            f"Requested InterMat scan would create approximately {estimated} candidates, "
            f"above --max-candidates={max_candidates}"
        )

    output_root = Path(output).expanduser().resolve()
    _validate_output(output_root, force)

    runtime = _runtime()
    atoms_type = runtime["Atoms"]
    try:
        film_atoms = atoms_type.from_poscar(str(film_path))
        substrate_atoms = atoms_type.from_poscar(str(substrate_path))
        generator = runtime["InterfaceCombi"](
            film_mats=[film_atoms],
            subs_mats=[substrate_atoms],
            film_ids=[film_path.stem],
            subs_ids=[substrate_path.stem],
            film_kplengths=[0],
            subs_kplengths=[0],
            film_indices=[film_index],
            subs_indices=[substrate_index],
            film_thicknesses=[float(film_thickness)],
            subs_thicknesses=[float(substrate_thickness)],
            seperations=separation_values,
            disp_intvl=float(displacement_interval),
            vacuum_interface=float(vacuum),
            max_area=float(max_area),
            ltol=float(length_tolerance),
            atol=float(angle_tolerance),
            apply_strain=bool(apply_strain),
            lowest_mismatch=True,
            from_conventional_structure_film=bool(use_conventional_film),
            from_conventional_structure_subs=bool(use_conventional_substrate),
            generated_interfaces=[],
            # InterMat otherwise downloads its full default JARVIS dataset even
            # though both local structures are already supplied.
            dataset=[{"jid": "__interfaceforge_local__"}],
            working_dir=str(output_root),
        )
        generated = generator.generate()
    except (IndexError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise SafetyError(
            "InterMat could not construct this crystalline interface. Check that both inputs are "
            "valid bulk POSCARs, then review Miller indices, max area, and tolerances. "
            f"Upstream error: {exc}"
        ) from exc
    if not generated:
        raise SafetyError(
            "InterMat found no commensurate interface. Increase --max-area or tolerances intentionally."
        )

    _prepare_output(output_root, force)
    structures_root = output_root / "structures"
    structures_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicates = 0
    for source_index, candidate in enumerate(generated):
        atoms = atoms_type.from_dict(candidate["generated_interface"])
        fingerprint = _structure_fingerprint(atoms)
        if fingerprint in seen:
            duplicates += 1
            continue
        seen.add(fingerprint)
        candidate_index = len(records)
        filename = f"interface_{candidate_index:04d}.vasp"
        path = structures_root / filename
        atoms.write_poscar(str(path))
        name = str(candidate.get("interface_name", ""))
        separation, displacement = _candidate_coordinates(name)
        records.append(
            {
                "id": f"intermat_interface_{candidate_index:04d}",
                "file": str(path.relative_to(output_root)),
                "sha256": sha256_file(path),
                "fingerprint": fingerprint,
                "source_index": source_index,
                "intermat_name": name,
                "separation_a": separation,
                "fractional_displacement": displacement,
                "mismatch_u": _jsonable(candidate.get("mismatch_u")),
                "mismatch_v": _jsonable(candidate.get("mismatch_v")),
                "mismatch_angle_deg": _jsonable(candidate.get("mismatch_angle")),
                "film_area_a2": _jsonable(candidate.get("area2")),
                "substrate_area_a2": _jsonable(candidate.get("area1")),
                "natoms": int(getattr(atoms, "num_atoms", len(atoms.elements))),
            }
        )

    fragment = {
        "systems": [
            {
                "id": record["id"],
                "kind": "interface",
                "structure": record["file"],
                "tags": {
                    "generator": "intermat",
                    "separation_a": record["separation_a"],
                    "fractional_displacement": record["fractional_displacement"],
                    "mismatch_u": record["mismatch_u"],
                    "mismatch_v": record["mismatch_v"],
                },
            }
            for record in records
        ]
    }
    fragment_path = output_root / "campaign_fragment.yaml"
    fragment_path.write_text(yaml.safe_dump(fragment, sort_keys=False), encoding="utf-8")

    manifest = {
        "adapter": ADAPTER_ID,
        "adapter_schema": 1,
        "intermat_version": _package_version("intermat"),
        "jarvis_tools_version": _package_version("jarvis-tools"),
        "inputs": {
            "film": str(film_path),
            "film_sha256": sha256_file(film_path),
            "substrate": str(substrate_path),
            "substrate_sha256": sha256_file(substrate_path),
        },
        "parameters": {
            "film_miller": film_index,
            "substrate_miller": substrate_index,
            "film_thickness_a": float(film_thickness),
            "substrate_thickness_a": float(substrate_thickness),
            "separations_a": separation_values,
            "vacuum_a": float(vacuum),
            "displacement_interval_fractional": float(displacement_interval),
            "max_area_a2": float(max_area),
            "length_tolerance": float(length_tolerance),
            "angle_tolerance_deg": float(angle_tolerance),
            "apply_strain": bool(apply_strain),
            "use_conventional_film": bool(use_conventional_film),
            "use_conventional_substrate": bool(use_conventional_substrate),
        },
        "estimated_candidates": estimated,
        "raw_candidates": len(generated),
        "unique_candidates": len(records),
        "periodic_duplicates_removed": duplicates,
        "campaign_fragment": str(fragment_path.relative_to(output_root)),
        "candidates": records,
        "scientific_caution": (
            "Candidates are unrelaxed crystalline registries. Inspect terminations, strain, polarity, "
            "magnetism, and stoichiometry before creating VASP calculations."
        ),
    }
    manifest_path = output_root / "intermat_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "output": str(output_root),
        "manifest": str(manifest_path),
        "campaign_fragment": str(fragment_path),
        "raw_candidates": len(generated),
        "unique_candidates": len(records),
        "periodic_duplicates_removed": duplicates,
    }
