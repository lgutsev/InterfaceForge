"""Canonical NiO (+ phosphonate) AIMD dataset export with leakage-safe splits.

The NiO campaigns are organised as VASP stage trees below one or more project
heads::

    <head>/Step1/<case>/                     preheat (INCAR, POSCAR, OUTCAR[.gz], OSZICAR)
    <head>/Step2_300K/<case>/                production AIMD at 300 K
    <head>/Step2_450K/<case>/                ... (step2_sample.json in each Step2 root)

where ``<case>`` is e.g. ``OH50/NiO_m110_Big_U46_OH50_clustered_capped_Me4PACz_boundary``.
Every ``Step2_<T>K/<case>`` run starts from the ``Step1/<case>`` CONTCAR, so the
temperature series of one case is a single correlated lineage. The generic
``iface collect`` / leaf collectors group by parent directory and would put
``Step2_300K/<case>`` and ``Step2_450K/<case>`` into different splits; this
module therefore groups whole *cases* (and anything linked to them by an
identical starting structure or a CONTCAR -> POSCAR hand-off) before splitting.

One export writes the single canonical dataset consumed by every backend:

* ``{train,valid,test}.extxyz`` -- MACE and NequIP (``REF_energy`` /
  ``REF_forces``; exact float64 round trip via the packaging writer);
* ``deepmd/{train,valid,test}/<trajectory>/`` -- DeePMD/DPA systems with
  ``move_mask.npy``, ``system_meta.json`` and a ``frame_map.csv`` whose
  ``relative_leaf`` column matches the extxyz ``IF_leaf`` key, so
  ``iface mlip-compare`` can prove matched-frame membership;
* ``frames.csv`` / ``trajectories.csv`` / ``rejected_frames.csv`` /
  ``split_manifest.json`` / ``manifest.json`` -- provenance, QC and hashes.

Nothing here decides chemistry by directory name alone: composition comes from
the frames, and a name/composition disagreement is reported.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .errors import ConfigurationError, DependencyError, SafetyError
from .provenance import interfaceforge_commit, stable_json_hash
from .state import sha256_file, utc_now

SCHEMA = "interfaceforge-canonical-dataset"
SPLITS = ("train", "valid", "test")
KNOWN_LIGANDS = ("Me4PACz", "MeO2PACz", "MeO4PADBC", "DCZ4P")
KNOWN_ANCHORS = ("bare", "boundary", "hbond", "oh")
KNOWN_PATTERNS = ("clustered", "scattered", "full")
KNOWN_MOTIFS = ("capped", "dissoc")
GROUP_BY = ("case", "surface", "coverage")
SPLIT_METHODS = ("balanced", "hash")
STAGES = ("Step1", "Step2", "OPT", "unstaged")

_STAGE_DIR = re.compile(r"^(?P<stage>Step1|Step2|OPT)(?:_(?P<temp>\d+(?:\.\d+)?)K)?$", re.IGNORECASE)
_EXCLUDED_PARTS = {"archive", "backup", ".interfaceforge", "precondition", ".git", "smoke"}
_EXCLUDED_PREFIXES = ("restart_archive_", "refit_archive_", "stability_archive_")
_COMPLETION_MARKER = "General timing and accounting informations"
_IONIC_BLOCK = "TOTAL-FORCE"
_ERROR_MARKERS = (
    "VERY BAD NEWS",
    "ZBRENT: fatal error",
    "internal error",
    "Error EDDDAV",
    "The distance between some ions is very small",
)

LABEL_CONVENTION = {
    "energy": "ASE VASP OUTCAR 'energy(sigma->0)' in eV (REF_energy); TOTEN kept as REF_free_energy",
    "forces": "raw VASP TOTAL-FORCE in eV/angstrom; constrained atoms are NOT zeroed",
    "constraints": "per-atom move_mask (1 mobile, 0 fixed) from POSCAR/CONTCAR selective dynamics",
    "positions": "cartesian angstrom, atom order exactly as in the VASP run",
    "virial": "not exported unless include_virial (then -volume * ASE stress, eV)",
}


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass
class ExportConfig:
    """Every choice that changes which frames land in which split."""

    stages: tuple[str, ...] = ("Step2",)
    use_step2_sample: bool = True
    stride: int = 1
    include_incomplete: bool = False
    reject_scf_unconverged: bool = True
    reject_post_runaway: bool = True
    max_force_ev_ang: float | None = None
    max_md_temperature_k: float | None = None
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    seed: int = 20260730
    group_by: str = "case"
    stratify_by: tuple[str, ...] = ("ligand",)
    split_method: str = "balanced"
    type_map: tuple[str, ...] = ()
    include_virial: bool = False
    layout: str = "nio"

    def __post_init__(self) -> None:
        self.stages = tuple(_normalize_stage(stage) for stage in self.stages)
        if not self.stages:
            raise ConfigurationError("At least one stage must be selected")
        if int(self.stride) < 1:
            raise ConfigurationError("stride must be >= 1")
        self.stride = int(self.stride)
        ratios = tuple(float(value) for value in self.ratios)
        if len(ratios) != 3 or any(value < 0 for value in ratios) or sum(ratios) <= 0:
            raise ConfigurationError("ratios must be three non-negative numbers with a positive sum")
        total = sum(ratios)
        self.ratios = tuple(value / total for value in ratios)  # type: ignore[assignment]
        if self.group_by not in GROUP_BY:
            raise ConfigurationError(f"group_by must be one of {GROUP_BY}")
        if self.split_method not in SPLIT_METHODS:
            raise ConfigurationError(f"split_method must be one of {SPLIT_METHODS}")
        allowed_strata = {"ligand", "coverage", "motif", "pattern", "anchor", "stage", "none"}
        self.stratify_by = tuple(str(key) for key in self.stratify_by)
        unknown = sorted(set(self.stratify_by) - allowed_strata)
        if unknown:
            raise ConfigurationError(f"Unknown stratify_by keys {unknown}; allowed {sorted(allowed_strata)}")
        if self.layout not in {"nio", "generic"}:
            raise ConfigurationError("layout must be 'nio' or 'generic'")
        for key in ("max_force_ev_ang", "max_md_temperature_k"):
            value = getattr(self, key)
            if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
                raise ConfigurationError(f"{key} must be positive when set")
        self.type_map = tuple(str(value) for value in self.type_map)
        if len(set(self.type_map)) != len(self.type_map):
            raise ConfigurationError("type_map contains duplicates")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> ExportConfig:
        known = {name for name in cls.__dataclass_fields__}
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ConfigurationError(f"Unknown dataset export keys: {', '.join(unknown)}")
        values = dict(mapping)
        for key in ("stages", "stratify_by", "type_map", "ratios"):
            if key in values and values[key] is not None:
                values[key] = tuple(values[key])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in payload.items():
            if isinstance(value, tuple):
                payload[key] = list(value)
        return payload


def _normalize_stage(value: str) -> str:
    lowered = str(value).strip().lower()
    for stage in STAGES:
        if stage.lower() == lowered:
            return stage
    raise ConfigurationError(f"Unknown stage {value!r}; choose from {', '.join(STAGES)}")


def load_export_config(path: str | Path | None, **overrides: Any) -> ExportConfig:
    """Load an optional YAML/JSON export config and apply non-None overrides."""

    values: dict[str, Any] = {}
    if path is not None:
        import yaml

        config_path = Path(path).expanduser().resolve()
        if not config_path.is_file():
            raise ConfigurationError(f"Dataset export config not found: {config_path}")
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, Mapping):
            raise ConfigurationError("Dataset export config must be a mapping")
        # Accept a previous export manifest directly ("config" block).
        values.update(loaded.get("config", loaded) if "schema" in loaded else loaded)
    values.update({key: value for key, value in overrides.items() if value is not None})
    return ExportConfig.from_mapping(values)


# --------------------------------------------------------------------------- #
# chemistry parsing (provenance only -- composition comes from the frames)
# --------------------------------------------------------------------------- #
def _norm_token(token: str) -> str:
    return re.sub(r"[^a-z0-9]", "", token.lower())


_LIGAND_LOOKUP = {_norm_token(name): name for name in KNOWN_LIGANDS}


def parse_nio_case(name: str) -> dict[str, Any]:
    """Parse the NiO(110) hydroxylation/phosphonate naming convention.

    ``NiO_m110_Big_U46[_OH<pct>_<pattern>_<motif>][_<ligand>[_<anchor>]]``.
    Unrecognised tokens are kept in ``unparsed_tokens`` rather than guessed.
    """

    tokens = [token for token in re.split(r"[_]+", name) if token]
    result: dict[str, Any] = {
        "coverage_pct": None,
        "pattern": None,
        "motif": None,
        "ligand": None,
        "anchor": None,
        "surface": None,
        "unparsed_tokens": [],
        "parse": "none",
    }
    if not tokens:
        return result
    coverage_index = next(
        (index for index, token in enumerate(tokens) if re.fullmatch(r"OH\d+", token, re.IGNORECASE)),
        None,
    )
    consumed: set[int] = set()
    if coverage_index is not None:
        result["coverage_pct"] = int(tokens[coverage_index][2:])
        consumed.add(coverage_index)
    for index, token in enumerate(tokens):
        lower = token.lower()
        if lower in KNOWN_PATTERNS and result["pattern"] is None:
            result["pattern"] = lower
            consumed.add(index)
        elif lower in KNOWN_MOTIFS and result["motif"] is None:
            result["motif"] = lower
            consumed.add(index)
        elif _norm_token(token) in _LIGAND_LOOKUP and result["ligand"] is None:
            result["ligand"] = _LIGAND_LOOKUP[_norm_token(token)]
            consumed.add(index)
        elif lower in KNOWN_ANCHORS and index == len(tokens) - 1 and index > 0:
            result["anchor"] = lower
            consumed.add(index)
    # Base surface stem: every token before the first recognised decoration.
    first_decoration = min(consumed) if consumed else len(tokens)
    base = tokens[:first_decoration]
    if base and base[0].lower().startswith("nio"):
        if result["coverage_pct"] is None:
            result["coverage_pct"] = 0
        surface_tokens = list(base)
        if coverage_index is not None:
            surface_tokens.append(tokens[coverage_index])
            for key in ("pattern", "motif"):
                if result[key]:
                    surface_tokens.append(result[key])
        result["surface"] = "_".join(surface_tokens)
    result["unparsed_tokens"] = [
        token for index, token in enumerate(tokens) if index not in consumed and index >= first_decoration
    ]
    if result["ligand"] is None and result["unparsed_tokens"]:
        result["parse"] = "partial"
    elif result["surface"] is not None:
        result["parse"] = "full"
    else:
        result["parse"] = "partial" if consumed else "none"
    return result


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
@dataclass
class Trajectory:
    root: Path
    root_label: str
    outcar: Path
    directory: Path
    relative_dir: str
    trajectory_id: str
    stage: str
    stage_root: Path | None
    head: str
    case: str
    temperature_k: float | None = None
    temperature_source: str | None = None
    chemistry: dict[str, Any] = field(default_factory=dict)


def _excluded(parts: Sequence[str]) -> bool:
    for part in parts:
        lowered = part.lower()
        if lowered in _EXCLUDED_PARTS or lowered.startswith(_EXCLUDED_PREFIXES):
            return True
        if "backup" in lowered or part.startswith("X"):
            return True
    return False


def _root_labels(roots: Sequence[Path]) -> list[str]:
    labels: list[str] = []
    seen: Counter[str] = Counter()
    for root in roots:
        base = root.name or "root"
        seen[base] += 1
        labels.append(base if seen[base] == 1 else f"{base}__{seen[base]}")
    return labels


def _incar_float(incar: Mapping[str, str], key: str) -> float | None:
    value = incar.get(key)
    if value is None:
        return None
    try:
        number = float(str(value).split()[0].replace("d", "e").replace("D", "E"))
    except (ValueError, IndexError):
        return None
    return number if math.isfinite(number) else None


def discover_trajectories(roots: Sequence[str | Path], *, layout: str = "nio") -> list[Trajectory]:
    """Find every VASP trajectory (OUTCAR or OUTCAR.gz) below ``roots``.

    Archive/backup/precondition/``X*`` branches are skipped, matching the rest of
    InterfaceForge. Leaves are not required to be terminal directories: a
    conservative NiO Step1 leaf legitimately contains a ``precondition/`` child.
    """

    from .vasp import parse_incar

    resolved = [Path(root).expanduser().resolve() for root in roots]
    if not resolved:
        raise ConfigurationError("At least one dataset source root is required")
    for root in resolved:
        if not root.is_dir():
            raise SafetyError(f"Dataset source root is not a directory: {root}")
    labels = _root_labels(resolved)
    found: list[Trajectory] = []
    seen_dirs: set[Path] = set()
    for root, label in zip(resolved, labels, strict=True):
        candidates = sorted(set(root.rglob("OUTCAR")) | set(root.rglob("OUTCAR.gz")))
        for outcar in candidates:
            if not outcar.is_file():
                continue
            if outcar.name.endswith(".gz") and outcar.with_name("OUTCAR").is_file():
                continue
            directory = outcar.parent
            if directory in seen_dirs:
                continue
            relative = directory.relative_to(root)
            if _excluded(relative.parts):
                continue
            seen_dirs.add(directory)
            parts = relative.parts
            stage = "unstaged"
            stage_index: int | None = None
            temperature: float | None = None
            temperature_source: str | None = None
            for index in range(len(parts) - 1, -1, -1):
                match = _STAGE_DIR.match(parts[index])
                if match:
                    stage = {"step1": "Step1", "step2": "Step2", "opt": "OPT"}[match.group("stage").lower()]
                    stage_index = index
                    if match.group("temp"):
                        temperature = float(match.group("temp"))
                        temperature_source = "stage directory name"
                    break
            if stage_index is not None:
                head = "/".join(parts[:stage_index])
                case = "/".join(parts[stage_index + 1 :]) or (parts[stage_index - 1] if stage_index else label)
                stage_root: Path | None = root.joinpath(*parts[: stage_index + 1])
            else:
                head = ""
                case = "/".join(parts) or label
                stage_root = None
            if temperature is None:
                incar = parse_incar(directory / "INCAR")
                teend = _incar_float(incar, "TEEND")
                tebeg = _incar_float(incar, "TEBEG")
                if teend is not None:
                    temperature, temperature_source = teend, "INCAR TEEND"
                elif tebeg is not None:
                    temperature, temperature_source = tebeg, "INCAR TEBEG"
            relative_dir = relative.as_posix()
            trajectory_id = label if relative_dir in {"", "."} else f"{label}/{relative_dir}"
            leaf_name = Path(case).name
            chemistry = parse_nio_case(leaf_name) if layout == "nio" else {"parse": "not-attempted"}
            found.append(
                Trajectory(
                    root=root,
                    root_label=label,
                    outcar=outcar,
                    directory=directory,
                    relative_dir=relative_dir,
                    trajectory_id=trajectory_id,
                    stage=stage,
                    stage_root=stage_root,
                    head=head,
                    case=case,
                    temperature_k=temperature,
                    temperature_source=temperature_source,
                    chemistry=chemistry,
                )
            )
    return sorted(found, key=lambda item: item.trajectory_id)


# --------------------------------------------------------------------------- #
# per-trajectory analysis
# --------------------------------------------------------------------------- #
@dataclass
class FrameRecord:
    source_frame: int
    symbols: list[str]
    positions: np.ndarray
    cell: np.ndarray
    energy: float
    free_energy: float | None
    forces: np.ndarray
    virial: np.ndarray | None
    md_temperature_k: float | None
    scf_iterations: int | None


@dataclass
class TrajectoryAnalysis:
    trajectory: Trajectory
    status: str = "ok"
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source_sha256: str | None = None
    ionic_blocks: int = 0
    frames_parsed: int = 0
    complete: bool = False
    error_markers: list[str] = field(default_factory=list)
    oszicar_steps: int | None = None
    scf_nelm: int | None = None
    scf_unconverged_steps: int = 0
    first_bad_step: int | None = None
    first_bad_reasons: list[str] = field(default_factory=list)
    md_temperature_mean_k: float | None = None
    md_temperature_max_k: float | None = None
    selection: str = "stride"
    sampled_indices: int | None = None
    frames_valid: int = 0
    frames_selected: int = 0
    rejected: list[dict[str, Any]] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    formula: str | None = None
    elements: list[str] = field(default_factory=list)
    natoms: int | None = None
    move_mask: np.ndarray | None = None
    constraint_source: str | None = None
    start_fingerprint: str | None = None
    end_fingerprint: str | None = None
    max_abs_force_ev_ang: float | None = None
    frames: list[FrameRecord] = field(default_factory=list)
    duplicate_of: str | None = None


def _open_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def scan_outcar_markers(path: Path) -> dict[str, Any]:
    """Count ionic-step force blocks and completion/error markers independently of ASE.

    ASE's OUTCAR reader silently stops at a truncated final ionic step; this
    count is what lets the exporter *report* that truncation.
    """

    blocks = 0
    complete = False
    errors: set[str] = set()
    with _open_text(path) as handle:
        for line in handle:
            if _IONIC_BLOCK in line and "POSITION" in line:
                blocks += 1
            elif _COMPLETION_MARKER in line:
                complete = True
            else:
                for marker in _ERROR_MARKERS:
                    if marker in line:
                        errors.add(marker)
    return {"ionic_blocks": blocks, "complete": complete, "error_markers": sorted(errors)}


def _hill_formula(symbols: Sequence[str]) -> str:
    counts = Counter(symbols)
    order = (["C", "H"] if "C" in counts else []) + sorted(
        element for element in counts if element not in ({"C", "H"} if "C" in counts else set())
    )
    return "".join(f"{element}{counts[element] if counts[element] > 1 else ''}" for element in order)


def structure_fingerprint(symbols: Sequence[str], cell: np.ndarray, scaled_positions: np.ndarray) -> str:
    """Hash of species, cell (1e-3 A) and wrapped fractional coordinates (1e-4)."""

    wrapped = np.mod(np.round(np.asarray(scaled_positions, dtype=float), 4), 1.0)
    wrapped = np.round(np.where(np.isclose(wrapped, 1.0, atol=5e-5), 0.0, wrapped), 4)
    payload = {
        "symbols": list(symbols),
        "cell": np.round(np.asarray(cell, dtype=float), 3).reshape(-1).tolist(),
        "frac": wrapped.reshape(-1).tolist(),
    }
    return stable_json_hash(payload)


def _read_poscar_like(path: Path) -> Any | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        from ase.io import read
    except ModuleNotFoundError as exc:  # pragma: no cover - ASE is a hard requirement here
        raise DependencyError("ASE is required: pip install 'interfaceforge[vasp]'") from exc
    try:
        return read(str(path), format="vasp")
    except Exception:
        return None


def _mask_from_atoms(atoms: Any) -> np.ndarray:
    mask = np.ones(len(atoms), dtype=np.int8)
    for constraint in getattr(atoms, "constraints", []) or []:
        try:
            indices = np.asarray(constraint.get_indices(), dtype=int)
        except (AttributeError, TypeError, ValueError):
            continue
        mask[indices] = 0
    return mask


def _step2_sample_indices(trajectory: Trajectory) -> list[int] | None:
    if trajectory.stage != "Step2" or trajectory.stage_root is None:
        return None
    sample = trajectory.stage_root / "step2_sample.json"
    if not sample.is_file():
        return None
    try:
        payload = json.loads(sample.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    case_relative = Path(trajectory.directory).relative_to(trajectory.stage_root).as_posix()
    for row in payload.get("runs", []) or []:
        if str(row.get("relative_path", "")).strip("/") == case_relative and row.get("status") == "OK":
            indices = row.get("indices")
            if isinstance(indices, list):
                return sorted({int(value) for value in indices})
    return None


def analyze_trajectory(trajectory: Trajectory, config: ExportConfig) -> TrajectoryAnalysis:
    """Validate every frame of one trajectory and select the exported ones."""

    from .step1_repair import diagnose_step1_run, parse_step1_oszicar
    from .vasp import parse_incar

    analysis = TrajectoryAnalysis(trajectory=trajectory)
    analysis.source_sha256 = sha256_file(trajectory.outcar)
    markers = scan_outcar_markers(trajectory.outcar)
    analysis.ionic_blocks = markers["ionic_blocks"]
    analysis.complete = markers["complete"]
    analysis.error_markers = markers["error_markers"]
    if analysis.error_markers:
        analysis.warnings.append("OUTCAR error markers: " + "; ".join(analysis.error_markers))

    incar = parse_incar(trajectory.directory / "INCAR")
    nelm_value = _incar_float(incar, "NELM")
    analysis.scf_nelm = int(nelm_value) if nelm_value else 60
    oszicar = trajectory.directory / "OSZICAR"
    steps: list[dict[str, Any]] = []
    if oszicar.is_file():
        steps = parse_step1_oszicar(oszicar, nelm=analysis.scf_nelm)["steps"]
        analysis.oszicar_steps = len(steps)
        temperatures = [row["temperature_k"] for row in steps if row["temperature_k"] is not None]
        if temperatures:
            analysis.md_temperature_mean_k = float(np.mean(temperatures))
            analysis.md_temperature_max_k = float(np.max(temperatures))
        if (trajectory.directory / "INCAR").is_file() and trajectory.stage in {"Step1", "Step2"}:
            diagnosis = diagnose_step1_run(trajectory.directory)
            analysis.first_bad_step = diagnosis["first_bad_step"]
            analysis.first_bad_reasons = list(diagnosis["first_bad_reasons"])
    else:
        analysis.warnings.append("no OSZICAR: SCF convergence and MD temperature cannot be checked")

    temperature_runaway = any("temperature" in reason for reason in analysis.first_bad_reasons)
    if analysis.first_bad_step is not None and temperature_runaway:
        analysis.warnings.append(
            f"temperature runaway from MD step {analysis.first_bad_step} "
            f"({'; '.join(analysis.first_bad_reasons)})"
            + (
                "; later frames rejected"
                if config.reject_post_runaway
                else "; NOT rejected (reject_post_runaway=False)"
            )
        )
    if analysis.first_bad_step is not None and not temperature_runaway:
        analysis.warnings.append(
            f"energy-reference excursion at MD step {analysis.first_bad_step} "
            f"({'; '.join(analysis.first_bad_reasons)}); reported, not rejected "
            "(see docs/nio-aimd.md: the |F-Fref| heuristic has known false positives)"
        )

    # Constraints: the input POSCAR is what VASP ran with; CONTCAR repeats its flags.
    poscar = _read_poscar_like(trajectory.directory / "POSCAR")
    contcar = _read_poscar_like(trajectory.directory / "CONTCAR")
    reference = poscar if poscar is not None else contcar
    if reference is not None:
        analysis.move_mask = _mask_from_atoms(reference)
        analysis.constraint_source = "POSCAR selective dynamics" if poscar is not None else "CONTCAR selective dynamics"
        analysis.start_fingerprint = structure_fingerprint(
            reference.get_chemical_symbols(), reference.cell.array, reference.get_scaled_positions(wrap=False)
        )
    if contcar is not None:
        analysis.end_fingerprint = structure_fingerprint(
            contcar.get_chemical_symbols(), contcar.cell.array, contcar.get_scaled_positions(wrap=False)
        )

    sampled = _step2_sample_indices(trajectory) if config.use_step2_sample else None
    if sampled is not None:
        analysis.selection = "step2_sample"
        analysis.sampled_indices = len(sampled)
    sampled_set = set(sampled) if sampled is not None else None

    try:
        from ase.io import iread
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise DependencyError("ASE is required: pip install 'interfaceforge[vasp]'") from exc

    reference_symbols: list[str] | None = reference.get_chemical_symbols() if reference is not None else None
    first_symbols: list[str] | None = None
    max_force = 0.0
    iterator = iread(str(trajectory.outcar), index=":")
    index = -1
    while True:
        try:
            atoms = next(iterator)
        except StopIteration:
            break
        except Exception as exc:  # malformed chunk: iteration cannot continue past it
            analysis.rejected.append(
                {"source_frame": index + 1, "reason": f"unreadable OUTCAR chunk ({type(exc).__name__}: {exc})"}
            )
            analysis.warnings.append(f"OUTCAR parsing stopped at ionic step {index + 2}: {exc}")
            break
        index += 1
        analysis.frames_parsed += 1
        reasons: list[str] = []
        try:
            symbols = atoms.get_chemical_symbols()
            energy = float(atoms.get_potential_energy())
            try:
                free_energy: float | None = float(atoms.get_potential_energy(force_consistent=True))
            except Exception:
                free_energy = None
            forces = np.asarray(atoms.get_forces(apply_constraint=False), dtype=np.float64)
            positions = np.asarray(atoms.positions, dtype=np.float64)
            cell = np.asarray(atoms.cell.array, dtype=np.float64)
        except Exception as exc:
            analysis.rejected.append({"source_frame": index, "reason": f"missing label ({type(exc).__name__}: {exc})"})
            continue
        if first_symbols is None:
            first_symbols = symbols
        if symbols != first_symbols:
            reasons.append("atom identity/order changed within trajectory")
        if reference_symbols is not None and symbols != reference_symbols:
            reasons.append("atom order differs from POSCAR/CONTCAR")
        if forces.shape != (len(symbols), 3):
            reasons.append(f"force array shape {forces.shape}")
        if not math.isfinite(energy):
            reasons.append("non-finite energy")
        if not np.isfinite(forces).all() or not np.isfinite(positions).all():
            reasons.append("non-finite forces/positions")
        if not np.isfinite(cell).all() or abs(float(np.linalg.det(cell))) < 1e-6:
            reasons.append("degenerate cell")
        step = steps[index] if index < len(steps) else None
        md_temperature = step["temperature_k"] if step else None
        scf_iterations = step["electronic_iterations"] if step else None
        if step and step["hit_nelm"]:
            analysis.scf_unconverged_steps += 1
            if config.reject_scf_unconverged:
                reasons.append(f"SCF not converged within NELM={analysis.scf_nelm}")
        if (
            config.reject_post_runaway
            and temperature_runaway
            and analysis.first_bad_step is not None
            and index + 1 >= analysis.first_bad_step
        ):
            reasons.append(f"at/after temperature runaway (MD step {analysis.first_bad_step})")
        frame_max_force = (
            float(np.max(np.linalg.norm(forces, axis=1))) if forces.size and np.isfinite(forces).all() else math.inf
        )
        if math.isfinite(frame_max_force):
            max_force = max(max_force, frame_max_force)
        if config.max_force_ev_ang is not None and frame_max_force > config.max_force_ev_ang:
            reasons.append(f"max |F| {frame_max_force:.2f} > {config.max_force_ev_ang:g} eV/A")
        if (
            config.max_md_temperature_k is not None
            and md_temperature is not None
            and md_temperature > config.max_md_temperature_k
        ):
            reasons.append(f"MD temperature {md_temperature:.0f} K > {config.max_md_temperature_k:g} K")
        if reasons:
            analysis.rejected.append({"source_frame": index, "reason": "; ".join(reasons)})
            continue
        analysis.frames_valid += 1
        if sampled_set is not None:
            if index not in sampled_set:
                continue
        elif index % config.stride:
            continue
        virial = None
        if config.include_virial:
            try:
                stress = np.asarray(atoms.get_stress(voigt=False), dtype=np.float64)
                virial = -float(atoms.get_volume()) * stress
            except Exception:
                analysis.rejected.append({"source_frame": index, "reason": "virial requested but stress missing"})
                analysis.frames_valid -= 1
                continue
        analysis.frames.append(
            FrameRecord(
                source_frame=index,
                symbols=symbols,
                positions=positions,
                cell=cell,
                energy=energy,
                free_energy=free_energy,
                forces=forces,
                virial=virial,
                md_temperature_k=md_temperature,
                scf_iterations=scf_iterations,
            )
        )
        if analysis.move_mask is None:
            analysis.move_mask = _mask_from_atoms(atoms)
            analysis.constraint_source = "ASE OUTCAR reader" if atoms.constraints else "none found (all mobile)"
        if analysis.start_fingerprint is None and len(analysis.frames) == 1 and index == 0:
            analysis.start_fingerprint = structure_fingerprint(symbols, cell, atoms.get_scaled_positions(wrap=False))

    analysis.frames_selected = len(analysis.frames)
    analysis.max_abs_force_ev_ang = max_force if analysis.frames_parsed else None
    if first_symbols is not None:
        analysis.symbols = first_symbols
        analysis.natoms = len(first_symbols)
        analysis.formula = _hill_formula(first_symbols)
        analysis.elements = sorted(set(first_symbols))
    if analysis.move_mask is not None and analysis.natoms is not None and len(analysis.move_mask) != analysis.natoms:
        analysis.warnings.append("constraint mask length differs from atom count; treating every atom as mobile")
        analysis.move_mask = np.ones(analysis.natoms, dtype=np.int8)
        analysis.constraint_source = "invalid mask ignored (all mobile)"
    if analysis.move_mask is None and analysis.natoms is not None:
        analysis.move_mask = np.ones(analysis.natoms, dtype=np.int8)
        analysis.constraint_source = "none found (all mobile)"
    if analysis.constraint_source and analysis.constraint_source.startswith("none"):
        analysis.warnings.append("no POSCAR/CONTCAR selective dynamics: move_mask marks every atom mobile")

    truncated = analysis.ionic_blocks - analysis.frames_parsed
    if truncated > 0:
        analysis.warnings.append(
            f"{truncated} ionic step(s) in OUTCAR could not be parsed (truncated/incomplete final step)"
        )
        for offset in range(truncated):
            analysis.rejected.append(
                {"source_frame": analysis.frames_parsed + offset, "reason": "truncated ionic step (no energy block)"}
            )
    if analysis.oszicar_steps is not None and analysis.oszicar_steps != analysis.ionic_blocks:
        analysis.warnings.append(
            f"OSZICAR has {analysis.oszicar_steps} MD steps but OUTCAR has {analysis.ionic_blocks} ionic blocks"
        )
    if sampled is not None:
        rejected_sampled = sorted(set(sampled) & {row["source_frame"] for row in analysis.rejected})
        missing = sorted(set(sampled) - set(range(analysis.frames_parsed)))
        if rejected_sampled:
            analysis.warnings.append(f"{len(rejected_sampled)} step2_sample frame(s) failed QC and were dropped")
        if missing:
            analysis.warnings.append(f"{len(missing)} step2_sample index(es) beyond the parsed OUTCAR")

    chemistry = trajectory.chemistry or {}
    if chemistry.get("ligand") and analysis.elements and "P" not in analysis.elements:
        analysis.warnings.append(f"name says ligand {chemistry['ligand']} but the frames contain no P")
    if (
        analysis.elements
        and "P" in analysis.elements
        and not chemistry.get("ligand")
        and chemistry.get("parse") != "not-attempted"
    ):
        analysis.warnings.append("frames contain P but no known ligand token was parsed from the case name")

    if analysis.frames_parsed == 0:
        analysis.status = "empty"
        analysis.reasons.append("no readable labelled frames")
    elif not analysis.complete:
        analysis.status = "incomplete"
        analysis.reasons.append("OUTCAR has no completion marker (running, killed or crashed)")
    elif analysis.frames_selected == 0:
        analysis.status = "no_usable_frames"
        analysis.reasons.append("every frame failed QC or selection")
    return analysis


# --------------------------------------------------------------------------- #
# grouping and splitting
# --------------------------------------------------------------------------- #
class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: str, second: str) -> None:
        a, b = self.find(first), self.find(second)
        if a != b:
            # Deterministic representative: lexicographically smallest key.
            low, high = sorted((a, b))
            self.parent[high] = low


def _base_group_key(analysis: TrajectoryAnalysis, group_by: str) -> str:
    trajectory = analysis.trajectory
    if trajectory.stage == "unstaged":
        return f"unstaged:{trajectory.trajectory_id}"
    chemistry = trajectory.chemistry or {}
    case_parent = str(Path(trajectory.case).parent.as_posix())
    case_parent = "" if case_parent == "." else case_parent + "/"
    if group_by == "surface" and chemistry.get("surface"):
        return f"surface:{case_parent}{chemistry['surface']}"
    if group_by == "coverage" and chemistry.get("coverage_pct") is not None:
        return f"coverage:OH{chemistry['coverage_pct']}"
    return f"case:{trajectory.case}"


def assign_groups(analyses: Sequence[TrajectoryAnalysis], group_by: str = "case") -> dict[str, str]:
    """Map trajectory_id -> leakage group.

    Beyond the configured key, trajectories are merged when they start from an
    identical structure, or when one trajectory's final CONTCAR is another's
    starting POSCAR (the Step1 -> Step2 hand-off), so a lineage is never split
    even when its directories do not share a case name.
    """

    union = _UnionFind()
    keys: dict[str, str] = {}
    by_start: dict[str, list[str]] = defaultdict(list)
    for analysis in analyses:
        tid = analysis.trajectory.trajectory_id
        key = _base_group_key(analysis, group_by)
        keys[tid] = key
        union.find(key)
        if analysis.start_fingerprint:
            by_start[analysis.start_fingerprint].append(key)
    for members in by_start.values():
        for other in members[1:]:
            union.union(members[0], other)
    for analysis in analyses:
        if analysis.end_fingerprint and analysis.end_fingerprint in by_start:
            for other in by_start[analysis.end_fingerprint]:
                union.union(keys[analysis.trajectory.trajectory_id], other)
    return {tid: union.find(key) for tid, key in keys.items()}


def _stratum(analyses: Sequence[TrajectoryAnalysis], keys: Sequence[str]) -> str:
    if not keys or keys == ("none",):
        return "all"
    parts: list[str] = []
    for key in keys:
        values: set[str] = set()
        for analysis in analyses:
            chemistry = analysis.trajectory.chemistry or {}
            if key == "stage":
                values.add(analysis.trajectory.stage)
            elif key == "coverage":
                value = chemistry.get("coverage_pct")
                values.add("NA" if value is None else f"OH{value}")
            else:
                value = chemistry.get(key)
                values.add("none" if value in (None, "") else str(value))
        parts.append(f"{key}={'+'.join(sorted(values))}")
    return "|".join(parts)


def _unit_hash(seed: int, key: str) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    return int(digest[:16], 16) / float(1 << 64)


def split_groups(
    group_frames: Mapping[str, int],
    group_strata: Mapping[str, str],
    ratios: Sequence[float],
    *,
    seed: int,
    method: str = "balanced",
) -> dict[str, str]:
    """Assign whole groups to splits, deterministically.

    ``balanced`` walks each stratum in a seeded hash order and gives every group
    to the split with the largest frame deficit (stratum first, then global);
    ``hash`` assigns each group from its own seeded hash alone, so adding new
    groups never moves existing ones.
    """

    normalized = [float(value) for value in ratios]
    total_ratio = sum(normalized)
    normalized = [value / total_ratio for value in normalized]
    target = dict(zip(SPLITS, normalized, strict=True))
    active = [split for split in SPLITS if target[split] > 0]
    if not active:
        raise ConfigurationError("At least one split ratio must be positive")
    assignment: dict[str, str] = {}
    if method == "hash":
        for group in sorted(group_frames):
            value = _unit_hash(seed, group)
            cumulative = 0.0
            chosen = active[-1]
            for split in SPLITS:
                cumulative += target[split]
                if target[split] > 0 and value < cumulative:
                    chosen = split
                    break
            assignment[group] = chosen
        return assignment

    global_assigned = {split: 0.0 for split in SPLITS}
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for group in group_frames:
        by_stratum[group_strata.get(group, "all")].append(group)
    for stratum in sorted(by_stratum):
        stratum_assigned = {split: 0.0 for split in SPLITS}
        members = sorted(by_stratum[stratum], key=lambda group: (_unit_hash(seed, group), group))
        for group in members:
            weight = float(max(group_frames[group], 1))
            stratum_total = sum(stratum_assigned.values()) + weight
            global_total = sum(global_assigned.values()) + weight
            chosen = max(
                active,
                key=lambda split: (
                    round(target[split] * stratum_total - stratum_assigned[split], 9),
                    round(target[split] * global_total - global_assigned[split], 9),
                    -SPLITS.index(split),
                ),
            )
            assignment[group] = chosen
            stratum_assigned[chosen] += weight
            global_assigned[chosen] += weight

    # With enough independent groups, never leave a requested split empty: move
    # the smallest group out of the split that is furthest above its target.
    if len(group_frames) >= len(active):
        for empty in [split for split in active if not any(value == split for value in assignment.values())]:
            total = sum(float(max(value, 1)) for value in group_frames.values())
            donors = sorted(
                active,
                key=lambda split: global_assigned[split] - target[split] * total,
                reverse=True,
            )
            for donor in donors:
                donor_groups = [group for group, split in assignment.items() if split == donor]
                if len(donor_groups) <= 1:
                    continue
                moved = min(donor_groups, key=lambda group: (group_frames[group], _unit_hash(seed, group), group))
                assignment[moved] = empty
                weight = float(max(group_frames[moved], 1))
                global_assigned[donor] -= weight
                global_assigned[empty] += weight
                break
    return assignment


def split_hash(frame_splits: Iterable[tuple[str, str]]) -> str:
    """Location-independent identity of a split: sha256 over sorted (frame_id, split)."""

    digest = hashlib.sha256()
    for frame_id, split in sorted(frame_splits):
        digest.update(f"{frame_id}\t{split}\n".encode())
    return digest.hexdigest()


def leakage_report(
    analyses: Sequence[TrajectoryAnalysis],
    groups: Mapping[str, str],
    trajectory_split: Mapping[str, str],
) -> dict[str, Any]:
    """Independent post-hoc checks that no correlated lineage spans splits."""

    problems: list[dict[str, Any]] = []
    group_splits: dict[str, set[str]] = defaultdict(set)
    for tid, split in trajectory_split.items():
        group_splits[groups[tid]].add(split)
    for group, splits in sorted(group_splits.items()):
        if len(splits) > 1:
            problems.append({"check": "group spans splits", "group": group, "splits": sorted(splits)})
    by_id = {analysis.trajectory.trajectory_id: analysis for analysis in analyses}
    start_splits: dict[str, set[str]] = defaultdict(set)
    case_splits: dict[str, set[str]] = defaultdict(set)
    for tid, split in trajectory_split.items():
        analysis = by_id[tid]
        if analysis.start_fingerprint:
            start_splits[analysis.start_fingerprint].add(split)
        case_splits[analysis.trajectory.case].add(split)
    for fingerprint, splits in sorted(start_splits.items()):
        if len(splits) > 1:
            problems.append(
                {
                    "check": "identical starting structure in several splits",
                    "fingerprint": fingerprint,
                    "splits": sorted(splits),
                }
            )
    for tid, split in trajectory_split.items():
        analysis = by_id[tid]
        if not analysis.end_fingerprint:
            continue
        for other_id, other_split in trajectory_split.items():
            if other_split != split and by_id[other_id].start_fingerprint == analysis.end_fingerprint:
                problems.append(
                    {
                        "check": "CONTCAR->POSCAR lineage crosses splits",
                        "from": tid,
                        "to": other_id,
                        "splits": [split, other_split],
                    }
                )
    # Informational: stricter surface-family view (ligand variants of one hydroxylated slab).
    surface_splits: dict[str, set[str]] = defaultdict(set)
    for tid, split in trajectory_split.items():
        surface = (by_id[tid].trajectory.chemistry or {}).get("surface")
        if surface:
            surface_splits[surface].add(split)
    shared_surfaces = sorted(surface for surface, splits in surface_splits.items() if len(splits) > 1)
    return {
        "leakage_detected": bool(problems),
        "problems": problems,
        "groups_checked": len(group_splits),
        "cases_spanning_splits": sorted(case for case, splits in case_splits.items() if len(splits) > 1),
        "informational": {
            "surface_families_shared_across_splits": len(shared_surfaces),
            "note": (
                "Surface families (one hydroxylated slab decorated by different ligands/anchors) "
                "shared across splits are allowed under group_by='case'; use group_by='surface' "
                "for the stricter ligand-transfer benchmark."
            ),
            "examples": shared_surfaces[:10],
        },
    }


# --------------------------------------------------------------------------- #
# planning (shared by export and readiness)
# --------------------------------------------------------------------------- #
@dataclass
class ExportPlan:
    config: ExportConfig
    roots: list[Path]
    trajectories: list[Trajectory]
    analyses: list[TrajectoryAnalysis]
    usable: list[TrajectoryAnalysis]
    groups: dict[str, str]
    group_split: dict[str, str]
    trajectory_split: dict[str, str]
    strata: dict[str, str]
    leakage: dict[str, Any]
    type_map: list[str]


def plan_export(roots: Sequence[str | Path], config: ExportConfig) -> ExportPlan:
    trajectories = discover_trajectories(roots, layout=config.layout)
    if not trajectories:
        raise SafetyError(f"No VASP OUTCAR trajectories found below {', '.join(map(str, roots))}")
    analyses: list[TrajectoryAnalysis] = []
    seen_sha: dict[str, str] = {}
    for trajectory in trajectories:
        if trajectory.stage not in config.stages:
            analysis = TrajectoryAnalysis(trajectory=trajectory, status="stage_not_selected")
            analysis.reasons.append(f"stage {trajectory.stage} not in selected stages {list(config.stages)}")
            analysis.source_sha256 = None
            analyses.append(analysis)
            continue
        analysis = analyze_trajectory(trajectory, config)
        if analysis.source_sha256 in seen_sha:
            analysis.duplicate_of = seen_sha[analysis.source_sha256]
            analysis.status = "duplicate"
            analysis.reasons.append(f"byte-identical OUTCAR already exported as {analysis.duplicate_of}")
            analysis.frames = []
        elif analysis.source_sha256:
            seen_sha[analysis.source_sha256] = trajectory.trajectory_id
        if analysis.status == "incomplete" and config.include_incomplete and analysis.frames:
            analysis.status = "ok_incomplete"
        analyses.append(analysis)

    usable = [analysis for analysis in analyses if analysis.status in {"ok", "ok_incomplete"} and analysis.frames]
    if not usable:
        raise SafetyError("No trajectory has usable frames after QC; see the discovery report")
    groups = assign_groups(usable, config.group_by)
    group_members: dict[str, list[TrajectoryAnalysis]] = defaultdict(list)
    for analysis in usable:
        group_members[groups[analysis.trajectory.trajectory_id]].append(analysis)
    group_frames = {group: sum(len(item.frames) for item in members) for group, members in group_members.items()}
    strata = {group: _stratum(members, config.stratify_by) for group, members in group_members.items()}
    group_split = split_groups(group_frames, strata, config.ratios, seed=config.seed, method=config.split_method)
    trajectory_split = {
        analysis.trajectory.trajectory_id: group_split[groups[analysis.trajectory.trajectory_id]] for analysis in usable
    }
    leakage = leakage_report(usable, groups, trajectory_split)

    elements = sorted({element for analysis in usable for element in analysis.elements})
    if config.type_map:
        missing = sorted(set(elements) - set(config.type_map))
        if missing:
            raise SafetyError(f"Explicit type_map is missing elements present in the data: {missing}")
        type_map = list(config.type_map)
    else:
        type_map = elements
    return ExportPlan(
        config=config,
        roots=[Path(root).expanduser().resolve() for root in roots],
        trajectories=trajectories,
        analyses=analyses,
        usable=usable,
        groups=groups,
        group_split=group_split,
        trajectory_split=trajectory_split,
        strata=strata,
        leakage=leakage,
        type_map=type_map,
    )


def _trajectory_row(analysis: TrajectoryAnalysis, plan: ExportPlan | None) -> dict[str, Any]:
    trajectory = analysis.trajectory
    chemistry = trajectory.chemistry or {}
    tid = trajectory.trajectory_id
    return {
        "trajectory_id": tid,
        "status": analysis.status,
        "reasons": "; ".join(analysis.reasons),
        "warnings": "; ".join(analysis.warnings),
        "stage": trajectory.stage,
        "temperature_k": trajectory.temperature_k,
        "temperature_source": trajectory.temperature_source,
        "head": trajectory.head,
        "case": trajectory.case,
        "group": plan.groups.get(tid) if plan else None,
        "split": plan.trajectory_split.get(tid) if plan else None,
        "formula": analysis.formula,
        "natoms": analysis.natoms,
        "coverage_pct": chemistry.get("coverage_pct"),
        "pattern": chemistry.get("pattern"),
        "motif": chemistry.get("motif"),
        "ligand": chemistry.get("ligand"),
        "anchor": chemistry.get("anchor"),
        "surface": chemistry.get("surface"),
        "name_parse": chemistry.get("parse"),
        "complete": analysis.complete,
        "ionic_blocks": analysis.ionic_blocks,
        "frames_parsed": analysis.frames_parsed,
        "frames_valid": analysis.frames_valid,
        "frames_rejected": len(analysis.rejected),
        "frames_selected": len(analysis.frames),
        "selection": analysis.selection if analysis.status != "stage_not_selected" else None,
        "step2_sample_indices": analysis.sampled_indices,
        "oszicar_steps": analysis.oszicar_steps,
        "scf_nelm": analysis.scf_nelm,
        "scf_unconverged_steps": analysis.scf_unconverged_steps,
        "first_bad_step": analysis.first_bad_step,
        "first_bad_reasons": "; ".join(analysis.first_bad_reasons),
        "md_temperature_mean_k": analysis.md_temperature_mean_k,
        "md_temperature_max_k": analysis.md_temperature_max_k,
        "max_abs_force_ev_ang": analysis.max_abs_force_ev_ang,
        "constraint_source": analysis.constraint_source,
        "fixed_atoms": int(len(analysis.move_mask) - analysis.move_mask.sum())
        if analysis.move_mask is not None
        else None,
        "start_fingerprint": analysis.start_fingerprint,
        "end_fingerprint": analysis.end_fingerprint,
        "duplicate_of": analysis.duplicate_of,
        "source_path": str(trajectory.outcar),
        "source_sha256": analysis.source_sha256,
    }


def _counts_by(rows: Iterable[Mapping[str, Any]], *keys: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        label = "|".join("none" if row.get(key) in (None, "") else str(row.get(key)) for key in keys)
        counter[label] += int(row.get("frames_selected") or 0)
    return dict(sorted(counter.items()))


def _rejection_reasons(analyses: Iterable[TrajectoryAnalysis]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for analysis in analyses:
        for row in analysis.rejected:
            for reason in str(row["reason"]).split("; "):
                counter[re.sub(r"\d+(?:\.\d+)?", "#", reason)] += 1
    return dict(counter.most_common())


def _plan_summary(plan: ExportPlan) -> dict[str, Any]:
    rows = [_trajectory_row(analysis, plan) for analysis in plan.analyses]
    usable_rows = [row for row in rows if row["status"] in {"ok", "ok_incomplete"}]
    frames_per_split = {split: 0 for split in SPLITS}
    trajectories_per_split = {split: 0 for split in SPLITS}
    groups_per_split = {split: 0 for split in SPLITS}
    for row in usable_rows:
        frames_per_split[row["split"]] += int(row["frames_selected"])
        trajectories_per_split[row["split"]] += 1
    for split in plan.group_split.values():
        groups_per_split[split] += 1
    status_counts = Counter(row["status"] for row in rows)
    total_frames = sum(frames_per_split.values())
    return {
        "trajectories_discovered": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "stage_counts": dict(sorted(Counter(row["stage"] for row in rows).items())),
        "usable_trajectories": len(usable_rows),
        "frames_parsed": sum(int(row["frames_parsed"] or 0) for row in rows),
        "frames_rejected": sum(int(row["frames_rejected"] or 0) for row in rows),
        "rejection_reasons": _rejection_reasons(plan.analyses),
        "frames_selected": total_frames,
        "frames_per_split": frames_per_split,
        "achieved_frame_fractions": {
            split: (frames_per_split[split] / total_frames if total_frames else None) for split in SPLITS
        },
        "target_fractions": dict(zip(SPLITS, plan.config.ratios, strict=True)),
        "trajectories_per_split": trajectories_per_split,
        "groups": len(plan.group_split),
        "groups_per_split": groups_per_split,
        "frames_by_stage_temperature": _counts_by(usable_rows, "stage", "temperature_k"),
        "frames_by_system_temperature": _counts_by(usable_rows, "case", "temperature_k"),
        "frames_by_split_ligand": _counts_by(usable_rows, "split", "ligand"),
        "frames_by_split_coverage": _counts_by(usable_rows, "split", "coverage_pct"),
        "type_map": plan.type_map,
        "leakage": plan.leakage,
        "problem_trajectories": [
            {
                "trajectory_id": row["trajectory_id"],
                "status": row["status"],
                "reasons": row["reasons"],
                "warnings": row["warnings"],
            }
            for row in rows
            if row["status"] not in {"ok", "stage_not_selected"} or row["warnings"]
        ],
    }


def discover_report(roots: Sequence[str | Path], config: ExportConfig) -> dict[str, Any]:
    """Read-only inventory plus the split the export would produce."""

    plan = plan_export(roots, config)
    return {
        "schema": f"{SCHEMA}-plan",
        "schema_version": 1,
        "roots": [str(root) for root in plan.roots],
        "config": config.to_dict(),
        "summary": _plan_summary(plan),
        "trajectories": [_trajectory_row(analysis, plan) for analysis in plan.analyses],
    }


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #
def _prepare_output(path: Path, *, force: bool, sources: Sequence[Path]) -> None:
    for source in sources:
        if path == source or source in path.parents or path in source.parents:
            raise SafetyError(f"Dataset output {path} must not overlap source root {source}")
    if path.exists() and any(path.iterdir()):
        if not force:
            raise SafetyError(f"Dataset output is not empty: {path} (use --force to replace)")
        if len(path.parts) < 3:
            raise SafetyError(f"Refusing broad destructive output replacement: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _safe_part(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return re.sub(r"_+", "_", cleaned).strip("_") or "unnamed"


def _frame_info(analysis: TrajectoryAnalysis, frame: FrameRecord, split: str, group: str) -> dict[str, Any]:
    trajectory = analysis.trajectory
    chemistry = trajectory.chemistry or {}
    info: dict[str, Any] = {
        "frame_id": f"{trajectory.trajectory_id}:{frame.source_frame}",
        "IF_leaf": trajectory.trajectory_id,
        "source_run": trajectory.trajectory_id,
        "source_path": str(trajectory.outcar),
        "source_frame": frame.source_frame,
        "split": split,
        "IF_stage": trajectory.stage,
        "IF_temperature_k": trajectory.temperature_k,
        "IF_case": trajectory.case,
        "IF_group": group,
        "IF_formula": analysis.formula,
        "IF_selection": analysis.selection,
        "IF_md_temperature_k": frame.md_temperature_k,
        "IF_scf_iterations": frame.scf_iterations,
        "REF_free_energy": frame.free_energy,
        "IF_coverage_pct": chemistry.get("coverage_pct"),
        "IF_pattern": chemistry.get("pattern"),
        "IF_motif": chemistry.get("motif"),
        "IF_ligand": chemistry.get("ligand"),
        "IF_anchor": chemistry.get("anchor"),
        "IF_surface": chemistry.get("surface"),
    }
    return {key: value for key, value in info.items() if value is not None}


def _write_deepmd(
    split_root: Path,
    analysis: TrajectoryAnalysis,
    split: str,
    group: str,
    type_map: Sequence[str],
) -> Path:
    trajectory = analysis.trajectory
    system = split_root.joinpath(*(_safe_part(part) for part in trajectory.trajectory_id.split("/")))
    set_dir = system / "set.000"
    set_dir.mkdir(parents=True, exist_ok=False)
    frames = analysis.frames
    symbols = frames[0].symbols
    atom_types = [type_map.index(symbol) for symbol in symbols]
    (system / "type.raw").write_text("\n".join(map(str, atom_types)) + "\n", encoding="utf-8")
    (system / "type_map.raw").write_text("\n".join(type_map) + "\n", encoding="utf-8")
    np.save(set_dir / "coord.npy", np.asarray([frame.positions.reshape(-1) for frame in frames]))
    np.save(set_dir / "box.npy", np.asarray([frame.cell.reshape(-1) for frame in frames]))
    np.save(set_dir / "energy.npy", np.asarray([[frame.energy] for frame in frames]))
    np.save(set_dir / "force.npy", np.asarray([frame.forces.reshape(-1) for frame in frames]))
    if all(frame.virial is not None for frame in frames) and frames[0].virial is not None:
        np.save(set_dir / "virial.npy", np.asarray([frame.virial.reshape(-1) for frame in frames]))
    mask = analysis.move_mask if analysis.move_mask is not None else np.ones(len(symbols), dtype=np.int8)
    np.save(system / "move_mask.npy", np.asarray([mask for _ in frames], dtype=np.int8))
    chemistry = trajectory.chemistry or {}
    meta = {
        "schema_version": 1,
        "run_id": trajectory.trajectory_id,
        "category": trajectory.case,
        "group": group,
        "kind": "surface",
        "tebeg_k": trajectory.temperature_k,
        "high_temperature": False,
        "split": split,
        "stage": trajectory.stage,
        "temperature_k": trajectory.temperature_k,
        "formula": analysis.formula,
        "chemistry": {
            key: chemistry.get(key) for key in ("coverage_pct", "pattern", "motif", "ligand", "anchor", "surface")
        },
        "source_outcar": str(trajectory.outcar),
        "source_sha256": analysis.source_sha256,
        "constraint_source": analysis.constraint_source,
        "selection": analysis.selection,
    }
    (system / "system_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    with (system / "frame_map.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["local_frame", "source_frame", "source_path", "relative_leaf", "frame_id"])
        for index, frame in enumerate(frames):
            writer.writerow(
                [
                    index,
                    frame.source_frame,
                    str(trajectory.outcar),
                    trajectory.trajectory_id,
                    f"{trajectory.trajectory_id}:{frame.source_frame}",
                ]
            )
    return system


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    names = list(fieldnames) if fieldnames else (list(rows[0]) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in names})


def _file_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            hashes[path.relative_to(root).as_posix()] = sha256_file(path)
    return hashes


def dataset_content_hash(file_hashes: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(file_hashes):
        digest.update(f"{name}\t{file_hashes[name]}\n".encode())
    return digest.hexdigest()


def export_dataset(
    roots: Sequence[str | Path],
    output: str | Path,
    config: ExportConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Write the canonical multi-backend dataset and its provenance."""

    from .packaging import _extxyz_frame_text

    plan = plan_export(roots, config)
    if plan.leakage["leakage_detected"]:
        raise SafetyError(f"Split leakage detected; refusing to write: {plan.leakage['problems'][:3]}")
    out = Path(output).expanduser().resolve()
    _prepare_output(out, force=force, sources=plan.roots)
    deepmd_root = out / "deepmd"
    for split in SPLITS:
        (deepmd_root / split).mkdir(parents=True, exist_ok=True)

    frame_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    frame_splits: list[tuple[str, str]] = []
    texts: dict[str, list[str]] = {split: [] for split in SPLITS}
    systems: dict[str, list[str]] = {split: [] for split in SPLITS}
    for analysis in plan.analyses:
        for row in analysis.rejected:
            rejected_rows.append({"trajectory_id": analysis.trajectory.trajectory_id, **row})
    for analysis in sorted(plan.usable, key=lambda item: item.trajectory.trajectory_id):
        tid = analysis.trajectory.trajectory_id
        split = plan.trajectory_split[tid]
        group = plan.groups[tid]
        mask = analysis.move_mask if analysis.move_mask is not None else np.ones(analysis.natoms or 0, dtype=np.int8)
        for frame in analysis.frames:
            info = _frame_info(analysis, frame, split, group)
            texts[split].append(
                _extxyz_frame_text(
                    symbols=frame.symbols,
                    positions=frame.positions,
                    forces=frame.forces,
                    move_mask=mask,
                    cell=frame.cell,
                    energy=frame.energy,
                    virial=frame.virial,
                    info=info,
                )
            )
            frame_splits.append((info["frame_id"], split))
            norms = np.linalg.norm(frame.forces, axis=1)
            frame_rows.append(
                {
                    "frame_id": info["frame_id"],
                    "split": split,
                    "trajectory_id": tid,
                    "group": group,
                    "stage": analysis.trajectory.stage,
                    "temperature_k": analysis.trajectory.temperature_k,
                    "case": analysis.trajectory.case,
                    "source_frame": frame.source_frame,
                    "natoms": len(frame.symbols),
                    "formula": analysis.formula,
                    "energy_ev": frame.energy,
                    "energy_per_atom_ev": frame.energy / len(frame.symbols),
                    "free_energy_ev": frame.free_energy,
                    "max_abs_force_ev_ang": float(norms.max()) if norms.size else None,
                    "rms_force_ev_ang": float(np.sqrt(np.mean(frame.forces**2))) if frame.forces.size else None,
                    "md_temperature_k": frame.md_temperature_k,
                    "scf_iterations": frame.scf_iterations,
                    "mobile_atoms": int(mask.sum()),
                    "fixed_atoms": int(len(mask) - mask.sum()),
                    "selection": analysis.selection,
                    "ligand": (analysis.trajectory.chemistry or {}).get("ligand"),
                    "coverage_pct": (analysis.trajectory.chemistry or {}).get("coverage_pct"),
                    "source_path": str(analysis.trajectory.outcar),
                }
            )
        system = _write_deepmd(deepmd_root / split, analysis, split, group, plan.type_map)
        systems[split].append(system.relative_to(out).as_posix())

    extxyz: dict[str, str | None] = {}
    for split in SPLITS:
        if texts[split]:
            path = out / f"{split}.extxyz"
            path.write_text("".join(texts[split]), encoding="utf-8")
            extxyz[split] = str(path)
        else:
            extxyz[split] = None
    empty = [split for split in SPLITS if plan.config.ratios[SPLITS.index(split)] > 0 and not texts[split]]

    _write_csv(out / "frames.csv", frame_rows)
    trajectory_rows = [_trajectory_row(analysis, plan) for analysis in plan.analyses]
    _write_csv(out / "trajectories.csv", trajectory_rows)
    _write_csv(out / "rejected_frames.csv", rejected_rows, ["trajectory_id", "source_frame", "reason"])
    split_identity = split_hash(frame_splits)
    split_manifest = {
        "schema": f"{SCHEMA}-split",
        "schema_version": 1,
        "split_hash": split_identity,
        "method": plan.config.split_method,
        "seed": plan.config.seed,
        "ratios": list(plan.config.ratios),
        "group_by": plan.config.group_by,
        "stratify_by": list(plan.config.stratify_by),
        "unit": "whole leakage group (case lineage); frames of one trajectory never span splits",
        "groups": {
            group: {
                "split": plan.group_split[group],
                "stratum": plan.strata[group],
                "trajectories": sorted(tid for tid, value in plan.groups.items() if value == group),
                "frames": sum(
                    len(item.frames) for item in plan.usable if plan.groups[item.trajectory.trajectory_id] == group
                ),
            }
            for group in sorted(plan.group_split)
        },
        "leakage": plan.leakage,
        "per_split_membership_hash": {
            split: split_hash([(frame_id, value) for frame_id, value in frame_splits if value == split])
            for split in SPLITS
        },
    }
    (out / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8")

    file_hashes = _file_hashes(out)
    frame_counts = {split: len(texts[split]) for split in SPLITS}
    commit = interfaceforge_commit()
    manifest = {
        "schema": SCHEMA,
        "schema_version": 1,
        "method": "nio-canonical-export",
        "created_at": utc_now(),
        "interfaceforge_version": __version__,
        "interfaceforge_commit": commit,
        "source_roots": [str(root) for root in plan.roots],
        "output_root": str(out),
        "config": plan.config.to_dict(),
        "label_convention": LABEL_CONVENTION,
        "preserve_raw_forces": True,
        "include_virial": plan.config.include_virial,
        "type_map": plan.type_map,
        "frame_counts": frame_counts,
        "empty_splits": empty,
        "trajectories": len(plan.usable),
        "extxyz": extxyz,
        "deepmd": {split: str(deepmd_root / split) for split in SPLITS},
        "deepmd_systems": systems,
        "split_hash": split_identity,
        "file_hashes": file_hashes,
        "dataset_hash": dataset_content_hash(file_hashes),
        "summary": _plan_summary(plan),
        "backends": {
            "mace": {
                "train_file": extxyz["train"],
                "valid_file": extxyz["valid"],
                "test_file": extxyz["test"],
                "energy_key": "REF_energy",
                "forces_key": "REF_forces",
            },
            "deepmd": {"dataset_root": str(deepmd_root), "type_map": plan.type_map},
            "nequip": {
                "train_file_path": extxyz["train"],
                "val_file_path": extxyz["valid"],
                "test_file_path": extxyz["test"],
                "key_mapping": {"REF_energy": "total_energy", "REF_forces": "forces"},
                "type_names": plan.type_map,
            },
        },
        "files": {
            "frames": "frames.csv",
            "trajectories": "trajectories.csv",
            "rejected_frames": "rejected_frames.csv",
            "split_manifest": "split_manifest.json",
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return {
        "output_root": str(out),
        "manifest": str(out / "manifest.json"),
        "dataset_hash": manifest["dataset_hash"],
        "split_hash": split_identity,
        "frame_counts": frame_counts,
        "empty_splits": empty,
        "trajectories_exported": len(plan.usable),
        "trajectories_discovered": len(plan.analyses),
        "frames_rejected": len(rejected_rows),
        "leakage_detected": plan.leakage["leakage_detected"],
        "type_map": plan.type_map,
    }


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
def load_dataset_manifest(dataset: str | Path) -> tuple[Path, dict[str, Any]]:
    root = Path(dataset).expanduser().resolve()
    manifest_path = root / "manifest.json" if root.is_dir() else root
    if not manifest_path.is_file():
        raise SafetyError(f"No canonical dataset manifest at {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SafetyError(f"Invalid dataset manifest: {manifest_path}") from exc
    if manifest.get("schema") != SCHEMA:
        raise SafetyError(f"{manifest_path} is not an InterfaceForge canonical NiO dataset manifest")
    return manifest_path.parent, manifest


def verify_dataset(dataset: str | Path, *, check_membership: bool = True) -> dict[str, Any]:
    """Re-derive hashes, frame identity, split identity and cross-backend consistency."""

    from ase.io import iread

    from .training import validate_deepmd_dataset

    root, manifest = load_dataset_manifest(dataset)
    problems: list[str] = []
    recorded = manifest.get("file_hashes", {})
    current = _file_hashes(root)
    for name, digest in recorded.items():
        if name not in current:
            problems.append(f"missing file {name}")
        elif current[name] != digest:
            problems.append(f"hash mismatch {name}")
    extra = sorted(set(current) - set(recorded))
    if extra:
        problems.append(f"files not in manifest: {extra[:5]}")
    if dataset_content_hash(recorded) != manifest.get("dataset_hash"):
        problems.append("dataset_hash does not match file_hashes")

    frame_splits: list[tuple[str, str]] = []
    seen: set[str] = set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    counts = {split: 0 for split in SPLITS}
    for split in SPLITS:
        path = root / f"{split}.extxyz"
        if not path.is_file():
            continue
        for atoms in iread(str(path), index=":"):
            frame_id = str(atoms.info.get("frame_id", ""))
            if not frame_id:
                problems.append(f"{split}: frame without frame_id")
                continue
            if frame_id in seen:
                problems.append(f"duplicate frame_id {frame_id}")
            seen.add(frame_id)
            if atoms.info.get("split") != split:
                problems.append(f"{frame_id}: info split {atoms.info.get('split')} != file {split}")
            frame_splits.append((frame_id, split))
            group_splits[str(atoms.info.get("IF_group", ""))].add(split)
            counts[split] += 1
    if counts != manifest.get("frame_counts"):
        problems.append(f"frame counts {counts} != manifest {manifest.get('frame_counts')}")
    derived = split_hash(frame_splits)
    if derived != manifest.get("split_hash"):
        problems.append("split_hash does not match extxyz membership")
    leaking = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if leaking:
        problems.append(f"groups spanning splits: {leaking[:5]}")

    deepmd_type_map: list[str] | None = None
    try:
        deepmd_type_map, _ = validate_deepmd_dataset(root / "deepmd")
        if deepmd_type_map != manifest.get("type_map"):
            problems.append(f"DeePMD type_map {deepmd_type_map} != manifest {manifest.get('type_map')}")
    except SafetyError as exc:
        problems.append(f"DeePMD validation: {exc}")

    membership: dict[str, Any] = {}
    if check_membership:
        from .mlip_compare import validate_membership

        for split in SPLITS:
            path = root / f"{split}.extxyz"
            if not path.is_file():
                continue
            import tempfile

            with tempfile.TemporaryDirectory() as scratch:
                try:
                    _, summary = validate_membership(path, root / "deepmd" / split, Path(scratch))
                    membership[split] = {
                        "exact_membership": summary["exact_membership"],
                        "frames": summary["frames"],
                        "max_absolute_delta": summary["max_absolute_delta"],
                    }
                    if not summary["exact_membership"]:
                        problems.append(f"{split}: extxyz/DeePMD membership differs")
                except SafetyError as exc:
                    problems.append(f"{split}: extxyz/DeePMD membership: {exc}")
    return {
        "dataset": str(root),
        "valid": not problems,
        "problems": problems,
        "frame_counts": counts,
        "split_hash": derived,
        "dataset_hash": manifest.get("dataset_hash"),
        "groups": len(group_splits),
        "deepmd_type_map": deepmd_type_map,
        "extxyz_deepmd_membership": membership,
    }


def dataset_identity(dataset: str | Path) -> dict[str, Any]:
    """Compact identity block recorded by every training backend."""

    root, manifest = load_dataset_manifest(dataset)
    return {
        "dataset_root": str(root),
        "manifest": str(root / "manifest.json"),
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "dataset_hash": manifest.get("dataset_hash"),
        "split_hash": manifest.get("split_hash"),
        "type_map": manifest.get("type_map"),
        "frame_counts": manifest.get("frame_counts"),
    }
