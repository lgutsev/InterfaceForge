"""Campaign and scheduler-profile loading with conservative validation."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigurationError, SafetyError

SYSTEM_KINDS = {"bulk", "surface", "interface", "molecule", "adsorbate", "other"}
SPLITS = ("train", "valid", "test")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class SystemSpec:
    """One physical system in a campaign."""

    id: str
    kind: str
    structure: Path
    temperature: float | None = None
    tags: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Campaign:
    """Validated, path-resolved campaign configuration."""

    path: Path
    root: Path
    name: str
    description: str
    profile_path: Path
    systems: tuple[SystemSpec, ...]
    reference: dict[str, Any]
    stages: dict[str, Any]
    dataset: dict[str, Any]
    models: dict[str, Any]
    exploration: dict[str, Any]
    validation: dict[str, Any]
    raw: dict[str, Any]


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{where} must be a mapping")
    return dict(value)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _validate_ratios(dataset: dict[str, Any]) -> None:
    values = dataset.get("ratios", [0.8, 0.1, 0.1])
    if not isinstance(values, list) or len(values) != 3:
        raise ConfigurationError("dataset.ratios must contain train, valid, test")
    try:
        ratios = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("dataset.ratios must be numeric") from exc
    if any(value < 0 for value in ratios) or sum(ratios) <= 0:
        raise ConfigurationError("dataset.ratios must be non-negative with a positive sum")
    dataset["ratios"] = ratios


def _validate_dataset(dataset: dict[str, Any]) -> None:
    strategy = str(dataset.get("strategy", "grouped")).lower()
    if strategy not in {"grouped", "guarded"}:
        raise ConfigurationError("dataset.strategy must be 'grouped' or 'guarded'")
    dataset["strategy"] = strategy
    _validate_ratios(dataset)

    for key in ("stride", "guard_frames"):
        default = 5 if key == "stride" else 20
        try:
            value = int(dataset.get(key, default))
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"dataset.{key} must be an integer") from exc
        if key == "stride" and value < 1:
            raise ConfigurationError("dataset.stride must be positive")
        if key == "guard_frames" and value < 0:
            raise ConfigurationError("dataset.guard_frames cannot be negative")
        dataset[key] = value

    # This is an explicit scientific invariant of InterfaceForge. Constraints
    # may be stored as masks, but reference force labels remain untouched.
    if dataset.get("preserve_raw_forces", True) is not True:
        raise SafetyError(
            "InterfaceForge does not zero reference forces on constrained atoms. "
            "Set dataset.preserve_raw_forces: true and store constraints as metadata."
        )
    dataset["preserve_raw_forces"] = True
    dataset["include_virial"] = bool(dataset.get("include_virial", False))

    type_map = dataset.get("type_map", [])
    if type_map is None:
        type_map = []
    if not isinstance(type_map, list) or any(not str(item).strip() for item in type_map):
        raise ConfigurationError("dataset.type_map must be a list of element symbols")
    if len(set(type_map)) != len(type_map):
        raise ConfigurationError("dataset.type_map contains duplicates")
    dataset["type_map"] = [str(item) for item in type_map]


def _validate_models(models: dict[str, Any]) -> None:
    deepmd = _mapping(models.get("deepmd"), "models.deepmd")
    if deepmd:
        committee = int(deepmd.get("committee", 4))
        seeds = [int(seed) for seed in deepmd.get("seeds", [11, 23, 37, 53])]
        if committee < 1:
            raise ConfigurationError("models.deepmd.committee must be positive")
        if len(seeds) < committee:
            raise ConfigurationError("models.deepmd.seeds must cover every committee member")
        if len(set(seeds[:committee])) != committee:
            raise ConfigurationError("models.deepmd committee seeds must be unique")
        backend = str(deepmd.get("backend", "tensorflow")).lower()
        if backend not in {"tensorflow", "pytorch", "pt_expt"}:
            raise ConfigurationError(
                "models.deepmd.backend must be tensorflow, pytorch, or pt_expt"
            )
        descriptor = str(deepmd.get("descriptor", "dpa1")).lower()
        architectures = [
            str(item).lower()
            for item in deepmd.get("architectures", [descriptor])
        ]
        supported = {"dpa1", "dpa2", "dpa3", "dpa4", "se_e2_a"}
        if not architectures or any(item not in supported for item in architectures):
            raise ConfigurationError(
                "models.deepmd.architectures supports dpa1, dpa2, dpa3, "
                "dpa4, or se_e2_a"
            )
        if len(set(architectures)) != len(architectures):
            raise ConfigurationError("models.deepmd.architectures contains duplicates")
        if any(item in {"dpa2", "dpa3", "dpa4"} for item in architectures) and backend == "tensorflow":
            raise ConfigurationError("DPA-2/3/4 campaigns require a PyTorch backend")
        deepmd.update(
            {
                "committee": committee,
                "seeds": seeds,
                "backend": backend,
                "descriptor": descriptor,
                "architectures": architectures,
            }
        )
        models["deepmd"] = deepmd

    mace = _mapping(models.get("mace"), "models.mace")
    if mace:
        mace["batch_size"] = int(mace.get("batch_size", 16))
        mace["max_num_epochs"] = int(mace.get("max_num_epochs", 200))
        models["mace"] = mace


def load_campaign(path: str | Path) -> Campaign:
    """Load and validate a campaign YAML file."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Campaign file does not exist: {config_path}")
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ConfigurationError("Campaign root must be a mapping")
    raw = copy.deepcopy(dict(data))

    if int(data.get("schema_version", 0)) != 1:
        raise ConfigurationError("Only schema_version: 1 is supported")

    root = config_path.parent
    project = _mapping(data.get("project"), "project")
    name = str(project.get("name", "")).strip()
    if not name:
        raise ConfigurationError("project.name is required")
    description = str(project.get("description", "")).strip()

    profile_value = data.get("profile")
    if not profile_value:
        raise ConfigurationError("profile is required")
    profile_path = _resolve(root, str(profile_value)).resolve()

    systems_data = data.get("systems")
    if not isinstance(systems_data, list) or not systems_data:
        raise ConfigurationError("systems must be a non-empty list")
    systems: list[SystemSpec] = []
    seen: set[str] = set()
    for index, item in enumerate(systems_data):
        entry = _mapping(item, f"systems[{index}]")
        system_id = str(entry.get("id", "")).strip()
        if not system_id or not _SAFE_ID.match(system_id):
            raise ConfigurationError(
                f"systems[{index}].id must contain only letters, digits, '.', '_' or '-'"
            )
        if system_id in seen:
            raise ConfigurationError(f"Duplicate system id: {system_id}")
        seen.add(system_id)
        kind = str(entry.get("kind", "other")).lower()
        if kind not in SYSTEM_KINDS:
            raise ConfigurationError(
                f"Unknown system kind {kind!r}; choose from {sorted(SYSTEM_KINDS)}"
            )
        structure = entry.get("structure")
        if not structure:
            raise ConfigurationError(f"systems[{index}].structure is required")
        temperature = entry.get("temperature")
        systems.append(
            SystemSpec(
                id=system_id,
                kind=kind,
                structure=_resolve(root, str(structure)).resolve(),
                temperature=float(temperature) if temperature is not None else None,
                tags=_mapping(entry.get("tags"), f"systems[{index}].tags"),
            )
        )

    reference = _mapping(data.get("reference"), "reference")
    if str(reference.get("engine", "vasp")).lower() != "vasp":
        raise ConfigurationError("The reference engine currently must be vasp")
    reference["engine"] = "vasp"
    reference["inputs"] = _mapping(reference.get("inputs"), "reference.inputs")

    stages = _mapping(data.get("stages"), "stages")
    dataset = _mapping(data.get("dataset"), "dataset")
    models = _mapping(data.get("models"), "models")
    exploration = _mapping(data.get("exploration"), "exploration")
    validation = _mapping(data.get("validation"), "validation")
    _validate_dataset(dataset)
    _validate_models(models)

    return Campaign(
        path=config_path,
        root=root,
        name=name,
        description=description,
        profile_path=profile_path,
        systems=tuple(systems),
        reference=reference,
        stages=stages,
        dataset=dataset,
        models=models,
        exploration=exploration,
        validation=validation,
        raw=raw,
    )


def load_profile(path: str | Path) -> dict[str, Any]:
    """Load a scheduler profile without interpreting engine-specific commands."""

    profile_path = Path(path).expanduser().resolve()
    if not profile_path.is_file():
        raise ConfigurationError(f"Scheduler profile does not exist: {profile_path}")
    try:
        profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid profile YAML in {profile_path}: {exc}") from exc
    if not isinstance(profile, Mapping):
        raise ConfigurationError("Scheduler profile root must be a mapping")
    result = dict(profile)
    scheduler = str(result.get("scheduler", "")).lower()
    if scheduler not in {"slurm", "local"}:
        raise ConfigurationError("profile.scheduler must be slurm or local")
    jobs = result.get("jobs")
    if not isinstance(jobs, Mapping) or not jobs:
        raise ConfigurationError("profile.jobs must be a non-empty mapping")
    result["scheduler"] = scheduler
    result["jobs"] = {str(key): _mapping(value, f"profile.jobs.{key}") for key, value in jobs.items()}
    result["_path"] = str(profile_path)
    return result


def write_default_campaign(destination: str | Path, *, force: bool = False) -> Path:
    """Write the packaged starter campaign."""

    output = Path(destination).expanduser().resolve()
    if output.exists() and not force:
        raise SafetyError(f"Refusing to overwrite existing file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    template = resources.files("interfaceforge").joinpath("templates/campaign.yaml")
    output.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    return output
