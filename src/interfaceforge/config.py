"""Campaign and scheduler-profile loading with conservative validation."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fnmatch import fnmatch
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigurationError, SafetyError

SYSTEM_KINDS = {"bulk", "surface", "interface", "molecule", "adsorbate", "defect", "other"}
SPLITS = ("train", "valid", "test")
_STACKING_AXES = {"a", "b", "c"}
_REFERENCE_QUANTITIES = {"work_of_adhesion", "interface_energy", "surface_energy"}
# A system id may be a single segment or several "/"-separated segments (so
# generated run directories can mirror a source tree's own nesting, e.g.
# "Real/N_Term/SiN_TiN_N-term"). Every segment must independently start with
# a letter/digit, which blocks ".." (and "." alone) as a segment -- system.id
# is joined directly onto campaign.root elsewhere, so this is what keeps a
# nested id from being able to escape the campaign root.
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)*$")


@dataclass(frozen=True)
class SystemSpec:
    """One physical system in a campaign."""

    id: str
    kind: str
    structure: Path
    temperature: float | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, Path] = field(default_factory=dict)
    run_glob: str | None = None
    """fnmatch pattern (matched against a collected trajectory's path relative
    to the dataset source root, e.g. "*/interface_*/*") identifying which
    collected OUTCAR trajectories belong to this system, for geometry-class
    stratified error reporting. A trajectory matching no system's run_glob is
    classified "unclassified" rather than silently guessed."""


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
    active_learning: dict[str, Any]
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
        supported = {"dpa1", "dpa2", "dpa2_ft", "dpa3", "dpa4", "se_e2_a"}
        if not architectures or any(item not in supported for item in architectures):
            raise ConfigurationError(
                "models.deepmd.architectures supports dpa1, dpa2, dpa2_ft, dpa3, "
                "dpa4, or se_e2_a"
            )
        if len(set(architectures)) != len(architectures):
            raise ConfigurationError("models.deepmd.architectures contains duplicates")
        if (
            any(item in {"dpa2", "dpa2_ft", "dpa3", "dpa4"} for item in architectures)
            and backend == "tensorflow"
        ):
            raise ConfigurationError("DPA-2/3/4 campaigns require a PyTorch backend")
        finetune: dict[str, Any] = {}
        if "dpa2_ft" in architectures:
            finetune = _mapping(deepmd.get("finetune"), "models.deepmd.finetune")
            pretrained = str(finetune.get("pretrained", "")).strip()
            if not pretrained:
                raise ConfigurationError(
                    "models.deepmd.finetune.pretrained is required when "
                    "models.deepmd.architectures includes dpa2_ft"
                )
            finetune = {
                "pretrained": pretrained,
                "model_branch": str(finetune.get("model_branch", "RANDOM")).strip()
                or "RANDOM",
            }
        deepmd.update(
            {
                "committee": committee,
                "seeds": seeds,
                "backend": backend,
                "descriptor": descriptor,
                "architectures": architectures,
                "finetune": finetune,
            }
        )
        models["deepmd"] = deepmd

    mace = _mapping(models.get("mace"), "models.mace")
    if mace:
        mace["batch_size"] = int(mace.get("batch_size", 16))
        mace["max_num_epochs"] = int(mace.get("max_num_epochs", 200))
        mace["stage2_max_num_epochs"] = int(mace.get("stage2_max_num_epochs", 100))
        if mace["batch_size"] < 1:
            raise ConfigurationError("models.mace.batch_size must be positive")
        if mace["max_num_epochs"] < 1:
            raise ConfigurationError("models.mace.max_num_epochs must be positive")
        if mace["stage2_max_num_epochs"] < 1:
            raise ConfigurationError("models.mace.stage2_max_num_epochs must be positive")

        roi = _mapping(mace.get("roi"), "models.mace.roi")
        if roi:
            roi["enabled"] = bool(roi.get("enabled", False))
            roi["cutoff"] = float(roi.get("cutoff", 3.5))
            roi["interface_multiplier"] = float(roi.get("interface_multiplier", 4.0))
            roi["shell_depth"] = int(roi.get("shell_depth", 0))
            if not math.isfinite(roi["cutoff"]) or roi["cutoff"] <= 0:
                raise ConfigurationError("models.mace.roi.cutoff must be positive")
            if (
                not math.isfinite(roi["interface_multiplier"])
                or roi["interface_multiplier"] < 1
            ):
                raise ConfigurationError(
                    "models.mace.roi.interface_multiplier must be at least one"
                )
            if roi["shell_depth"] < 0:
                raise ConfigurationError("models.mace.roi.shell_depth cannot be negative")

            for key in ("cycle_weight", "stage1_cycle_weight", "stage2_cycle_weight"):
                if key not in roi:
                    continue
                roi[key] = float(roi[key])
                if not math.isfinite(roi[key]) or roi[key] < 0:
                    raise ConfigurationError(
                        f"models.mace.roi.{key} must be finite and non-negative"
                    )

            ranges = roi.get("component_ranges", [])
            range_groups = ranges.values() if isinstance(ranges, Mapping) else [ranges]
            if not isinstance(ranges, (list, Mapping)):
                raise ConfigurationError(
                    "models.mace.roi.component_ranges must be a range list or source-pattern mapping"
                )
            if isinstance(ranges, Mapping) and any(not str(pattern) for pattern in ranges):
                raise ConfigurationError(
                    "models.mace.roi.component_ranges source patterns cannot be empty"
                )
            for group in range_groups:
                if not isinstance(group, list):
                    raise ConfigurationError(
                        "Each models.mace.roi.component_ranges value must be a list"
                    )
                for bounds in group:
                    if not isinstance(bounds, list) or len(bounds) != 2:
                        raise ConfigurationError(
                            "MACE-ROI component ranges must be [start, stop] pairs"
                        )
                    try:
                        start, stop = int(bounds[0]), int(bounds[1])
                    except (TypeError, ValueError) as exc:
                        raise ConfigurationError(
                            "MACE-ROI component range bounds must be integers"
                        ) from exc
                    if any(
                        isinstance(value, bool)
                        or (isinstance(value, float) and not value.is_integer())
                        for value in bounds
                    ):
                        raise ConfigurationError(
                            "MACE-ROI component range bounds must be integers"
                        )
                    if start < 0 or stop <= start:
                        raise ConfigurationError(
                            "MACE-ROI component ranges must satisfy 0 <= start < stop"
                        )
            roi["component_key"] = str(roi.get("component_key", "IF_component"))
            if not roi["component_key"]:
                raise ConfigurationError("models.mace.roi.component_key cannot be empty")
            mace["roi"] = roi
        models["mace"] = mace


def _validate_active_learning(active_learning: dict[str, Any], models: dict[str, Any]) -> None:
    allowed = {
        "enabled",
        "engine",
        "approval_required",
        "max_iterations",
        "output_root",
        "ai2kit",
    }
    unknown = sorted(set(active_learning) - allowed)
    if unknown:
        raise ConfigurationError(f"Unknown active_learning keys: {', '.join(unknown)}")

    enabled = active_learning.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ConfigurationError("active_learning.enabled must be a boolean")
    active_learning["enabled"] = enabled
    active_learning["engine"] = str(active_learning.get("engine", "ai2kit")).lower()
    active_learning["approval_required"] = active_learning.get("approval_required", True)
    try:
        active_learning["max_iterations"] = int(active_learning.get("max_iterations", 1))
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("active_learning.max_iterations must be an integer") from exc
    active_learning["output_root"] = str(
        active_learning.get("output_root", "runs/active_learning/ai2kit")
    )
    adapter = _mapping(active_learning.get("ai2kit"), "active_learning.ai2kit")
    active_learning["ai2kit"] = adapter

    adapter_allowed = {
        "workflow",
        "version",
        "omb_version",
        "executor_name",
        "trainer",
        "explorer",
        "labeler",
        "selector",
        "architecture",
        "backend",
        "model_count",
        "training_artifacts",
        "validation_artifacts",
        "exploration_artifacts",
        "trust_force_low",
        "trust_force_high",
        "selection_limit",
        "experimental_compatibility",
        "committee_models",
        "committee_seeds",
        "md_steps",
        "sample_frequency",
        "timestep_fs",
        "friction_ps",
        "equilibration_frames",
        "max_force_ev_ang",
        "default_dtype",
        "use_poor_frames",
        "update_md_structures",
        "md_workers",
        "label_workers",
    }
    adapter_unknown = sorted(set(adapter) - adapter_allowed)
    if adapter_unknown:
        raise ConfigurationError(
            f"Unknown active_learning.ai2kit keys: {', '.join(adapter_unknown)}"
        )

    if not active_learning["enabled"]:
        return
    if active_learning["engine"] != "ai2kit":
        raise ConfigurationError("active_learning.engine must be ai2kit")
    if active_learning["approval_required"] is not True:
        raise SafetyError("AI2-kit active learning requires approval_required: true")
    if active_learning["max_iterations"] < 1:
        raise ConfigurationError("active_learning.max_iterations must be positive")
    workflow = str(adapter.get("workflow", "cll_deepmd")).lower()
    if workflow not in {"cll_deepmd", "tesla_mace"}:
        raise ConfigurationError(
            "active_learning.ai2kit.workflow must be 'cll_deepmd' or 'tesla_mace'"
        )
    adapter["workflow"] = workflow
    fixed = {
        "version": "1.0.9",
        "trainer": "mace" if workflow == "tesla_mace" else "deepmd",
        "explorer": "openmm" if workflow == "tesla_mace" else "lammps",
        "labeler": "vasp",
        "selector": "model_deviation",
    }
    for key, expected in fixed.items():
        actual = str(adapter.get(key, expected)).lower()
        if actual != expected:
            raise ConfigurationError(
                f"active_learning.ai2kit.{key} must be {expected!r}; got {actual!r}"
            )
        adapter[key] = actual
    adapter["executor_name"] = str(adapter.get("executor_name", "loni")).strip()
    if not adapter["executor_name"]:
        raise ConfigurationError("active_learning.ai2kit.executor_name is required")

    for key in ("trust_force_low", "trust_force_high"):
        if key not in adapter:
            raise ConfigurationError(f"active_learning.ai2kit.{key} is required")
        try:
            adapter[key] = float(adapter[key])
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"active_learning.ai2kit.{key} must be numeric") from exc
    if not 0 <= adapter["trust_force_low"] < adapter["trust_force_high"]:
        raise ConfigurationError(
            "AI2-kit trust thresholds require 0 <= trust_force_low < trust_force_high"
        )

    for key, default in (("model_count", 4), ("selection_limit", 20)):
        try:
            adapter[key] = int(adapter.get(key, default))
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"active_learning.ai2kit.{key} must be an integer") from exc
        if adapter[key] < 1:
            raise ConfigurationError(f"active_learning.ai2kit.{key} must be positive")

    for key in ("training_artifacts", "validation_artifacts", "exploration_artifacts"):
        values = adapter.get(key)
        if not isinstance(values, list) or not values or any(not str(value).strip() for value in values):
            raise ConfigurationError(f"active_learning.ai2kit.{key} must be a non-empty list")
        adapter[key] = [str(value) for value in values]

    if workflow == "tesla_mace":
        adapter["omb_version"] = str(adapter.get("omb_version", "0.7.2"))
        committee_models = adapter.get("committee_models")
        if (
            not isinstance(committee_models, list)
            or not committee_models
            or any(not str(value).strip() for value in committee_models)
        ):
            raise ConfigurationError(
                "active_learning.ai2kit.committee_models must list the ready MACE model files"
            )
        adapter["committee_models"] = [str(value) for value in committee_models]
        committee_seeds = adapter.get("committee_seeds", [11, 23, 37, 53])
        if not isinstance(committee_seeds, list):
            raise ConfigurationError("active_learning.ai2kit.committee_seeds must be a list")
        try:
            committee_seeds = [int(value) for value in committee_seeds]
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                "active_learning.ai2kit.committee_seeds must contain integers"
            ) from exc
        if len(committee_seeds) < adapter["model_count"]:
            raise ConfigurationError(
                "active_learning.ai2kit.committee_seeds must cover every MACE model"
            )
        if len(set(committee_seeds[: adapter["model_count"]])) != adapter["model_count"]:
            raise ConfigurationError("AI2-kit MACE committee seeds must be unique")
        if len(adapter["committee_models"]) != adapter["model_count"]:
            raise ConfigurationError(
                "active_learning.ai2kit.committee_models must match model_count"
            )
        adapter["committee_seeds"] = committee_seeds[: adapter["model_count"]]
        integer_defaults = {
            "md_steps": 10000,
            "sample_frequency": 20,
            "equilibration_frames": 10,
            "use_poor_frames": 0,
            "update_md_structures": 0,
            "md_workers": 1,
            "label_workers": 1,
        }
        for key, default in integer_defaults.items():
            try:
                value = int(adapter.get(key, default))
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(f"active_learning.ai2kit.{key} must be an integer") from exc
            if value < 0 or (
                key in {"md_steps", "sample_frequency", "md_workers", "label_workers"}
                and value < 1
            ):
                raise ConfigurationError(f"active_learning.ai2kit.{key} has an invalid value")
            adapter[key] = value
        saved_frames = adapter["md_steps"] // adapter["sample_frequency"]
        if saved_frames < 1:
            raise ConfigurationError(
                "active_learning.ai2kit.sample_frequency cannot exceed md_steps"
            )
        if adapter["equilibration_frames"] >= saved_frames:
            raise ConfigurationError(
                "active_learning.ai2kit.equilibration_frames must leave at least one saved frame"
            )
        for key, default in {
            "timestep_fs": 0.5,
            "friction_ps": 1.0,
            "max_force_ev_ang": 50.0,
        }.items():
            try:
                value = float(adapter.get(key, default))
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(f"active_learning.ai2kit.{key} must be numeric") from exc
            if not math.isfinite(value) or value <= 0:
                raise ConfigurationError(f"active_learning.ai2kit.{key} must be positive")
            adapter[key] = value
        dtype = str(adapter.get("default_dtype", "float64")).lower()
        if dtype not in {"float32", "float64"}:
            raise ConfigurationError(
                "active_learning.ai2kit.default_dtype must be float32 or float64"
            )
        adapter["default_dtype"] = dtype

        mace = _mapping(models.get("mace"), "models.mace")
        if not mace.get("enabled", False):
            raise ConfigurationError("AI2-kit TESLA MACE requires models.mace.enabled: true")
        return

    architecture = str(adapter.get("architecture", "se_e2_a")).lower()
    backend = str(adapter.get("backend", "tensorflow")).lower()
    experimental = adapter.get("experimental_compatibility", False)
    if not isinstance(experimental, bool):
        raise ConfigurationError(
            "active_learning.ai2kit.experimental_compatibility must be a boolean"
        )
    if (architecture, backend) != ("se_e2_a", "tensorflow") and not experimental:
        raise ConfigurationError(
            "AI2-kit 1.0.9 MVP supports se_e2_a/tensorflow; other combinations "
            "require experimental_compatibility: true"
        )
    adapter.update(
        {
            "architecture": architecture,
            "backend": backend,
            "experimental_compatibility": experimental,
        }
    )

    deepmd = _mapping(models.get("deepmd"), "models.deepmd")
    if not deepmd.get("enabled", False):
        raise ConfigurationError("AI2-kit requires models.deepmd.enabled: true")
    architectures = [str(value).lower() for value in deepmd.get("architectures", [])]
    if architectures != [architecture]:
        raise ConfigurationError(
            "AI2-kit requires exactly one models.deepmd architecture matching "
            "active_learning.ai2kit.architecture"
        )
    if str(deepmd.get("backend", "tensorflow")).lower() != backend:
        raise ConfigurationError(
            "active_learning.ai2kit.backend must match models.deepmd.backend"
        )
    if adapter["model_count"] != deepmd["committee"]:
        raise ConfigurationError(
            "active_learning.ai2kit.model_count must match models.deepmd.committee"
        )
    seeds = list(deepmd.get("seeds", []))
    if adapter["model_count"] > len(seeds):
        raise ConfigurationError("models.deepmd.seeds must cover every AI2-kit model")
    if len(set(seeds[: adapter["model_count"]])) != adapter["model_count"]:
        raise ConfigurationError("AI2-kit committee seeds must be unique")


def _normalize_interface_metadata(interfaces: Any) -> list[dict[str, Any]]:
    if not isinstance(interfaces, list):
        raise ConfigurationError("validation.interfaces must be a list")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(interfaces):
        entry = _mapping(item, f"validation.interfaces[{index}]")
        pattern = str(entry.get("match", "")).strip()
        if not pattern:
            raise ConfigurationError(
                f"validation.interfaces[{index}].match is required "
                "(an fnmatch pattern against the interface leaf / system id)"
            )
        clean: dict[str, Any] = {"match": pattern}
        axis = entry.get("stacking_axis")
        if axis is not None:
            axis = str(axis).strip().lower()
            if axis not in _STACKING_AXES:
                raise ConfigurationError(
                    f"validation.interfaces[{index}].stacking_axis must be a, b, or c"
                )
            clean["stacking_axis"] = axis
        if entry.get("n_interfaces") is not None:
            try:
                n_interfaces = int(entry["n_interfaces"])
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(
                    f"validation.interfaces[{index}].n_interfaces must be an integer"
                ) from exc
            if n_interfaces < 1:
                raise ConfigurationError(
                    f"validation.interfaces[{index}].n_interfaces must be positive"
                )
            clean["n_interfaces"] = n_interfaces
        if "polar_termination" in entry:
            if not isinstance(entry["polar_termination"], bool):
                raise ConfigurationError(
                    f"validation.interfaces[{index}].polar_termination must be a boolean"
                )
            clean["polar_termination"] = entry["polar_termination"]
        for key in ("orientation", "termination", "family", "surface"):
            if entry.get(key) is not None:
                clean[key] = str(entry[key]).strip()
        components = entry.get("components")
        if components is not None:
            if not isinstance(components, list):
                raise ConfigurationError(
                    f"validation.interfaces[{index}].components must be a list"
                )
            clean["components"] = [
                dict(_mapping(component, f"validation.interfaces[{index}].components[{c}]"))
                for c, component in enumerate(components)
            ]
        normalized.append(clean)
    return normalized


def _normalize_reference_entry(entry: Mapping[str, Any], index: int) -> dict[str, Any]:
    where = f"validation.references[{index}]"
    key = str(entry.get("key", "")).strip()
    if not key:
        raise ConfigurationError(f"{where}.key is required")
    quantity = str(entry.get("quantity", "")).strip().lower()
    if quantity not in _REFERENCE_QUANTITIES:
        raise ConfigurationError(
            f"{where}.quantity must be one of {sorted(_REFERENCE_QUANTITIES)}"
        )
    values = entry.get("values")
    if not isinstance(values, list) or not values:
        raise ConfigurationError(f"{where}.values must be a non-empty list")
    clean_values: list[dict[str, Any]] = []
    for v_index, value in enumerate(values):
        value_entry = _mapping(value, f"{where}.values[{v_index}]")
        try:
            magnitude = float(value_entry["value_j_per_m2"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"{where}.values[{v_index}].value_j_per_m2 must be a number"
            ) from exc
        if not math.isfinite(magnitude):
            raise ConfigurationError(
                f"{where}.values[{v_index}].value_j_per_m2 must be finite"
            )
        selector = _mapping(value_entry.get("match"), f"{where}.values[{v_index}].match")
        clean_value: dict[str, Any] = {
            "match": {str(k): str(val).strip() for k, val in selector.items()},
            "value_j_per_m2": magnitude,
        }
        if value_entry.get("note") is not None:
            clean_value["note"] = str(value_entry["note"]).strip()
        clean_values.append(clean_value)
    tolerance = entry.get("tolerance_j_per_m2", 0.5)
    try:
        tolerance = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{where}.tolerance_j_per_m2 must be a number") from exc
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ConfigurationError(
            f"{where}.tolerance_j_per_m2 must be finite and non-negative"
        )
    clean: dict[str, Any] = {
        "key": key,
        "quantity": quantity,
        "tolerance_j_per_m2": tolerance,
        "values": clean_values,
    }
    for extra in ("doi", "citation", "note"):
        if entry.get(extra) is not None:
            clean[extra] = str(entry[extra]).strip()
    if entry.get("method") is not None:
        clean["method"] = dict(_mapping(entry.get("method"), f"{where}.method"))
    return clean


def _validate_validation(validation: dict[str, Any]) -> None:
    """Validate the structured ``validation.interfaces`` / ``validation.references``
    blocks and expand ``validation.reference_profiles``.

    Legacy boolean keys (``parity``, ``work_of_adhesion``, ...) are untouched --
    they predate this and are still only advisory.
    """

    if validation.get("interfaces") is not None:
        validation["interfaces"] = _normalize_interface_metadata(validation["interfaces"])

    profile_entries: list[dict[str, Any]] = []
    profiles = validation.get("reference_profiles")
    if profiles is not None:
        if not isinstance(profiles, list) or any(not str(name).strip() for name in profiles):
            raise ConfigurationError(
                "validation.reference_profiles must be a list of profile names"
            )
        names = [str(name).strip() for name in profiles]
        from .reference_import import resolve_reference_profiles

        profile_entries = resolve_reference_profiles(names)
        validation["reference_profiles"] = names

    hand_written = validation.get("references")
    if hand_written is None:
        hand_written = []
    if not isinstance(hand_written, list):
        raise ConfigurationError("validation.references must be a list")

    if not profile_entries and not hand_written:
        return

    merged: dict[tuple[str, str], dict[str, Any]] = {}
    order: list[tuple[str, str]] = []
    for entry in [*profile_entries, *hand_written]:
        mapping = _mapping(entry, "validation.references[]")
        ident = (
            str(mapping.get("key", "")).strip(),
            str(mapping.get("quantity", "")).strip().lower(),
        )
        if ident not in merged:
            order.append(ident)
        merged[ident] = dict(mapping)  # hand-written entries come last and win
    validation["references"] = [
        _normalize_reference_entry(merged[ident], index)
        for index, ident in enumerate(order)
    ]


def merge_interface_metadata(
    entries: Sequence[Mapping[str, Any]] | None, leaf: str
) -> dict[str, Any]:
    """Merge every ``validation.interfaces`` entry whose fnmatch ``match``
    matches ``leaf``. Earlier entries win on a key conflict; ``{}`` if none match.
    """

    merged: dict[str, Any] = {}
    for entry in entries or []:
        pattern = entry.get("match")
        if pattern and fnmatch(leaf, str(pattern)):
            for key, value in entry.items():
                if key != "match":
                    merged.setdefault(key, value)
    return merged


def references_for(
    references: Sequence[Mapping[str, Any]] | None, quantity: str
) -> list[dict[str, Any]]:
    """The normalized ``validation.references`` entries for one quantity."""

    return [
        dict(entry)
        for entry in references or []
        if str(entry.get("quantity", "")).strip().lower() == quantity
    ]


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
        run_glob = entry.get("run_glob")
        if run_glob is not None and not isinstance(run_glob, str):
            raise ConfigurationError(f"systems[{index}].run_glob must be a string")
        systems.append(
            SystemSpec(
                id=system_id,
                kind=kind,
                structure=_resolve(root, str(structure)).resolve(),
                temperature=float(temperature) if temperature is not None else None,
                tags=_mapping(entry.get("tags"), f"systems[{index}].tags"),
                inputs={
                    str(name): _resolve(root, str(value)).resolve()
                    for name, value in _mapping(
                        entry.get("inputs"), f"systems[{index}].inputs"
                    ).items()
                },
                run_glob=run_glob,
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
    active_learning = _mapping(data.get("active_learning"), "active_learning")
    exploration = _mapping(data.get("exploration"), "exploration")
    validation = _mapping(data.get("validation"), "validation")
    _validate_dataset(dataset)
    _validate_models(models)
    _validate_active_learning(active_learning, models)
    _validate_validation(validation)

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
        active_learning=active_learning,
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
