"""Resolve trained MACE / DeePMD models into ASE calculators for ``iface properties``.

Models come from repeatable ``--model`` paths or from a collected InterfaceForge
committee bundle (``iface committee collect``); the bundle is verified with
:func:`interfaceforge.committee.verify_committee_bundle` and read as-is. One
calculator is built per member: nothing here averages a committee.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .committee import verify_committee_bundle
from .errors import ConfigurationError, DependencyError, SafetyError

ENGINES = ("mace", "deepmd")
_SEED = re.compile(r"seed[_-]?(-?\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class ModelSpec:
    """One committee member: where it lives and who it is."""

    path: Path
    engine: str
    label: str
    seed: int | None = None
    source: str = "--model"


def _seed_from_path(path: Path) -> int | None:
    for part in reversed(path.with_suffix("").parts):
        match = _SEED.search(part)
        if match:
            return int(match.group(1))
    return None


def _labels(paths: list[Path], seeds: list[int | None]) -> list[str]:
    return [f"member_{index:03d}" + (f"_seed_{seed}" if seed is not None else "") for index, seed in enumerate(seeds)]


def models_from_paths(paths: list[str], engine: str) -> list[ModelSpec]:
    resolved: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Model not found: {path}")
        if path in resolved:
            raise SafetyError(f"Duplicate model path: {path}")
        resolved.append(path)
    seeds = [_seed_from_path(path) for path in resolved]
    return [
        ModelSpec(path=path, engine=engine, label=label, seed=seed)
        for path, seed, label in zip(resolved, seeds, _labels(resolved, seeds), strict=True)
    ]


def models_from_committee(bundle: str, engine: str) -> list[ModelSpec]:
    root = Path(bundle).expanduser().resolve()
    if root.suffix.lower() == ".zip":
        raise ConfigurationError(
            f"{root} is a committee archive; extract it and pass the bundle directory to --committee"
        )
    verified = verify_committee_bundle(root)
    if verified["engine"] != engine:
        raise ConfigurationError(f"Committee {root} is a {verified['engine']} bundle, but --engine is {engine}")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    members = manifest["members"]
    paths = [(root / str(member["stored_model"])).resolve() for member in members]
    seeds = [member.get("seed") for member in members]
    return [
        ModelSpec(path=path, engine=engine, label=label, seed=seed, source=f"committee:{root}")
        for path, seed, label in zip(paths, seeds, _labels(paths, seeds), strict=True)
    ]


def resolve_dtype(engine: str, dtype: str | None) -> str | None:
    """MACE runs in the requested precision (default float64); DeePMD precision is fixed by the frozen model."""

    if engine == "mace":
        value = dtype or "float64"
        if value not in ("float32", "float64"):
            raise ConfigurationError("--dtype must be float32 or float64")
        return value
    if dtype is not None:
        raise ConfigurationError(
            "--dtype is not supported for --engine deepmd: a frozen DeePMD model's precision is set at "
            "training/freeze time and the ASE calculator cannot change it"
        )
    return None


def build_calculator(spec: ModelSpec, *, device: str, dtype: str | None) -> Any:
    """Instantiate the trained local model as an ASE calculator (never a foundation-model download)."""

    if spec.engine == "mace":
        try:
            from mace.calculators import MACECalculator
        except ImportError as exc:
            raise DependencyError("--engine mace needs mace-torch (pip install -e '.[mace-roi]')") from exc
        return MACECalculator(model_paths=[str(spec.path)], device=device, default_dtype=dtype)
    if spec.engine == "deepmd":
        try:
            from deepmd.calculator import DP
        except ImportError as exc:
            raise DependencyError("--engine deepmd needs deepmd-kit in this environment") from exc
        # DP takes no device argument (see device_note). Native neighbour lists, as in swap-mc and
        # separation-energy, avoid the DeePMD 3.2.0b0 vesin/TorchScript failure.
        return DP(model=str(spec.path), nlist_backend="native")
    raise ConfigurationError(f"Unknown engine {spec.engine!r}; expected one of {ENGINES}")


def precision_record(spec: ModelSpec, dtype: str | None) -> dict[str, Any]:
    """The checkpoint's own precision next to the requested one.

    MACECalculator silently casts a float32 checkpoint to float64 (and back);
    casting changes the arithmetic, not the precision the weights were trained
    in, so the record keeps both.
    """

    if spec.engine != "mace":
        return {"model_native_dtype": None, "dtype_converted": None}
    import torch

    model = torch.load(str(spec.path), map_location="cpu", weights_only=False)
    native = str(next(model.parameters()).dtype).removeprefix("torch.")
    return {"model_native_dtype": native, "dtype_converted": native != dtype}


def device_note(engine: str) -> str | None:
    if engine == "deepmd":
        return "deepmd.calculator.DP has no device argument; it uses CUDA when visible (CUDA_VISIBLE_DEVICES)"
    return None


def engine_packages(engine: str) -> tuple[str, ...]:
    return {"mace": ("mace-torch", "torch"), "deepmd": ("deepmd-kit", "torch")}.get(engine, ())
