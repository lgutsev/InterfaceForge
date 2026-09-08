# ruff: noqa: E501
"""Fixed-composition N/O swap Monte Carlo over MLIP-relaxed interface structures.

This is a chemical-ordering search: with the oxygen content of an oxynitride
interface held fixed, N and O are exchanged on an explicit set of anion sites,
each proposed arrangement is relaxed with a MACE or DeePMD committee, and a
Metropolis walk over the relaxed structures is used to find low-energy
arrangements. It produces a provenance-stamped archive of candidate interfaces
and a diverse shortlist for DFT verification -- the input to a "random vs
searched" ordering/adhesion comparison.

Method and workflow after PAIPAI (S. Zhu & R. Arroyave, "Ground-State Structure
Search of Defective High-Entropy Alloys Using Machine-Learning Potentials and
Monte Carlo Sampling", Computational Materials Science 270 (2026) 114752;
https://github.com/siyazhu/PAIPAI). InterfaceForge adapts it: swaps are
restricted to explicit anion sites with the Si/Ti sublattices frozen, the cell
is fixed and the interface's own ``FixAtoms`` constraints are kept, the
calculator is an InterfaceForge MACE/DeePMD committee rather than
``GraceCalculator``, and ``kT`` is supplied directly as an eV energy scale.

The search is a *low-energy structure search*, not equilibrium sampling -- see
``CAVEATS``. The MLIP/relaxation backend is isolated behind :class:`Relaxer` so
the search, archive, and selection logic run without ``mace-torch`` or
``deepmd-kit`` installed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .errors import DependencyError, SafetyError
from .state import sha256_file, utc_now

SCHEMA_VERSION = 1
QUANTITY = "chemical_ordering_search"
_AXES = {"a": 0, "b": 1, "c": 2}
_REGIONS = ("all", "lower", "upper", "interface")

CITATION: dict[str, Any] = {
    "method": "Monte Carlo over MLIP-relaxed anion-swap configurations, after PAIPAI",
    "software": {
        "name": "PAIPAI",
        "url": "https://github.com/siyazhu/PAIPAI",
        "source_reviewed_commit": "00ff18b8",
    },
    "paper": {
        "authors": "S. Zhu and R. Arroyave",
        "title": (
            "Ground-State Structure Search of Defective High-Entropy Alloys Using "
            "Machine-Learning Potentials and Monte Carlo Sampling"
        ),
        "journal": "Computational Materials Science",
        "volume": "270",
        "pages": "114752",
        "year": 2026,
    },
    "adaptations": [
        "N<->O swaps restricted to explicit anion sites; Si/Ti sublattice identities fixed",
        "fixed cell; the interface's own FixAtoms constraints are preserved during relaxation",
        "InterfaceForge MACE / DeePMD committee calculators in place of GraceCalculator",
        "kT supplied directly as an eV energy scale (no implicit Boltzmann/Kelvin conversion)",
        "coarse screen relaxation then a strict refinement of the shortlist",
    ],
}

CAVEATS: tuple[str, ...] = (
    "kT is an eV energy scale, not a temperature: acceptance uses exp(-dE / kT) with "
    "dE and kT both in eV and no Boltzmann constant. 0.025 eV corresponds to ~300 K "
    "only when dE is a genuine total-energy difference.",
    "This is a low-energy structure search. The greedy screen relaxation and the walk "
    "over relaxed structures bias the archived population toward low energy; it is not a "
    "finite-temperature equilibrium ensemble and the populations must not be read as "
    "equilibrium site occupancies.",
    "Equilibrium ordering, even where it is sampled, does not by itself predict the "
    "arrangement produced by deposition kinetics.",
    "Relative energies are from a single MLIP potential-energy surface. DFT-check the "
    "selected low-energy and high-committee-spread candidates before drawing any "
    "ordering or adhesion conclusion.",
)


# --------------------------------------------------------------------------- ASE


def _ase() -> dict[str, Any]:
    try:
        from ase.constraints import FixAtoms  # noqa: F401
        from ase.io import read, write
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without ASE
        raise DependencyError(
            "swap-mc needs ASE to read/write structures; install interfaceforge[vasp]"
        ) from exc
    return {"read": read, "write": write, "FixAtoms": FixAtoms}


def _read_atoms(path: str | Path) -> Any:
    atoms = _ase()["read"](str(Path(path).expanduser().resolve()))
    if not atoms.cell.rank:
        raise SafetyError(f"{path} has no periodic cell; swap-mc needs the interface cell")
    return atoms


def _frozen_mask(atoms: Any) -> np.ndarray:
    mask = np.zeros(len(atoms), dtype=bool)
    for constraint in getattr(atoms, "constraints", []):
        indices = getattr(constraint, "index", getattr(constraint, "a", None))
        if indices is not None:
            mask[np.atleast_1d(np.asarray(indices, dtype=int))] = True
    return mask


# ---------------------------------------------------------------- site resolution


@dataclass(frozen=True)
class SwapSites:
    """Anion sites eligible for N<->O exchange and their move classes."""

    substituent: str
    host: str
    stacking_axis: str
    eligible: tuple[int, ...]
    groups: dict[str, tuple[int, ...]]
    layer_of: dict[int, int]
    region_of: dict[int, str]
    interface_z: float | None
    initial_substituent: tuple[int, ...]
    excluded_frozen: tuple[int, ...]

    @property
    def n_substituent(self) -> int:
        return len(self.initial_substituent)

    def to_dict(self) -> dict[str, Any]:
        return {
            "substituent": self.substituent,
            "host": self.host,
            "stacking_axis": self.stacking_axis,
            "n_eligible": len(self.eligible),
            "eligible": list(self.eligible),
            "groups": {name: list(idx) for name, idx in self.groups.items()},
            "layer_of": {str(k): v for k, v in self.layer_of.items()},
            "region_of": {str(k): v for k, v in self.region_of.items()},
            "interface_z_a": self.interface_z,
            "initial_substituent": list(self.initial_substituent),
            "initial_substituent_count": self.n_substituent,
            "excluded_frozen": list(self.excluded_frozen),
        }


def _layers(z: np.ndarray, order: Sequence[int], tolerance: float) -> dict[int, int]:
    """Group atom indices into layers by their stacking coordinate (running-mean anchor)."""

    layer_of: dict[int, int] = {}
    current = 0
    members: list[float] = []
    for index in order:
        value = float(z[index])
        if members and abs(value - sum(members) / len(members)) > tolerance:
            current += 1
            members = []
        members.append(value)
        layer_of[int(index)] = current
    return layer_of


def _interface_z(atoms: Any, axis: int, cations: Sequence[str]) -> float | None:
    """Cation contact plane between the two half-slabs (needs exactly two cations)."""

    if len(cations) != 2:
        return None
    symbols = np.asarray(atoms.get_chemical_symbols())
    z = atoms.positions[:, axis]
    lo = z[symbols == cations[0]]
    hi = z[symbols == cations[1]]
    if not len(lo) or not len(hi):
        return None
    if float(np.median(lo)) <= float(np.median(hi)):
        return float((lo.max() + hi.min()) / 2.0)
    return float((hi.max() + lo.min()) / 2.0)


def resolve_swap_sites(
    atoms: Any,
    *,
    substituent: str = "O",
    host: str = "N",
    cations: Sequence[str] = ("Si", "Ti"),
    stacking_axis: str = "c",
    z_window: tuple[float, float] | None = None,
    interface_band: float | None = None,
    region: str = "all",
    layers: Sequence[int] | None = None,
    layer_tolerance: float = 0.6,
    move_classes: str = "layer",
) -> SwapSites:
    """Select the anion sites that may exchange N<->O and partition them into move classes.

    ``region`` (needs two ``cations``) keeps only anions whose nearest cation is in
    the lower slab (``lower``) or the upper slab (``upper``), or -- when an
    ``interface_band`` is set or ``region="interface"`` -- those within that A band
    of the cation contact plane (the ``interface`` label is then carved out of
    ``lower``/``upper``). ``z_window`` and ``layers`` further restrict the set.
    Si/Ti sites are never eligible. Frozen atoms (interface ``FixAtoms``) are
    excluded and reported.
    """

    if substituent == host:
        raise SafetyError("substituent and host elements must differ")
    if region not in _REGIONS:
        raise SafetyError(f"region must be one of {_REGIONS}")
    axis = _AXES.get(stacking_axis.lower())
    if axis is None:
        raise SafetyError("stacking_axis must be a, b, or c")

    symbols = np.asarray(atoms.get_chemical_symbols())
    z = atoms.positions[:, axis]
    anion_mask = np.isin(symbols, [substituent, host])
    if not anion_mask.any():
        raise SafetyError(f"structure has no {host} or {substituent} atoms")
    frozen = _frozen_mask(atoms)

    cations = tuple(str(c) for c in cations)
    cation_idx = np.where(np.isin(symbols, list(cations)))[0]
    interface_z = _interface_z(atoms, axis, cations)
    if interface_band is not None and interface_z is not None and z_window is None:
        z_window = (interface_z - interface_band / 2.0, interface_z + interface_band / 2.0)

    lower_cat: str | None = None
    if len(cations) == 2 and all((symbols == c).any() for c in cations):
        medians = {c: float(np.median(z[symbols == c])) for c in cations}
        lower_cat = min(medians, key=medians.get)

    # Nearest-cation assignment (minimum image) gives each anion a slab label; an
    # interface band, when requested, is carved out of it near the contact plane.
    carve_interface = interface_band is not None or region == "interface"
    band = interface_band if interface_band is not None else 2.5
    region_of: dict[int, str] = {}
    for anion in np.where(anion_mask)[0]:
        anion = int(anion)
        label = "other"
        if len(cation_idx):
            d = atoms.get_distances(anion, cation_idx, mic=True)
            nearest = str(symbols[cation_idx[int(np.argmin(d))]])
            if lower_cat is not None:
                label = "lower" if nearest == lower_cat else "upper"
        if carve_interface and interface_z is not None and abs(float(z[anion]) - interface_z) <= band / 2.0:
            label = "interface"
        region_of[anion] = label

    order = list(np.argsort(z))
    layer_of_all = _layers(z, order, layer_tolerance)
    anion_layers = sorted({layer_of_all[int(i)] for i in np.where(anion_mask)[0]})
    layer_rank = {layer: rank for rank, layer in enumerate(anion_layers)}

    eligible: list[int] = []
    excluded_frozen: list[int] = []
    for index in np.where(anion_mask)[0]:
        index = int(index)
        if frozen[index]:
            excluded_frozen.append(index)
            continue
        if z_window is not None and not (z_window[0] <= float(z[index]) <= z_window[1]):
            continue
        if region == "lower" and region_of.get(index) != "lower":
            continue
        if region == "upper" and region_of.get(index) != "upper":
            continue
        if region == "interface" and region_of.get(index) != "interface":
            continue
        if layers is not None and layer_rank.get(layer_of_all[index]) not in set(layers):
            continue
        eligible.append(index)

    if len(eligible) < 2:
        raise SafetyError(
            f"only {len(eligible)} eligible anion site(s) after filtering; need at least 2 to swap"
        )

    layer_of = {i: layer_rank[layer_of_all[i]] for i in eligible}
    groups: dict[str, tuple[int, ...]] = {}
    if move_classes == "layer":
        for i in eligible:
            groups.setdefault(f"layer_{layer_of[i]:02d}", tuple())
        groups = {name: tuple(sorted(i for i in eligible if f"layer_{layer_of[i]:02d}" == name)) for name in groups}
    elif move_classes == "region":
        for i in eligible:
            groups.setdefault(region_of.get(i, "other"), tuple())
        groups = {name: tuple(sorted(i for i in eligible if region_of.get(i, "other") == name)) for name in groups}
    elif move_classes == "none":
        groups = {"all": tuple(eligible)}
    else:
        raise SafetyError("move_classes must be 'layer', 'region', or 'none'")

    initial_sub = tuple(sorted(i for i in eligible if symbols[i] == substituent))
    return SwapSites(
        substituent=substituent,
        host=host,
        stacking_axis=stacking_axis.lower(),
        eligible=tuple(eligible),
        groups=groups,
        layer_of=layer_of,
        region_of={i: region_of.get(i, "other") for i in eligible},
        interface_z=interface_z,
        initial_substituent=initial_sub,
        excluded_frozen=tuple(excluded_frozen),
    )


# --------------------------------------------------------------- occupation moves


def _apply_occupation(base: Any, sites: SwapSites, substituent_sites: frozenset[int]) -> Any:
    trial = base.copy()
    syms = list(trial.get_chemical_symbols())
    for index in sites.eligible:
        syms[index] = sites.substituent if index in substituent_sites else sites.host
    trial.set_chemical_symbols(syms)
    return trial


def initial_occupation(
    sites: SwapSites,
    *,
    mode: str = "keep",
    introduce: int | None = None,
    seed: int = 0,
) -> frozenset[int]:
    """Starting substituent placement over the eligible sites."""

    eligible = list(sites.eligible)
    rng = np.random.default_rng(seed)
    if mode == "keep":
        if introduce is not None:
            raise SafetyError("introduce is only valid with mode='introduce'")
        return frozenset(sites.initial_substituent)
    if mode == "randomize":
        count = len(sites.initial_substituent)
        return frozenset(int(i) for i in rng.choice(eligible, size=count, replace=False))
    if mode == "introduce":
        if introduce is None or introduce < 1:
            raise SafetyError("mode='introduce' needs a positive --introduce-oxygen count")
        if sites.initial_substituent:
            raise SafetyError(
                f"{len(sites.initial_substituent)} eligible sites already hold {sites.substituent}; "
                "use mode='keep' or 'randomize', or narrow the site selection"
            )
        if introduce >= len(eligible):
            raise SafetyError(f"cannot introduce {introduce} {sites.substituent} on {len(eligible)} eligible sites")
        return frozenset(int(i) for i in rng.choice(eligible, size=introduce, replace=False))
    raise SafetyError("mode must be 'keep', 'randomize', or 'introduce'")


def propose_swap(
    rng: np.random.Generator,
    substituent_sites: frozenset[int],
    sites: SwapSites,
    *,
    p_interclass: float,
) -> tuple[int, int, str]:
    """Pick one substituent site and one host site to exchange (composition-conserving).

    With probability ``p_interclass`` the two sites are drawn from different move
    classes (inter-layer redistribution); otherwise from the same class (lateral
    reordering). Falls back to any unlike pair when the chosen mode is impossible.
    """

    host_sites = frozenset(sites.eligible) - substituent_sites
    if not substituent_sites or not host_sites:
        raise SafetyError("eligible sites are all one species; nothing to swap at fixed composition")

    class_of = {i: name for name, members in sites.groups.items() for i in members}
    subs = sorted(substituent_sites)
    hosts = sorted(host_sites)

    want_inter = rng.random() < p_interclass and len(sites.groups) > 1
    for _ in range(64):
        s = int(rng.choice(subs))
        h = int(rng.choice(hosts))
        same_class = class_of.get(s) == class_of.get(h)
        if want_inter and not same_class:
            return s, h, "inter-class"
        if not want_inter and same_class:
            return s, h, "intra-class"
    # No pair of the preferred kind exists; take any unlike pair.
    s = int(rng.choice(subs))
    h = int(rng.choice(hosts))
    return s, h, "fallback"


def random_configurations(
    sites: SwapSites, *, count: int, seed: int, n_substituent: int | None = None
) -> list[frozenset[int]]:
    """Independent fixed-composition random arrangements, for the random baseline."""

    n_sub = len(sites.initial_substituent) if n_substituent is None else n_substituent
    if n_sub <= 0 or n_sub >= len(sites.eligible):
        raise SafetyError("random baseline needs 0 < substituent count < number of eligible sites")
    rng = np.random.default_rng(seed)
    eligible = list(sites.eligible)
    seen: set[tuple[int, ...]] = set()
    out: list[frozenset[int]] = []
    for _ in range(count * 40):
        if len(out) >= count:
            break
        pick = tuple(sorted(int(i) for i in rng.choice(eligible, size=n_sub, replace=False)))
        if pick not in seen:
            seen.add(pick)
            out.append(frozenset(pick))
    return out


# ------------------------------------------------------------------- relax backend


@dataclass(frozen=True)
class RelaxTier:
    """One relaxation budget (PAIPAI's fast screen / slow refinement split)."""

    fmax: float
    max_steps: int
    label: str


@dataclass
class RelaxOutcome:
    atoms: Any
    energy: float
    converged: bool
    steps: int
    max_force: float


class Relaxer(Protocol):
    def relax(self, atoms: Any, tier: RelaxTier) -> RelaxOutcome: ...

    def uncertainty(self, atoms: Any) -> float | None: ...

    def identity(self) -> dict[str, Any]: ...


def _build_calculator(family: str, model_paths: Sequence[str], *, device: str, dtype: str) -> Any:
    if family == "mace":
        try:
            from mace.calculators import MACECalculator
        except ModuleNotFoundError as exc:
            raise DependencyError(
                "swap-mc --mace-model needs mace-torch; run where the committee's environment is importable"
            ) from exc
        return MACECalculator(model_paths=list(model_paths), device=device, default_dtype=dtype)
    if family == "deepmd":
        try:
            from deepmd.calculator import DP
        except ModuleNotFoundError as exc:
            raise DependencyError(
                "swap-mc --deepmd-model needs deepmd-kit; run where the committee's environment is importable"
            ) from exc
        if len(model_paths) != 1:
            raise SafetyError(
                "the DeePMD ASE calculator takes a single model; pass one --deepmd-model for the driving PES "
                "(committee spread for DeePMD is not yet wired into the search)"
            )
        # Keep compatibility with the LONI DeePMD 3.2.0b0 module: its auto
        # backend selects vesin and may fail on frozen PyTorch models because
        # ModelOutputDef is unavailable after TorchScript deserialization.
        return DP(model=model_paths[0], nlist_backend="native")
    raise SafetyError("family must be 'mace' or 'deepmd'")


class MlipRelaxer:
    """Default backend: one persistent committee calculator, fixed cell, kept constraints."""

    def __init__(
        self,
        family: str,
        model_paths: Sequence[str],
        *,
        device: str = "cpu",
        dtype: str = "float64",
        optimizer: str = "fire",
        committee_uncertainty: bool = True,
    ) -> None:
        self.family = family
        self.model_paths = [str(Path(p).expanduser().resolve()) for p in model_paths]
        if not self.model_paths:
            raise SafetyError("MlipRelaxer needs at least one model path")
        self.device = device
        self.dtype = dtype
        self.optimizer = optimizer.lower()
        self._calc = _build_calculator(family, self.model_paths, device=device, dtype=dtype)
        self._members: list[Any] = []
        if committee_uncertainty and family == "mace" and len(self.model_paths) > 1:
            self._members = [
                _build_calculator("mace", [path], device=device, dtype=dtype) for path in self.model_paths
            ]

    def _optimizer(self):
        try:
            from ase.optimize import BFGS, FIRE, LBFGS
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise DependencyError("swap-mc relaxation needs ase.optimize") from exc
        return {"fire": FIRE, "bfgs": BFGS, "lbfgs": LBFGS}.get(self.optimizer, FIRE)

    def relax(self, atoms: Any, tier: RelaxTier) -> RelaxOutcome:
        probe = atoms.copy()
        probe.calc = self._calc
        if tier.max_steps <= 0:
            forces = np.asarray(probe.get_forces(), dtype=float)
            return RelaxOutcome(probe, float(probe.get_potential_energy()), False, 0,
                                float(np.linalg.norm(forces, axis=1).max()))
        opt = self._optimizer()(probe, logfile=None)
        converged = bool(opt.run(fmax=tier.fmax, steps=tier.max_steps))
        forces = np.asarray(probe.get_forces(), dtype=float)
        return RelaxOutcome(
            probe,
            float(probe.get_potential_energy()),
            converged,
            int(getattr(opt, "nsteps", 0)),
            float(np.linalg.norm(forces, axis=1).max()),
        )

    def uncertainty(self, atoms: Any) -> float | None:
        if not self._members:
            return None
        stacks = []
        for calc in self._members:
            probe = atoms.copy()
            probe.calc = calc
            stacks.append(np.asarray(probe.get_forces(), dtype=float))
        per_atom = np.std(np.asarray(stacks), axis=0)
        return float(np.linalg.norm(per_atom, axis=1).max())

    def identity(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "device": self.device,
            "default_dtype": self.dtype,
            "optimizer": self.optimizer,
            "committee_uncertainty": bool(self._members),
            "models": [
                {"path": path, "sha256": sha256_file(path) if Path(path).is_file() else None}
                for path in self.model_paths
            ],
        }


# ------------------------------------------------------------------ search driver


@dataclass
class CandidateRecord:
    occupation: tuple[int, ...]
    energy_screen: float
    energy_refine: float | None = None
    screen_converged: bool = False
    refine_converged: bool | None = None
    relax_steps: int = 0
    force_std_ev_ang: float | None = None
    first_seen_step: int = 0
    seen_count: int = 0
    roles: set[str] = field(default_factory=set)
    seeds: set[int] = field(default_factory=set)
    atoms: Any = None

    @property
    def energy(self) -> float:
        return self.energy_refine if self.energy_refine is not None else self.energy_screen

    def to_row(self, cand_id: str) -> dict[str, Any]:
        return {
            "cand_id": cand_id,
            "n_substituent": len(self.occupation),
            "occupation": list(self.occupation),
            "energy_ev": self.energy,
            "energy_screen_ev": self.energy_screen,
            "energy_refine_ev": self.energy_refine,
            "screen_converged": self.screen_converged,
            "refine_converged": self.refine_converged,
            "relax_steps": self.relax_steps,
            "force_std_ev_ang": self.force_std_ev_ang,
            "first_seen_step": self.first_seen_step,
            "seen_count": self.seen_count,
            "roles": sorted(self.roles),
            "seeds": sorted(self.seeds),
        }


@dataclass
class SearchResult:
    seed: int
    steps: int
    initial_energy: float
    best_occupation: tuple[int, ...]
    best_energy: float
    acceptance_rate: float
    screen_converged_rate: float
    n_evaluated: int
    trajectory: list[dict[str, Any]]
    warnings: list[str]


def _key(occupation: frozenset[int]) -> tuple[int, ...]:
    return tuple(sorted(occupation))


def _record(
    candidates: dict[tuple[int, ...], CandidateRecord],
    occupation: frozenset[int],
    outcome: RelaxOutcome,
    *,
    step: int,
    seed: int,
    role: str,
) -> CandidateRecord:
    key = _key(occupation)
    record = candidates.get(key)
    if record is None:
        record = CandidateRecord(
            occupation=key,
            energy_screen=outcome.energy,
            screen_converged=outcome.converged,
            relax_steps=outcome.steps,
            first_seen_step=step,
        )
        candidates[key] = record
    elif outcome.energy < record.energy_screen:
        record.energy_screen = outcome.energy
        record.screen_converged = outcome.converged
        record.relax_steps = outcome.steps
    record.seen_count += 1
    record.roles.add(role)
    record.seeds.add(seed)
    if record.atoms is None or outcome.energy <= record.energy_screen:
        record.atoms = outcome.atoms
    return record


def run_search(
    atoms: Any,
    sites: SwapSites,
    relaxer: Relaxer,
    *,
    steps: int,
    kt_ev: float,
    seed: int,
    screen: RelaxTier,
    refine: RelaxTier,
    p_interclass: float = 0.3,
    start_mode: str = "keep",
    introduce: int | None = None,
    refine_accepted: bool = False,
    refine_cap: int = 12,
    candidates: dict[tuple[int, ...], CandidateRecord] | None = None,
    progress: Callable[[str], None] | None = None,
) -> SearchResult:
    """One seeded fixed-composition Metropolis walk over screen-relaxed swaps."""

    if steps < 1:
        raise SafetyError("steps must be positive")
    if not math.isfinite(kt_ev) or kt_ev <= 0:
        raise SafetyError("kt_ev must be a positive, finite eV energy scale")
    candidates = candidates if candidates is not None else {}
    rng = np.random.default_rng(seed)

    current = initial_occupation(sites, mode=start_mode, introduce=introduce, seed=seed)
    current_outcome = relaxer.relax(_apply_occupation(atoms, sites, current), screen)
    current_energy = current_outcome.energy
    initial_energy = current_energy
    _record(candidates, current, current_outcome, step=-1, seed=seed, role="initial")

    best = current
    best_energy = current_energy
    accepted = 0
    converged_hits = int(current_outcome.converged)
    accepted_keys: set[tuple[int, ...]] = {_key(current)}
    trajectory: list[dict[str, Any]] = []

    for step in range(steps):
        s, h, mode = propose_swap(rng, current, sites, p_interclass=p_interclass)
        trial = frozenset((current - {s}) | {h})
        outcome = relaxer.relax(_apply_occupation(atoms, sites, trial), screen)
        converged_hits += int(outcome.converged)
        delta = outcome.energy - current_energy
        accept = delta <= 0.0 or rng.random() < math.exp(-delta / kt_ev)
        _record(candidates, trial, outcome, step=step, seed=seed, role="sampled")
        if accept:
            current, current_energy = trial, outcome.energy
            accepted += 1
            accepted_keys.add(_key(current))
            if current_energy < best_energy:
                best, best_energy = current, current_energy
        trajectory.append(
            {
                "seed": seed,
                "step": step,
                "sub_site": s,
                "host_site": h,
                "move": mode,
                "delta_ev": delta,
                "accepted": bool(accept),
                "energy_current_ev": current_energy,
                "energy_best_ev": best_energy,
                "screen_converged": bool(outcome.converged),
            }
        )
        if progress and (step + 1) % max(1, steps // 10) == 0:
            progress(f"seed {seed}: step {step + 1}/{steps}  E_best={best_energy:.4f} eV  acc={accepted / (step + 1):.2f}")

    # Strict refinement of the shortlist: the global best plus, optionally, every
    # arrangement the walk accepted -- the lowest screen-energy ones first.
    refine_keys = [_key(best)]
    if refine_accepted:
        for key in sorted(accepted_keys, key=lambda k: candidates[k].energy_screen):
            if key not in refine_keys:
                refine_keys.append(key)
    for key in refine_keys[: max(1, refine_cap)]:
        record = candidates[key]
        source = record.atoms if record.atoms is not None else _apply_occupation(atoms, sites, frozenset(key))
        outcome = relaxer.relax(source, refine)
        if record.energy_refine is None or outcome.energy < record.energy_refine:
            record.energy_refine = outcome.energy
            record.refine_converged = outcome.converged
            record.atoms = outcome.atoms
        record.roles.add("refined")

    for record in candidates.values():
        if record.force_std_ev_ang is None and record.atoms is not None and "refined" in record.roles:
            record.force_std_ev_ang = relaxer.uncertainty(record.atoms)

    best_record = candidates[_key(best)]
    best_record.roles.add("best")
    warnings = list(CAVEATS)
    conv_rate = converged_hits / (steps + 1)
    if conv_rate < 0.5:
        warnings.append(
            f"only {conv_rate:.0%} of screen relaxations reached fmax={screen.fmax} eV/A in "
            f"{screen.max_steps} steps; raise --screen-steps or relax --screen-fmax and re-check dE stability"
        )
    return SearchResult(
        seed=seed,
        steps=steps,
        initial_energy=initial_energy,
        best_occupation=_key(best),
        best_energy=best_record.energy,
        acceptance_rate=accepted / steps,
        screen_converged_rate=conv_rate,
        n_evaluated=steps + 1,
        trajectory=trajectory,
        warnings=warnings,
    )


def run_searches(
    atoms: Any,
    sites: SwapSites,
    relaxer: Relaxer,
    *,
    seeds: Sequence[int],
    steps: int,
    kt_ev: float,
    screen: RelaxTier,
    refine: RelaxTier,
    p_interclass: float = 0.3,
    start_mode: str = "keep",
    introduce: int | None = None,
    refine_accepted: bool = False,
    random_baseline: int = 0,
    baseline_seed: int = 0,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Independent seeded searches plus an optional random baseline, aggregated."""

    if not seeds:
        raise SafetyError("run_searches needs at least one seed")
    candidates: dict[tuple[int, ...], CandidateRecord] = {}
    per_seed: list[SearchResult] = []
    for seed in seeds:
        if progress:
            progress(f"--- search seed {seed} ---")
        per_seed.append(
            run_search(
                atoms,
                sites,
                relaxer,
                steps=steps,
                kt_ev=kt_ev,
                seed=int(seed),
                screen=screen,
                refine=refine,
                p_interclass=p_interclass,
                start_mode=start_mode,
                introduce=introduce,
                refine_accepted=refine_accepted,
                candidates=candidates,
                progress=progress,
            )
        )

    if random_baseline:
        if progress:
            progress(f"--- random baseline ({random_baseline}) ---")
        n_sub = len(per_seed[0].best_occupation)
        for occ in random_configurations(sites, count=random_baseline, seed=baseline_seed, n_substituent=n_sub):
            outcome = relaxer.relax(_apply_occupation(atoms, sites, occ), refine)
            record = _record(candidates, occ, outcome, step=-1, seed=baseline_seed, role="random-baseline")
            record.energy_refine = outcome.energy
            record.refine_converged = outcome.converged
            record.atoms = outcome.atoms
            record.force_std_ev_ang = relaxer.uncertainty(outcome.atoms)

    ranked = sorted(candidates.values(), key=lambda rec: rec.energy)
    consensus = ranked[0]
    seed_bests = {res.seed: res.best_occupation for res in per_seed}
    consensus_seed_hits = sum(1 for occ in seed_bests.values() if occ == consensus.occupation)

    def _hamming(a: Sequence[int], b: Sequence[int]) -> int:
        return len(set(a) ^ set(b)) // 2

    pairwise = [
        {"seeds": [x, y], "hamming": _hamming(seed_bests[x], seed_bests[y])}
        for i, x in enumerate(sorted(seed_bests))
        for y in sorted(seed_bests)[i + 1 :]
    ]
    return {
        "candidates": candidates,
        "per_seed": per_seed,
        "consensus": {
            "occupation": list(consensus.occupation),
            "energy_ev": consensus.energy,
            "energy_refine_ev": consensus.energy_refine,
            "force_std_ev_ang": consensus.force_std_ev_ang,
            "independent_seed_hits": consensus_seed_hits,
            "n_seeds": len(per_seed),
        },
        "seed_best_energies_ev": {str(res.seed): res.best_energy for res in per_seed},
        "seed_best_pairwise_hamming": pairwise,
        "acceptance_rate": float(np.mean([res.acceptance_rate for res in per_seed])),
        "screen_converged_rate": float(np.mean([res.screen_converged_rate for res in per_seed])),
    }


# --------------------------------------------------------------------- archive IO


def _cand_id(occupation: Sequence[int]) -> str:
    digest = hashlib.sha1(json.dumps(sorted(int(i) for i in occupation)).encode()).hexdigest()
    return f"cand_{digest[:10]}"


def write_archive(
    run_dir: str | Path,
    aggregate: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    sites: SwapSites,
    relaxer_identity: Mapping[str, Any],
    input_structure: str | Path,
    geom_cap: int = 40,
    force: bool = False,
) -> dict[str, Any]:
    """Write manifest, per-candidate index, kept geometries, and per-seed trajectories."""

    out = Path(run_dir).expanduser().resolve()
    manifest_path = out / "manifest.json"
    if out.exists() and not force:
        # a lone plan.json from an earlier --dry-run of this same command is fine
        leftovers = [p for p in out.iterdir() if p.name != "plan.json"]
        prior_run = manifest_path.is_file() and json.loads(manifest_path.read_text(encoding="utf-8")).get("quantity") == QUANTITY
        if leftovers and not prior_run:
            raise SafetyError(f"swap-mc output directory is not empty and not a prior run: {out}; pass --force to reuse")
    (out / "candidates").mkdir(parents=True, exist_ok=True)

    candidates: dict[tuple[int, ...], CandidateRecord] = dict(aggregate["candidates"])
    ranked = sorted(candidates.values(), key=lambda rec: rec.energy)
    keep_geom = {rec.occupation for rec in ranked[:geom_cap]}
    keep_geom |= {rec.occupation for rec in candidates.values() if rec.roles & {"refined", "best", "random-baseline", "initial"}}

    write = _ase()["write"]
    index_rows: list[dict[str, Any]] = []
    for rec in ranked:
        cid = _cand_id(rec.occupation)
        row = rec.to_row(cid)
        if rec.occupation in keep_geom and rec.atoms is not None:
            geom_path = out / "candidates" / f"{cid}.extxyz"
            probe = rec.atoms.copy()
            probe.calc = None
            probe.info.update({"cand_id": cid, "swap_mc_energy_ev": rec.energy})
            write(str(geom_path), probe, format="extxyz")
            row["geometry"] = f"candidates/{cid}.extxyz"
        index_rows.append(row)

    (out / "candidates.jsonl").write_text(
        "".join(json.dumps(row, default=str) + "\n" for row in index_rows), encoding="utf-8"
    )

    for result in aggregate["per_seed"]:
        traj_path = out / f"trajectory_seed_{result.seed}.csv"
        _write_csv(
            traj_path,
            result.trajectory,
            ("seed", "step", "sub_site", "host_site", "move", "delta_ev", "accepted",
             "energy_current_ev", "energy_best_ev", "screen_converged"),
        )

    warnings = sorted({w for result in aggregate["per_seed"] for w in result.warnings})
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "quantity": QUANTITY,
        "created_at": utc_now(),
        "input_structure": {
            "path": str(Path(input_structure).expanduser().resolve()),
            "sha256": sha256_file(input_structure) if Path(input_structure).is_file() else None,
        },
        "config": dict(config),
        "sites": sites.to_dict(),
        "relaxer": dict(relaxer_identity),
        "consensus": aggregate["consensus"],
        "seed_best_energies_ev": aggregate["seed_best_energies_ev"],
        "seed_best_pairwise_hamming": aggregate["seed_best_pairwise_hamming"],
        "acceptance_rate": aggregate["acceptance_rate"],
        "screen_converged_rate": aggregate["screen_converged_rate"],
        "n_candidates": len(candidates),
        "citation": CITATION,
        "caveats": list(CAVEATS),
        "warnings": warnings,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    _write_summary_md(out / "summary.md", manifest, index_rows)
    return {
        "run_dir": str(out),
        "manifest": str(manifest_path),
        "candidates_index": str(out / "candidates.jsonl"),
        "n_candidates": len(candidates),
        "geometries_written": sum(1 for row in index_rows if "geometry" in row),
        "consensus": manifest["consensus"],
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_md(path: Path, manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    con = manifest["consensus"]
    lines = [
        "# Chemical-ordering search",
        "",
        f"Input: `{manifest['input_structure']['path']}`",
        "",
        f"- eligible anion sites: {manifest['sites']['n_eligible']}  "
        f"(fixed {manifest['sites']['substituent']} count: {manifest['sites']['initial_substituent_count']})",
        f"- seeds: {list(manifest['seed_best_energies_ev'])}  |  mean acceptance {manifest['acceptance_rate']:.2f}  "
        f"|  screen convergence {manifest['screen_converged_rate']:.0%}",
        f"- consensus best: {con['energy_ev']:.4f} eV, found independently by "
        f"{con['independent_seed_hits']}/{con['n_seeds']} seeds"
        + (f", committee force σ {con['force_std_ev_ang']:.3f} eV/A" if con.get("force_std_ev_ang") else ""),
        "",
        "| candidate | n_sub | E (eV) | ΔE vs best (eV) | refined | committee σ | roles |",
        "|---|---:|---:|---:|:--:|---:|---|",
    ]
    best_e = rows[0]["energy_ev"] if rows else 0.0
    for row in rows[:15]:
        sigma = f"{row['force_std_ev_ang']:.3f}" if row.get("force_std_ev_ang") is not None else "—"
        refined = "✓" if row.get("energy_refine_ev") is not None else "—"
        lines.append(
            f"| {row['cand_id']} | {row['n_substituent']} | {row['energy_ev']:.4f} | "
            f"{row['energy_ev'] - best_e:+.4f} | {refined} | {sigma} | {', '.join(row['roles'])} |"
        )
    lines += ["", "## Caveats", ""]
    lines += [f"- {c}" for c in manifest["caveats"]]
    lines += [
        "",
        f"Method after PAIPAI — {manifest['citation']['paper']['authors']}, "
        f"*{manifest['citation']['paper']['journal']}* {manifest['citation']['paper']['volume']} "
        f"({manifest['citation']['paper']['year']}) {manifest['citation']['paper']['pages']}; "
        f"{manifest['citation']['software']['url']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_index(run_dir: Path) -> list[dict[str, Any]]:
    index = run_dir / "candidates.jsonl"
    if not index.is_file():
        raise SafetyError(f"no candidates.jsonl in {run_dir}; run 'iface swap-mc run' first")
    return [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line.strip()]


# ------------------------------------------------------------------- DFT shortlist


def select_candidates(
    run_dir: str | Path,
    *,
    count: int = 12,
    low_fraction: float = 0.6,
    include_baseline: bool = True,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Diverse DFT-verification shortlist: lowest-energy picks plus a spread over the archive.

    The low-energy picks are exactly the arrangements the search calls favorable
    -- the ones whose MLIP energy must be checked against DFT before any ordering
    claim. The diverse picks (farthest-point in occupation-Hamming space) and the
    baseline/random arrangements give the "random vs searched" contrast.
    """

    run_dir = Path(run_dir).expanduser().resolve()
    all_rows = _load_index(run_dir)
    if count < 1:
        raise SafetyError("count must be positive")
    if not 0.0 <= low_fraction <= 1.0:
        raise SafetyError("low_fraction must be in [0, 1]")

    # Only candidates with an archived relaxed geometry can be DFT-checked.
    rows = [row for row in all_rows if row.get("geometry")]
    if not rows:
        raise SafetyError(
            f"no candidate in {run_dir} has an archived geometry; re-run with a larger --geom-cap"
        )
    excluded_no_geometry = len(all_rows) - len(rows)
    by_energy = sorted(rows, key=lambda r: r["energy_ev"])
    n_low = min(len(by_energy), max(1, round(count * low_fraction)))
    picked: dict[str, dict[str, Any]] = {}
    for row in by_energy[:n_low]:
        picked[row["cand_id"]] = {**row, "role": "low-energy"}

    def _farthest(pool: list[dict[str, Any]]) -> dict[str, Any] | None:
        chosen_occ = [set(p["occupation"]) for p in picked.values()]
        best_row, best_gap = None, -1
        for row in pool:
            if row["cand_id"] in picked:
                continue
            occ = set(row["occupation"])
            gap = min((len(occ ^ c) for c in chosen_occ), default=0)
            if gap > best_gap:
                best_row, best_gap = row, gap
        return best_row

    while len(picked) < min(count, len(rows)):
        row = _farthest(by_energy)
        if row is None:
            break
        picked[row["cand_id"]] = {**row, "role": "diverse"}

    if include_baseline:
        for row in rows:
            if {"initial", "random-baseline"} & set(row.get("roles", [])):
                picked.setdefault(row["cand_id"], {**row, "role": "baseline"})

    # High committee spread is an independent reason to spend a DFT check.
    flagged = sorted(
        (r for r in rows if r.get("force_std_ev_ang") is not None),
        key=lambda r: r["force_std_ev_ang"],
        reverse=True,
    )[:3]
    for row in flagged:
        picked.setdefault(row["cand_id"], {**row, "role": "high-committee-spread"})

    shortlist = sorted(picked.values(), key=lambda r: r["energy_ev"])
    best_e = by_energy[0]["energy_ev"]
    for row in shortlist:
        row["delta_vs_best_ev"] = row["energy_ev"] - best_e

    payload = {
        "schema_version": SCHEMA_VERSION,
        "quantity": "chemical_ordering_dft_shortlist",
        "run_dir": str(run_dir),
        "count_requested": count,
        "low_fraction": low_fraction,
        "n_selected": len(shortlist),
        "candidates_without_archived_geometry": excluded_no_geometry,
        "shortlist": shortlist,
        "note": (
            "DFT-check relative energies and rankings with the same fixed-cell, "
            "same-constraint relaxation convention used here. Low-energy picks are "
            "the search's own favorable arrangements; confirm them before trusting "
            "any ordering or adhesion conclusion."
        ),
        "citation": CITATION,
    }
    out = Path(output).expanduser().resolve() if output else run_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "dft_shortlist.json").write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    _write_csv(
        out / "dft_shortlist.csv",
        shortlist,
        ("cand_id", "role", "n_substituent", "energy_ev", "delta_vs_best_ev",
         "energy_refine_ev", "force_std_ev_ang", "geometry"),
    )
    payload["outputs"] = {
        "json": str(out / "dft_shortlist.json"),
        "csv": str(out / "dft_shortlist.csv"),
    }
    return payload


def export_candidates(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    shortlist: str | Path | None = None,
    cand_ids: Sequence[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Write selected candidates as POSCARs (one directory each) for VASP / separation-energy."""

    run_dir = Path(run_dir).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()) and not force:
        raise SafetyError(f"export directory is not empty: {out}; pass --force to overwrite")
    rows = {row["cand_id"]: row for row in _load_index(run_dir)}

    selected: list[dict[str, Any]]
    if shortlist is not None:
        payload = json.loads(Path(shortlist).expanduser().resolve().read_text(encoding="utf-8"))
        selected = [rows[item["cand_id"]] | {"role": item.get("role")} for item in payload["shortlist"] if item["cand_id"] in rows]
    elif cand_ids:
        missing = [cid for cid in cand_ids if cid not in rows]
        if missing:
            raise SafetyError(f"unknown candidate id(s): {missing}")
        selected = [rows[cid] for cid in cand_ids]
    else:
        shortlist_json = run_dir / "dft_shortlist.json"
        if not shortlist_json.is_file():
            raise SafetyError("no --shortlist / --cand-id given and no dft_shortlist.json in the run dir")
        payload = json.loads(shortlist_json.read_text(encoding="utf-8"))
        selected = [rows[item["cand_id"]] | {"role": item.get("role")} for item in payload["shortlist"] if item["cand_id"] in rows]

    read, write = _ase()["read"], _ase()["write"]
    out.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, Any]] = []
    for row in selected:
        geometry = row.get("geometry")
        if not geometry:
            raise SafetyError(
                f"{row['cand_id']} has no archived geometry (increase --geom-cap on the run, or pick a refined candidate)"
            )
        atoms = read(str(run_dir / geometry))
        target = out / row["cand_id"]
        target.mkdir(parents=True, exist_ok=True)
        write(str(target / "POSCAR"), atoms, format="vasp", direct=True, vasp5=True)
        (target / "ordering.json").write_text(
            json.dumps(
                {
                    "cand_id": row["cand_id"],
                    "role": row.get("role"),
                    "n_substituent": row["n_substituent"],
                    "substituent_sites": row["occupation"],
                    "energy_ev": row["energy_ev"],
                    "energy_refine_ev": row.get("energy_refine_ev"),
                    "force_std_ev_ang": row.get("force_std_ev_ang"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        written.append({"cand_id": row["cand_id"], "role": row.get("role"), "directory": str(target)})

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "quantity": "chemical_ordering_export",
        "run_dir": str(run_dir),
        "created_at": utc_now(),
        "n_exported": len(written),
        "candidates": written,
        "next_steps": [
            "DFT ordering benchmark: 'iface vasp opt-prepare' over the exported POSCARs "
            "(fixed cell, same frozen layers) and compare relative energies / rankings.",
            "Adhesion contrast: 'iface vasp adhesion prepare' per candidate, then "
            "'iface validate separation-energy' random vs searched at identical composition.",
        ],
        "citation": CITATION,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return {"export_dir": str(out), "manifest": str(out / "manifest.json"), "n_exported": len(written), "candidates": written}


# ----------------------------------------------------------------------- CLI glue


def _sites_from_args(atoms: Any, args: argparse.Namespace) -> SwapSites:
    return resolve_swap_sites(
        atoms,
        substituent=args.substituent,
        host=args.host,
        cations=tuple(args.cation) if args.cation else ("Si", "Ti"),
        stacking_axis=args.stacking_axis,
        z_window=tuple(args.z_window) if args.z_window else None,
        interface_band=args.interface_band,
        region=args.region,
        layers=args.layer or None,
        layer_tolerance=args.layer_tolerance,
        move_classes=args.move_classes,
    )


def _config_from_args(args: argparse.Namespace, seeds: Sequence[int]) -> dict[str, Any]:
    return {
        "steps": args.steps,
        "kt_ev": args.kt_ev,
        "seeds": list(seeds),
        "p_interclass": args.p_interclass,
        "start_mode": args.start_mode,
        "introduce_oxygen": args.introduce_oxygen,
        "screen": {"fmax": args.screen_fmax, "max_steps": args.screen_steps},
        "refine": {"fmax": args.refine_fmax, "max_steps": args.refine_steps},
        "refine_accepted": args.refine_accepted,
        "random_baseline": args.random_baseline,
        "region": args.region,
        "move_classes": args.move_classes,
    }


def cmd_sites(args: argparse.Namespace) -> int:
    atoms = _read_atoms(args.structure)
    print(json.dumps(_sites_from_args(atoms, args).to_dict(), indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    atoms = _read_atoms(args.structure)
    sites = _sites_from_args(atoms, args)
    seeds = list(args.seed) if args.seed else [args.base_seed + 7919 * i for i in range(args.searches)]
    config = _config_from_args(args, seeds)

    if args.dry_run:
        out = Path(args.output).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        plan = {
            "schema_version": SCHEMA_VERSION,
            "quantity": QUANTITY,
            "mode": "dry-run",
            "input_structure": str(Path(args.structure).expanduser().resolve()),
            "config": config,
            "sites": sites.to_dict(),
            "citation": CITATION,
            "caveats": list(CAVEATS),
        }
        (out / "plan.json").write_text(json.dumps(plan, indent=2, default=str) + "\n", encoding="utf-8")
        print(json.dumps(plan, indent=2, default=str))
        return 0

    if not (args.mace_model or args.deepmd_model):
        raise SafetyError("swap-mc run needs --mace-model or --deepmd-model (or --dry-run to only resolve sites)")
    family = "mace" if args.mace_model else "deepmd"
    relaxer = MlipRelaxer(
        family,
        args.mace_model or args.deepmd_model,
        device=args.device,
        dtype=args.dtype,
        optimizer=args.optimizer,
    )
    started = time.time()
    aggregate = run_searches(
        atoms,
        sites,
        relaxer,
        seeds=seeds,
        steps=args.steps,
        kt_ev=args.kt_ev,
        screen=RelaxTier(args.screen_fmax, args.screen_steps, "screen"),
        refine=RelaxTier(args.refine_fmax, args.refine_steps, "refine"),
        p_interclass=args.p_interclass,
        start_mode=args.start_mode,
        introduce=args.introduce_oxygen,
        refine_accepted=args.refine_accepted,
        random_baseline=args.random_baseline,
        baseline_seed=args.base_seed,
        progress=lambda message: print(message, flush=True),
    )
    config["wall_seconds"] = round(time.time() - started, 1)
    result = write_archive(
        args.output,
        aggregate,
        config=config,
        sites=sites,
        relaxer_identity=relaxer.identity(),
        input_structure=args.structure,
        geom_cap=args.geom_cap,
        force=args.force,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_select(args: argparse.Namespace) -> int:
    print(json.dumps(
        select_candidates(
            args.run_dir,
            count=args.count,
            low_fraction=args.low_fraction,
            include_baseline=not args.no_baseline,
            output=args.output,
        ),
        indent=2,
        default=str,
    ))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    print(json.dumps(
        export_candidates(
            args.run_dir,
            args.output_dir,
            shortlist=args.shortlist,
            cand_ids=args.cand_id or None,
            force=args.force,
        ),
        indent=2,
        default=str,
    ))
    return 0


def _add_site_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--substituent", default="O", help="Element introduced onto anion sites (default O)")
    parser.add_argument("--host", default="N", help="Anion element it replaces (default N)")
    parser.add_argument("--cation", action="append", default=[], help="Fixed cation element; repeat (default: Si Ti)")
    parser.add_argument("--stacking-axis", default="c", choices=("a", "b", "c"))
    parser.add_argument("--z-window", nargs=2, type=float, metavar=("LO", "HI"), help="Keep anions with LO<=z<=HI (A)")
    parser.add_argument("--interface-band", type=float, help="Keep anions within this A band of the cation contact plane")
    parser.add_argument("--region", default="all", choices=_REGIONS, help="Restrict by nearest-cation slab")
    parser.add_argument("--layer", action="append", type=int, default=[], help="Anion-layer index to include; repeat")
    parser.add_argument("--layer-tolerance", type=float, default=0.6, help="A tolerance for layer grouping")
    parser.add_argument("--move-classes", default="layer", choices=("layer", "region", "none"))


def register_commands(commands: Any) -> None:
    """Attach the sites/run/select/export subcommands to a subparsers object."""

    sites = commands.add_parser("sites", help="Resolve and print the eligible anion sites and move classes")
    sites.add_argument("structure")
    _add_site_options(sites)
    sites.set_defaults(func=cmd_sites)

    run = commands.add_parser("run", help="Run seeded searches; writes a candidate archive (needs a committee, or --dry-run)")
    run.add_argument("structure")
    run.add_argument("output", help="Run directory for manifest.json, candidates.jsonl, candidates/, trajectories")
    _add_site_options(run)
    run.add_argument("--steps", type=int, default=2000, help="Metropolis steps per seed (default 2000)")
    run.add_argument("--kt-ev", type=float, default=0.05, help="Acceptance energy scale in eV (NOT Kelvin; default 0.05)")
    run.add_argument("--seed", action="append", type=int, default=[], help="Explicit seed; repeat for independent searches")
    run.add_argument("--searches", type=int, default=3, help="Number of seeds when --seed is not given (default 3)")
    run.add_argument("--base-seed", type=int, default=20260905, help="Base for generated seeds and the random baseline")
    run.add_argument("--start-mode", default="keep", choices=("keep", "randomize", "introduce"))
    run.add_argument("--introduce-oxygen", type=int, help="With --start-mode introduce: how many O to place")
    run.add_argument("--screen-fmax", type=float, default=0.10, help="Screen relaxation fmax, eV/A (default 0.10)")
    run.add_argument("--screen-steps", type=int, default=30, help="Screen relaxation step cap (default 30)")
    run.add_argument("--refine-fmax", type=float, default=0.01, help="Refinement fmax, eV/A (default 0.01)")
    run.add_argument("--refine-steps", type=int, default=400, help="Refinement step cap (default 400)")
    run.add_argument("--refine-accepted", action="store_true", help="Refine every accepted arrangement, not just the best")
    run.add_argument("--p-interclass", type=float, default=0.3, help="Probability a swap crosses move classes (default 0.3)")
    run.add_argument("--random-baseline", type=int, default=0, help="Also relax N random arrangements at the same composition")
    run.add_argument("--mace-model", action="append", default=[], help="MACE committee member; repeat (committee mean drives the walk)")
    run.add_argument("--deepmd-model", action="append", default=[], help="A single DeePMD model for the driving PES")
    run.add_argument("--device", default="cpu")
    run.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    run.add_argument("--optimizer", default="fire", choices=("fire", "bfgs", "lbfgs"))
    run.add_argument("--geom-cap", type=int, default=40, help="Archive relaxed geometries for the N lowest-energy candidates (default 40)")
    run.add_argument("--dry-run", action="store_true", help="Only resolve sites and write plan.json; import no MLIP backend")
    run.add_argument("--force", action="store_true", help="Reuse a non-empty prior run directory")
    run.set_defaults(func=cmd_run)

    select = commands.add_parser("select", help="Build a diverse DFT-verification shortlist from a run")
    select.add_argument("run_dir")
    select.add_argument("-n", "--count", type=int, default=12)
    select.add_argument("--low-fraction", type=float, default=0.6, help="Fraction of the shortlist taken as lowest-energy (default 0.6)")
    select.add_argument("--no-baseline", action="store_true", help="Do not force-include the initial / random-baseline arrangements")
    select.add_argument("-o", "--output", help="Directory for dft_shortlist.{json,csv} (default: the run dir)")
    select.set_defaults(func=cmd_select)

    export = commands.add_parser("export", help="Write selected candidates as POSCAR directories for VASP")
    export.add_argument("run_dir")
    export.add_argument("output_dir")
    export.add_argument("--shortlist", help="A dft_shortlist.json (default: the one in the run dir)")
    export.add_argument("--cand-id", action="append", default=[], help="Export exactly these candidate ids instead")
    export.add_argument("--force", action="store_true")
    export.set_defaults(func=cmd_export)


def main(argv: Sequence[str] | None = None) -> int:
    """Standalone entry point for isolated MACE/DeePMD environments.

    ``python -m interfaceforge.swap_mc run ...`` mirrors ``iface swap-mc run ...``.
    """

    parser = argparse.ArgumentParser(prog="python -m interfaceforge.swap_mc", description=__doc__)
    register_commands(parser.add_subparsers(dest="swap_mc_command", required=True))
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (SafetyError, DependencyError, ValueError) as exc:
        print(f"ERROR: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
