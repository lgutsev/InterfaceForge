"""Density-initializer abstraction and the authoritative-magnetism policy.

A :class:`DensityInitializer` turns a validated VASP calculation directory into
an *initial* electronic state (today: a ``CHGCAR``).  It never decides the DFT
method, never edits the run's inputs, and never owns the final result: VASP
still converges the density self-consistently.

Backends:

``standard``
    VASP's own start (superposition of atomic densities, ``ICHARG=2``).  It
    writes nothing and exists so the benchmark and provenance can treat both
    arms of a comparison identically.
``neural-paw``
    The ``neural_paw_dft`` Complete Neural Electronic Initializer
    (ELECTRAFI + AugNet [+ CHGNet]); see :mod:`.neural_paw`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import SafetyError

MAGMOM_SOURCES = ("incar", "initializer", "none")
SPIN_CHANNEL_MODES = ("auto", "off", "model")


@dataclass
class InitializationPlan:
    """Everything an initializer needs, resolved from the run's own inputs."""

    run_dir: Path
    species: list[str]
    grid: tuple[int, int, int] | None
    grid_source: str
    nelect: float
    nelect_source: str
    lmaxmix: int
    magnetism: dict[str, Any]
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def n_ions(self) -> int:
        return len(self.species)


@dataclass
class BackendResult:
    """What a backend produced in its private scratch directory."""

    chgcar: Path | None
    backend_info: dict[str, Any]
    timing: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    initializer_moments: list[float] | None = None
    spin_channel_written: bool = False
    log: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class DensityInitializer(ABC):
    """One strategy for producing a starting electronic state."""

    name: str = ""
    writes_chgcar: bool = True
    #: Whether the backend can produce a model spin-density channel.
    supports_spin_channel: bool = False
    #: Whether the backend has its own automatic magnetic-moment estimate.
    has_automatic_moments: bool = False
    #: Whether the backend needs the FFT grid of the target run.
    needs_grid: bool = True

    @abstractmethod
    def probe(self) -> dict[str, Any]:
        """``{"available": bool, "detail": str, ...}`` without running inference."""

    @abstractmethod
    def generate(self, plan: InitializationPlan, workdir: Path) -> BackendResult:
        """Produce the initial state inside ``workdir`` (never the run directory)."""


class StandardInitializer(DensityInitializer):
    """VASP's default atomic-superposition start: nothing is generated."""

    name = "standard"
    writes_chgcar = False
    needs_grid = False

    def probe(self) -> dict[str, Any]:
        return {"available": True, "detail": "VASP built-in atomic-density start (ICHARG=2)"}

    def generate(self, plan: InitializationPlan, workdir: Path) -> BackendResult:
        return BackendResult(
            chgcar=None,
            backend_info={"backend": self.name, "method": "VASP superposition of atomic densities"},
            timing={"inference_s": 0.0, "total_s": 0.0},
        )


def resolve_magnetism(
    settings: dict[str, Any],
    *,
    magmom_source: str,
    spin_channel: str,
    backend: DensityInitializer,
) -> dict[str, Any]:
    """Decide which magnetic information the initializer may use.

    The INCAR magnetic configuration is authoritative.  In particular a
    mixed-sign (antiferro/ferrimagnetic, e.g. NiO AFM-II) ``MAGMOM`` is never
    replaced by an initializer's own estimate, and no spin channel is written
    that could not reproduce it: VASP uses ``MAGMOM`` for the initial moments
    only when the ``CHGCAR`` carries *no* magnetization density, so a
    charge-only seed is what keeps the signed pattern in control.
    """

    if magmom_source not in MAGMOM_SOURCES:
        raise SafetyError(f"--magmom-source must be one of {', '.join(MAGMOM_SOURCES)}")
    if spin_channel not in SPIN_CHANNEL_MODES:
        raise SafetyError(f"--spin-channel must be one of {', '.join(SPIN_CHANNEL_MODES)}")

    ispin = settings["ispin"]
    order = settings["order"]
    magmom = settings["incar_magmom"]
    notes: list[str] = []
    warnings: list[str] = []
    decision: dict[str, Any] = {
        **settings,
        "magmom_source": magmom_source,
        "spin_channel_requested": spin_channel,
        "spin_channel": False,
        "site_moments_passed": None,
        "use_initializer_moments": False,
        "automatic_magnetic_initialization": "not_applicable",
        "initial_moments_in_vasp_from": "not_applicable",
    }

    if ispin != 2:
        if magmom_source == "initializer":
            raise SafetyError(
                "--magmom-source initializer requested for a non-spin-polarized INCAR "
                "(ISPIN != 2); the initializer cannot add magnetism to it"
            )
        if spin_channel == "model":
            raise SafetyError("--spin-channel model requires ISPIN = 2")
        notes.append("non-spin-polarized run: charge-only seed")
        decision.update(notes=notes, warnings=warnings)
        return decision

    if magmom_source == "none":
        raise SafetyError(
            "ISPIN = 2 needs a magnetic source; use --magmom-source incar (authoritative "
            "signed INCAR MAGMOM) or, for non-antiferromagnetic cases only, initializer"
        )

    if magmom_source == "incar":
        decision["automatic_magnetic_initialization"] = (
            "overridden: INCAR MAGMOM is authoritative; the initializer's automatic "
            "moment estimate was not used" if backend.has_automatic_moments else "not_available"
        )
        uniform = order == "sign-uniform"
        if spin_channel == "model" and not uniform:
            raise SafetyError(
                f"--spin-channel model refused for a {order} MAGMOM: the neural spin "
                "channel is constrained only by the net moment (sum of MAGMOM), which "
                "cannot represent a signed antiferromagnetic pattern such as NiO AFM-II. "
                "Use --spin-channel off/auto so VASP initializes the moments from MAGMOM"
            )
        if spin_channel == "model" and not backend.supports_spin_channel:
            raise SafetyError(f"backend {backend.name!r} cannot write a spin channel")
        use_spin = (
            uniform and backend.supports_spin_channel and spin_channel in {"auto", "model"}
        )
        decision["spin_channel"] = use_spin
        decision["site_moments_passed"] = list(magmom) if (use_spin and magmom) else None
        if use_spin:
            decision["initial_moments_in_vasp_from"] = (
                "CHGCAR magnetization density (model spatial distribution constrained to "
                "the INCAR net moment); VASP uses MAGMOM for symmetry only"
            )
            warnings.append(
                "sign-uniform MAGMOM: the model spin channel is constrained to the INCAR net "
                f"moment ({settings['net_moment']:.4g} muB); site magnitudes come from the model"
            )
        else:
            decision["initial_moments_in_vasp_from"] = (
                "INCAR MAGMOM (charge-only CHGCAR carries no magnetization density)"
            )
            if order == "mixed-sign":
                notes.append(
                    "mixed-sign (e.g. AFM-II) MAGMOM preserved exactly: charge-only seed, "
                    "signed INCAR moments initialize the spin state"
                )
            elif order.startswith("vasp-default"):
                warnings.append(
                    "ISPIN = 2 without MAGMOM: VASP will start from its default NIONS*1.0 "
                    "ferromagnetic moments; set MAGMOM explicitly for magnetic oxides"
                )
            elif order == "zero":
                warnings.append("ISPIN = 2 with an all-zero MAGMOM: spin state starts unpolarized")
        decision.update(notes=notes, warnings=warnings)
        return decision

    # magmom_source == "initializer"
    if not backend.has_automatic_moments:
        raise SafetyError(f"backend {backend.name!r} has no automatic magnetic-moment estimate")
    if settings["mixed_sign"]:
        raise SafetyError(
            "--magmom-source initializer refused: the INCAR MAGMOM has mixed signs "
            "(antiferro/ferrimagnetic intent). The initializer's unsigned moments would "
            "silently change the magnetic branch; use --magmom-source incar"
        )
    if spin_channel == "off":
        raise SafetyError(
            "--magmom-source initializer with --spin-channel off would discard the "
            "initializer moments; use --magmom-source incar for a charge-only seed"
        )
    decision["spin_channel"] = True
    decision["use_initializer_moments"] = True
    decision["automatic_magnetic_initialization"] = "used (explicit --magmom-source initializer)"
    decision["initial_moments_in_vasp_from"] = (
        "CHGCAR magnetization density constrained to the initializer's net moment; "
        "the INCAR MAGMOM is not rewritten and VASP uses it for symmetry only"
    )
    warnings.append(
        "the initial magnetic state comes from the initializer, not the INCAR MAGMOM; "
        "audit the converged moments with 'iface vasp density-init-audit'"
    )
    decision.update(notes=notes, warnings=warnings)
    return decision
