"""Optional pre-SCF electronic initialization for VASP runs.

The initializer only changes where VASP's SCF *starts*; the INCAR-defined DFT
method, the converged density, energies, forces and stresses remain VASP's.
See ``docs/density-init.md``.
"""

from __future__ import annotations

from .base import (
    MAGMOM_SOURCES,
    SPIN_CHANNEL_MODES,
    BackendResult,
    DensityInitializer,
    InitializationPlan,
    StandardInitializer,
    resolve_magnetism,
)
from .neural_paw import InferenceError, NeuralPawInitializer
from .workflow import (
    BACKENDS,
    INCAR_BACKUP_NAME,
    REPORT_NAME,
    STAGED_NAME,
    hook_command,
    initialize_density,
    make_backend,
    register_backend,
)

__all__ = [
    "BACKENDS",
    "INCAR_BACKUP_NAME",
    "MAGMOM_SOURCES",
    "REPORT_NAME",
    "SPIN_CHANNEL_MODES",
    "STAGED_NAME",
    "BackendResult",
    "DensityInitializer",
    "InferenceError",
    "InitializationPlan",
    "NeuralPawInitializer",
    "StandardInitializer",
    "hook_command",
    "initialize_density",
    "make_backend",
    "register_backend",
    "resolve_magnetism",
]
