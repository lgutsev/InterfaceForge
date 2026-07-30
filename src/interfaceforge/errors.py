"""Public exception types used by InterfaceForge."""

from __future__ import annotations


class InterfaceForgeError(RuntimeError):
    """Base error for an actionable campaign failure."""


class ConfigurationError(InterfaceForgeError):
    """Raised when a campaign or scheduler profile is invalid."""


class SafetyError(InterfaceForgeError):
    """Raised when an operation would overwrite or mutate unsafe state."""


class DependencyError(InterfaceForgeError):
    """Raised when an optional runtime dependency is required."""
