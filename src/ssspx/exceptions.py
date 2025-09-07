"""Custom exception types used across :mod:`ssspx`."""

from __future__ import annotations


class SSSPXError(Exception):
    """Base class for all package-specific errors."""


class InputError(SSSPXError, ValueError):
    """Raised for invalid user input such as malformed edges."""


class GraphFormatError(InputError):
    """Raised when parsing a graph file fails."""


class ConfigError(SSSPXError, ValueError):
    """Raised for invalid configuration options."""


class NotSupportedError(SSSPXError):
    """Raised when requesting a feature that is not implemented."""


class GraphError(InputError):
    """Alias of :class:`InputError` for backward compatibility."""


class AlgorithmError(SSSPXError, RuntimeError):
    """Raised when algorithm invariants are violated at runtime."""


__all__ = [
    "SSSPXError",
    "InputError",
    "GraphFormatError",
    "ConfigError",
    "NotSupportedError",
    "GraphError",
    "AlgorithmError",
]
