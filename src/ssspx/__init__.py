"""Public package exports for :mod:`ssspx`."""

from __future__ import annotations

from .dijkstra import dijkstra_reference
from .exceptions import (
    AlgorithmError,
    ConfigError,
    GraphError,
    GraphFormatError,
    InputError,
    NotSupportedError,
)
from .graph import Graph

try:  # pragma: no cover
    from .graph_numpy import NumpyGraph
except ModuleNotFoundError:  # pragma: no cover
    NumpyGraph = None  # type: ignore[misc, assignment]
from .io import load_graph, read_graph, write_graph
from .logger import Logger, NoopLogger, StdLogger
from .solver import SolverConfig, SolverMetrics, SSSPResult, SSSPSolver
from .transform import constant_outdegree_transform

__version__ = "0.1.0"

__all__ = [
    "Graph",
    "NumpyGraph",
    "SSSPSolver",
    "SSSPResult",
    "SolverConfig",
    "SolverMetrics",
    "constant_outdegree_transform",
    "dijkstra_reference",
    "Logger",
    "NoopLogger",
    "StdLogger",
    "read_graph",
    "write_graph",
    "load_graph",
    "GraphError",
    "AlgorithmError",
    "InputError",
    "ConfigError",
    "GraphFormatError",
    "NotSupportedError",
]
