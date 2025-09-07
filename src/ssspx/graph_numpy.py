"""NumPy-backed graph representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from .exceptions import GraphFormatError, InputError
from .graph import Edge, Float, Graph, Vertex


@dataclass
class NumpyGraph:
    """Directed graph using NumPy arrays for adjacency lists.

    Negative weights are disallowed and will trigger
    :class:`~ssspx.exceptions.GraphFormatError` with the exact offending edge.
    """

    n: int

    def __post_init__(self) -> None:
        """Validate initialization arguments and allocate adjacency storage."""
        if not isinstance(self.n, int) or self.n <= 0:
            raise InputError("Graph.n must be a positive integer.")
        self.adj: List[np.ndarray] = [np.zeros((0, 2), dtype=float) for _ in range(self.n)]

    def add_edge(self, u: Vertex, v: Vertex, w: Float) -> None:
        """Add a directed edge from ``u`` to ``v``."""
        if not (0 <= u < self.n and 0 <= v < self.n):
            raise InputError("u and v must be vertex ids in [0, n).")
        if not isinstance(w, (int, float)):
            raise GraphFormatError(f"non-numeric weight {w!r} on edge ({u}, {v})")
        if w < 0:
            raise GraphFormatError(f"negative weight {w} on edge ({u}, {v})")
        arr = self.adj[u]
        self.adj[u] = np.vstack([arr, np.array([[v, float(w)]])])

    @classmethod
    def from_edges(cls, n: int, edges: Iterable[Edge]) -> "NumpyGraph":
        """Construct a graph from an iterable of ``(u, v, w)`` edges."""
        g = cls(n)
        for u, v, w in edges:
            g.add_edge(int(u), int(v), float(w))
        return g

    def out_degree(self, u: Vertex) -> int:
        """Return the out-degree of vertex ``u``."""
        return int(self.adj[u].shape[0])

    # Utility for tests: convert to standard Graph
    def to_graph(self) -> Graph:
        """Return a standard :class:`~ssspx.graph.Graph` copy of this graph."""
        edges: List[Edge] = []
        for u in range(self.n):
            arr = self.adj[u]
            for v, w in arr:
                edges.append((int(u), int(v), float(w)))
        return Graph.from_edges(self.n, edges)


__all__ = ["NumpyGraph"]
