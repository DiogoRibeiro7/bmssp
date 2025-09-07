"""Simple directed graph representation used by the solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

from .exceptions import GraphFormatError, InputError

Vertex = int
Float = float
Edge = Tuple[Vertex, Vertex, Float]


@dataclass
class Graph:
    """Directed graph with non-negative edge weights.

    Negative weights are not supported: attempting to insert an edge with
    ``w < 0`` raises :class:`~ssspx.exceptions.GraphFormatError` that cites the
    offending edge.

    Attributes:
        n: Number of vertices in the range ``0`` .. ``n-1``.
        adj: Outgoing adjacency lists.
    """

    n: int

    def __post_init__(self) -> None:
        """Validate vertex count and initialize adjacency lists."""
        if not isinstance(self.n, int) or self.n <= 0:
            raise InputError("Graph.n must be a positive integer.")
        self.adj: List[List[Tuple[Vertex, Float]]] = [[] for _ in range(self.n)]

    def add_edge(self, u: Vertex, v: Vertex, w: Float) -> None:
        """Add a directed edge from ``u`` to ``v``.

        Args:
            u: Tail vertex.
            v: Head vertex.
            w: Non-negative edge weight.

        Raises:
            InputError: If ``u`` or ``v`` are out of range.
            GraphFormatError: If ``w`` is negative.

        Examples:
            ```python
            >>> g = Graph(2)
            >>> g.add_edge(0, 1, 1.5)
            >>> g.adj
            [[(1, 1.5)], []]
            ```
        """
        if not (0 <= u < self.n and 0 <= v < self.n):
            raise InputError("u and v must be vertex ids in [0, n).")
        if not isinstance(w, (int, float)):
            raise GraphFormatError(f"non-numeric weight {w!r} on edge ({u}, {v})")
        if w < 0:
            raise GraphFormatError(f"negative weight {w} on edge ({u}, {v})")
        self.adj[u].append((int(v), float(w)))

    @classmethod
    def from_edges(cls, n: int, edges: Iterable[Edge]) -> "Graph":
        """Create a graph from an iterable of edges.

        Args:
            n: Number of vertices.
            edges: Iterable of ``(u, v, w)`` tuples.

        Returns:
            A graph populated with the provided edges.
        """
        g = cls(n)
        for u, v, w in edges:
            g.add_edge(int(u), int(v), float(w))
        return g

    def out_degree(self, u: Vertex) -> int:
        """Return the out-degree of vertex ``u``.

        Args:
            u: Vertex identifier.

        Returns:
            Number of outgoing edges from ``u``.
        """
        return len(self.adj[u])
