"""Reference Dijkstra implementation used in tests."""

from __future__ import annotations

import heapq
from typing import Iterable, List, Optional, Set, Tuple

from .graph import Float, Graph, Vertex
from .solver import SSSPResult


def dijkstra_reference(G: Graph, sources: Iterable[Vertex]) -> SSSPResult:
    """Run the standard Dijkstra algorithm.

    Args:
        G: Input graph with non-negative edge weights.
        sources: Iterable of source vertex identifiers.

    Returns:
        Distances and predecessors from running Dijkstra.
    """
    n = G.n
    dist: List[Float] = [float("inf")] * n
    pred: List[Optional[Vertex]] = [None] * n
    pq: List[Tuple[Float, Vertex]] = []
    for s in sources:
        dist[s] = 0.0
        pq.append((0.0, s))
    heapq.heapify(pq)
    seen: Set[Vertex] = set()
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u] or u in seen:
            continue
        seen.add(u)
        for v, w in G.adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                pred[v] = u
                heapq.heappush(pq, (nd, v))
    return SSSPResult(distances=dist, predecessors=pred)
