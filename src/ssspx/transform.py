"""Graph transformations used by the solver."""

from __future__ import annotations

from typing import Dict, List, Tuple

from .exceptions import ConfigError
from .graph import Float, Graph, Vertex


def constant_outdegree_transform(G: Graph, delta: int) -> Tuple[Graph, Dict[Vertex, List[Vertex]]]:
    """Split vertices so that every vertex has out-degree at most ``delta``.

    The transformation replaces a vertex with a chain of clones. Each clone
    except the last holds ``delta - 1`` of the original outgoing edges and a
    zero-weight edge to the next clone. The last clone holds the remaining
    edges (at most ``delta``). Incoming edges of a vertex are redirected to its
    first clone.

    Args:
        G: Original graph to transform.
        delta: Maximum out-degree allowed for each vertex (``delta > 0``).

    Returns:
        A tuple ``(G2, mapping)`` where ``G2`` is the transformed graph and
        ``mapping`` maps each original vertex id to a list of its clone ids in
        ``G2``.

    Raises:
        ConfigError: If ``delta`` is not positive.
    """
    if delta <= 0:
        raise ConfigError("delta must be positive")

    # First pass: determine clones and partition outgoing edges.
    mapping: Dict[Vertex, List[Vertex]] = {}
    partitions: Dict[Vertex, List[List[Tuple[Vertex, Float]]]] = {}
    next_id = 0
    for u in range(G.n):
        edges = list(G.adj[u])
        if len(edges) <= delta:
            mapping[u] = [next_id]
            partitions[u] = [edges]
            next_id += 1
            continue

        chunks: List[List[Tuple[Vertex, Float]]] = []
        remaining = edges
        while len(remaining) > delta:
            chunks.append(remaining[: delta - 1])
            remaining = remaining[delta - 1 :]
        chunks.append(remaining)  # last chunk size <= delta
        clones = [next_id + i for i in range(len(chunks))]
        mapping[u] = clones
        partitions[u] = chunks
        next_id += len(clones)

    # Second pass: build edge list in transformed space.
    edges2: List[Tuple[Vertex, Vertex, Float]] = []
    for u in range(G.n):
        clones = mapping[u]
        chunks = partitions[u]
        for i, chunk in enumerate(chunks):
            cu = clones[i]
            for v, w in chunk:
                edges2.append((cu, mapping[v][0], w))
            if i < len(clones) - 1:
                edges2.append((cu, clones[i + 1], 0.0))

    G2 = Graph.from_edges(next_id, edges2)
    return G2, mapping


__all__ = ["constant_outdegree_transform"]
