"""Utilities for reconstructing paths from predecessor arrays."""

from __future__ import annotations

from typing import Iterable, List, Optional

Vertex = int


def reconstruct_path_basic(
    predecessors: List[Optional[Vertex]],
    source: Vertex,
    target: Vertex,
) -> List[Vertex]:
    """Return the path from ``source`` to ``target`` using a predecessor array.

    Args:
        predecessors: Predecessor of each vertex or ``None`` if unknown or
            unreached.
        source: Source vertex identifier.
        target: Target vertex identifier.

    Returns:
        Vertices from source to target (inclusive). Returns an empty list if no
        path exists.

    Notes:
        Works for the non-transformed case (one-level graph). It is safe when
        ``predecessors[source]`` is ``None`` (root).
    """
    if source == target:
        return [source]

    if target < 0 or target >= len(predecessors) or source < 0 or source >= len(predecessors):
        raise ValueError("source/target out of range.")

    # Walk backwards from target to source
    chain: List[Vertex] = []
    cur: Optional[Vertex] = target
    seen = set()
    while cur is not None:
        chain.append(cur)
        if cur == source:
            chain.reverse()
            return chain
        if cur in seen:  # defensive: detect cycles in preds
            break
        seen.add(cur)
        cur = predecessors[cur]

    return []  # unreachable


def compress_original_path_from_clones(
    clone_path: Iterable[Vertex],
    clone2orig: List[Vertex],
) -> List[Vertex]:
    """Map a clone-level path back to original vertices.

    Consecutive duplicates created by zero-weight cycles are removed.

    Args:
        clone_path: Path expressed in the transformed (clone) graph.
        clone2orig: Mapping from each clone identifier to its original vertex
            identifier.

    Returns:
        Path in the original graph with consecutive duplicates removed.
    """
    path_o: List[Vertex] = []
    last: Optional[Vertex] = None
    for c in clone_path:
        o = clone2orig[c]
        if last is None or o != last:
            path_o.append(o)
            last = o
    return path_o
