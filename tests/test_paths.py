from __future__ import annotations

from typing import List, Optional

from ssspx import Graph, SSSPSolver, SolverConfig, dijkstra_reference
from ssspx.path import reconstruct_path_basic


def _reconstruct_basic(pred: List[Optional[int]], s: int, t: int) -> list[int]:
    return reconstruct_path_basic(pred, s, t)


def test_path_no_transform() -> None:
    G = Graph.from_edges(6, [
        (0, 1, 2.0), (0, 2, 5.0),
        (1, 2, 1.0), (1, 3, 2.0),
        (2, 3, 1.0), (2, 4, 3.0),
        (3, 4, 1.0), (4, 5, 1.0),
    ])
    cfg = SolverConfig(use_transform=False, frontier="heap")
    solver = SSSPSolver(G, 0, cfg)
    res = solver.solve()

    ref = dijkstra_reference(G, [0])
    # Check path to 5 exists and begins at 0
    path = solver.path(5)
    assert path[0] == 0 and path[-1] == 5
    # Length (in edges) should be <= Dijkstra's path length in edges (not a strong assertion, but a sanity check)
    d_path = _reconstruct_basic(ref.predecessors, 0, 5)
    assert len(path) == len(d_path)


def test_path_with_transform() -> None:
    # Create a node with many outgoing edges to trigger clone split
    edges = [(0, i, 1.0) for i in range(1, 8)] + [(i, 8, 1.0) for i in range(1, 8)]
    G = Graph.from_edges(9, edges)
    cfg = SolverConfig(use_transform=True, target_outdeg=2, frontier="block")
    solver = SSSPSolver(G, 0, cfg)
    res = solver.solve()

    # Path to 8 must exist and map back to original ids, without duplicates
    path = solver.path(8)
    assert path[0] == 0 and path[-1] == 8
    # Distances should be finite and match number of hops (weights all 1)
    assert res.distances[8] == 2.0
