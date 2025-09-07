from __future__ import annotations

import math

from ssspx import Graph, SolverConfig, SSSPSolver, dijkstra_reference


def test_multi_source_paths_and_distances() -> None:
    G = Graph.from_edges(
        5,
        [
            (0, 1, 1.0),
            (2, 1, 1.0),
            (1, 3, 1.0),
            (2, 3, 2.0),
        ],
    )
    cfg = SolverConfig(use_transform=False, frontier="heap")
    solver = SSSPSolver(G, source=0, sources=[0, 2], config=cfg)
    res = solver.solve()
    ref = dijkstra_reference(G, [0, 2])
    for a, b in zip(res.distances, ref.distances):
        aa = a if a < float("inf") else 1e18
        bb = b if b < float("inf") else 1e18
        assert abs(aa - bb) < 1e-9
    assert solver.path(3) == [0, 1, 3]
    assert math.isinf(res.distances[4])
    assert solver.path(4) == []
