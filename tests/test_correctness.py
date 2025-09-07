from __future__ import annotations

import random
from typing import List, Tuple

import pytest

from ssspx import Graph, SSSPSolver, SolverConfig, dijkstra_reference

Edge = Tuple[int, int, float]


def random_graph(n: int, m: int, seed: int = 0) -> Graph:
    rnd = random.Random(seed)
    edges: List[Edge] = []
    for _ in range(m):
        u = rnd.randrange(n)
        v = rnd.randrange(n)
        w = rnd.random() * 10.0
        edges.append((u, v, w))
    return Graph.from_edges(n, edges)


@pytest.mark.parametrize("n,m", [(20, 60), (50, 200)])
@pytest.mark.parametrize("use_transform", [False, True])
@pytest.mark.parametrize("frontier", ["heap", "block"])
def test_vs_dijkstra(n: int, m: int, use_transform: bool, frontier: str) -> None:
    G = random_graph(n, m, seed=42)
    cfg = SolverConfig(use_transform=use_transform, frontier=frontier)
    solver = SSSPSolver(G, source=0, config=cfg)
    res = solver.solve()

    ref = dijkstra_reference(G, [0])
    for a, b in zip(res.distances, ref.distances):
        aa = a if a < float("inf") else 1e18
        bb = b if b < float("inf") else 1e18
        assert abs(aa - bb) < 1e-7
