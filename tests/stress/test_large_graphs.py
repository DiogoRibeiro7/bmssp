"""Stress tests for large sparse graphs.

These tests are skipped by default. Set the environment variable
``SSSPX_RUN_STRESS=1`` to execute the heavy variant locally.
"""

from __future__ import annotations

import os
import random

import pytest

from ssspx.graph import Graph
from ssspx.solver import SSSPSolver

pytestmark = [
    pytest.mark.stress,
    pytest.mark.skipif(
        not os.getenv("SSSPX_RUN_STRESS"),
        reason="Stress tests disabled; set SSSPX_RUN_STRESS=1 to enable",
    ),
]


def _rand_graph(n: int, m: int, seed: int) -> Graph:
    rnd = random.Random(seed)
    edges = [(rnd.randrange(n), rnd.randrange(n), rnd.random()) for _ in range(m)]
    return Graph.from_edges(n, edges)


def test_stress_smoke() -> None:
    """Small smoke test to ensure stress harness works."""
    G = _rand_graph(100, 200, seed=0)
    solver = SSSPSolver(G, 0)
    res = solver.solve()
    assert len(res.distances) == 100


def test_large_sparse_graph() -> None:
    """Run solver on a graph with ~1e6 edges to check memory stability."""
    n, m = 100_000, 1_000_000
    G = _rand_graph(n, m, seed=1)
    solver = SSSPSolver(G, 0)
    res = solver.solve()
    assert len(res.distances) == n
