import json
import math
from pathlib import Path

import pytest

from ssspx.cli import _build_random_graph
from ssspx.solver import SolverConfig, SSSPSolver

FIXTURES = Path(__file__).parent / "fixtures"
SNAPSHOTS = sorted(FIXTURES.glob("*.json"))


@pytest.mark.regression
@pytest.mark.parametrize("snapshot_path", SNAPSHOTS, ids=lambda p: p.name)
def test_random_graph_snapshots(snapshot_path):
    data = json.loads(snapshot_path.read_text())
    G = _build_random_graph(data["n"], data["m"], data["seed"])
    cfg = SolverConfig(frontier=data["frontier"], use_transform=data["transform"])
    solver = SSSPSolver(G, 0, cfg)
    result = solver.solve()
    expected = [math.inf if v is None else float(v) for v in data["distances"]]
    assert len(result.distances) == len(expected)
    for got, want in zip(result.distances, expected):
        if math.isinf(want):
            assert math.isinf(got)
        else:
            assert math.isclose(got, want, abs_tol=1e-9)
