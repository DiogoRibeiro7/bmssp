import json
import time
from dataclasses import asdict

import pytest

from ssspx import Graph, SSSPSolver


def test_solver_metrics_json_schema(tmp_path):
    G = Graph.from_edges(3, [(0, 1, 1.0)])
    solver = SSSPSolver(G, 0)
    t0 = time.perf_counter()
    solver.solve()
    wall = (time.perf_counter() - t0) * 1000.0
    metrics = solver.metrics(wall_ms=wall, peak_mib=1.5)
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(asdict(metrics)))
    data = json.loads(path.read_text())
    assert set(data) == {
        "n",
        "m",
        "frontier",
        "transform",
        "counters",
        "wall_ms",
        "peak_mib",
    }
    assert data["n"] == 3 and data["m"] == 1
    assert data["peak_mib"] == pytest.approx(1.5)
    assert "edges_relaxed" in data["counters"]
