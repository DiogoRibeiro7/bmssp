"""Refresh random graph distance snapshots."""

import json
import math
from pathlib import Path

from ssspx.cli import _build_random_graph
from ssspx.solver import SolverConfig, SSSPSolver

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "regressions" / "fixtures"


def update() -> None:
    for path in sorted(FIXTURES.glob("*.json")):
        data = json.loads(path.read_text())
        G = _build_random_graph(data["n"], data["m"], data["seed"])
        cfg = SolverConfig(frontier=data["frontier"], use_transform=data["transform"])
        solver = SSSPSolver(G, 0, cfg)
        result = solver.solve()
        data["distances"] = [
            None if math.isinf(d) else float(f"{d:.12f}") for d in result.distances
        ]
        path.write_text(json.dumps(data, indent=2) + "\n")
        print(f"updated {path}")


if __name__ == "__main__":
    update()
