from __future__ import annotations

import json
from typing import List

from ssspx.cli import main
from ssspx.graph import Graph
from ssspx.solver import SolverConfig, SSSPSolver


def test_cli_random_invocation(monkeypatch, capsys) -> None:
    # Mimic `ssspx --random --n 10 --m 20 --source 0 --target 5 --no-transform --frontier heap`
    args: List[str] = [
        "--random",
        "--n",
        "10",
        "--m",
        "20",
        "--source",
        "0",
        "--target",
        "5",
        "--no-transform",
        "--frontier",
        "heap",
    ]
    rc = main(args)
    assert rc == 200
    out = json.loads(capsys.readouterr().out)
    assert "distances" in out
    assert out["sources"] == [0]
    assert out["frontier"] == "heap"
