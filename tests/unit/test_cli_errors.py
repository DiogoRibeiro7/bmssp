import subprocess
import sys

import pytest

from ssspx import Graph, SolverConfig, SSSPSolver
from ssspx.exceptions import ConfigError


def test_cli_bad_edges_file(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("nonsense\n")
    proc = subprocess.run(
        [sys.executable, "-m", "ssspx.cli", "--edges", str(p)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 64
    assert "no edges parsed" in proc.stderr.lower()


def test_cli_missing_edges_file(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "ssspx.cli", "--edges", str(tmp_path / "none.csv")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 64
    assert "edges file not found" in proc.stderr.lower()


def test_cli_negative_weight(tmp_path):
    p = tmp_path / "neg.csv"
    p.write_text("0,1,-1\n")
    proc = subprocess.run(
        [sys.executable, "-m", "ssspx.cli", "--edges", str(p)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 64
    assert "edge (0, 1)" in proc.stderr
    assert "negative weight" in proc.stderr.lower()


def test_cli_success_exit_code(tmp_path):
    p = tmp_path / "edges.csv"
    p.write_text("0,1,1\n")
    proc = subprocess.run(
        [sys.executable, "-m", "ssspx.cli", "--edges", str(p)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 200


def test_solver_invalid_frontier():
    G = Graph.from_edges(2, [(0, 1, 1.0)])
    cfg = SolverConfig(frontier="bogus")
    solver = SSSPSolver(G, 0, config=cfg)
    with pytest.raises(ConfigError) as excinfo:
        solver.solve()
    assert "unknown frontier" in str(excinfo.value)
