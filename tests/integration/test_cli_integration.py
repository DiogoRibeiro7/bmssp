import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

import pytest

from ssspx import Graph, write_graph

pytestmark = pytest.mark.integration


def run_cli(args: List[object], cwd: Path) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "ssspx.cli", *map(str, args)]
    project_src = Path(__file__).resolve().parents[2] / "src"
    env = {**os.environ, "PYTHONPATH": str(project_src)}
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env, timeout=10)


def test_edges_json_output(tmp_path: Path) -> None:
    csv = tmp_path / "edges.csv"
    csv.write_text("0,1,1\n1,2,2\n")
    result = run_cli(["--edges", csv, "--source", 0], tmp_path)
    assert result.returncode == 200
    data = json.loads(result.stdout)
    assert set(data.keys()) == {"sources", "frontier", "use_transform", "distances"}
    assert data["sources"] == [0]
    assert all(isinstance(d, (int, float)) for d in data["distances"])


def test_edges_multiple_sources(tmp_path: Path) -> None:
    csv = tmp_path / "edges.csv"
    csv.write_text("0,1,1\n2,1,1\n")
    result = run_cli(["--edges", csv, "--sources", "0,2"], tmp_path)
    assert result.returncode == 200
    data = json.loads(result.stdout)
    assert data["sources"] == [0, 2]


def test_random_profile_creates_file(tmp_path: Path) -> None:
    prof = tmp_path / "run.cprof"
    result = run_cli(
        ["--random", "--n", 5, "--m", 10, "--profile", "--profile-out", prof],
        tmp_path,
    )
    assert result.returncode == 200
    assert prof.is_file() and prof.stat().st_size > 0


def test_export_json_and_graphml(tmp_path: Path) -> None:
    csv = tmp_path / "edges.csv"
    csv.write_text("0,1,1\n1,2,2\n")
    json_out = tmp_path / "dag.json"
    graphml_out = tmp_path / "dag.graphml"
    result = run_cli(
        [
            "--edges",
            csv,
            "--source",
            0,
            "--export-json",
            json_out,
            "--export-graphml",
            graphml_out,
        ],
        tmp_path,
    )
    assert result.returncode == 200
    data = json.loads(json_out.read_text())
    assert set(data.keys()) == {"nodes", "edges"}
    assert all("id" in node for node in data["nodes"])
    assert all({"source", "target"} <= edge.keys() for edge in data["edges"])
    ET.parse(graphml_out)


def test_log_json(tmp_path: Path) -> None:
    result = run_cli(["--random", "--n", 3, "--m", 3, "--log-json"], tmp_path)
    assert result.returncode == 200
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["n"] == 3 and data["m"] == 3
    assert data["frontier"] == "block"
    assert "edges_relaxed" in data


@pytest.mark.parametrize(
    "fmt, ext",
    [("csv", ".csv"), ("jsonl", ".jsonl"), ("mtx", ".mtx"), ("graphml", ".graphml")],
)
def test_cli_reads_formats(tmp_path: Path, fmt: str, ext: str) -> None:
    G = Graph.from_edges(3, [(0, 1, 1.0), (1, 2, 2.0)])
    path = tmp_path / f"graph{ext}"
    write_graph(G, path, fmt)
    result = run_cli(["--edges", path, "--source", 0], tmp_path)
    assert result.returncode == 200
    data = json.loads(result.stdout)
    assert data["distances"][2] == pytest.approx(3.0)


def test_format_override(tmp_path: Path) -> None:
    G = Graph.from_edges(2, [(0, 1, 1.0)])
    path = tmp_path / "edges"
    write_graph(G, path, "csv")
    result = run_cli(["--edges", path, "--format", "csv", "--source", 0], tmp_path)
    assert result.returncode == 200
    data = json.loads(result.stdout)
    assert data["distances"][1] == pytest.approx(1.0)


def test_metrics_out(tmp_path: Path) -> None:
    csv = tmp_path / "edges.csv"
    csv.write_text("0,1,1\n1,2,2\n")
    out = tmp_path / "metrics.json"
    result = run_cli(["--edges", csv, "--metrics-out", out], tmp_path)
    assert result.returncode == 200
    data = json.loads(out.read_text())
    assert data["n"] == 3 and data["m"] == 2
    assert set(data["counters"].keys()) >= {"edges_relaxed", "pulls"}
    assert "peak_mib" in data and data["peak_mib"] >= 0.0
