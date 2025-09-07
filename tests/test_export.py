from __future__ import annotations

from ssspx import Graph
from ssspx.export import export_dag_json, export_dag_graphml
from ssspx.solver import SSSPSolver, SolverConfig


def test_export_json_graphml() -> None:
    G = Graph.from_edges(5, [
        (0, 1, 1.0),
        (1, 2, 1.0),
        (0, 2, 2.0),  # both paths tie
        (2, 3, 1.0),
        (3, 4, 1.0),
    ])
    cfg = SolverConfig(use_transform=False, frontier="heap")
    res = SSSPSolver(G, 0, cfg).solve()

    s_json = export_dag_json(G, res.distances)
    assert '"nodes":' in s_json and '"edges":' in s_json

    s_gml = export_dag_graphml(G, res.distances)
    assert "<graphml" in s_gml and "<graph" in s_gml and "<node" in s_gml and "<edge" in s_gml
