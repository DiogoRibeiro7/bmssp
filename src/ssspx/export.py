"""Export utilities for shortest-path DAGs."""

from __future__ import annotations

import json
from typing import List, Tuple

from .graph import Float, Graph


def shortest_path_dag(
    G: Graph, distances: List[Float], eps: float = 1e-12
) -> List[Tuple[int, int]]:
    """Return edges of the shortest-path DAG.

    Args:
        G: Original graph.
        distances: Distances from the source to each vertex (``inf`` for
            unreachable vertices).
        eps: Numerical tolerance.

    Returns:
        Edges ``(u, v)`` satisfying ``d[v] == d[u] + w(u, v)``.
    """
    dag: List[Tuple[int, int]] = []
    for u in range(G.n):
        du = distances[u]
        if not (du < float("inf")):
            continue
        for v, w in G.adj[u]:
            dv = distances[v]
            if dv < float("inf") and abs(dv - (du + w)) <= eps:
                dag.append((u, v))
    return dag


def export_dag_json(G: Graph, distances: List[Float]) -> str:
    """Return a JSON string with nodes and DAG edges."""
    edges = shortest_path_dag(G, distances)
    data = {
        "nodes": [{"id": i} for i in range(G.n)],
        "edges": [{"source": u, "target": v} for (u, v) in edges],
    }
    return json.dumps(data)


def export_dag_graphml(G: Graph, distances: List[Float]) -> str:
    """Return a minimal GraphML string for the shortest-path DAG."""
    edges = shortest_path_dag(G, distances)
    lines: List[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">')
    lines.append('  <graph id="G" edgedefault="directed">')
    for i in range(G.n):
        lines.append(f'    <node id="n{i}"/>')
    for u, v in edges:
        lines.append(f'    <edge source="n{u}" target="n{v}"/>')
    lines.append("  </graph>")
    lines.append("</graphml>")
    return "\n".join(lines)
