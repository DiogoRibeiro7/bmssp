from pathlib import Path

import pytest

from ssspx import Graph, read_graph, write_graph


def _edges_from_graph(G: Graph):
    return sorted((u, v, w) for u in range(G.n) for v, w in G.adj[u])


@pytest.mark.parametrize(
    "fmt, ext",
    [
        ("csv", ".csv"),
        ("jsonl", ".jsonl"),
        ("mtx", ".mtx"),
        ("graphml", ".graphml"),
    ],
)
def test_round_trip(tmp_path: Path, fmt: str, ext: str) -> None:
    edges = [(0, 1, 1.0), (1, 2, 2.5)]
    G = Graph.from_edges(3, edges)
    first = tmp_path / f"g1{ext}"
    write_graph(G, first, fmt)
    G1 = read_graph(first)
    second = tmp_path / f"g2{ext}"
    write_graph(G1, second)
    G2 = read_graph(second)
    assert _edges_from_graph(G2) == sorted(edges)
    assert G2.n == 3
