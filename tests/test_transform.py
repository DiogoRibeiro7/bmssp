from __future__ import annotations

from ssspx import Graph
from ssspx.transform import constant_outdegree_transform


def test_outdegree_bound() -> None:
    edges = [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (0, 4, 1.0), (0, 5, 1.0)]
    G = Graph.from_edges(6, edges)
    G2, mapping = constant_outdegree_transform(G, delta=2)

    # Every vertex in G2 must have outdegree <= 2
    assert all(len(G2.adj[u]) <= 2 for u in range(G2.n))
    # Vertex 0 has 5 edges, with delta=2 it needs 4 clones:
    # Clone 1: 1 edge + link to clone 2 = 2 edges total
    # Clone 2: 1 edge + link to clone 3 = 2 edges total  
    # Clone 3: 1 edge + link to clone 4 = 2 edges total
    # Clone 4: 2 edges (final) = 2 edges total
    assert 0 in mapping and len(mapping[0]) == 4
