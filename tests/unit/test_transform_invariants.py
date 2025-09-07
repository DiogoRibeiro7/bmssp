import random

from ssspx import Graph, constant_outdegree_transform


def test_transform_zero_weight_and_bound() -> None:
    random.seed(0)
    n = 5
    edges = []
    for u in range(n):
        for v in range(n):
            if u != v:
                edges.append((u, v, 1.0))
    G = Graph.from_edges(n, edges)
    delta = 3
    G2, mapping = constant_outdegree_transform(G, delta)
    for u in range(G2.n):
        assert G2.out_degree(u) <= delta
    for clones in mapping.values():
        for i in range(len(clones) - 1):
            cu, cv = clones[i], clones[i + 1]
            weights = [w for v, w in G2.adj[cu] if v == cv]
            assert weights == [0.0]
