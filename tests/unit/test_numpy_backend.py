import pytest


def test_numpy_graph_matches_graph() -> None:
    from ssspx import Graph, NumpyGraph, SolverConfig, SSSPSolver

    if NumpyGraph is None:
        pytest.skip("numpy backend not installed")

    edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 3.0)]
    G = Graph.from_edges(3, edges)
    NG = NumpyGraph.from_edges(3, edges)
    assert NG.out_degree(0) == G.out_degree(0)
    solver1 = SSSPSolver(G, 0, SolverConfig(use_transform=False, frontier="heap"))
    solver2 = SSSPSolver(NG.to_graph(), 0, SolverConfig(use_transform=False, frontier="heap"))
    assert solver1.solve().distances == solver2.solve().distances
