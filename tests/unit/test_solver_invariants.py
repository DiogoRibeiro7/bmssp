from ssspx import Graph, SolverConfig, SSSPSolver


def solve_distances(edges):
    G = Graph.from_edges(3, edges)
    solver = SSSPSolver(G, 0, SolverConfig(use_transform=False, frontier="heap"))
    return solver.solve().distances


def test_monotonic_distances_under_extra_edges() -> None:
    base = [(0, 1, 2.0), (1, 2, 2.0)]
    d1 = solve_distances(base)
    d2 = solve_distances(base + [(0, 2, 10.0)])
    assert d2 == d1
    d3 = solve_distances(base + [(0, 2, 1.0)])
    assert d3[2] <= d1[2]


def test_relax_tie_returns_true() -> None:
    G = Graph.from_edges(2, [(0, 1, 1.0)])
    solver = SSSPSolver(G, 0, SolverConfig(use_transform=False, frontier="heap"))
    assert solver._relax(0, 1, 1.0) is True
    assert solver._relax(0, 1, 1.0) is True
    assert solver.dhat[1] == 1.0 and solver.pred[1] == 0
