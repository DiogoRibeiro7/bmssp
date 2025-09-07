from ssspx import Graph, SSSPSolver


def test_counters_increase():
    G = Graph.from_edges(2, [(0, 1, 1.0)])
    solver = SSSPSolver(G, 0)
    before = solver.summary()
    assert all(v == 0 for v in before.values())
    solver.solve()
    after = solver.summary()
    assert after["edges_relaxed"] > 0
    assert after["pulls"] > 0
    assert after["findpivots_rounds"] > 0
    assert after["basecase_pops"] > 0
