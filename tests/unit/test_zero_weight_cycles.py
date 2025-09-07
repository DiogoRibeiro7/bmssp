import pytest
from ssspx import Graph, SSSPSolver, SolverConfig


@pytest.mark.parametrize("use_transform", [False, True])
def test_zero_weight_cycles_and_duplicates(use_transform: bool) -> None:
    edges = [
        (0, 1, 0.0),
        (1, 2, 0.0),
        (2, 0, 0.0),
        (0, 2, 0.0),
        (2, 1, 0.0),
        (1, 0, 0.0),
    ]
    G = Graph.from_edges(3, edges)
    cfg = SolverConfig(use_transform=use_transform, frontier="heap")
    solver = SSSPSolver(G, 0, cfg)
    res = solver.solve()
    assert res.distances == [0.0, 0.0, 0.0]
    path = solver.path(2)
    assert path[0] == 0 and path[-1] == 2
    assert set(path).issubset({0, 1, 2})
