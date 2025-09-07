import pytest
from ssspx import Graph, SSSPSolver, SolverConfig


@pytest.mark.parametrize("use_transform", [False, True])
def test_path_edge_cases(use_transform: bool) -> None:
    G = Graph.from_edges(5, [(0, 1, 1.0), (1, 2, 1.0)])
    cfg = SolverConfig(use_transform=use_transform, frontier="heap")
    solver = SSSPSolver(G, 0, cfg)
    solver.solve()
    assert solver.path(0) == [0]
    assert solver.path(2) == [0, 1, 2]
    assert solver.path(4) == []


def test_path_compression_with_transform() -> None:
    G = Graph.from_edges(
        5,
        [
            (0, 1, 1.0),
            (0, 2, 1.0),
            (0, 3, 1.0),
            (1, 4, 1.0),
            (2, 4, 2.0),
            (3, 4, 3.0),
        ],
    )
    cfg = SolverConfig(use_transform=True, target_outdeg=2, frontier="heap")
    solver = SSSPSolver(G, 0, cfg)
    solver.solve()
    assert solver.path(4) == [0, 1, 4]
