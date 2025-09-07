import math

import pytest

from ssspx.frontier import BlockFrontier, HeapFrontier


@pytest.mark.parametrize("Frontier", [BlockFrontier, HeapFrontier])
def test_duplicate_and_stale_entries(Frontier) -> None:
    f = Frontier(M=10, B=math.inf)
    f.insert(1, 5.0)
    f.insert(1, 3.0)  # update to better value
    f.insert(2, 4.0)
    f.insert(1, 4.0)  # stale larger value
    f.batch_prepend([(3, 1.0), (3, 0.5)])  # duplicate in batch
    s, x = f.pull()
    assert s == {1, 2, 3}
    s2, x2 = f.pull()
    assert s2 == set() and math.isinf(x2)


@pytest.mark.parametrize("Frontier", [BlockFrontier, HeapFrontier])
def test_empty_pull_and_large_m(Frontier) -> None:
    f = Frontier(M=5, B=99.0)
    s, x = f.pull()
    assert s == set() and x == 99.0
    for i in range(3):
        f.insert(i, float(i))
    s2, x2 = f.pull()
    assert s2 == {0, 1, 2} and x2 == 99.0
