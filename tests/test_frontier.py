from __future__ import annotations

from ssspx.frontier import BlockFrontier, HeapFrontier


def _exercise(fr):
    fr.insert(5, 5.0)
    fr.insert(3, 3.0)
    fr.insert(7, 7.0)
    fr.batch_prepend([(2, 2.0), (1, 1.0)])
    s, x = fr.pull()
    assert len(s) <= 4
    assert x >= 0.0


def test_block_frontier():
    fr = BlockFrontier(M=4, B=1e18)
    _exercise(fr)


def test_heap_frontier():
    fr = HeapFrontier(M=4, B=1e18)
    _exercise(fr)
