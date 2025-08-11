import os
import sys
import math
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from bmssp import Graph, run_sssp

def test_run_sssp():
    g = Graph()
    g.add_edge("s", "a", 1)
    g.add_edge("s", "b", 4)
    g.add_edge("a", "b", 2)
    g.add_edge("a", "c", 5)
    g.add_edge("b", "c", 1)
    dist = run_sssp(g, "s")
    assert dist["s"] == 0
    assert dist["a"] == 1
    assert dist["b"] == 4
    assert math.isinf(dist["c"])
