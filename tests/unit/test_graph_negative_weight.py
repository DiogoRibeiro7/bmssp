import pytest

from ssspx import Graph
from ssspx.exceptions import GraphFormatError


def test_negative_weight_error_mentions_edge() -> None:
    g = Graph(2)
    with pytest.raises(GraphFormatError) as excinfo:
        g.add_edge(0, 1, -1.0)
    msg = str(excinfo.value)
    assert "edge (0, 1)" in msg
    assert "-1.0" in msg
