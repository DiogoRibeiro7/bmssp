from __future__ import annotations

import warnings

from ssspx.io import load_graph
from ssspx.deprecation import warn_once


def test_load_graph_warns_once(tmp_path):
    path = tmp_path / "g.csv"
    path.write_text("0,1,1\n", encoding="utf-8")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        load_graph(str(path), fmt="csv")
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter("always")
        load_graph(str(path), fmt="csv")
        assert w2 == []


def test_warn_once_direct():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_once("deprecated", since="0.1.0", remove_in="0.2.0")
        warn_once("deprecated", since="0.1.0", remove_in="0.2.0")
        assert len(w) == 1
