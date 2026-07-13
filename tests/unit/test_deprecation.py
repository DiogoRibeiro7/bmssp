from __future__ import annotations

import warnings

from ssspx.deprecation import warn_once


def test_warn_once_direct():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_once("deprecated", since="0.1.0", remove_in="0.2.0")
        warn_once("deprecated", since="0.1.0", remove_in="0.2.0")
        assert len(w) == 1
