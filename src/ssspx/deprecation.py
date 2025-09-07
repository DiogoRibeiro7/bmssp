"""Helpers for emitting one-time deprecation warnings."""

from __future__ import annotations

import warnings
from typing import Set, Tuple

_warned: Set[Tuple[str, str, str, type]] = set()


def warn_once(
    message: str,
    *,
    category: type[Warning] = DeprecationWarning,
    since: str,
    remove_in: str,
) -> None:
    """Issue ``message`` once per process.

    Parameters
    ----------
    message:
        The deprecation message to emit.
    category:
        Warning type, defaults to :class:`DeprecationWarning`.
    since:
        Version in which the deprecation was introduced.
    remove_in:
        Version in which the deprecated name will be removed.
    """
    key = (message, since, remove_in, category)
    if key in _warned:
        return
    _warned.add(key)
    warnings.warn(
        f"{message} (deprecated since {since}; will be removed in {remove_in})",
        category,
        stacklevel=2,
    )
