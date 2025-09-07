"""Lightweight helpers for optional cProfile integration."""

from __future__ import annotations

import cProfile
import pstats
from dataclasses import dataclass
from io import StringIO
from types import TracebackType
from typing import Optional


@dataclass
class ProfileReport:
    """Summary of profiling statistics."""

    profile: cProfile.Profile

    def to_text(self, lines: int = 20) -> str:
        """Return a formatted statistics table.

        Args:
            lines: Number of rows from the profile to include.

        Returns:
            Human readable text report.
        """
        buffer = StringIO()
        stats = pstats.Stats(self.profile, stream=buffer)
        stats.strip_dirs().sort_stats("cumulative").print_stats(lines)
        return buffer.getvalue()


class ProfileSession:
    """Context manager that records a profiling session using :mod:`cProfile`."""

    def __init__(self, dump_path: Optional[str] = None) -> None:
        """Initialize the profiling session.

        Args:
            dump_path: Optional path where raw stats should be dumped when the
                session exits.
        """
        self.dump_path = dump_path
        self._prof = cProfile.Profile()
        self._profile: Optional[cProfile.Profile] = None

    def __enter__(self) -> "ProfileSession":
        """Start the profiling session."""
        self._prof.enable()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        """Stop profiling and finalize statistics."""
        self._prof.disable()
        if self.dump_path:
            self._prof.dump_stats(self.dump_path)
        self._profile = self._prof

    def report(self) -> ProfileReport:
        """Return profiling statistics collected so far."""
        if self._profile is None:
            raise RuntimeError("profiling session not finished")
        return ProfileReport(self._profile)


__all__: list[str] = ["ProfileReport", "ProfileSession"]
