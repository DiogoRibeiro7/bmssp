"""Lightweight logging helpers for optional structured output."""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, Protocol


class Logger(Protocol):
    """Protocol for minimal logger implementations."""

    def info(self, event: str, **fields: Any) -> None:
        """Emit an ``INFO``-level event."""
        ...

    def debug(self, event: str, **fields: Any) -> None:
        """Emit a ``DEBUG``-level event."""
        ...


class NoopLogger:
    """Logger that discards all events."""

    def info(self, event: str, **fields: Any) -> None:  # pragma: no cover - noop
        """Ignore an ``INFO`` event."""
        return

    def debug(self, event: str, **fields: Any) -> None:  # pragma: no cover - noop
        """Ignore a ``DEBUG`` event."""
        return


class StdLogger:
    """Minimal logger with optional JSON output."""

    _levels: Dict[str, int] = {"debug": 10, "info": 20, "warning": 30}

    def __init__(
        self,
        level: str = "warning",
        json_fmt: bool = False,
        stream: Any | None = None,
    ) -> None:
        """Initialize the logger."""
        self.level = level
        self.json_fmt = json_fmt
        self.stream = stream or sys.stderr

    def _enabled(self, level: str) -> bool:  # pragma: no cover - thin wrapper
        return self._levels[level] >= self._levels.get(self.level, 20)

    def log(self, level: str, event: str, **fields: Any) -> None:  # pragma: no cover - simple I/O
        """Emit a log ``event`` at ``level`` with additional ``fields``."""
        if not self._enabled(level):
            return
        if self.json_fmt:
            obj = {"level": level, "event": event}
            obj.update(fields)
            self.stream.write(json.dumps(obj) + "\n")
        else:
            kv = " ".join(f"{k}={v}" for k, v in fields.items())
            msg = f"{level} {event} {kv}".rstrip()
            self.stream.write(msg + "\n")

    def info(self, event: str, **fields: Any) -> None:  # pragma: no cover - passthrough
        """Emit an ``INFO`` event."""
        self.log("info", event, **fields)

    def debug(self, event: str, **fields: Any) -> None:  # pragma: no cover - passthrough
        """Emit a ``DEBUG`` event."""
        self.log("debug", event, **fields)
