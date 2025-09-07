#!/usr/bin/env python3
"""Fail if deprecated aliases have expired."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import ssspx

CURRENT = tuple(int(p) for p in ssspx.__version__.split("."))
PATTERN = re.compile(r"remove_in=['\"]([0-9]+\.[0-9]+\.[0-9]+)['\"]")

errors: list[str] = []
for path in Path("src/ssspx").rglob("*.py"):
    text = path.read_text(encoding="utf-8")
    for match in PATTERN.finditer(text):
        ver = tuple(int(p) for p in match.group(1).split("."))
        if ver <= CURRENT:
            errors.append(
                f"{path}: remove_in {match.group(1)} <= current {ssspx.__version__}"
            )

if errors:
    for e in errors:
        print(e)
    sys.exit(1)
