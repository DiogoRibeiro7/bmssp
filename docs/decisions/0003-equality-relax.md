# ADR 0003: Equality Relaxation

## Status
Accepted

## Context
Floating‑point distances can tie due to rounding. Using a strict `<` comparison
when relaxing edges risks missing alternative shortest paths or producing
nondeterministic results.

## Decision
Relax edges with `<=` instead of `<` and treat equal distances as valid updates.
This ensures all minimal parents are captured when reconstructing the shortest
path DAG.

## Consequences
More than one predecessor may be stored for a vertex, slightly increasing
memory usage, but distances remain correct and stable across platforms.

