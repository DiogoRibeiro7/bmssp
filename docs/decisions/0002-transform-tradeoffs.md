# ADR 0002: Constant Out-degree Transform

## Status
Accepted

## Context
Vertices with very high out-degree slow frontier operations. The BMSSP transform
clones such vertices to cap the out-degree by a configurable value.

## Decision
Provide an optional transform (enabled via `use_transform`) that enforces a
bound on out-degree. The cap defaults to four but can be tuned per run.

## Consequences
The transform increases the number of vertices and edges, which can raise memory
usage and complicate path reconstruction. However, it enables predictable
frontier performance on graphs with skewed degree distributions.

