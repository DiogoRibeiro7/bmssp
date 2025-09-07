# ADR 0001: Choose Block Frontier

## Status
Accepted

## Context
Selecting a frontier data structure affects memory locality and the cost of
processing duplicate or stale entries. A block frontier groups vertices by
distance buckets, whereas a binary heap orders every push individually.

## Decision
Use a block frontier as the default implementation. Keep a heap frontier as a
simpler baseline and for debugging.

## Consequences
The block frontier improves cache behaviour and reduces priority‑queue churn but
requires more bookkeeping and is tuned for non‑negative weights.

