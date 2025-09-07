---
title: 'ssspx: Production-grade Single-Source Shortest Paths'
tags:
  - Python
  - graphs
  - shortest paths
  - algorithms
  - benchmarking
authors:
  - name: Diogo Ribeiro
    orcid: 0009-0001-2022-7072
    affiliation: 1
affiliations:
  - name: ESMAD - Instituto Politécnico do Porto
    index: 1
date: 2024-09-01
bibliography: paper.bib
---

# Summary

`ssspx` provides a production-ready implementation of the BMSSP single-source shortest paths
algorithm. The library includes a switchable frontier, an optional constant-outdegree
transform, a command-line interface, and a benchmarking harness. The codebase is fully
typed, tested, and documented.

# Statement of need

Efficient shortest-path solvers are foundational in graph analysis, route planning, and
network optimization. Existing Python libraries such as NetworkX provide Dijkstra's
algorithm but often lack deterministic behaviour, configurable frontiers, and constant
outdegree transforms. `ssspx` addresses these gaps with a clean API and reproducible
results.

# State of the field

Classical algorithms like Dijkstra remain the baseline for non-negative edge weights
[@dijkstra1959]. For general-purpose graph analysis, NetworkX offers a rich suite of
routines [@hagberg2008exploring]. However, these frameworks do not expose BMSSP-style
optimisations or benchmarking utilities, motivating a focused package.

# Research quality

The solver implements the BMSSP divide-and-conquer structure and validates invariants
around frontiers, transforms, and path reconstruction. Google-style docstrings and static
typing clarify the API surface and expected behaviour.

# Implementation

`ssspx` targets Python 3.9+ and optionally leverages NumPy for a CSR graph backend. The
package depends on `pytest`, `mypy`, `flake8`, `black`, and `isort` for quality checks,
and uses MkDocs for documentation. A small CLI offers CSV input, random graph generation,
and export formats.

# Validation

The project ships unit tests, property-based checks, integration scenarios, and snapshot
regressions. Distances are cross-checked against NetworkX and a Dijkstra baseline with a
numerical tolerance of 1e-9, ensuring semantic parity.

# Performance notes

A benchmark harness compares the solver against a heap-based Dijkstra implementation,
reporting median and 95th percentile runtimes over multiple trials. The constant-outdegree
transform and block frontier yield predictable performance across large graphs.

To aid reproducibility, the repository includes `scripts/reproduce.sh` which generates the
benchmark CSV, derived figures, and a small archive of inputs and outputs. A zipped copy of
these artifacts is published with each release and referenced in the project’s GitHub
release assets.

# Acknowledgements

The author thanks the open-source community for tooling and prior art that made this
project possible.

# References
