# BMSSP – Breaking the Sorting Barrier for Directed SSSP (Multi-language Implementations)

This repository contains **multi-language reference implementations** of the algorithms from:

> **Breaking the Sorting Barrier for Directed Single-Source Shortest Paths**
> [arXiv:2504.17033v2](https://arxiv.org/abs/2504.17033)
> David H. Yu, et al., 2025

Currently implemented in:
 - ✅ Python
 - ✅ Go
 - ✅ Fortran
 - ✅ C
 - ✅ Rust
 - ✅ Java

---

## Algorithms Implemented
1. **FindPivots** (Lemma 3.2) – bounded Bellman–Ford expansion with pivot selection.
2. **BaseCase** – small-instance solver using Dijkstra.
3. **BMSSP** – recursive bounded multi-source shortest path solver.

---

## Structure
Each language has its own folder with:
- `bmssp` source file(s)
- A `tests/` folder for correctness verification against known shortest paths
- Build/run instructions (Makefile, scripts, etc.)

---

## Running Python Version
```bash
cd python
python bmssp.py
```

## Running Go Version
```bash
cd go
go run bmssp.go
```

## Running Fortran Version
```bash
cd fortran
gfortran bmssp.f90 -o bmssp
./bmssp
```

## Running C Version
```bash
cd c
gcc bmssp.c -lm -o bmssp
./bmssp
```

## Running Rust Version
```bash
cd rust
cargo run
```

## Running Java Version
```bash
cd java
javac BMSSP.java
java BMSSP
```
