# BMSSP – Breaking the Sorting Barrier for Directed SSSP (Multi-language Implementations)

This repository contains **multi-language reference implementations** of the algorithms from:

> **Breaking the Sorting Barrier for Directed Single-Source Shortest Paths**  
> [arXiv:2504.17033v2](https://arxiv.org/abs/2504.17033)  
> Ran Duan, Jiayi Mao, Xiao Mao, Xinkai Shu, Longhui Yin, 2025

**🎯 Breakthrough Result**: This paper presents the first deterministic algorithm to break Dijkstra's O(m + n log n) time bound for directed single-source shortest paths, achieving **O(m log^(2/3) n)** time complexity.

Currently implemented in:

 - ✅ Python
 - ✅ Go
 - ✅ Fortran
 - ✅ C
 - ✅ Rust
 - ✅ Java

---

## Algorithm Overview

The BMSSP (Bounded Multi-Source Shortest Path) algorithm uses a novel recursive partitioning technique that merges ideas from both Dijkstra's algorithm and the Bellman-Ford algorithm:

### Core Innovation

- **Frontier Reduction**: Instead of maintaining a frontier of size Θ(n), the algorithm reduces it to |U|/log^Ω(1)(n)
- **Pivot Selection**: Uses bounded Bellman-Ford expansion to identify "pivot" vertices that cover large subtrees
- **Recursive Structure**: Employs O(log n / t) levels of recursion with specialized data structures

### Key Components

1. **FindPivots** (Lemma 3.2) – Bounded Bellman-Ford expansion with pivot selection
2. **BaseCase** – Small-instance solver using Dijkstra-like approach  
3. **BMSSP** – Main recursive bounded multi-source shortest path solver
4. **DQueue** – Specialized data structure supporting bounded batch operations

---

## Important Notes

⚠️ **These are research algorithm implementations**, not general-purpose shortest path solvers:

- **Theoretical Focus**: Optimized for asymptotic complexity O(m log^(2/3) n), not practical performance on small graphs
- **Bounded Computation**: May not compute complete shortest path trees due to algorithmic bounds and early termination conditions
- **Parameter Sensitivity**: Uses specific parameters k = ⌊log^(1/3) n⌋ and t = ⌊log^(2/3) n⌋ derived from theoretical analysis
- **Research Code**: Prioritizes algorithmic clarity over production optimizations

For practical shortest path computation, use standard implementations of Dijkstra's algorithm or other established methods.

---

## Repository Structure

Each language implementation includes:

```
<language>/
├── bmssp.<ext>          # Main algorithm implementation
├── tests/               # Correctness verification
├── README.md           # Language-specific instructions
└── ...                 # Build files (Makefile, package files, etc.)
```

---

## Running the Implementations

### Python

```bash
cd python
python bmssp.py
```

### Go

```bash
cd go
go run bmssp.go
```

### Fortran

```bash
cd fortran
gfortran bmssp.f90 -o bmssp
./bmssp
```

### C

```bash
cd c
gcc bmssp.c -lm -o bmssp
./bmssp
```

### Rust

```bash
cd rust
cargo run
```

### Java

```bash
cd java
javac BMSSP.java
java BMSSP
```

---

## Testing

Each implementation includes test cases that verify the algorithm's behavior on small example graphs. The tests validate:

- Correct implementation of the recursive structure
- Proper handling of bounds and early termination
- Expected algorithmic behavior under the BMSSP framework

Run tests using the respective language's testing framework (pytest, go test, etc.).

---

## Technical Details

### Time Complexity

- **Main Result**: O(m log^(2/3) n) deterministic time
- **Comparison**: Breaks Dijkstra's O(m + n log n) barrier on sparse graphs
- **Model**: Comparison-addition model with real non-negative edge weights

### Key Parameters

- `k = ⌊log^(1/3) n⌋` - Controls pivot selection and base case size
- `t = ⌊log^(2/3) n⌋` - Determines recursion branching factor  
- `l = ⌈log n / t⌉` - Number of recursion levels

### Graph Preprocessing

All implementations assume constant-degree graphs. For general graphs, the algorithm applies a standard vertex-splitting transformation to achieve constant in-degree and out-degree while preserving shortest paths.

---

## Research Context

This work represents a significant theoretical breakthrough in graph algorithms:

- **First** to break the sorting barrier for directed SSSP in the comparison-addition model
- **Deterministic** improvement over previous randomized results
- **Novel techniques** combining Dijkstra and Bellman-Ford through recursive partitioning

The algorithm demonstrates that Dijkstra's approach, while optimal when vertex ordering is required, is not optimal for computing distances alone.

---

## Citation

```bibtex
@article{duan2025breaking,
  title={Breaking the Sorting Barrier for Directed Single-Source Shortest Paths},
  author={Duan, Ran and Mao, Jiayi and Mao, Xiao and Shu, Xinkai and Yin, Longhui},
  journal={arXiv preprint arXiv:2504.17033},
  year={2025}
}
```

---

## Contributing

When contributing to this repository:

1. **Maintain algorithmic fidelity** - Preserve the theoretical structure of BMSSP
2. **Follow paper notation** - Use variable names and structure consistent with the research paper
3. **Test carefully** - Ensure implementations match expected BMSSP behavior, not standard SSSP
4. **Document clearly** - Explain any implementation-specific choices or optimizations

For questions about the algorithm itself, refer to the original research paper.
