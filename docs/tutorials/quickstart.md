# Quickstart

## Python
```python
>>> from ssspx.io import read_graph
>>> from ssspx import SSSPSolver, SolverConfig
>>> G = read_graph("docs/examples/small.csv")
>>> solver = SSSPSolver(G, 0, config=SolverConfig())
>>> solver.solve().distances
[0.0, 1.0, 3.0, 4.0]
```

## CLI
```bash
ssspx --edges docs/examples/small.csv --source 0

# formats: .csv, .jsonl, .mtx, .graphml (auto-detected)
ssspx --edges path/to/graph.mtx --source 0

# Structured logging with counters
ssspx --random --n 10 --m 20 --log-json

# Reproducible random graph
ssspx --random --n 10 --m 20 --seed 123
```
