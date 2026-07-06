# ssspx — Production-grade Single-Source Shortest Paths (Python)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17381904.svg)](https://doi.org/10.5281/zenodo.17381904)

<!-- GitHub stars/forks -->

[![GitHub stars](https://img.shields.io/github/stars/diogoribeiro7/bmssp.svg?style=social)](https://github.com/DiogoRibeiro7/bmssp/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/diogoribeiro7/bmssp.svg?style=social)](https://github.com/DiogoRibeiro7/bmssp/network/members)

`ssspx` is a clean, typed, and tested implementation of a deterministic **Single-Source Shortest Paths** solver for directed graphs with non-negative weights.
It follows a BMSSP-style divide-and-conquer design (levels, `FindPivots`, bounded base case) and includes a switchable frontier, an optional constant-outdegree transform, a CLI, exports, and a small benchmark harness.

> Research context: this is an implementation of the algorithmic approach from
> **“Breaking the Sorting Barrier for Directed Single-Source Shortest Paths” (arXiv:2504.17033)** by Duan, Mao, Mao, Shu, and Yin.
> Distances are exact. For the advertised asymptotics, use the **block frontier** with the **constant-outdegree transform** enabled.

---

## Features

* Two frontiers: `block` (paper-style) and `heap` (baseline)
* Optional constant-outdegree transform (cap configurable, default 4)
* Path reconstruction in original vertex IDs (works with or without transform)
* Shortest-path DAG export to **JSON** and **GraphML**
* CLI with CSV/TSV/JSONL/MTX/GraphML input, random graphs, and optional cProfile
* Benchmark harness vs. Dijkstra
* Typed (mypy), linted (flake8/black/isort), tested (pytest + Hypothesis)

## Limitations

* Edge weights must be **non-negative**. Supplying a negative weight raises a
  `GraphFormatError` that points to the exact offending edge.
* The BMSSP solver includes **safety limits** (caps on iterations, recursion
  depth, and frontier size) to prevent runaway behavior on large graphs. If a
  cap is hit, the solver may terminate early and leave some reachable vertices
  at `inf` while `dijkstra_reference` continues to find them. You can detect
  this via `SSSPSolver.summary()` and checking whether
  `iterations_protected > 0`. These safety limits are not currently exposed as
  public configuration knobs.

## Large graphs

If you see unexpectedly many `inf` distances on large networks, the solver may
have hit its safety limits. To diagnose:
* Call `SSSPSolver.summary()` and check if `iterations_protected > 0`.

Workarounds (until limits are configurable):
* For guaranteed reachability parity, use `dijkstra_reference`.
* Try `frontier="heap"` with `use_transform=False`, which can reduce the chance
  of hitting safety caps at the cost of performance.

## Documentation

Full guides and the API reference live at the
[project documentation site](https://DiogoRibeiro7.github.io/ssspx/).
Algorithmic details are covered in the
[BMSSP design notes](docs/design/ssspx.md), and key trade-offs are tracked in
[Architecture Decision Records](docs/decisions/).

## Stability policy

Public modules and classes are listed in our
[API stability policy](docs/policy/api.md). Deprecated names emit warnings and
are removed after the stated `remove_in` release.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and
testing instructions.

---

## Installation

```bash
# in your repo
poetry install
pre-commit install  # optional but recommended
```

Python ≥ 3.9. NumPy is optional (for a CSR backend).

Install with NumPy extras:

```bash
poetry install -E numpy-backend
```

---

## Quick start (Python)

```python
>>> from ssspx import Graph, SSSPSolver, SolverConfig
>>> G = Graph.from_edges(4, [(0, 1, 1.0), (1, 2, 2.0), (0, 2, 4.0), (2, 3, 1.0)])
>>> solver = SSSPSolver(G, 0, config=SolverConfig())
>>> solver.solve().distances  # doctest: +NORMALIZE_WHITESPACE
[0.0, 1.0, 3.0, 4.0]
```

---

## CLI

```bash
# Edges from file (auto-detected: CSV, JSONL, MTX, GraphML)
poetry run ssspx --edges docs/examples/small.csv --source 0 \
  --frontier block --export-graphml out.graphml --export-json out.json

# Override format if needed
poetry run ssspx --edges docs/examples/small --format csv --source 0

# Random graph
poetry run ssspx --random --n 1000 --m 5000 --source 0 \
  --frontier heap --no-transform

# Reproducible random graph
poetry run ssspx --random --n 10 --m 20 --seed 123

# Multiple sources + a path to a target
poetry run ssspx --edges docs/examples/small.csv --sources 0,2 --target 3

# Emit structured log line and write metrics JSON
poetry run ssspx --edges docs/examples/small.csv --log-json --metrics-out metrics.json

# Print a sample CSV to stdout
poetry run ssspx --example

# Profiling to file + brief text report on stderr
poetry run ssspx --random --n 5000 --m 20000 --source 0 \
  --frontier block --profile --profile-out prof.cprof 2> prof.txt

# Structured log line with counters
poetry run ssspx --random --n 10 --m 20 --log-json
```

CLI notes:
* Results are emitted as JSON to stdout unless `--log-json` is set.
* With `--log-json`, the logger writes a structured line to stdout and the
  distances payload is suppressed.
* `--format csv` accepts comma- or tab-separated files (so `.tsv` works).
* `--format jsonl` accepts `.json` and `.jsonl` line-delimited inputs.

CLI reference (grouped):

Input:
| Flag | Default | Description |
| --- | --- | --- |
| `--edges PATH` | required (unless `--random`/`--example`) | Input edges file. |
| `--random` | false | Generate a random graph. |
| `--example` | false | Print a sample CSV and exit. |
| `--format {csv,jsonl,mtx,graphml}` | auto | Input format override. |
| `--n` | 10 | Vertex count for random graphs. |
| `--m` | 20 | Edge count for random graphs. |
| `--seed` | 0 | Random graph seed. |

Solver:
| Flag | Default | Description |
| --- | --- | --- |
| `--source ID` | 0 | Single source vertex id. |
| `--sources LIST` | none | Comma-separated source ids. |
| `--target ID` | none | Target id for path output. |
| `--frontier {block,heap}` | `block` | Frontier implementation. |
| `--no-transform` | false | Disable constant-outdegree transform. |
| `--target-outdeg N` | 4 | Outdegree cap when transforming. |

Output:
| Flag | Default | Description |
| --- | --- | --- |
| `--export-json PATH` | none | Write DAG as JSON. |
| `--export-graphml PATH` | none | Write DAG as GraphML. |
| `--metrics-out PATH` | none | Write run metrics JSON. |
| `--profile` | false | Enable cProfile. |
| `--profile-out PATH` | none | Write `.prof` file. |

Logging:
| Flag | Default | Description |
| --- | --- | --- |
| `--log-json` | false | Emit structured log line. |
| `--log-level {debug,info,warning}` | `warning` | Log verbosity. |
| `--verbose` | false | Show full tracebacks. |

---

## Path reconstruction

After `solve()`, extract a path in **original** vertex IDs:

```python
>>> from ssspx import Graph, SSSPSolver
>>> G = Graph.from_edges(3, [(0,1,1.0), (1,2,2.0)])
>>> solver = SSSPSolver(G, 0)
>>> _ = solver.solve()
>>> solver.path(2)
[0, 1, 2]
```

With the transform enabled, clone cycles are compressed away. Empty list means unreachable.

---

## Exports

```python
from ssspx.export import export_dag_json, export_dag_graphml

json_str = export_dag_json(G, res.distances)
gml_str  = export_dag_graphml(G, res.distances)

with open("sp_dag.json","w") as f: f.write(json_str)
with open("sp_dag.graphml","w") as f: f.write(gml_str)
```

The DAG contains edges `(u → v)` where `d[v] == d[u] + w(u,v)`.

---

## NumPy CSR backend (optional)

```python
from ssspx.graph_numpy import NumpyGraph

edges = [(0, 1, 1.0), (1, 2, 2.0), (0, 2, 4.0)]
G_np = NumpyGraph.from_edges(3, edges)
res = SSSPSolver(G_np, 0).solve()
```

`NumpyGraph` is drop-in because the solver only needs `G.n` and iteration over `G.adj[u] -> (v, w)`.

---

## Benchmarks

```bash
poetry run python -m ssspx.bench
```

This uses a small default graph set; provide `--sizes` to benchmark custom
graphs. Treat results as sanity checks. For serious evaluations, pin seeds and
sizes and run multiple trials.

---

## Testing & quality

```bash
# Unit + property tests
poetry run pytest -q --maxfail=1 --disable-warnings --cov=ssspx --cov-report=term-missing

# Static checks
poetry run mypy src/ssspx
poetry run flake8 src/ssspx
poetry run black --check src tests
poetry run isort --check-only src tests

# All hooks locally
pre-commit run --all-files
```

---

## Security

Continuous integration runs [Bandit](https://bandit.readthedocs.io/) on `src/` and
[pip-audit](https://pypi.org/project/pip-audit/) on the installed dependencies. The build
fails if Bandit reports any **HIGH** severity issues; medium and low findings are logged for
manual review. To reproduce locally:

```bash
poetry run bandit -r src/ssspx -c bandit.yaml --severity-level high
poetry run pip-audit
```

Resolve flagged problems by updating dependencies or refactoring code before committing.
If pip-audit reports an unfixed issue, temporarily suppress it with `--ignore-vuln <ID>` and
track the upstream resolution.

---

## Versioning, releases & archiving

Configured for **semantic-release** with Conventional Commits (Angular style). On pushes to `main`, it bumps the version, updates `CHANGELOG.md`, and creates a GitHub Release.

Releases are archived on **Zenodo**. The archive DOI is **[10.5281/zenodo.17381904](https://doi.org/10.5281/zenodo.17381904)** (badge at the top).

Common prefixes: `feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`, `chore:`.

---

## Cite this work

If you use `ssspx` in academic work, please cite:

**The algorithmic paper**

> Duan, R., Mao, J., Mao, X., Shu, X., & Yin, L. (2025). *Breaking the Sorting Barrier for Directed Single-Source Shortest Paths.* arXiv:2504.17033.

**This software**

```bibtex
@software{ribeiro_ssspx_2025,
  author    = {Ribeiro, Diogo},
  title     = {ssspx — Production-grade Single-Source Shortest Paths (Python)},
  year      = {2025},
  doi       = {10.5281/zenodo.17381904},
  url       = {https://github.com/DiogoRibeiro7/bmssp},
  publisher = {Zenodo}
}
```

---

## License

MIT — see `LICENSE`.
