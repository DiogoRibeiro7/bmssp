# Benchmarking

The `ssspx.bench` module provides micro-benchmarks comparing the solver
against a plain Dijkstra implementation. It is optional and not executed in
CI but can help track performance locally.

Run benchmarks across multiple graph sizes and trials and write per-trial
results to a CSV file. If no sizes are provided, a small demo ``10,20`` and
``20,40`` is used by default:

```bash
python -m ssspx.bench --trials 5 --sizes 1000,5000 2000,10000 --out-csv out.csv
```

Each size pair `n,m` is tested for both frontier implementations and with and
without the outdegree transform. The command prints a summary table with
median and 95th percentile timings along with internal solver counters
(`edges`, `pulls`, `fp_rounds`, `bc_pops`). Individual measurements including
these counters and wall-clock timings are written to the specified CSV file.
Use `--seed-base` to control random graph generation.

Pass `--mem` to enable a lightweight tracemalloc profiler that records peak
memory usage in MiB. When enabled, a `peak_mib` column is added to the CSV and
median and 95th percentile memory statistics are printed in the summary table.
