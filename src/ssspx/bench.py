"""Micro-benchmark utilities for the solver.

Run this module as a script to benchmark ``ssspx`` against a reference
implementation across multiple random graphs.

Example:
```bash
python -m ssspx.bench --trials 5 --sizes 1000,5000 2000,10000 --out-csv out.csv
```

Use ``--mem`` to record peak memory usage during solver runs.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from .dijkstra import dijkstra_reference
from .graph import Graph
from .solver import SolverConfig, SolverMetrics, SSSPSolver


@dataclass
class BenchResult:
    """Result of a single benchmarking run."""

    metrics: SolverMetrics
    dijkstra_ms: float
    max_abs_err: float


def _random_graph(n: int, m: int, seed: int) -> Graph:
    """Generate a random graph with ``n`` vertices and ``m`` edges."""
    import random

    rnd = random.Random(seed)
    edges: List[Tuple[int, int, float]] = []
    for _ in range(m):
        u = rnd.randrange(n)
        v = rnd.randrange(n)
        w = rnd.random() * 10.0
        edges.append((u, v, w))
    return Graph.from_edges(n, edges)


def run_once(
    n: int,
    m: int,
    frontier: str,
    use_transform: bool,
    seed: int = 0,
    track_mem: bool = False,
) -> BenchResult:
    """Run the solver once and compare against a Dijkstra reference.

    Args:
        n: Number of vertices.
        m: Number of edges.
        frontier: Frontier implementation name (``"heap"`` or ``"block"``).
        use_transform: Whether to apply the constant-outdegree transform.
        seed: Seed for the random graph generator.

    Returns:
        Timing information and maximum absolute distance error.
    """
    G = _random_graph(n, m, seed)
    s = 0

    cfg = SolverConfig(use_transform=use_transform, frontier=frontier)
    if track_mem:
        import tracemalloc

        tracemalloc.start()
        t0 = time.perf_counter()
        solver = SSSPSolver(G, s, cfg)
        res = solver.solve()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    else:
        t0 = time.perf_counter()
        solver = SSSPSolver(G, s, cfg)
        res = solver.solve()
        peak = None
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    ref = dijkstra_reference(G, [s])
    t3 = time.perf_counter()

    # Compare distance vectors
    max_err = 0.0
    for a, b in zip(res.distances, ref.distances):
        aa = a if a < float("inf") else 1e18
        bb = b if b < float("inf") else 1e18
        max_err = max(max_err, abs(aa - bb))

    wall_ms = (t1 - t0) * 1000.0
    peak_mib = (peak / (1024 * 1024)) if peak is not None else None
    metrics = solver.metrics(wall_ms=wall_ms, peak_mib=peak_mib)
    return BenchResult(
        metrics=metrics,
        dijkstra_ms=(t3 - t2) * 1000.0,
        max_abs_err=max_err,
    )


def main(argv: List[str] | None = None) -> None:
    """Run benchmarking trials and optionally record results.

    Args:
        argv: Optional argument list for testing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=1, help="Number of trials per configuration")
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["10,20", "20,40"],
        help="Size pairs as n,m (e.g. 1000,5000). Defaults to a small demo.",
    )
    parser.add_argument("--seed-base", type=int, default=0, help="Base seed for random graphs")
    parser.add_argument("--out-csv", type=Path, help="Optional path to write per-trial CSV data")
    parser.add_argument(
        "--mem",
        action="store_true",
        help="Profile peak memory usage (MiB) using tracemalloc",
    )
    args = parser.parse_args(argv)

    sizes: List[Tuple[int, int]] = []
    for spec in args.sizes:
        try:
            n_str, m_str = spec.split(",")
            sizes.append((int(n_str), int(m_str)))
        except ValueError:  # pragma: no cover - argparse handles
            parser.error(f"invalid size specification '{spec}'")

    rows: List[List[object]] = []
    aggregates: dict[
        Tuple[int, int, str, bool],
        Tuple[
            List[float],
            List[float],
            List[int],
            List[int],
            List[int],
            List[int],
            List[float],
        ],
    ] = {}

    for n, m in sizes:
        for frontier in ("heap", "block"):
            for use_transform in (False, True):
                s_times: List[float] = []
                d_times: List[float] = []
                e_counts: List[int] = []
                p_counts: List[int] = []
                fp_counts: List[int] = []
                bc_counts: List[int] = []
                mem_counts: List[float] = []
                for trial in range(args.trials):
                    seed = args.seed_base + trial
                    res = run_once(
                        n=n,
                        m=m,
                        frontier=frontier,
                        use_transform=use_transform,
                        seed=seed,
                        track_mem=args.mem,
                    )
                    mtx = res.metrics
                    row = [
                        mtx.n,
                        mtx.m,
                        mtx.frontier,
                        int(mtx.transform),
                        trial,
                        f"{mtx.wall_ms:.6f}",
                        f"{res.dijkstra_ms:.6f}",
                        mtx.counters["edges_relaxed"],
                        mtx.counters["pulls"],
                        mtx.counters["findpivots_rounds"],
                        mtx.counters["basecase_pops"],
                    ]
                    if args.mem:
                        row.append(f"{(mtx.peak_mib or 0.0):.6f}")
                        mem_counts.append(mtx.peak_mib or 0.0)
                    rows.append(row)
                    s_times.append(mtx.wall_ms)
                    d_times.append(res.dijkstra_ms)
                    e_counts.append(mtx.counters["edges_relaxed"])
                    p_counts.append(mtx.counters["pulls"])
                    fp_counts.append(mtx.counters["findpivots_rounds"])
                    bc_counts.append(mtx.counters["basecase_pops"])
                aggregates[(n, m, frontier, use_transform)] = (
                    s_times,
                    d_times,
                    e_counts,
                    p_counts,
                    fp_counts,
                    bc_counts,
                    mem_counts,
                )

    if args.out_csv:
        with args.out_csv.open("w", newline="") as fh:
            writer = csv.writer(fh)
            csv_header = [
                "n",
                "m",
                "frontier",
                "transform",
                "trial",
                "ssspx_ms",
                "dijkstra_ms",
                "edges_relaxed",
                "pulls",
                "findpivots_rounds",
                "basecase_pops",
            ]
            if args.mem:
                csv_header.append("peak_mib")
            writer.writerow(csv_header)
            writer.writerows(rows)

    header = (
        f"{'n':>6} {'m':>7} {'frontier':>8} {'tf':>2}"
        f" {'edges':>10} {'pulls':>10} {'fp_rounds':>10} {'bc_pops':>9}"
        f" {'ssspx_med':>11} {'ssspx_p95':>11}"
        f" {'dijk_med':>11} {'dijk_p95':>11}"
    )
    if args.mem:
        header += f" {'mem_med':>8} {'mem_p95':>8}"
    print(header)
    for (n, m, frontier, use_transform), (
        s_times,
        d_times,
        e_counts,
        p_counts,
        fp_counts,
        bc_counts,
        mem_counts,
    ) in aggregates.items():
        s_med = statistics.median(s_times)
        s_p95 = (
            statistics.quantiles(s_times, n=100, method="inclusive")[94]
            if len(s_times) > 1
            else s_times[0]
        )
        d_med = statistics.median(d_times)
        d_p95 = (
            statistics.quantiles(d_times, n=100, method="inclusive")[94]
            if len(d_times) > 1
            else d_times[0]
        )
        e_med = statistics.median(e_counts)
        p_med = statistics.median(p_counts)
        fp_med = statistics.median(fp_counts)
        bc_med = statistics.median(bc_counts)
        line = (
            f"{n:6d} {m:7d} {frontier:>8} {int(use_transform):2d}"
            f" {int(e_med):10d} {int(p_med):10d} {int(fp_med):10d} {int(bc_med):9d}"
            f" {s_med:11.2f} {s_p95:11.2f} {d_med:11.2f} {d_p95:11.2f}"
        )
        if args.mem and mem_counts is not None:
            mem_med = statistics.median(mem_counts)
            mem_p95 = (
                statistics.quantiles(mem_counts, n=100, method="inclusive")[94]
                if len(mem_counts) > 1
                else mem_counts[0]
            )
            line += f" {mem_med:8.2f} {mem_p95:8.2f}"
        print(line)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
