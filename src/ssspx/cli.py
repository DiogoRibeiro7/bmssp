"""Command-line interface for running the solver."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

from .exceptions import ConfigError, GraphFormatError, InputError, NotSupportedError, SSSPXError
from .export import export_dag_graphml, export_dag_json
from .graph import Graph
from .io import read_graph
from .logger import StdLogger
from .profiling import ProfileSession
from .solver import SolverConfig, SSSPSolver

EXAMPLE_CSV = """# u,v,w
0,1,1.0
1,2,2.0
0,2,4.0
2,3,1.0
"""


def _build_graph_from_file(path: str, fmt: Optional[str]) -> Tuple[Graph, int, int]:
    """Build a :class:`Graph` from an edges file."""
    p = Path(path)
    if not p.exists():
        raise InputError(f"edges file not found: {path}")
    G = read_graph(path, fmt)
    m = sum(len(lst) for lst in G.adj)
    return G, G.n, m


def _build_random_graph(n: int, m: int, seed: int) -> Graph:
    """Generate a random graph for quick experiments."""
    import random

    rnd = random.Random(seed)
    edges: List[Tuple[int, int, float]] = []
    for _ in range(m):
        u = rnd.randrange(n)
        v = rnd.randrange(n)
        w = rnd.random() * 10.0
        edges.append((u, v, w))
    return Graph.from_edges(n, edges)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the ``ssspx`` command-line tool."""
    examples = (
        "Examples:\n"
        "  ssspx --edges graph.csv --source 0\n"
        "  ssspx --random --n 100 --m 500\n"
        "  ssspx --edges graph.csv --export-json dag.json\n"
    )
    p = argparse.ArgumentParser(
        prog="ssspx",
        description="Deterministic SSSP (BMSSP-style) runner",
        epilog=examples,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--verbose", action="store_true", help="Show full tracebacks")
    p.add_argument("--log-json", action="store_true", help="Emit structured log line")
    p.add_argument(
        "--log-level",
        choices=["debug", "info", "warning"],
        default="warning",
        help="Log verbosity",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--edges", type=str, help="Path to edges file")
    src.add_argument("--random", action="store_true", help="Use a random graph")
    src.add_argument(
        "--example",
        action="store_true",
        help="Print a sample edges CSV to stdout and exit",
    )

    p.add_argument(
        "--format",
        choices=["csv", "jsonl", "mtx", "graphml"],
        default=None,
        help="Edge file format (auto-detected from extension)",
    )

    p.add_argument("--n", type=int, default=10, help="Vertices (random mode)")
    p.add_argument("--m", type=int, default=20, help="Edges (random mode)")
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed controlling random graph generation",
    )
    src_group = p.add_mutually_exclusive_group()
    src_group.add_argument("--source", type=int, default=0, help="Source vertex id")
    src_group.add_argument(
        "--sources",
        type=str,
        default=None,
        help="Comma-separated list of source vertex ids",
    )
    p.add_argument("--target", type=int, default=None, help="Target vertex id for path output")

    p.add_argument("--no-transform", action="store_true", help="Disable outdegree transform")
    p.add_argument("--target-outdeg", type=int, default=4, help="Outdegree cap when transforming")
    p.add_argument("--frontier", choices=["block", "heap"], default="block")

    # Profiling + export
    p.add_argument("--profile", action="store_true", help="Enable cProfile")
    p.add_argument("--profile-out", type=str, default=None, help="Dump .prof file to this path")
    p.add_argument("--export-json", type=str, default=None, help="Write shortest-path DAG as JSON")
    p.add_argument(
        "--export-graphml",
        type=str,
        default=None,
        help="Write shortest-path DAG as GraphML",
    )
    p.add_argument(
        "--metrics-out",
        type=str,
        default=None,
        help="Write run metrics to this JSON file",
    )

    args = p.parse_args(argv)

    if args.example:
        sys.stdout.write(EXAMPLE_CSV)
        return 200

    try:
        # Build graph
        if args.random:
            G = _build_random_graph(args.n, args.m, args.seed)
            n, m = args.n, args.m
        else:
            G, n, m = _build_graph_from_file(args.edges, args.format)

        cfg = SolverConfig(
            use_transform=not args.no_transform,
            target_outdeg=args.target_outdeg,
            frontier=args.frontier,
            k_t_auto=True,
        )

        stream = sys.stdout if args.log_json else sys.stderr
        level = "info" if args.log_json and args.log_level == "warning" else args.log_level
        logger = StdLogger(level=level, json_fmt=args.log_json, stream=stream)

        if args.sources is not None:
            try:
                sources = [int(x) for x in args.sources.split(",") if x.strip()]
            except ValueError as exc:  # pragma: no cover - arg parsing
                raise InputError("invalid --sources list") from exc
        else:
            sources = [args.source]

        if args.verbose and not args.log_json:
            sys.stderr.write(
                f"config: n={n} m={m} frontier={args.frontier} "
                f"transform={not args.no_transform} seed={args.seed} "
                f"sources={sources}\n"
            )

        import time

        if args.metrics_out:
            import tracemalloc

            tracemalloc.start()
            t0 = time.perf_counter()
            if args.profile:
                with ProfileSession(dump_path=args.profile_out) as prof:
                    solver = SSSPSolver(G, sources[0], config=cfg, logger=logger, sources=sources)
                    res = solver.solve()
                sys.stderr.write(prof.report().to_text(lines=40))
            else:
                solver = SSSPSolver(G, sources[0], config=cfg, logger=logger, sources=sources)
                res = solver.solve()
            wall_ms = (time.perf_counter() - t0) * 1000.0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mib = peak / (1024 * 1024)
        else:
            t0 = time.perf_counter()
            if args.profile:
                with ProfileSession(dump_path=args.profile_out) as prof:
                    solver = SSSPSolver(G, sources[0], config=cfg, logger=logger, sources=sources)
                    res = solver.solve()
                sys.stderr.write(prof.report().to_text(lines=40))
            else:
                solver = SSSPSolver(G, sources[0], config=cfg, logger=logger, sources=sources)
                res = solver.solve()
            wall_ms = (time.perf_counter() - t0) * 1000.0
            peak_mib = None

        out = {
            "sources": sources,
            "frontier": args.frontier,
            "use_transform": not args.no_transform,
            "distances": res.distances,
        }

        if args.target is not None:
            path = solver.path(args.target)
            out["target"] = args.target
            out["path"] = path

        if args.export_json:
            with open(args.export_json, "w", encoding="utf-8") as fh:
                fh.write(export_dag_json(G, res.distances))
        if args.export_graphml:
            with open(args.export_graphml, "w", encoding="utf-8") as fh:
                fh.write(export_dag_graphml(G, res.distances))

        if args.metrics_out:
            metrics = solver.metrics(wall_ms=wall_ms, peak_mib=peak_mib)
            with open(args.metrics_out, "w", encoding="utf-8") as fh:
                json.dump(asdict(metrics), fh)

        summary = solver.summary()
        if args.log_level in ("info", "debug") or args.log_json:
            logger.info(
                "run",
                n=n,
                m=m,
                frontier=args.frontier,
                transform=not args.no_transform,
                sources=sources,
                **summary,
            )
        if not args.log_json:
            print(json.dumps(out))
        return 200

    except (InputError, ConfigError, GraphFormatError, NotSupportedError) as exc:
        if args.verbose:
            traceback.print_exc()
        else:
            sys.stderr.write(f"error: {exc}\n")
        return 64
    except SSSPXError as exc:
        if args.verbose:
            traceback.print_exc()
        else:
            sys.stderr.write(f"internal error: {exc}\n")
        return 70
    except Exception as exc:  # pragma: no cover - unexpected
        if args.verbose:
            traceback.print_exc()
        else:
            sys.stderr.write(f"internal error: {exc}\n")
        return 70


if __name__ == "__main__":
    sys.exit(main())
