#!/usr/bin/env python3
"""Plot benchmark runtimes from a CSV file.

Usage:
    python plot_bench.py bench.csv output_prefix

The script reads the benchmark CSV produced by ``ssspx.bench`` and writes
``output_prefix`` with ``.png`` and ``.svg`` extensions.
"""
from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def load(csv_path: Path):
    data = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["n"]), int(row["m"]))
            data.setdefault(key, {"ssspx_ms": [], "dijkstra_ms": []})
            data[key]["ssspx_ms"].append(float(row["ssspx_ms"]))
            data[key]["dijkstra_ms"].append(float(row["dijkstra_ms"]))
    return data


def plot(data: dict[tuple[int, int], dict[str, list[float]]], out_prefix: Path) -> None:
    keys = sorted(data, key=lambda t: t[1])
    edges = [m for (_, m) in keys]
    ssspx = [statistics.median(data[k]["ssspx_ms"]) for k in keys]
    dijkstra = [statistics.median(data[k]["dijkstra_ms"]) for k in keys]

    plt.figure()
    plt.plot(edges, ssspx, marker="o", label="ssspx")
    plt.plot(edges, dijkstra, marker="o", label="dijkstra")
    plt.xlabel("edges (m)")
    plt.ylabel("runtime (ms)")
    plt.legend()
    plt.tight_layout()

    png = out_prefix.with_suffix(".png")
    svg = out_prefix.with_suffix(".svg")
    plt.savefig(png)
    plt.savefig(svg)
    print(f"wrote {png} and {svg}")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 1
    csv_path = Path(argv[1])
    out_prefix = Path(argv[2])
    data = load(csv_path)
    plot(data, out_prefix)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
