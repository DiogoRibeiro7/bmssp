#!/usr/bin/env bash
set -euo pipefail

OUTDIR="paper/figures"
mkdir -p "$OUTDIR"
CSV="$OUTDIR/bench.csv"

# ensure local package is importable
export PYTHONPATH="${PYTHONPATH:-}:src"

# Run a small benchmark and capture metrics
python -m ssspx.bench --trials 3 --sizes 50,100 100,200 --out-csv "$CSV"

# Export a small DAG from the example graph
python -m ssspx.cli --edges docs/examples/small.csv --export-json "$OUTDIR/example.json" --source 0 >/dev/null || true

# Generate runtime plots
python paper/figures/plot_bench.py "$CSV" "$OUTDIR/runtime"

# Package inputs and outputs for release asset

zip -j "$OUTDIR/repro.zip" "$CSV" "$OUTDIR/example.json" "$OUTDIR/runtime.png" "$OUTDIR/runtime.svg"

echo "Artifacts written to $OUTDIR"
