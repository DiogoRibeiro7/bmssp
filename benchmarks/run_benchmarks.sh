#!/bin/bash
echo "🚀 Running BMSSP Performance Benchmarks"
cd "$(dirname "$0")"

# Check dependencies
python -c "import numpy, matplotlib, networkx, psutil" || {
    echo "Installing dependencies..."
    pip install -r requirements.txt
}

# Run analysis
python performance_analysis.py

echo "✅ Benchmarks complete. Check results/ directory."
