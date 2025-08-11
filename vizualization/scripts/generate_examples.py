#!/usr/bin/env python3
"""
Generate Standard BMSSP Visualization Examples

This script creates a comprehensive set of standard visualization examples
demonstrating BMSSP behavior on different graph types and sizes.
"""

import os
import sys
import networkx as nx
import random
import math
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from bmssp_visualization import BMSSPVisualizationSuite
except ImportError:
    print(
        "Error: Could not import bmssp_visualization. Make sure it's in the parent directory."
    )
    sys.exit(1)


class ExampleGraphGenerator:
    """Generate various types of example graphs for visualization."""

    @staticmethod
    def create_simple_path(n: int) -> nx.DiGraph:
        """Create a simple directed path graph."""
        G = nx.DiGraph()
        for i in range(n - 1):
            G.add_edge(f"v{i}", f"v{i + 1}", weight=random.uniform(1, 3))
        return G

    @staticmethod
    def create_binary_tree(depth: int) -> nx.DiGraph:
        """Create a directed binary tree."""
        G = nx.DiGraph()

        def add_tree_nodes(node_id, current_depth, max_depth):
            if current_depth >= max_depth:
                return

            left_child = f"{node_id}_L"
            right_child = f"{node_id}_R"

            G.add_edge(node_id, left_child, weight=random.uniform(1, 4))
            G.add_edge(node_id, right_child, weight=random.uniform(1, 4))

            add_tree_nodes(left_child, current_depth + 1, max_depth)
            add_tree_nodes(right_child, current_depth + 1, max_depth)

        add_tree_nodes("root", 0, depth)
        return G

    @staticmethod
    def create_diamond_graph() -> nx.DiGraph:
        """Create a diamond-shaped graph showing multiple paths."""
        G = nx.DiGraph()
        edges = [
            ("s", "a", 1),
            ("s", "b", 4),  # Source to first level
            ("a", "c", 2),
            ("a", "d", 6),  # From a
            ("b", "c", 1),
            ("b", "d", 2),  # From b
            ("c", "t", 3),
            ("d", "t", 1),  # To target
        ]

        for u, v, w in edges:
            G.add_edge(u, v, weight=w)
        return G

    @staticmethod
    def create_layered_graph(layers: int, nodes_per_layer: int) -> nx.DiGraph:
        """Create a layered graph with controlled structure."""
        G = nx.DiGraph()

        # Create layers
        layer_nodes = []
        for layer in range(layers):
            layer_nodes.append([f"L{layer}_N{i}" for i in range(nodes_per_layer)])

        # Add source
        source = "source"
        G.add_node(source)

        # Connect source to first layer
        for node in layer_nodes[0]:
            G.add_edge(source, node, weight=random.uniform(1, 3))

        # Connect between layers
        for layer in range(len(layer_nodes) - 1):
            current_layer = layer_nodes[layer]
            next_layer = layer_nodes[layer + 1]

            for curr_node in current_layer:
                # Connect to 1-3 nodes in next layer
                connections = random.sample(
                    next_layer, min(random.randint(1, 3), len(next_layer))
                )
                for next_node in connections:
                    G.add_edge(curr_node, next_node, weight=random.uniform(1, 5))

        return G

    @staticmethod
    def create_scale_free_small(n: int) -> nx.DiGraph:
        """Create a small scale-free graph."""
        G = nx.scale_free_graph(n, alpha=0.4, beta=0.5, gamma=0.1, seed=42)

        # Convert to directed with weights
        DG = nx.DiGraph()
        for u, v in G.edges():
            DG.add_edge(u, v, weight=random.uniform(1, 6))
            # Add some reverse edges
            if random.random() < 0.3:
                DG.add_edge(v, u, weight=random.uniform(1, 6))

        return DG

    @staticmethod
    def create_grid_graph(width: int, height: int) -> nx.DiGraph:
        """Create a grid graph representing spatial networks."""
        G = nx.grid_2d_graph(width, height, create_using=nx.DiGraph)

        # Convert node labels and add weights
        DG = nx.DiGraph()
        node_mapping = {node: f"({node[0]},{node[1]})" for node in G.nodes()}

        for u, v in G.edges():
            u_new, v_new = node_mapping[u], node_mapping[v]
            DG.add_edge(u_new, v_new, weight=random.uniform(1, 3))

        return DG


def generate_standard_examples():
    """Generate the complete set of standard examples."""

    output_base = Path("examples")
    output_base.mkdir(exist_ok=True)

    examples = [
        # Basic examples for understanding core concepts
        {
            "name": "simple_path",
            "description": "Simple 5-node path - shows basic recursion",
            "graph_func": lambda: ExampleGraphGenerator.create_simple_path(5),
            "source": "v0",
        },
        {
            "name": "diamond_graph",
            "description": "Diamond structure - shows multiple path selection",
            "graph_func": ExampleGraphGenerator.create_diamond_graph,
            "source": "s",
        },
        {
            "name": "binary_tree",
            "description": "Binary tree - shows hierarchical exploration",
            "graph_func": lambda: ExampleGraphGenerator.create_binary_tree(3),
            "source": "root",
        },
        # Medium complexity examples
        {
            "name": "layered_graph",
            "description": "Layered structure - shows frontier evolution",
            "graph_func": lambda: ExampleGraphGenerator.create_layered_graph(4, 3),
            "source": "source",
        },
        {
            "name": "small_scale_free",
            "description": "Scale-free network - shows hub-based pivoting",
            "graph_func": lambda: ExampleGraphGenerator.create_scale_free_small(12),
            "source": 0,
        },
        {
            "name": "grid_3x3",
            "description": "Grid network - shows spatial locality effects",
            "graph_func": lambda: ExampleGraphGenerator.create_grid_graph(3, 3),
            "source": "(0,0)",
        },
        # Parameter demonstration examples
        {
            "name": "parameter_demo_small",
            "description": "Small graph for parameter effect demonstration",
            "graph_func": lambda: ExampleGraphGenerator.create_layered_graph(3, 4),
            "source": "source",
        },
        {
            "name": "parameter_demo_medium",
            "description": "Medium graph showing parameter scaling",
            "graph_func": lambda: ExampleGraphGenerator.create_scale_free_small(16),
            "source": 0,
        },
    ]

    print("🎨 Generating Standard BMSSP Visualization Examples")
    print("=" * 60)

    generated_files = {}

    for example in examples:
        print(f"\n📊 Creating {example['name']}...")
        print(f"   Description: {example['description']}")

        try:
            # Generate graph
            graph = example["graph_func"]()
            source = example["source"]

            # Create output directory
            example_dir = output_base / example["name"]
            example_dir.mkdir(exist_ok=True)

            # Generate visualization suite
            suite = BMSSPVisualizationSuite()
            outputs = suite.create_complete_analysis(graph, source, str(example_dir))

            # Save graph structure info
            graph_info_path = example_dir / "graph_info.txt"
            with open(graph_info_path, "w") as f:
                f.write(f"Graph: {example['name']}\n")
                f.write(f"Description: {example['description']}\n")
                f.write(f"Nodes: {len(graph.nodes())}\n")
                f.write(f"Edges: {len(graph.edges())}\n")
                f.write(f"Source: {source}\n")
                f.write(
                    f"Density: {len(graph.edges()) / (len(graph.nodes()) * (len(graph.nodes()) - 1)):.4f}\n"
                )

                # Algorithm parameters
                n = len(graph.nodes())
                k = max(1, int(math.log(n, 2) ** (1 / 3)))
                t = max(1, int(math.log(n, 2) ** (2 / 3)))
                l = math.ceil(math.log(n, 2) / t)
                f.write(f"\nBMSSP Parameters:\n")
                f.write(f"k (pivot control): {k}\n")
                f.write(f"t (branching factor): {t}\n")
                f.write(f"l (recursion levels): {l}\n")

            outputs["graph_info"] = str(graph_info_path)
            generated_files[example["name"]] = outputs

            print(f"   ✅ Generated {len(outputs)} files in {example_dir}")

        except Exception as e:
            print(f"   ❌ Failed to generate {example['name']}: {str(e)}")
            continue

    # Generate index file
    create_examples_index(output_base, examples, generated_files)

    print(f"\n🎯 Standard examples complete!")
    print(f"📁 Generated {len(generated_files)} example sets in {output_base}/")
    print(f"📄 See {output_base}/README.md for detailed descriptions")

    return generated_files


def create_examples_index(output_dir: Path, examples: list, generated_files: dict):
    """Create an index file describing all examples."""

    index_path = output_dir / "README.md"

    with open(index_path, "w") as f:
        f.write("# BMSSP Visualization Examples\n\n")
        f.write(
            "This directory contains standard examples demonstrating BMSSP algorithm behavior "
        )
        f.write("on various graph types and structures.\n\n")

        f.write("## Quick Start\n\n")
        f.write("Each example directory contains:\n")
        f.write("- `recursion_tree.png` - Complete recursion analysis\n")
        f.write("- `frontier_evolution.png` - Frontier behavior over time\n")
        f.write("- `pivot_selection.png` - Step-by-step pivot selection\n")
        f.write("- `interactive_analysis.html` - Interactive web visualization\n")
        f.write("- `analysis_report.md` - Detailed written analysis\n")
        f.write("- `graph_info.txt` - Graph statistics and parameters\n\n")

        f.write("## Examples Overview\n\n")

        # Basic examples
        f.write("### 🎓 Basic Examples (Educational)\n\n")
        basic_examples = ["simple_path", "diamond_graph", "binary_tree"]

        for example_name in basic_examples:
            if example_name in generated_files:
                example_info = next(ex for ex in examples if ex["name"] == example_name)
                f.write(f"#### {example_name.replace('_', ' ').title()}\n")
                f.write(f"**Location**: `{example_name}/`\n")
                f.write(f"**Description**: {example_info['description']}\n")
                f.write(
                    f"**Best for**: Understanding {example_name.replace('_', ' ')} behavior\n\n"
                )

        # Medium examples
        f.write("### 🔬 Intermediate Examples (Analysis)\n\n")
        intermediate_examples = ["layered_graph", "small_scale_free", "grid_3x3"]

        for example_name in intermediate_examples:
            if example_name in generated_files:
                example_info = next(ex for ex in examples if ex["name"] == example_name)
                f.write(f"#### {example_name.replace('_', ' ').title()}\n")
                f.write(f"**Location**: `{example_name}/`\n")
                f.write(f"**Description**: {example_info['description']}\n")
                f.write(
                    f"**Best for**: Analyzing {example_name.replace('_', ' ')} characteristics\n\n"
                )

        # Parameter examples
        f.write("### ⚙️ Parameter Demonstration Examples\n\n")
        param_examples = ["parameter_demo_small", "parameter_demo_medium"]

        for example_name in param_examples:
            if example_name in generated_files:
                example_info = next(ex for ex in examples if ex["name"] == example_name)
                f.write(f"#### {example_name.replace('_', ' ').title()}\n")
                f.write(f"**Location**: `{example_name}/`\n")
                f.write(f"**Description**: {example_info['description']}\n")
                f.write(f"**Best for**: Understanding parameter effects\n\n")

        f.write("## Usage Recommendations\n\n")
        f.write("### For Learning BMSSP\n")
        f.write("1. Start with `simple_path` - understand basic recursion\n")
        f.write("2. Examine `diamond_graph` - see multiple path handling\n")
        f.write("3. Study `binary_tree` - observe hierarchical exploration\n\n")

        f.write("### For Research Analysis\n")
        f.write("1. Use `layered_graph` - analyze frontier evolution patterns\n")
        f.write("2. Study `small_scale_free` - understand hub-based pivoting\n")
        f.write("3. Examine `grid_3x3` - observe spatial locality effects\n\n")

        f.write("### For Parameter Studies\n")
        f.write("1. Compare `parameter_demo_small` with different k, t, l values\n")
        f.write("2. Scale up to `parameter_demo_medium` for larger effects\n\n")

        f.write("## Viewing Instructions\n\n")
        f.write("### Static Images\n")
        f.write("All `.png` files can be viewed directly in any image viewer.\n\n")

        f.write("### Interactive Analysis\n")
        f.write(
            "Open `interactive_analysis.html` files in a web browser for dynamic exploration.\n\n"
        )

        f.write("### Analysis Reports\n")
        f.write(
            "Read `analysis_report.md` files for detailed algorithmic insights and interpretation.\n\n"
        )

        f.write("## Regenerating Examples\n\n")
        f.write("```bash\n")
        f.write("cd benchmarks/visualization/scripts\n")
        f.write("python generate_examples.py\n")
        f.write("```\n\n")

        f.write("This will regenerate all examples with fresh random seeds.\n")


def main():
    """Main function to generate all standard examples."""

    # Set random seed for reproducible examples
    random.seed(42)

    try:
        generated_files = generate_standard_examples()

        print("\n" + "=" * 60)
        print("📋 Generation Summary:")
        for name, files in generated_files.items():
            print(f"   {name}: {len(files)} files")

        print("\n🎯 Next Steps:")
        print("   1. Open examples/README.md for overview")
        print("   2. Start with examples/simple_path/ for basic understanding")
        print("   3. Open any interactive_analysis.html in a web browser")
        print("   4. Read analysis_report.md files for detailed insights")

    except KeyboardInterrupt:
        print("\n🛑 Generation interrupted by user")
    except Exception as e:
        print(f"\n❌ Generation failed: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
