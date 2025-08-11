#!/usr/bin/env python3
"""
BMSSP Educational Demo Generator

This script creates an educational sequence of visualizations designed for
teaching the BMSSP algorithm, from basic concepts to advanced analysis.
"""

import os
import sys
import networkx as nx
import random
import math
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from bmssp_visualization import BMSSPVisualizationSuite
    from compare_algorithms import AlgorithmComparator
except ImportError:
    print(
        "Error: Could not import required modules. Make sure they're in the parent directory."
    )
    sys.exit(1)


@dataclass
class LessonPlan:
    """Structure for organizing educational content."""

    lesson_id: str
    title: str
    description: str
    learning_objectives: List[str]
    graph_description: str
    key_concepts: List[str]
    follow_up_questions: List[str]


class EducationalGraphs:
    """Collection of graphs designed for educational purposes."""

    @staticmethod
    def lesson_1_linear_path() -> nx.DiGraph:
        """Simple linear path - introduces basic recursion."""
        G = nx.DiGraph()
        edges = [("A", "B", 1), ("B", "C", 2), ("C", "D", 1), ("D", "E", 3)]
        for u, v, w in edges:
            G.add_edge(u, v, weight=w)
        return G

    @staticmethod
    def lesson_2_branching_paths() -> nx.DiGraph:
        """Y-shaped graph - shows pivot selection."""
        G = nx.DiGraph()
        edges = [
            ("S", "A", 1),
            ("S", "B", 3),  # Source branches
            ("A", "C", 2),
            ("B", "C", 1),  # Paths converge
            ("C", "D", 1),  # Continue
        ]
        for u, v, w in edges:
            G.add_edge(u, v, weight=w)
        return G

    @staticmethod
    def lesson_3_multiple_paths() -> nx.DiGraph:
        """Diamond pattern - demonstrates path selection."""
        G = nx.DiGraph()
        edges = [
            ("S", "A", 1),
            ("S", "B", 4),  # Two initial paths
            ("A", "C", 2),
            ("A", "D", 6),  # From A
            ("B", "C", 1),
            ("B", "D", 2),  # From B
            ("C", "T", 3),
            ("D", "T", 1),  # To target
        ]
        for u, v, w in edges:
            G.add_edge(u, v, weight=w)
        return G

    @staticmethod
    def lesson_4_tree_structure() -> nx.DiGraph:
        """Binary tree - shows hierarchical exploration."""
        G = nx.DiGraph()
        edges = [
            ("Root", "L1", 1),
            ("Root", "R1", 2),  # Level 1
            ("L1", "L2", 1),
            ("L1", "R2", 3),  # Left subtree
            ("R1", "L3", 2),
            ("R1", "R3", 1),  # Right subtree
            ("L2", "L4", 2),
            ("R2", "R4", 1),  # Leaves
            ("L3", "L5", 1),
            ("R3", "R5", 2),  # More leaves
        ]
        for u, v, w in edges:
            G.add_edge(u, v, weight=w)
        return G

    @staticmethod
    def lesson_5_complex_network() -> nx.DiGraph:
        """More complex network - combines concepts."""
        G = nx.DiGraph()
        edges = [
            # Central hub structure
            ("S", "Hub1", 1),
            ("S", "Hub2", 3),
            ("S", "Hub3", 5),
            # From Hub1
            ("Hub1", "A1", 2),
            ("Hub1", "A2", 1),
            ("Hub1", "Hub2", 1),
            # From Hub2
            ("Hub2", "B1", 1),
            ("Hub2", "B2", 3),
            ("Hub2", "Hub3", 2),
            # From Hub3
            ("Hub3", "C1", 1),
            ("Hub3", "C2", 2),
            # Cross connections
            ("A1", "B1", 2),
            ("A2", "C1", 4),
            ("B2", "C2", 1),
            # Terminal connections
            ("B1", "Target", 2),
            ("C1", "Target", 1),
            ("C2", "Target", 3),
        ]
        for u, v, w in edges:
            G.add_edge(u, v, weight=w)
        return G

    @staticmethod
    def lesson_6_parameter_demo() -> nx.DiGraph:
        """Medium graph for parameter effects."""
        G = nx.DiGraph()

        # Create layered structure with cross-connections
        layers = [
            ["S"],
            ["A1", "A2", "A3"],
            ["B1", "B2", "B3", "B4"],
            ["C1", "C2"],
            ["T"],
        ]

        # Add all nodes
        for layer in layers:
            for node in layer:
                G.add_node(node)

        # Connect layers with varying weights
        for i in range(len(layers) - 1):
            current_layer = layers[i]
            next_layer = layers[i + 1]

            for curr_node in current_layer:
                for next_node in next_layer:
                    # Add some connections with random weights
                    if random.random() < 0.6:  # 60% connection probability
                        weight = random.randint(1, 5)
                        G.add_edge(curr_node, next_node, weight=weight)

        # Add some cross-layer connections for complexity
        cross_connections = [
            ("A1", "C1", 4),
            ("A3", "B1", 2),
            ("B2", "C2", 1),
            ("B3", "T", 6),
            ("A2", "B4", 3),
        ]

        for u, v, w in cross_connections:
            if G.has_node(u) and G.has_node(v):
                G.add_edge(u, v, weight=w)

        return G


class EducationalSequence:
    """Manages the educational lesson sequence."""

    def __init__(self):
        self.lessons = self._create_lesson_plans()

    def _create_lesson_plans(self) -> List[LessonPlan]:
        """Define the complete educational sequence."""
        return [
            LessonPlan(
                lesson_id="lesson_1",
                title="Introduction to BMSSP: Linear Exploration",
                description="Understand how BMSSP explores a simple linear path",
                learning_objectives=[
                    "Understand basic BMSSP recursion structure",
                    "See how bounds limit exploration",
                    "Compare with intuitive shortest path",
                ],
                graph_description="Simple path A→B→C→D→E with varying edge weights",
                key_concepts=[
                    "Recursion levels (l parameter)",
                    "Base case behavior",
                    "Bound-limited exploration",
                ],
                follow_up_questions=[
                    "Why might BMSSP not reach all vertices?",
                    "How do the bounds affect reachability?",
                    "What role do the k, t, l parameters play?",
                ],
            ),
            LessonPlan(
                lesson_id="lesson_2",
                title="Pivot Selection: Branching Paths",
                description="Learn how BMSSP selects pivots when paths branch",
                learning_objectives=[
                    "Understand FindPivots algorithm",
                    "See pivot vs witness classification",
                    "Observe frontier evolution",
                ],
                graph_description="Y-shaped graph where paths from source branch and reconverge",
                key_concepts=[
                    "FindPivots algorithm",
                    "Pivot vs witness vertices",
                    "Frontier reduction strategy",
                    "k-step Bellman-Ford expansion",
                ],
                follow_up_questions=[
                    "Why are some vertices chosen as pivots?",
                    "How does the k parameter affect pivot selection?",
                    "What happens to vertices classified as witnesses?",
                ],
            ),
            LessonPlan(
                lesson_id="lesson_3",
                title="Multiple Paths: Diamond Structure",
                description="Analyze BMSSP behavior with multiple competing paths",
                learning_objectives=[
                    "See path competition resolution",
                    "Understand distance-based exploration",
                    "Compare optimal vs BMSSP paths",
                ],
                graph_description="Diamond-shaped graph with multiple paths between source and target",
                key_concepts=[
                    "Path competition",
                    "Distance-based bounds",
                    "Recursive subproblem division",
                    "Optimal vs bounded exploration",
                ],
                follow_up_questions=[
                    "Does BMSSP find the optimal path?",
                    "How do bounds affect path discovery?",
                    "When might BMSSP miss shorter paths?",
                ],
            ),
            LessonPlan(
                lesson_id="lesson_4",
                title="Hierarchical Exploration: Tree Structures",
                description="Examine BMSSP on tree-like hierarchical structures",
                learning_objectives=[
                    "Understand recursive tree exploration",
                    "See subtree-based pivot selection",
                    "Analyze depth vs breadth trade-offs",
                ],
                graph_description="Binary tree structure showing hierarchical relationships",
                key_concepts=[
                    "Tree-based exploration",
                    "Subtree size requirements",
                    "Recursive divide-and-conquer",
                    "Depth-first vs breadth-first tendencies",
                ],
                follow_up_questions=[
                    "How does tree structure affect algorithm behavior?",
                    "Why might some subtrees be unexplored?",
                    "How do tree depth and branching factor interact?",
                ],
            ),
            LessonPlan(
                lesson_id="lesson_5",
                title="Complex Networks: Hub-Based Structures",
                description="Apply BMSSP to realistic network structures with hubs",
                learning_objectives=[
                    "See hub vertex identification",
                    "Understand scale-free network effects",
                    "Analyze practical algorithm behavior",
                ],
                graph_description="Complex network with hub vertices and varied connectivity",
                key_concepts=[
                    "Hub vertex effects",
                    "Scale-free network properties",
                    "Practical performance characteristics",
                    "Real-world network applications",
                ],
                follow_up_questions=[
                    "How does hub structure affect pivot selection?",
                    "What advantages does BMSSP have on scale-free networks?",
                    "How do network properties influence algorithm efficiency?",
                ],
            ),
            LessonPlan(
                lesson_id="lesson_6",
                title="Parameter Effects: Tuning k, t, l",
                description="Explore how different parameter values affect algorithm behavior",
                learning_objectives=[
                    "Understand parameter interdependencies",
                    "See performance vs completeness trade-offs",
                    "Learn parameter tuning strategies",
                ],
                graph_description="Medium-sized graph suitable for parameter experimentation",
                key_concepts=[
                    "Parameter sensitivity",
                    "k-t-l relationships",
                    "Performance vs completeness trade-offs",
                    "Practical parameter tuning",
                ],
                follow_up_questions=[
                    "How do different k values affect results?",
                    "What's the relationship between t and recursion depth?",
                    "How should parameters be chosen for different graph types?",
                ],
            ),
        ]

    def generate_complete_sequence(self, output_dir: str = "educational_sequence"):
        """Generate the complete educational sequence."""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print("🎓 Generating BMSSP Educational Sequence")
        print("=" * 50)

        # Graph creation functions
        graph_creators = {
            "lesson_1": EducationalGraphs.lesson_1_linear_path,
            "lesson_2": EducationalGraphs.lesson_2_branching_paths,
            "lesson_3": EducationalGraphs.lesson_3_multiple_paths,
            "lesson_4": EducationalGraphs.lesson_4_tree_structure,
            "lesson_5": EducationalGraphs.lesson_5_complex_network,
            "lesson_6": EducationalGraphs.lesson_6_parameter_demo,
        }

        sources = {
            "lesson_1": "A",
            "lesson_2": "S",
            "lesson_3": "S",
            "lesson_4": "Root",
            "lesson_5": "S",
            "lesson_6": "S",
        }

        generated_lessons = {}

        for lesson in self.lessons:
            print(f"\n📚 Creating {lesson.title}...")

            try:
                # Create lesson directory
                lesson_dir = output_path / lesson.lesson_id
                lesson_dir.mkdir(exist_ok=True)

                # Generate graph
                graph = graph_creators[lesson.lesson_id]()
                source = sources[lesson.lesson_id]

                # Create visualizations
                suite = BMSSPVisualizationSuite()
                outputs = suite.create_complete_analysis(graph, source, str(lesson_dir))

                # Generate lesson guide
                guide_path = lesson_dir / "lesson_guide.md"
                self._create_lesson_guide(lesson, graph, source, outputs, guide_path)

                # Create comparison with Dijkstra for advanced lessons
                if lesson.lesson_id in ["lesson_3", "lesson_5", "lesson_6"]:
                    print(f"   🔬 Adding algorithm comparison...")
                    comparator = AlgorithmComparator()
                    bmssp_result, dijkstra_result = comparator.run_comparison(
                        graph, source
                    )

                    comparison_dir = lesson_dir / "comparison"
                    comparison_outputs = comparator.create_comparison_visualization(
                        graph,
                        source,
                        bmssp_result,
                        dijkstra_result,
                        str(comparison_dir),
                    )
                    outputs.update(
                        {f"comparison_{k}": v for k, v in comparison_outputs.items()}
                    )

                # Parameter exploration for lesson 6
                if lesson.lesson_id == "lesson_6":
                    print(f"   ⚙️ Adding parameter exploration...")
                    param_outputs = self._create_parameter_exploration(
                        graph, source, lesson_dir
                    )
                    outputs.update(param_outputs)

                generated_lessons[lesson.lesson_id] = {
                    "lesson": lesson,
                    "outputs": outputs,
                    "graph_info": {
                        "nodes": len(graph.nodes()),
                        "edges": len(graph.edges()),
                        "source": source,
                    },
                }

                print(f"   ✅ Generated {len(outputs)} files")

            except Exception as e:
                print(f"   ❌ Failed to create {lesson.lesson_id}: {str(e)}")
                continue

        # Create master index
        self._create_master_index(output_path, generated_lessons)

        # Create instructor guide
        self._create_instructor_guide(output_path, generated_lessons)

        print(f"\n🎯 Educational sequence complete!")
        print(f"📁 {len(generated_lessons)} lessons created in {output_path}/")
        print(f"📖 Start with {output_path}/README.md")

        return generated_lessons

    def _create_lesson_guide(
        self,
        lesson: LessonPlan,
        graph: nx.DiGraph,
        source: str,
        outputs: Dict,
        guide_path: Path,
    ):
        """Create detailed lesson guide."""

        with open(guide_path, "w") as f:
            f.write(f"# {lesson.title}\n\n")
            f.write(f"{lesson.description}\n\n")

            # Learning objectives
            f.write("## 🎯 Learning Objectives\n\n")
            for i, objective in enumerate(lesson.learning_objectives, 1):
                f.write(f"{i}. {objective}\n")
            f.write("\n")

            # Graph description
            f.write("## 📊 Graph Structure\n\n")
            f.write(f"{lesson.graph_description}\n\n")
            f.write(f"- **Vertices**: {len(graph.nodes())}\n")
            f.write(f"- **Edges**: {len(graph.edges())}\n")
            f.write(f"- **Source**: {source}\n")
            f.write(
                f"- **Density**: {len(graph.edges()) / (len(graph.nodes()) * (len(graph.nodes()) - 1)):.3f}\n\n"
            )

            # Key concepts
            f.write("## 🔑 Key Concepts\n\n")
            for concept in lesson.key_concepts:
                f.write(f"- **{concept}**\n")
            f.write("\n")

            # Algorithm parameters
            n = len(graph.nodes())
            k = max(1, int(math.log(n, 2) ** (1 / 3)))
            t = max(1, int(math.log(n, 2) ** (2 / 3)))
            l = math.ceil(math.log(n, 2) / t)

            f.write("## ⚙️ Algorithm Parameters\n\n")
            f.write(f"For this graph (n={n}):\n")
            f.write(f"- **k** = {k} (pivot selection parameter)\n")
            f.write(f"- **t** = {t} (branching factor)\n")
            f.write(f"- **l** = {l} (recursion levels)\n\n")

            # Viewing instructions
            f.write("## 👀 How to Explore\n\n")
            f.write("### 1. Start with the Graph State\n")
            f.write(
                "Open `pivot_selection.png` to see step-by-step algorithm execution.\n\n"
            )

            f.write("### 2. Understand the Recursion\n")
            f.write(
                "Examine `recursion_tree.png` to see how the algorithm breaks down the problem.\n\n"
            )

            f.write("### 3. Track Progress\n")
            f.write(
                "Study `frontier_evolution.png` to understand how the frontier changes over time.\n\n"
            )

            f.write("### 4. Interactive Exploration\n")
            f.write(
                "Open `interactive_analysis.html` in a web browser for dynamic exploration.\n\n"
            )

            f.write("### 5. Read the Analysis\n")
            f.write(
                "Review `analysis_report.md` for detailed algorithmic insights.\n\n"
            )

            # Discussion questions
            f.write("## 💭 Discussion Questions\n\n")
            for i, question in enumerate(lesson.follow_up_questions, 1):
                f.write(f"{i}. {question}\n")
            f.write("\n")

            # Expected observations
            f.write("## 🔍 What to Look For\n\n")

            if lesson.lesson_id == "lesson_1":
                f.write("- Simple linear recursion structure\n")
                f.write("- Base case handling at the leaves\n")
                f.write("- Bound effects on reachability\n")
            elif lesson.lesson_id == "lesson_2":
                f.write("- Pivot selection at the branching point\n")
                f.write("- Witness vertex classification\n")
                f.write("- Frontier reduction after FindPivots\n")
            elif lesson.lesson_id == "lesson_3":
                f.write("- Multiple path competition\n")
                f.write("- Distance-based exploration priorities\n")
                f.write("- Potential suboptimal path selection\n")
            elif lesson.lesson_id == "lesson_4":
                f.write("- Hierarchical tree exploration\n")
                f.write("- Subtree-based pivot selection\n")
                f.write("- Depth vs breadth exploration patterns\n")
            elif lesson.lesson_id == "lesson_5":
                f.write("- Hub vertex identification as pivots\n")
                f.write("- Network structure effects on algorithm\n")
                f.write("- Realistic performance characteristics\n")
            elif lesson.lesson_id == "lesson_6":
                f.write("- Parameter sensitivity effects\n")
                f.write("- Performance vs completeness trade-offs\n")
                f.write("- Optimal parameter selection strategies\n")

            f.write("\n## 📚 Additional Resources\n\n")
            f.write(
                "- Main paper: [Breaking the Sorting Barrier for Directed Single-Source Shortest Paths](https://arxiv.org/abs/2504.17033)\n"
            )
            f.write("- Algorithm walkthrough: `../ALGORITHM_WALKTHROUGH.md`\n")
            f.write("- Performance analysis: `../benchmarks/README.md`\n")

    def _create_parameter_exploration(
        self, graph: nx.DiGraph, source: str, lesson_dir: Path
    ) -> Dict[str, str]:
        """Create parameter exploration for lesson 6."""

        param_dir = lesson_dir / "parameter_exploration"
        param_dir.mkdir(exist_ok=True)

        n = len(graph.nodes())
        base_k = max(1, int(math.log(n, 2) ** (1 / 3)))
        base_t = max(1, int(math.log(n, 2) ** (2 / 3)))

        # Test different parameter combinations
        param_combinations = [
            (1, base_t, 3, "Low k - minimal pivot selection"),
            (base_k, base_t, None, "Standard parameters"),
            (base_k + 2, base_t, None, "High k - aggressive pivot selection"),
            (base_k, 1, None, "Low t - deep recursion"),
            (base_k, base_t * 2, None, "High t - shallow recursion"),
        ]

        results = []
        output_files = {}

        for i, (k, t, l_override, description) in enumerate(param_combinations):
            l = l_override if l_override else math.ceil(math.log(n, 2) / t)

            print(f"     Testing parameters k={k}, t={t}, l={l}")

            # Create modified visualizer (this would need implementation)
            # For now, create a placeholder analysis
            param_case_dir = param_dir / f"case_{i + 1}_k{k}_t{t}_l{l}"
            param_case_dir.mkdir(exist_ok=True)

            # Save parameter info
            info_path = param_case_dir / "parameters.txt"
            with open(info_path, "w") as f:
                f.write(f"Parameter Case {i + 1}\n")
                f.write(f"Description: {description}\n")
                f.write(f"k (pivot control): {k}\n")
                f.write(f"t (branching factor): {t}\n")
                f.write(f"l (recursion levels): {l}\n")
                f.write(f"Expected behavior: {description}\n")

            output_files[f"param_case_{i + 1}"] = str(info_path)

        # Create parameter comparison guide
        comparison_path = param_dir / "parameter_comparison.md"
        with open(comparison_path, "w") as f:
            f.write("# Parameter Exploration Guide\n\n")
            f.write(
                "This section explores how different parameter values affect BMSSP behavior.\n\n"
            )

            f.write("## Parameter Cases\n\n")
            for i, (k, t, l_override, description) in enumerate(param_combinations):
                l = l_override if l_override else math.ceil(math.log(n, 2) / t)
                f.write(f"### Case {i + 1}: {description}\n")
                f.write(f"- **Parameters**: k={k}, t={t}, l={l}\n")
                f.write(f"- **Expected effect**: {description}\n")
                f.write(f"- **Files**: `case_{i + 1}_k{k}_t{t}_l{l}/`\n\n")

            f.write("## Comparison Exercise\n\n")
            f.write(
                "1. Compare the recursion trees across different parameter settings\n"
            )
            f.write("2. Analyze how frontier evolution changes with parameters\n")
            f.write("3. Observe the trade-offs between performance and completeness\n")
            f.write("4. Identify optimal parameters for this graph type\n\n")

            f.write("## Questions to Explore\n\n")
            f.write("- How does increasing k affect the number of pivots selected?\n")
            f.write(
                "- What happens to recursion depth when t is very small or very large?\n"
            )
            f.write(
                "- Can you find parameter settings that achieve 100% reachability?\n"
            )
            f.write(
                "- Which parameters give the best performance/completeness balance?\n"
            )

        output_files["parameter_comparison"] = str(comparison_path)
        return output_files

    def _create_master_index(self, output_path: Path, generated_lessons: Dict):
        """Create master index for the educational sequence."""

        index_path = output_path / "README.md"

        with open(index_path, "w") as f:
            f.write("# BMSSP Educational Sequence\n\n")
            f.write(
                "Welcome to the comprehensive BMSSP (Bounded Multi-Source Shortest Path) "
            )
            f.write(
                "educational sequence. This series of lessons introduces the algorithm "
            )
            f.write("from basic concepts to advanced analysis.\n\n")

            f.write("## 🎓 Course Overview\n\n")
            f.write("This educational sequence is designed to:\n")
            f.write("- Introduce BMSSP concepts progressively\n")
            f.write("- Provide visual understanding of algorithm behavior\n")
            f.write("- Compare with classical algorithms like Dijkstra\n")
            f.write("- Explore practical applications and parameter tuning\n\n")

            f.write("## 📚 Lesson Sequence\n\n")

            for lesson_id, lesson_data in generated_lessons.items():
                lesson = lesson_data["lesson"]
                f.write(f"### {lesson.title}\n")
                f.write(f"**Directory**: `{lesson_id}/`\n")
                f.write(f"**Description**: {lesson.description}\n")
                f.write(f"**Graph**: {lesson_data['graph_info']['nodes']} vertices, ")
                f.write(f"{lesson_data['graph_info']['edges']} edges\n\n")

            f.write("## 🚀 Getting Started\n\n")
            f.write("### Prerequisites\n")
            f.write("- Basic understanding of graph theory\n")
            f.write("- Familiarity with shortest path problems\n")
            f.write(
                "- Knowledge of Dijkstra's algorithm (helpful but not required)\n\n"
            )

            f.write("### How to Use This Sequence\n")
            f.write("1. **Start with Lesson 1** - builds fundamental understanding\n")
            f.write(
                "2. **Follow the sequence** - each lesson builds on previous concepts\n"
            )
            f.write(
                "3. **Use multiple views** - combine static images with interactive visualizations\n"
            )
            f.write(
                "4. **Read lesson guides** - each lesson has detailed explanations\n"
            )
            f.write(
                "5. **Explore comparisons** - advanced lessons include algorithm comparisons\n\n"
            )

            f.write("### File Organization\n")
            f.write("Each lesson directory contains:\n")
            f.write("- `lesson_guide.md` - Detailed lesson explanation\n")
            f.write("- `recursion_tree.png` - Algorithm recursion analysis\n")
            f.write("- `frontier_evolution.png` - Frontier behavior over time\n")
            f.write("- `pivot_selection.png` - Step-by-step execution\n")
            f.write("- `interactive_analysis.html` - Interactive exploration\n")
            f.write("- `analysis_report.md` - Detailed algorithmic analysis\n\n")

            f.write("## 👨‍🏫 For Instructors\n\n")
            f.write("See `instructor_guide.md` for:\n")
            f.write("- Lesson timing recommendations\n")
            f.write("- Discussion prompts and answers\n")
            f.write("- Extension activities\n")
            f.write("- Assessment suggestions\n\n")

            f.write("## 🔧 Technical Requirements\n\n")
            f.write("- Web browser (for interactive visualizations)\n")
            f.write("- Image viewer (for static analyses)\n")
            f.write("- Markdown reader (for lesson guides)\n\n")

            f.write("## 📖 Additional Resources\n\n")
            f.write("- [Original Paper](https://arxiv.org/abs/2504.17033)\n")
            f.write("- [Algorithm Walkthrough](../ALGORITHM_WALKTHROUGH.md)\n")
            f.write("- [Performance Analysis](../benchmarks/README.md)\n")
            f.write("- [Implementation Repository](../implementations/)\n")

    def _create_instructor_guide(self, output_path: Path, generated_lessons: Dict):
        """Create instructor guide with teaching suggestions."""

        guide_path = output_path / "instructor_guide.md"

        with open(guide_path, "w") as f:
            f.write("# BMSSP Educational Sequence - Instructor Guide\n\n")
            f.write("This guide provides suggestions for teaching the BMSSP algorithm ")
            f.write("using the provided educational materials.\n\n")

            f.write("## 📅 Suggested Timeline\n\n")
            f.write("### Option 1: Intensive Workshop (1 day)\n")
            f.write("- **Introduction & Lesson 1**: 45 minutes\n")
            f.write("- **Lessons 2-3**: 90 minutes\n")
            f.write("- **Break**: 15 minutes\n")
            f.write("- **Lessons 4-5**: 90 minutes\n")
            f.write("- **Lesson 6 & Discussion**: 60 minutes\n\n")

            f.write("### Option 2: Multi-Session Course (6 sessions)\n")
            f.write("- **Session 1**: Introduction + Lesson 1 (60 min)\n")
            f.write("- **Session 2**: Lesson 2 + Discussion (60 min)\n")
            f.write("- **Session 3**: Lesson 3 + Algorithm Comparison (75 min)\n")
            f.write("- **Session 4**: Lesson 4 + Tree Analysis (60 min)\n")
            f.write("- **Session 5**: Lesson 5 + Complex Networks (75 min)\n")
            f.write(
                "- **Session 6**: Lesson 6 + Parameter Tuning Workshop (90 min)\n\n"
            )

            f.write("## 🎯 Learning Objectives by Lesson\n\n")

            for lesson_id, lesson_data in generated_lessons.items():
                lesson = lesson_data["lesson"]
                f.write(f"### {lesson.title}\n")
                f.write("**Objectives:**\n")
                for obj in lesson.learning_objectives:
                    f.write(f"- {obj}\n")

                f.write("\n**Teaching Suggestions:**\n")

                if lesson_id == "lesson_1":
                    f.write("- Start with intuitive explanation of shortest paths\n")
                    f.write(
                        "- Emphasize the difference between 'optimal' and 'bounded'\n"
                    )
                    f.write("- Use the linear path to show recursion concept clearly\n")
                elif lesson_id == "lesson_2":
                    f.write("- Focus on the pivot selection mechanism\n")
                    f.write(
                        "- Have students predict which vertices will become pivots\n"
                    )
                    f.write("- Discuss why frontier reduction is important\n")
                elif lesson_id == "lesson_3":
                    f.write("- Compare BMSSP results with Dijkstra\n")
                    f.write("- Highlight cases where BMSSP might miss optimal paths\n")
                    f.write("- Discuss the performance vs optimality trade-off\n")
                elif lesson_id == "lesson_4":
                    f.write("- Connect to binary search and divide-and-conquer\n")
                    f.write("- Show how tree structure influences exploration\n")
                    f.write("- Discuss subtree size requirements for pivots\n")
                elif lesson_id == "lesson_5":
                    f.write("- Introduce scale-free network concepts\n")
                    f.write("- Show how hubs naturally become pivots\n")
                    f.write("- Connect to real-world applications\n")
                elif lesson_id == "lesson_6":
                    f.write("- Make this hands-on with parameter experimentation\n")
                    f.write("- Have students predict parameter effects\n")
                    f.write("- Discuss practical parameter selection strategies\n")

                f.write("\n")

            f.write("## 💡 Discussion Prompts\n\n")

            f.write("### General Algorithm Understanding\n")
            f.write("1. How is BMSSP fundamentally different from Dijkstra?\n")
            f.write(
                "2. What are the advantages and disadvantages of bounded exploration?\n"
            )
            f.write("3. In what scenarios would you choose BMSSP over Dijkstra?\n")
            f.write(
                "4. How do the theoretical complexity benefits manifest in practice?\n\n"
            )

            f.write("### Parameter Analysis\n")
            f.write("1. How do k, t, and l interact with each other?\n")
            f.write("2. What happens when k is very small? Very large?\n")
            f.write("3. How should parameters be chosen for different graph types?\n")
            f.write(
                "4. Can you design parameters that guarantee 100% reachability?\n\n"
            )

            f.write("### Practical Applications\n")
            f.write("1. What real-world networks would benefit from BMSSP?\n")
            f.write("2. How would you implement BMSSP in a production system?\n")
            f.write("3. What modifications might improve practical performance?\n")
            f.write("4. How does BMSSP relate to other graph algorithms?\n\n")

            f.write("## 🔍 Assessment Ideas\n\n")

            f.write("### Formative Assessment\n")
            f.write(
                "- **Prediction exercises**: Given a graph, predict pivot selection\n"
            )
            f.write(
                "- **Parameter effects**: Estimate how changing k affects results\n"
            )
            f.write(
                "- **Comparison tasks**: Compare BMSSP vs Dijkstra on given graphs\n"
            )
            f.write(
                "- **Trace execution**: Step through algorithm on small examples\n\n"
            )

            f.write("### Summative Assessment\n")
            f.write("- **Algorithm analysis**: Analyze BMSSP behavior on novel graph\n")
            f.write(
                "- **Parameter tuning**: Design parameters for specific requirements\n"
            )
            f.write(
                "- **Implementation project**: Code simplified version of algorithm\n"
            )
            f.write(
                "- **Research paper**: Compare BMSSP with other shortest path algorithms\n\n"
            )

            f.write("## 🛠️ Extension Activities\n\n")

            f.write("### For Advanced Students\n")
            f.write("1. **Implement BMSSP**: Code the algorithm from scratch\n")
            f.write(
                "2. **Design variations**: Modify algorithm for specific graph types\n"
            )
            f.write("3. **Performance analysis**: Benchmark on large graphs\n")
            f.write(
                "4. **Theoretical analysis**: Prove properties about parameter choices\n\n"
            )

            f.write("### For Research Projects\n")
            f.write("1. **Parallel BMSSP**: Design parallel version of algorithm\n")
            f.write("2. **Dynamic graphs**: Adapt BMSSP for changing networks\n")
            f.write(
                "3. **Approximation quality**: Analyze how close BMSSP gets to optimal\n"
            )
            f.write("4. **Real-world evaluation**: Test on actual network datasets\n\n")

            f.write("## 📋 Troubleshooting Common Issues\n\n")

            f.write("### Student Confusion Points\n")
            f.write("- **'Why doesn't BMSSP find shortest paths?'**\n")
            f.write("  → Emphasize bounded vs optimal exploration trade-off\n")
            f.write("- **'What's the point if it's not optimal?'**\n")
            f.write(
                "  → Discuss theoretical complexity benefits and large-scale applications\n"
            )
            f.write("- **'How do I choose parameters?'**\n")
            f.write("  → Use Lesson 6 parameter exploration as hands-on exercise\n")
            f.write("- **'The recursion is confusing'**\n")
            f.write("  → Start with simple examples and build complexity gradually\n\n")

            f.write("### Technical Issues\n")
            f.write("- **Visualizations won't load**: Check browser compatibility\n")
            f.write(
                "- **Images too small**: Use browser zoom or open in image editor\n"
            )
            f.write(
                "- **Interactive features not working**: Ensure JavaScript is enabled\n\n"
            )

            f.write("## 📚 Additional Resources for Instructors\n\n")

            f.write("### Background Reading\n")
            f.write("- Original BMSSP paper (arXiv:2504.17033)\n")
            f.write("- Classic Dijkstra paper for comparison\n")
            f.write("- Survey of shortest path algorithms\n")
            f.write("- Graph algorithm complexity theory\n\n")

            f.write("### Related Topics to Explore\n")
            f.write("- A* algorithm and heuristic search\n")
            f.write("- Bidirectional shortest path algorithms\n")
            f.write("- All-pairs shortest paths (Floyd-Warshall)\n")
            f.write("- Network flow algorithms\n")
            f.write("- Graph neural networks for shortest paths\n\n")

            f.write("### Teaching Tools\n")
            f.write("- Graph visualization software (Gephi, Cytoscape)\n")
            f.write("- Algorithm animation tools\n")
            f.write("- Programming environments (Python, Java)\n")
            f.write("- Online graph theory resources\n")


def create_educational_sequence():
    """Main function to create the educational sequence."""

    # Set random seed for reproducible examples
    random.seed(42)

    print("🎓 Creating BMSSP Educational Demo Sequence")
    print("=" * 50)

    try:
        sequence = EducationalSequence()
        generated_lessons = sequence.generate_complete_sequence()

        print("\n" + "=" * 50)
        print("📊 Educational Sequence Summary:")
        for lesson_id, lesson_data in generated_lessons.items():
            lesson = lesson_data["lesson"]
            print(f"   {lesson.title}")
            print(f"      Files: {len(lesson_data['outputs'])}")
            print(
                f"      Graph: {lesson_data['graph_info']['nodes']} nodes, {lesson_data['graph_info']['edges']} edges"
            )

        print("\n🎯 Next Steps for Educators:")
        print("   1. Review educational_sequence/README.md for course overview")
        print("   2. Read educational_sequence/instructor_guide.md for teaching tips")
        print("   3. Start with educational_sequence/lesson_1/ for introduction")
        print("   4. Use interactive_analysis.html files for dynamic demonstrations")
        print("   5. Adapt lesson guides to your specific course needs")

        print(f"\n🎯 Next Steps for Students:")
        print("   1. Start with educational_sequence/README.md")
        print("   2. Follow lessons in sequence: lesson_1 → lesson_2 → ... → lesson_6")
        print("   3. Read lesson_guide.md in each directory")
        print("   4. Explore visualizations and interactive content")
        print("   5. Answer discussion questions in each lesson")

        return generated_lessons

    except KeyboardInterrupt:
        print("\n🛑 Educational sequence generation interrupted by user")
    except Exception as e:
        print(f"\n❌ Educational sequence generation failed: {str(e)}")
        import traceback

        traceback.print_exc()


def main():
    """Main function demonstrating educational sequence creation."""
    create_educational_sequence()


if __name__ == "__main__":
    main()
