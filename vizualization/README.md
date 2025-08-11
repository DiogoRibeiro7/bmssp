# BMSSP Visualization Tools

This module provides comprehensive visualization capabilities for understanding the BMSSP algorithm's behavior, including recursion trees, frontier evolution, and pivot selection processes.

## Quick Start

```python
from bmssp_visualization import BMSSPVisualizationSuite
import networkx as nx

# Create test graph
graph = nx.DiGraph()
graph.add_edge('s', 'a', weight=1)
graph.add_edge('s', 'b', weight=4)
graph.add_edge('a', 'b', weight=2)
graph.add_edge('a', 'c', weight=5)
graph.add_edge('b', 'c', weight=1)

# Generate complete visualization suite
suite = BMSSPVisualizationSuite()
outputs = suite.create_complete_analysis(graph, 's', 'output_directory')
```

This creates:
- **Recursion tree analysis** showing call hierarchy and complexity
- **Frontier evolution plots** tracking algorithm progress over time
- **Pivot selection animation** showing step-by-step vertex classification
- **Interactive HTML visualization** for dynamic exploration
- **Comprehensive analysis report** with insights and metrics

## Visualization Components

### 🌳 **Recursion Tree Visualizer**

Shows the complete BMSSP call hierarchy with detailed analysis:

#### **Tree Structure Plot**
- **Nodes**: Colored by recursion level, sized by source count
- **Edges**: Show parent-child call relationships  
- **Shapes**: Circles for recursive calls, squares for base cases
- **Labels**: Level, source count, and pivot information

#### **Level Analysis**
- **Call distribution** across recursion levels
- **Average source count** per level
- **Pivot selection efficiency** metrics

#### **Timeline Analysis**  
- **Execution flow** showing algorithm phases
- **Resource usage** evolution over time
- **Parameter impact** on algorithm behavior

```python
from bmssp_visualization import RecursionTreeVisualizer

# Detailed recursion analysis
visualizer = BMSSPVisualizer()
distances = visualizer.trace_execution(graph, source)

recursion_viz = RecursionTreeVisualizer(visualizer)
fig = recursion_viz.plot_recursion_tree('recursion_analysis.png')
```

### 🌊 **Frontier Evolution Visualizer**

Tracks how the algorithm's frontier (incomplete vertices) evolves:

#### **Frontier Size Evolution**
- **Actual frontier size** vs theoretical bounds
- **Complete vertices count** over time
- **Algorithm phase transitions** (FindPivots, recursive calls)

#### **Distance Distribution**
- **Histogram evolution** at key time points
- **Distribution spread** as algorithm progresses
- **Reachability analysis** showing explored regions

#### **Frontier Composition**
- **Pivot vs witness ratio** in frontier
- **Vertex classification evolution** over time
- **Bound effectiveness** analysis

#### **Bound Evolution**
- **Algorithm bounds** vs actual maximum distances
- **Bound tightening** during recursive calls
- **Convergence analysis** showing algorithm termination

```python
from bmssp_visualization import FrontierEvolutionVisualizer

frontier_viz = FrontierEvolutionVisualizer(visualizer)
fig = frontier_viz.plot_frontier_evolution('frontier_analysis.png')
```

### 🎯 **Graph State Visualizer**

Shows algorithm execution on the actual graph structure:

#### **Pivot Selection Animation**
- **Step-by-step visualization** of FindPivots execution
- **Color-coded vertices** showing current classification:
  - 🔴 **Red**: Source vertex
  - 🟠 **Orange**: Pivot vertices  
  - 🟡 **Yellow**: Witness vertices
  - 🟢 **Green**: Complete vertices
  - 🔵 **Blue**: Frontier vertices
  - ⚪ **Gray**: Reached but not in frontier
  - ⚫ **White**: Unreached vertices

#### **Distance Labels**
- **Current distance estimates** on each vertex
- **Edge weights** displayed on graph edges
- **Infinite distances** shown as ∞ symbol

#### **Interactive Features**
- **Timeline scrubbing** through algorithm execution
- **Hover information** showing vertex details
- **Dynamic layout** preserving graph structure

```python
from bmssp_visualization import GraphStateVisualizer

graph_viz = GraphStateVisualizer(visualizer)  
fig = graph_viz.plot_pivot_selection_animation('pivot_animation.png')
```

## Advanced Usage

### **Custom Graph Analysis**

```python
# Analyze specific graph types
def analyze_scale_free_graph():
    graph = nx.scale_free_graph(20, alpha=0.4, beta=0.5, gamma=0.1)
    # Add weights
    for u, v in graph.edges():
        graph[u][v]['weight'] = random.uniform(1, 10)
    
    suite = BMSSPVisualizationSuite()
    return suite.create_complete_analysis(graph, 0, 'scale_free_analysis')

def analyze_grid_graph():
    graph = nx.grid_2d_graph(5, 5, create_using=nx.DiGraph)
    # Convert to integer nodes and add weights
    mapping = {node: i for i, node in enumerate(graph.nodes())}
    graph = nx.relabel_nodes(graph, mapping)
    for u, v in graph.edges():
        graph[u][v]['weight'] = random.uniform(1, 3)
    
    suite = BMSSPVisualizationSuite()
    return suite.create_complete_analysis(graph, 0, 'grid_analysis')
```

### **Parameter Sensitivity Visualization**

```python
def visualize_parameter_effects():
    graph = create_test_graph(50)  # Medium-sized graph
    
    # Test different parameter combinations
    parameters = [
        (1, 2, 3),  # Small k, t, l
        (2, 4, 2),  # Medium k, t, l  
        (4, 6, 2),  # Large k, t, l
    ]
    
    results = []
    for k, t, l in parameters:
        # Run BMSSP with custom parameters
        visualizer = BMSSPVisualizer()
        distances = visualizer.trace_execution_with_params(graph, 's', k, t, l)
        
        results.append({
            'params': (k, t, l),
            'reachable': sum(1 for d in distances.values() if d < float('inf')),
            'max_frontier': max(len(state.frontier) for state in visualizer.execution_trace),
            'total_steps': len(visualizer.execution_trace)
        })
    
    # Visualize parameter impact
    plot_parameter_comparison(results)
```

### **Comparative Analysis**

```python
def compare_with_dijkstra():
    """Compare BMSSP behavior with Dijkstra's algorithm."""
    graph = create_test_graph(30)
    
    # Run BMSSP
    bmssp_suite = BMSSPVisualizationSuite()
    bmssp_distances = bmssp_suite.visualizer.trace_execution(graph, 's')
    
    # Run Dijkstra for comparison
    dijkstra_distances = nx.single_source_dijkstra_path_length(graph, 's')
    
    # Generate comparison visualization
    plot_algorithm_comparison(bmssp_distances, dijkstra_distances, 
                            bmssp_suite.visualizer.execution_trace)
```

## Output Analysis

### **Generated Files**

When running `create_complete_analysis()`, you get:

#### **Static Visualizations**
- `recursion_tree.png` - Complete recursion analysis (16x12 inches)
- `frontier_evolution.png` - Frontier behavior analysis (16x10 inches)  
- `pivot_selection.png` - Step-by-step pivot selection (20x15 inches)

#### **Interactive Content**
- `interactive_analysis.html` - Full interactive visualization
- Open in any web browser for dynamic exploration

#### **Analysis Report**
- `analysis_report.md` - Comprehensive analysis with:
  - Graph statistics and algorithm parameters
  - Execution results and distance computations
  - Algorithm insights and complexity analysis
  - File descriptions and interpretation guide

### **Interpretation Guide**

#### **Understanding Recursion Trees**
```
Level 0 (Top): Initial BMSSP call with full graph
├─ Level 1: Recursive calls on pivot subsets  
│  ├─ Level 2: Further subdivision
│  └─ Base cases: Dijkstra-like computation
└─ Leaf nodes: Final distance computations
```

**Key Metrics**:
- **Node count per level**: Shows divide-and-conquer effectiveness
- **Source reduction**: Indicates pivot selection quality  
- **Base case frequency**: Reveals algorithm termination patterns

#### **Reading Frontier Evolution**
- **Rising frontier**: Algorithm exploring new regions
- **Falling frontier**: Vertices being completed/bounded out
- **Plateau periods**: Algorithm in FindPivots or recursive setup
- **Sharp drops**: Bound tightening excluding vertices

#### **Interpreting Pivot Selection**
- **Red vertices** (sources): Starting points for exploration
- **Orange vertices** (pivots): Selected for recursive calls
- **Yellow vertices** (witnesses): Explored but not pivotal
- **Progression**: Shows how vertex classification evolves

## Performance Considerations

### **Memory Usage**
- **Execution tracing** stores complete algorithm state at each step
- **Large graphs** (n > 100) may require significant memory
- **Reduce visualization frequency** for very large graphs

### **Computation Time**
- **Full tracing** adds 2-3x overhead to algorithm execution
- **Layout computation** for graph visualization can be expensive
- **Use simplified visualizations** for quick analysis

### **Optimization Tips**
```python
# For large graphs, reduce tracing frequency
visualizer = BMSSPVisualizer()
visualizer.trace_frequency = 5  # Record every 5th step

# Use simpler layouts for faster rendering
pos = nx.spring_layout(graph, k=0.5, iterations=20)  # Faster layout

# Generate only specific visualizations
suite = BMSSPVisualizationSuite()
suite.recursion_visualizer.plot_recursion_tree()  # Only recursion analysis
```

## Integration with Analysis Framework

### **Combined with Performance Analysis**
```python
from performance_analysis import PerformanceAnalyzer
from bmssp_visualization import BMSSPVisualizationSuite

# Run performance analysis
analyzer = PerformanceAnalyzer()
perf_results = analyzer.run_comparison([64, 128, 256], ["sparse_random"])

# Generate visualizations for interesting cases
for result in perf_results["sparse_random"]:
    if result['runtime_improvement'] > 0.2:  # 20% improvement
        graph = GraphGenerator.sparse_random(result['n'])
        suite = BMSSPVisualizationSuite()
        suite.create_complete_analysis(graph, 0, f"analysis_n_{result['n']}")
```

### **Educational Use Cases**
```python
# Create step-by-step tutorial
def create_educational_sequence():
    """Generate educational visualization sequence."""
    graphs = [
        create_simple_path_graph(4),      # Linear structure
        create_binary_tree_graph(7),      # Tree structure  
        create_small_random_graph(8),     # General case
    ]
    
    for i, graph in enumerate(graphs):
        suite = BMSSPVisualizationSuite()
        suite.create_complete_analysis(graph, 0, f"tutorial_step_{i+1}")
        
        # Generate simplified explanation
        generate_step_explanation(graph, i+1)
```

This visualization suite provides comprehensive tools for understanding, analyzing, and teaching the BMSSP algorithm through rich, interactive visualizations that reveal the algorithm's sophisticated behavior and theoretical foundations.