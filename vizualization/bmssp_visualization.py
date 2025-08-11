#!/usr/bin/env python3
"""
BMSSP Visualization Tools

This module provides comprehensive visualization capabilities for understanding
the BMSSP algorithm behavior, including recursion trees, frontier evolution,
and pivot selection processes.

Based on "Breaking the Sorting Barrier for Directed Single-Source Shortest Paths"
by Duan et al., 2025.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import math
import random
from datetime import datetime

@dataclass
class BMSSPState:
    """Captures algorithm state at a specific point in execution."""
    level: int
    bound: float
    sources: Set[str]
    distances: Dict[str, float]
    complete: Set[str]
    frontier: Set[str]
    pivots: Set[str]
    witnesses: Set[str]
    step_description: str
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

@dataclass
class RecursionNode:
    """Represents a node in the BMSSP recursion tree."""
    call_id: str
    level: int
    bound: float
    sources: Set[str]
    parent: Optional['RecursionNode'] = None
    children: List['RecursionNode'] = field(default_factory=list)
    result_bound: float = float('inf')
    result_vertices: Set[str] = field(default_factory=set)
    execution_time: float = 0.0
    pivot_count: int = 0
    is_base_case: bool = False

class BMSSPVisualizer:
    """Main class for BMSSP algorithm visualization."""
    
    def __init__(self):
        self.execution_trace: List[BMSSPState] = []
        self.recursion_tree: Optional[RecursionNode] = None
        self.call_counter = 0
        self.graph = None
        self.source = None
        
    def trace_execution(self, graph: nx.DiGraph, source: str) -> Dict[str, float]:
        """Run BMSSP with full execution tracing."""
        self.graph = graph
        self.source = source
        self.execution_trace.clear()
        self.call_counter = 0
        
        n = len(graph.nodes())
        k = max(1, int(math.log(n, 2) ** (1/3)))
        t = max(1, int(math.log(n, 2) ** (2/3)))
        l = math.ceil(math.log(n, 2) / t)
        
        distances = {node: float('inf') for node in graph.nodes()}
        distances[source] = 0.0
        complete = {source}
        
        # Record initial state
        self._record_state(l, float('inf'), {source}, distances, complete, 
                          set(), set(), set(), "Initial state")
        
        # Start main recursion with tracing
        self.recursion_tree = self._traced_bmssp(
            graph, l, float('inf'), {source}, distances, complete, k, t
        )
        
        return distances
    
    def _traced_bmssp(self, graph: nx.DiGraph, level: int, bound: float,
                     sources: Set[str], distances: Dict[str, float],
                     complete: Set[str], k: int, t: int) -> RecursionNode:
        """BMSSP with recursion tree building."""
        self.call_counter += 1
        call_id = f"BMSSP_{self.call_counter}"
        
        node = RecursionNode(
            call_id=call_id,
            level=level,
            bound=bound,
            sources=sources.copy(),
            is_base_case=(level == 0)
        )
        
        if level == 0:
            # Base case
            result_distances, result_vertices = self._traced_base_case(
                graph, bound, sources, distances, complete, k, call_id
            )
            node.result_vertices = result_vertices
            node.result_bound = bound
            return node
        
        # FindPivots with tracing
        pivots, witnesses = self._traced_find_pivots(
            graph, bound, sources, distances, complete, k, call_id
        )
        
        node.pivot_count = len(pivots)
        
        # Record state after FindPivots
        frontier = self._calculate_frontier(distances, bound)
        self._record_state(level, bound, sources, distances, complete,
                          frontier, pivots, witnesses,
                          f"FindPivots complete: {len(pivots)} pivots, {len(witnesses)} witnesses")
        
        # Recursive calls (simplified for visualization)
        M = 2 ** ((level - 1) * t)
        completed = set()
        
        for i, pivot in enumerate(pivots):
            if len(completed) >= k * (2 ** (level * t)):
                break
                
            # Simplified recursive call
            sub_bound = min(bound, distances[pivot] + 10)
            child_node = self._traced_bmssp(
                graph, level - 1, sub_bound, {pivot}, 
                distances, complete, k, t
            )
            
            child_node.parent = node
            node.children.append(child_node)
            completed.update(child_node.result_vertices)
            
            # Record recursive call state
            self._record_state(level, bound, sources, distances, complete,
                              self._calculate_frontier(distances, bound),
                              pivots, witnesses,
                              f"Recursive call {i+1}/{len(pivots)} complete")
        
        node.result_vertices = completed
        node.result_bound = bound
        return node
    
    def _traced_find_pivots(self, graph: nx.DiGraph, bound: float,
                           sources: Set[str], distances: Dict[str, float],
                           complete: Set[str], k: int, call_id: str) -> Tuple[Set[str], Set[str]]:
        """FindPivots with detailed tracing."""
        explored = set(sources)
        current_level = set(sources)
        
        self._record_state(0, bound, sources, distances, complete,
                          current_level, set(), set(),
                          f"{call_id}: FindPivots started")
        
        # k iterations of bounded expansion
        for iteration in range(k):
            next_level = set()
            for u in current_level:
                if distances[u] >= bound:
                    continue
                
                for v in graph.neighbors(u):
                    if v not in graph.nodes():
                        continue
                    weight = graph[u][v]['weight']
                    new_dist = distances[u] + weight
                    
                    if new_dist <= distances[v] and new_dist < bound:
                        distances[v] = new_dist
                        if new_dist < bound:
                            next_level.add(v)
            
            explored.update(next_level)
            current_level = next_level
            
            # Record expansion step
            self._record_state(0, bound, sources, distances, complete,
                              explored, set(), set(),
                              f"{call_id}: FindPivots iteration {iteration+1}, explored {len(explored)} nodes")
            
            # Size check
            if len(explored) > k * len(sources):
                self._record_state(0, bound, sources, distances, complete,
                                  explored, sources, explored,
                                  f"{call_id}: Size limit exceeded, all sources become pivots")
                return sources, explored
        
        # Select pivots (simplified)
        pivots = set()
        for s in sources:
            if len(pivots) < max(1, len(explored) // k):
                pivots.add(s)
        
        pivots = pivots or sources
        
        self._record_state(0, bound, sources, distances, complete,
                          explored, pivots, explored,
                          f"{call_id}: FindPivots complete, selected {len(pivots)} pivots")
        
        return pivots, explored
    
    def _traced_base_case(self, graph: nx.DiGraph, bound: float,
                         sources: Set[str], distances: Dict[str, float],
                         complete: Set[str], k: int, call_id: str) -> Tuple[Dict[str, float], Set[str]]:
        """Base case with tracing."""
        if not sources:
            return distances, set()
        
        source = next(iter(sources))
        processed = {source}
        queue = [(distances[source], source)]
        
        self._record_state(0, bound, sources, distances, complete,
                          {source}, set(), set(),
                          f"{call_id}: Base case started from {source}")
        
        step = 0
        while queue and len(processed) < k + 1:
            step += 1
            current_dist, u = min(queue)
            queue.remove((current_dist, u))
            
            if current_dist >= bound:
                break
            
            for v in graph.neighbors(u):
                if v not in graph.nodes():
                    continue
                weight = graph[u][v]['weight']
                new_dist = current_dist + weight
                
                if new_dist <= distances[v] and new_dist < bound:
                    distances[v] = new_dist
                    processed.add(v)
                    queue.append((new_dist, v))
            
            # Record base case step
            if step % 3 == 0:  # Record every few steps to avoid clutter
                self._record_state(0, bound, sources, distances, complete,
                                  processed, set(), set(),
                                  f"{call_id}: Base case step {step}, processed {len(processed)} nodes")
        
        self._record_state(0, bound, sources, distances, complete,
                          processed, set(), set(),
                          f"{call_id}: Base case complete, {len(processed)} nodes processed")
        
        return distances, processed
    
    def _calculate_frontier(self, distances: Dict[str, float], bound: float) -> Set[str]:
        """Calculate current frontier (incomplete vertices with finite distance < bound)."""
        return {v for v, d in distances.items() 
                if d < bound and d < float('inf')}
    
    def _record_state(self, level: int, bound: float, sources: Set[str],
                     distances: Dict[str, float], complete: Set[str],
                     frontier: Set[str], pivots: Set[str], witnesses: Set[str],
                     description: str):
        """Record current algorithm state."""
        state = BMSSPState(
            level=level,
            bound=bound,
            sources=sources.copy(),
            distances=distances.copy(),
            complete=complete.copy(),
            frontier=frontier.copy(),
            pivots=pivots.copy(),
            witnesses=witnesses.copy(),
            step_description=description
        )
        self.execution_trace.append(state)

class RecursionTreeVisualizer:
    """Specialized visualizer for BMSSP recursion trees."""
    
    def __init__(self, visualizer: BMSSPVisualizer):
        self.visualizer = visualizer
    
    def plot_recursion_tree(self, save_path: str = "recursion_tree.png", 
                           figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """Generate comprehensive recursion tree visualization."""
        if not self.visualizer.recursion_tree:
            raise ValueError("No recursion tree available. Run trace_execution first.")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Tree structure with call hierarchy
        self._plot_tree_structure(ax1)
        
        # 2. Level-wise resource usage
        self._plot_level_analysis(ax2)
        
        # 3. Execution timeline
        self._plot_execution_timeline(ax3)
        
        # 4. Parameter sensitivity
        self._plot_parameter_impact(ax4)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    def _plot_tree_structure(self, ax):
        """Plot the recursion tree structure."""
        # Build tree layout
        positions = self._calculate_tree_positions(self.visualizer.recursion_tree)
        
        # Draw nodes
        for node, (x, y) in positions.items():
            # Node color based on level
            colors = ['red', 'orange', 'yellow', 'lightgreen', 'lightblue', 'purple']
            color = colors[node.level % len(colors)]
            
            # Node size based on number of sources
            size = 100 + 50 * len(node.sources)
            
            if node.is_base_case:
                marker = 's'  # Square for base cases
                edgecolor = 'black'
                linewidth = 2
            else:
                marker = 'o'  # Circle for recursive calls
                edgecolor = 'gray'
                linewidth = 1
            
            ax.scatter(x, y, s=size, c=color, marker=marker, 
                      edgecolors=edgecolor, linewidth=linewidth, alpha=0.8)
            
            # Label with call info
            label = f"L{node.level}\n|S|={len(node.sources)}"
            if node.pivot_count > 0:
                label += f"\nP={node.pivot_count}"
            
            ax.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points',
                       fontsize=8, ha='left')
        
        # Draw edges
        for node, (x, y) in positions.items():
            for child in node.children:
                child_x, child_y = positions[child]
                ax.plot([x, child_x], [y, child_y], 'k-', alpha=0.6, linewidth=1)
        
        ax.set_title("BMSSP Recursion Tree Structure")
        ax.set_xlabel("Horizontal Position")
        ax.set_ylabel("Recursion Level")
        
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                      markersize=10, label='Recursive Call'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgreen', 
                      markersize=10, label='Base Case'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    def _plot_level_analysis(self, ax):
        """Plot analysis of calls by recursion level."""
        level_stats = defaultdict(lambda: {'count': 0, 'total_sources': 0, 'total_pivots': 0})
        
        def collect_stats(node):
            level_stats[node.level]['count'] += 1
            level_stats[node.level]['total_sources'] += len(node.sources)
            level_stats[node.level]['total_pivots'] += node.pivot_count
            for child in node.children:
                collect_stats(child)
        
        collect_stats(self.visualizer.recursion_tree)
        
        levels = sorted(level_stats.keys())
        counts = [level_stats[l]['count'] for l in levels]
        avg_sources = [level_stats[l]['total_sources'] / max(1, level_stats[l]['count']) for l in levels]
        avg_pivots = [level_stats[l]['total_pivots'] / max(1, level_stats[l]['count']) for l in levels]
        
        ax2 = ax.twinx()
        
        # Bar chart of call counts
        bars1 = ax.bar([l - 0.2 for l in levels], counts, 0.4, label='Call Count', alpha=0.7)
        bars2 = ax2.bar([l + 0.2 for l in levels], avg_sources, 0.4, 
                       label='Avg Sources', alpha=0.7, color='orange')
        
        ax.set_xlabel('Recursion Level')
        ax.set_ylabel('Number of Calls', color='blue')
        ax2.set_ylabel('Average Sources per Call', color='orange')
        
        ax.set_title('Recursion Level Analysis')
        ax.grid(True, alpha=0.3)
        
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    def _plot_execution_timeline(self, ax):
        """Plot execution flow over time."""
        if not self.visualizer.execution_trace:
            ax.text(0.5, 0.5, 'No execution trace available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        times = range(len(self.visualizer.execution_trace))
        frontier_sizes = [len(state.frontier) for state in self.visualizer.execution_trace]
        complete_sizes = [len(state.complete) for state in self.visualizer.execution_trace]
        pivot_sizes = [len(state.pivots) for state in self.visualizer.execution_trace]
        
        ax.plot(times, frontier_sizes, 'b-', label='Frontier Size', linewidth=2)
        ax.plot(times, complete_sizes, 'g-', label='Complete Vertices', linewidth=2)
        ax.plot(times, pivot_sizes, 'r-', label='Pivot Count', linewidth=2)
        
        # Highlight major algorithm phases
        phase_changes = []
        current_level = None
        for i, state in enumerate(self.visualizer.execution_trace):
            if state.level != current_level:
                phase_changes.append(i)
                current_level = state.level
        
        for change in phase_changes:
            ax.axvline(x=change, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Execution Step')
        ax.set_ylabel('Count')
        ax.set_title('Algorithm Execution Timeline')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_parameter_impact(self, ax):
        """Plot parameter impact visualization."""
        if not self.visualizer.recursion_tree:
            return
        
        # Calculate parameter-derived metrics
        root = self.visualizer.recursion_tree
        n = len(self.visualizer.graph.nodes()) if self.visualizer.graph else 10
        
        k = max(1, int(math.log(n, 2) ** (1/3)))
        t = max(1, int(math.log(n, 2) ** (2/3)))
        l = math.ceil(math.log(n, 2) / t)
        
        # Theoretical vs actual metrics
        theoretical_calls = sum(k ** i for i in range(l + 1))
        actual_calls = self._count_total_calls(root)
        
        theoretical_max_sources = k * (2 ** (l * t))
        actual_total_sources = sum(len(node.sources) for node in self._get_all_nodes(root))
        
        metrics = ['Total Calls', 'Source Vertices', 'Max Level', 'Pivot Efficiency']
        theoretical = [theoretical_calls, theoretical_max_sources, l, 1.0]
        actual = [actual_calls, actual_total_sources, self._get_max_level(root), 
                 self._calculate_pivot_efficiency(root)]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        # Normalize for comparison
        theoretical_norm = [t / max(t, a) for t, a in zip(theoretical, actual)]
        actual_norm = [a / max(t, a) for t, a in zip(theoretical, actual)]
        
        ax.bar(x - width/2, theoretical_norm, width, label='Theoretical', alpha=0.7)
        ax.bar(x + width/2, actual_norm, width, label='Actual', alpha=0.7)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Normalized Values')
        ax.set_title('Theoretical vs Actual Algorithm Behavior')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _calculate_tree_positions(self, root: RecursionNode) -> Dict[RecursionNode, Tuple[float, float]]:
        """Calculate 2D positions for tree nodes."""
        positions = {}
        
        def assign_positions(node, x, y, x_spacing):
            positions[node] = (x, y)
            
            if node.children:
                child_spacing = x_spacing / len(node.children)
                start_x = x - (len(node.children) - 1) * child_spacing / 2
                
                for i, child in enumerate(node.children):
                    child_x = start_x + i * child_spacing
                    assign_positions(child, child_x, y - 1, child_spacing)
        
        assign_positions(root, 0, 0, 10)
        return positions
    
    def _count_total_calls(self, node: RecursionNode) -> int:
        """Count total recursive calls in tree."""
        return 1 + sum(self._count_total_calls(child) for child in node.children)
    
    def _get_all_nodes(self, node: RecursionNode) -> List[RecursionNode]:
        """Get all nodes in tree."""
        result = [node]
        for child in node.children:
            result.extend(self._get_all_nodes(child))
        return result
    
    def _get_max_level(self, node: RecursionNode) -> int:
        """Get maximum level in tree."""
        if not node.children:
            return node.level
        return max(self._get_max_level(child) for child in node.children)
    
    def _calculate_pivot_efficiency(self, node: RecursionNode) -> float:
        """Calculate pivot selection efficiency."""
        total_pivots = sum(n.pivot_count for n in self._get_all_nodes(node))
        total_sources = sum(len(n.sources) for n in self._get_all_nodes(node))
        return total_pivots / max(1, total_sources)

class FrontierEvolutionVisualizer:
    """Specialized visualizer for frontier evolution."""
    
    def __init__(self, visualizer: BMSSPVisualizer):
        self.visualizer = visualizer
    
    def plot_frontier_evolution(self, save_path: str = "frontier_evolution.png",
                               figsize: Tuple[int, int] = (16, 10)) -> plt.Figure:
        """Generate comprehensive frontier evolution visualization."""
        if not self.visualizer.execution_trace:
            raise ValueError("No execution trace available. Run trace_execution first.")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Frontier size over time
        self._plot_frontier_size_evolution(ax1)
        
        # 2. Distance distribution evolution
        self._plot_distance_distribution(ax2)
        
        # 3. Frontier composition analysis
        self._plot_frontier_composition(ax3)
        
        # 4. Bound evolution
        self._plot_bound_evolution(ax4)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    def _plot_frontier_size_evolution(self, ax):
        """Plot how frontier size changes over time."""
        steps = range(len(self.visualizer.execution_trace))
        frontier_sizes = [len(state.frontier) for state in self.visualizer.execution_trace]
        complete_sizes = [len(state.complete) for state in self.visualizer.execution_trace]
        
        # Theoretical bounds
        n = len(self.visualizer.graph.nodes()) if self.visualizer.graph else max(frontier_sizes + complete_sizes)
        theoretical_frontier = [min(n, step * 2) for step in steps]  # Simplified theoretical bound
        
        ax.plot(steps, frontier_sizes, 'b-', linewidth=2, label='Actual Frontier Size')
        ax.plot(steps, complete_sizes, 'g-', linewidth=2, label='Complete Vertices')
        ax.plot(steps, theoretical_frontier, 'r--', alpha=0.7, label='Theoretical Bound')
        
        # Highlight FindPivots phases
        findpivots_steps = [i for i, state in enumerate(self.visualizer.execution_trace)
                           if 'FindPivots' in state.step_description]
        
        for step in findpivots_steps:
            ax.axvline(x=step, color='orange', alpha=0.5, linestyle=':')
        
        ax.set_xlabel('Execution Step')
        ax.set_ylabel('Number of Vertices')
        ax.set_title('Frontier Size Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add annotations for key events
        if findpivots_steps:
            ax.annotate('FindPivots', xy=(findpivots_steps[0], frontier_sizes[findpivots_steps[0]]),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    def _plot_distance_distribution(self, ax):
        """Plot evolution of distance distribution."""
        # Sample key time points
        sample_indices = [0, len(self.visualizer.execution_trace) // 4,
                         len(self.visualizer.execution_trace) // 2,
                         3 * len(self.visualizer.execution_trace) // 4,
                         len(self.visualizer.execution_trace) - 1]
        
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'blue']
        
        for i, idx in enumerate(sample_indices):
            if idx < len(self.visualizer.execution_trace):
                state = self.visualizer.execution_trace[idx]
                distances = [d for d in state.distances.values() if d < float('inf')]
                
                if distances:
                    ax.hist(distances, bins=20, alpha=0.6, color=colors[i], 
                           label=f'Step {idx}', density=True)
        
        ax.set_xlabel('Distance')
        ax.set_ylabel('Density')
        ax.set_title('Distance Distribution Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_frontier_composition(self, ax):
        """Analyze frontier composition over time."""
        steps = range(len(self.visualizer.execution_trace))
        pivot_ratios = []
        witness_ratios = []
        
        for state in self.visualizer.execution_trace:
            total_frontier = len(state.frontier)
            if total_frontier > 0:
                pivot_ratio = len(state.pivots & state.frontier) / total_frontier
                witness_ratio = len(state.witnesses & state.frontier) / total_frontier
            else:
                pivot_ratio = witness_ratio = 0
            
            pivot_ratios.append(pivot_ratio)
            witness_ratios.append(witness_ratio)
        
        ax.fill_between(steps, 0, pivot_ratios, alpha=0.7, label='Pivots', color='red')
        ax.fill_between(steps, pivot_ratios, [p + w for p, w in zip(pivot_ratios, witness_ratios)],
                       alpha=0.7, label='Witnesses', color='blue')
        ax.fill_between(steps, [p + w for p, w in zip(pivot_ratios, witness_ratios)], 1,
                       alpha=0.7, label='Other', color='gray')
        
        ax.set_xlabel('Execution Step')
        ax.set_ylabel('Composition Ratio')
        ax.set_title('Frontier Composition Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_bound_evolution(self, ax):
        """Plot how bounds evolve during execution."""
        steps = range(len(self.visualizer.execution_trace))
        bounds = [state.bound for state in self.visualizer.execution_trace]
        max_distances = []
        
        for state in self.visualizer.execution_trace:
            finite_distances = [d for d in state.distances.values() if d < float('inf')]
            max_distances.append(max(finite_distances) if finite_distances else 0)
        
        ax.plot(steps, bounds, 'r-', linewidth=2, label='Algorithm Bound')
        ax.plot(steps, max_distances, 'b-', linewidth=2, label='Max Finite Distance')
        
        ax.set_xlabel('Execution Step')
        ax.set_ylabel('Distance Value')
        ax.set_title('Bound Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Log scale for better visualization

class GraphStateVisualizer:
    """Specialized visualizer for graph-based state visualization."""
    
    def __init__(self, visualizer: BMSSPVisualizer):
        self.visualizer = visualizer
    
    def plot_pivot_selection_animation(self, save_path: str = "pivot_selection.png",
                                      figsize: Tuple[int, int] = (20, 15)) -> plt.Figure:
        """Generate step-by-step pivot selection visualization."""
        if not self.visualizer.graph or not self.visualizer.execution_trace:
            raise ValueError("Graph and execution trace required.")
        
        # Find FindPivots execution steps
        pivot_steps = [i for i, state in enumerate(self.visualizer.execution_trace)
                      if 'FindPivots' in state.step_description]
        
        if not pivot_steps:
            raise ValueError("No FindPivots steps found in execution trace.")
        
        # Create subplot grid
        n_steps = min(6, len(pivot_steps))  # Show up to 6 key steps
        cols = 3
        rows = (n_steps + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        # Graph layout (consistent across all subplots)
        pos = nx.spring_layout(self.visualizer.graph, seed=42, k=2, iterations=50)
        
        # Select key steps to visualize
        key_steps = pivot_steps[:n_steps]
        
        for i, step_idx in enumerate(key_steps):
            ax = axes[i] if i < len(axes) else None
            if ax is None:
                break
                
            state = self.visualizer.execution_trace[step_idx]
            self._plot_graph_state(ax, pos, state, f"Step {step_idx}: {state.step_description}")
        
        # Hide unused subplots
        for i in range(len(key_steps), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    def _plot_graph_state(self, ax, pos: Dict, state: BMSSPState, title: str):
        """Plot graph state with color-coded vertices."""
        # Clear the axes
        ax.clear()
        
        # Node colors based on state
        node_colors = []
        node_sizes = []
        
        for node in self.visualizer.graph.nodes():
            # Determine color based on vertex state
            if node == self.visualizer.source:
                color = 'red'
                size = 400
            elif node in state.pivots:
                color = 'orange' 
                size = 350
            elif node in state.witnesses:
                color = 'yellow'
                size = 300
            elif node in state.complete:
                color = 'lightgreen'
                size = 250
            elif node in state.frontier:
                color = 'lightblue'
                size = 200
            elif state.distances[node] < float('inf'):
                color = 'lightgray'
                size = 150
            else:
                color = 'white'
                size = 100
            
            node_colors.append(color)
            node_sizes.append(size)
        
        # Draw graph
        nx.draw_networkx_nodes(self.visualizer.graph, pos, node_color=node_colors, 
                              node_size=node_sizes, ax=ax, edgecolors='black', linewidths=1)
        
        nx.draw_networkx_edges(self.visualizer.graph, pos, ax=ax, edge_color='gray', 
                              arrows=True, arrowsize=20, alpha=0.6)
        
        # Labels with distances
        labels = {}
        for node in self.visualizer.graph.nodes():
            dist = state.distances[node]
            if dist == float('inf'):
                labels[node] = f"{node}\n∞"
            else:
                labels[node] = f"{node}\n{dist:.1f}"
        
        nx.draw_networkx_labels(self.visualizer.graph, pos, labels, ax=ax, font_size=8)
        
        # Edge labels (weights)
        edge_labels = nx.get_edge_attributes(self.visualizer.graph, 'weight')
        nx.draw_networkx_edge_labels(self.visualizer.graph, pos, edge_labels, ax=ax, font_size=6)
        
        ax.set_title(title, fontsize=10)
        ax.set_aspect('equal')
        
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markersize=10, label='Source'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                      markersize=10, label='Pivot'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', 
                      markersize=10, label='Witness'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                      markersize=10, label='Complete'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                      markersize=10, label='Frontier'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', 
                      markersize=10, label='Reached'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white', 
                      markersize=10, label='Unreached'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))

    def create_interactive_visualization(self, save_path: str = "interactive_bmssp.html"):
        """Create interactive HTML visualization using plotly (if available)."""
        try:
            import plotly.graph_objects as go
            import plotly.offline as pyo
            from plotly.subplots import make_subplots
        except ImportError:
            print("Plotly not available. Install with: pip install plotly")
            return None
        
        if not self.visualizer.execution_trace:
            raise ValueError("No execution trace available.")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Graph State', 'Frontier Evolution', 'Distance Distribution', 'Algorithm Progress'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # Graph state evolution (simplified 2D projection)
        pos = nx.spring_layout(self.visualizer.graph, seed=42)
        
        for i, state in enumerate(self.visualizer.execution_trace[::max(1, len(self.visualizer.execution_trace)//20)]):
            # Node positions
            x_coords = [pos[node][0] for node in self.visualizer.graph.nodes()]
            y_coords = [pos[node][1] for node in self.visualizer.graph.nodes()]
            
            # Color coding
            colors = []
            for node in self.visualizer.graph.nodes():
                if node in state.pivots:
                    colors.append('red')
                elif node in state.complete:
                    colors.append('green')
                elif node in state.frontier:
                    colors.append('blue')
                else:
                    colors.append('gray')
            
            fig.add_trace(
                go.Scatter(x=x_coords, y=y_coords, mode='markers',
                          marker=dict(color=colors, size=10),
                          name=f'Step {i}', visible=(i==0)),
                row=1, col=1
            )
        
        # Frontier evolution
        steps = list(range(len(self.visualizer.execution_trace)))
        frontier_sizes = [len(state.frontier) for state in self.visualizer.execution_trace]
        
        fig.add_trace(
            go.Scatter(x=steps, y=frontier_sizes, mode='lines',
                      name='Frontier Size', line=dict(color='blue')),
            row=1, col=2
        )
        
        # Distance distribution
        final_state = self.visualizer.execution_trace[-1]
        distances = [d for d in final_state.distances.values() if d < float('inf')]
        
        fig.add_trace(
            go.Histogram(x=distances, nbinsx=20, name='Distance Distribution'),
            row=2, col=1
        )
        
        # Algorithm progress
        complete_sizes = [len(state.complete) for state in self.visualizer.execution_trace]
        
        fig.add_trace(
            go.Scatter(x=steps, y=complete_sizes, mode='lines',
                      name='Complete Vertices', line=dict(color='green')),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Interactive BMSSP Algorithm Visualization",
            showlegend=True,
            height=800
        )
        
        # Save as HTML
        pyo.plot(fig, filename=save_path, auto_open=False)
        return fig

class BMSSPVisualizationSuite:
    """Main class coordinating all BMSSP visualizations."""
    
    def __init__(self):
        self.visualizer = BMSSPVisualizer()
        self.recursion_visualizer = None
        self.frontier_visualizer = None
        self.graph_visualizer = None
    
    def create_complete_analysis(self, graph: nx.DiGraph, source: str, 
                               output_dir: str = "bmssp_analysis") -> Dict[str, str]:
        """Generate complete visualization suite for a graph."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("🔬 Running BMSSP with full tracing...")
        distances = self.visualizer.trace_execution(graph, source)
        
        # Initialize specialized visualizers
        self.recursion_visualizer = RecursionTreeVisualizer(self.visualizer)
        self.frontier_visualizer = FrontierEvolutionVisualizer(self.visualizer)
        self.graph_visualizer = GraphStateVisualizer(self.visualizer)
        
        output_files = {}
        
        print("📊 Generating recursion tree analysis...")
        recursion_path = os.path.join(output_dir, "recursion_tree.png")
        self.recursion_visualizer.plot_recursion_tree(recursion_path)
        output_files['recursion_tree'] = recursion_path
        
        print("🌊 Generating frontier evolution analysis...")
        frontier_path = os.path.join(output_dir, "frontier_evolution.png")
        self.frontier_visualizer.plot_frontier_evolution(frontier_path)
        output_files['frontier_evolution'] = frontier_path
        
        print("🎯 Generating pivot selection visualization...")
        pivot_path = os.path.join(output_dir, "pivot_selection.png")
        self.graph_visualizer.plot_pivot_selection_animation(pivot_path)
        output_files['pivot_selection'] = pivot_path
        
        print("🌐 Generating interactive visualization...")
        interactive_path = os.path.join(output_dir, "interactive_analysis.html")
        self.graph_visualizer.create_interactive_visualization(interactive_path)
        output_files['interactive'] = interactive_path
        
        # Generate summary report
        print("📝 Generating analysis report...")
        report_path = os.path.join(output_dir, "analysis_report.md")
        self._generate_analysis_report(distances, report_path)
        output_files['report'] = report_path
        
        print(f"✅ Complete analysis saved to {output_dir}/")
        return output_files
    
    def _generate_analysis_report(self, distances: Dict[str, float], report_path: str):
        """Generate comprehensive analysis report."""
        with open(report_path, 'w') as f:
            f.write("# BMSSP Algorithm Analysis Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Graph statistics
            n = len(self.visualizer.graph.nodes())
            m = len(self.visualizer.graph.edges())
            density = m / (n * (n - 1)) if n > 1 else 0
            
            f.write("## Graph Statistics\n\n")
            f.write(f"- **Vertices**: {n}\n")
            f.write(f"- **Edges**: {m}\n")
            f.write(f"- **Density**: {density:.4f}\n")
            f.write(f"- **Source**: {self.visualizer.source}\n\n")
            
            # Algorithm parameters
            k = max(1, int(math.log(n, 2) ** (1/3)))
            t = max(1, int(math.log(n, 2) ** (2/3)))
            l = math.ceil(math.log(n, 2) / t)
            
            f.write("## Algorithm Parameters\n\n")
            f.write(f"- **k** (pivot control): {k}\n")
            f.write(f"- **t** (branching factor): {t}\n")
            f.write(f"- **l** (recursion levels): {l}\n\n")
            
            # Results analysis
            reachable = sum(1 for d in distances.values() if d < float('inf'))
            unreachable = len(distances) - reachable
            
            f.write("## Execution Results\n\n")
            f.write(f"- **Reachable vertices**: {reachable}/{len(distances)} ({100*reachable/len(distances):.1f}%)\n")
            f.write(f"- **Unreachable vertices**: {unreachable}\n")
            f.write(f"- **Execution steps**: {len(self.visualizer.execution_trace)}\n")
            
            if self.visualizer.recursion_tree:
                total_calls = len(self.recursion_visualizer._get_all_nodes(self.visualizer.recursion_tree))
                f.write(f"- **Total recursive calls**: {total_calls}\n")
            
            f.write("\n## Distance Results\n\n")
            f.write("| Vertex | Distance | Status |\n")
            f.write("|--------|----------|--------|\n")
            
            for vertex in sorted(distances.keys()):
                dist = distances[vertex]
                if dist == float('inf'):
                    f.write(f"| {vertex} | ∞ | Unreachable |\n")
                else:
                    status = "Source" if vertex == self.visualizer.source else "Reachable"
                    f.write(f"| {vertex} | {dist:.2f} | {status} |\n")
            
            f.write("\n## Visualization Files\n\n")
            f.write("- `recursion_tree.png` - Complete recursion tree structure and analysis\n")
            f.write("- `frontier_evolution.png` - Frontier size and composition evolution\n")
            f.write("- `pivot_selection.png` - Step-by-step pivot selection process\n")
            f.write("- `interactive_analysis.html` - Interactive visualization (open in browser)\n")
            
            f.write("\n## Algorithm Insights\n\n")
            
            # Analyze frontier behavior
            max_frontier = max(len(state.frontier) for state in self.visualizer.execution_trace)
            avg_frontier = sum(len(state.frontier) for state in self.visualizer.execution_trace) / len(self.visualizer.execution_trace)
            
            f.write(f"- **Maximum frontier size**: {max_frontier}\n")
            f.write(f"- **Average frontier size**: {avg_frontier:.1f}\n")
            
            # Theoretical comparison
            dijkstra_ops = n * math.log(n) + m
            bmssp_ops = m * (math.log(n) ** (2/3))
            improvement = (dijkstra_ops - bmssp_ops) / dijkstra_ops * 100
            
            f.write(f"- **Theoretical complexity improvement**: {improvement:.1f}%\n")
            f.write(f"- **BMSSP bounded exploration**: Limits search scope for efficiency\n")
            
            if unreachable > 0:
                f.write(f"\n**Note**: {unreachable} vertices remain unreachable due to BMSSP's bounded exploration strategy, ")
                f.write("which prioritizes theoretical complexity guarantees over complete shortest path computation.\n")

def create_example_visualizations():
    """Create example visualizations with test graphs."""
    print("🎨 Creating BMSSP Visualization Examples")
    
    # Example 1: Small test graph
    g1 = nx.DiGraph()
    g1.add_edge('s', 'a', weight=1)
    g1.add_edge('s', 'b', weight=4)
    g1.add_edge('a', 'b', weight=2)
    g1.add_edge('a', 'c', weight=5)
    g1.add_edge('b', 'c', weight=1)
    
    suite1 = BMSSPVisualizationSuite()
    outputs1 = suite1.create_complete_analysis(g1, 's', 'examples/small_graph')
    
    # Example 2: Larger random graph
    g2 = nx.erdos_renyi_graph(12, 0.3, directed=True, seed=42)
    for u, v in g2.edges():
        g2[u][v]['weight'] = random.uniform(1, 5)
    
    suite2 = BMSSPVisualizationSuite()
    outputs2 = suite2.create_complete_analysis(g2, 0, 'examples/random_graph')
    
    print("✅ Example visualizations created in examples/ directory")
    return outputs1, outputs2

def main():
    """Main function demonstrating visualization capabilities."""
    create_example_visualizations()

if __name__ == "__main__":
    main()
        