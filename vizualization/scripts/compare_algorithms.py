#!/usr/bin/env python3
"""
BMSSP vs Dijkstra Comparison Visualizer

This script creates comprehensive visual comparisons between BMSSP and Dijkstra's
algorithm, highlighting differences in exploration patterns, performance, and results.
"""

import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from bmssp_visualization import BMSSPVisualizationSuite, BMSSPVisualizer
except ImportError:
    print("Error: Could not import bmssp_visualization. Make sure it's in the parent directory.")
    sys.exit(1)

@dataclass
class AlgorithmResult:
    """Container for algorithm execution results."""
    algorithm_name: str
    distances: Dict[str, float]
    execution_time: float
    vertices_processed: int
    edges_relaxed: int
    max_frontier_size: int
    memory_usage: float
    complete_percentage: float

class DijkstraTracer:
    """Dijkstra implementation with execution tracing for comparison."""
    
    def __init__(self):
        self.vertices_processed = 0
        self.edges_relaxed = 0
        self.max_frontier_size = 0
        self.execution_steps = []
        
    def shortest_paths(self, graph: nx.DiGraph, source: str) -> Dict[str, float]:
        """Run Dijkstra with detailed tracing."""
        import heapq
        
        self.vertices_processed = 0
        self.edges_relaxed = 0
        self.max_frontier_size = 0
        self.execution_steps = []
        
        distances = {node: float('inf') for node in graph.nodes()}
        distances[source] = 0.0
        heap = [(0, source)]
        visited = set()
        
        step = 0
        while heap:
            current_dist, u = heapq.heappop(heap)
            
            if u in visited:
                continue
                
            visited.add(u)
            self.vertices_processed += 1
            self.max_frontier_size = max(self.max_frontier_size, len(heap))
            
            # Record execution step
            if step % 5 == 0:  # Record every 5th step
                self.execution_steps.append({
                    'step': step,
                    'current_vertex': u,
                    'current_distance': current_dist,
                    'visited_count': len(visited),
                    'frontier_size': len(heap),
                    'distances': distances.copy()
                })
            
            for v in graph.neighbors(u):
                weight = graph[u][v]['weight']
                new_dist = current_dist + weight
                self.edges_relaxed += 1
                
                if new_dist < distances[v]:
                    distances[v] = new_dist
                    heapq.heappush(heap, (new_dist, v))
            
            step += 1
        
        return distances

class AlgorithmComparator:
    """Main class for comparing BMSSP and Dijkstra algorithms."""
    
    def __init__(self):
        self.bmssp_visualizer = BMSSPVisualizer()
        self.dijkstra_tracer = DijkstraTracer()
        
    def run_comparison(self, graph: nx.DiGraph, source: str) -> Tuple[AlgorithmResult, AlgorithmResult]:
        """Run both algorithms and collect detailed metrics."""
        
        print(f"🔬 Running algorithm comparison on graph with {len(graph.nodes())} nodes, {len(graph.edges())} edges")
        
        # Run BMSSP
        print("   Running BMSSP...")
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        
        bmssp_distances = self.bmssp_visualizer.trace_execution(graph, source)
        
        bmssp_time = time.perf_counter() - start_time
        bmssp_memory = self._get_memory_usage() - start_memory
        
        # Calculate BMSSP metrics
        bmssp_reachable = sum(1 for d in bmssp_distances.values() if d < float('inf'))
        bmssp_complete_pct = bmssp_reachable / len(bmssp_distances) * 100
        
        # Extract metrics from trace
        bmssp_vertices_processed = sum(1 for state in self.bmssp_visualizer.execution_trace 
                                     if 'processed' in state.step_description.lower())
        bmssp_max_frontier = max(len(state.frontier) for state in self.bmssp_visualizer.execution_trace)
        
        bmssp_result = AlgorithmResult(
            algorithm_name="BMSSP",
            distances=bmssp_distances,
            execution_time=bmssp_time,
            vertices_processed=bmssp_vertices_processed,
            edges_relaxed=0,  # Not directly tracked in current implementation
            max_frontier_size=bmssp_max_frontier,
            memory_usage=max(0, bmssp_memory),
            complete_percentage=bmssp_complete_pct
        )
        
        # Run Dijkstra
        print("   Running Dijkstra...")
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        
        dijkstra_distances = self.dijkstra_tracer.shortest_paths(graph, source)
        
        dijkstra_time = time.perf_counter() - start_time
        dijkstra_memory = self._get_memory_usage() - start_memory
        
        dijkstra_reachable = sum(1 for d in dijkstra_distances.values() if d < float('inf'))
        dijkstra_complete_pct = dijkstra_reachable / len(dijkstra_distances) * 100
        
        dijkstra_result = AlgorithmResult(
            algorithm_name="Dijkstra",
            distances=dijkstra_distances,
            execution_time=dijkstra_time,
            vertices_processed=self.dijkstra_tracer.vertices_processed,
            edges_relaxed=self.dijkstra_tracer.edges_relaxed,
            max_frontier_size=self.dijkstra_tracer.max_frontier_size,
            memory_usage=max(0, dijkstra_memory),
            complete_percentage=dijkstra_complete_pct
        )
        
        print(f"   ✅ BMSSP: {bmssp_time:.3f}s, {bmssp_complete_pct:.1f}% complete")
        print(f"   ✅ Dijkstra: {dijkstra_time:.3f}s, {dijkstra_complete_pct:.1f}% complete")
        
        return bmssp_result, dijkstra_result
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def create_comparison_visualization(self, graph: nx.DiGraph, source: str,
                                      bmssp_result: AlgorithmResult, 
                                      dijkstra_result: AlgorithmResult,
                                      output_dir: str = "comparison_analysis") -> Dict[str, str]:
        """Generate comprehensive comparison visualizations."""
        
        os.makedirs(output_dir, exist_ok=True)
        output_files = {}
        
        # 1. Performance metrics comparison
        print("📊 Generating performance comparison...")
        perf_path = os.path.join(output_dir, "performance_comparison.png")
        self._plot_performance_comparison(bmssp_result, dijkstra_result, perf_path)
        output_files['performance'] = perf_path
        
        # 2. Distance results comparison
        print("📏 Generating distance comparison...")
        dist_path = os.path.join(output_dir, "distance_comparison.png")
        self._plot_distance_comparison(bmssp_result, dijkstra_result, dist_path)
        output_files['distances'] = dist_path
        
        # 3. Graph state comparison
        print("🗺️ Generating graph state comparison...")
        graph_path = os.path.join(output_dir, "graph_state_comparison.png")
        self._plot_graph_state_comparison(graph, source, bmssp_result, dijkstra_result, graph_path)
        output_files['graph_states'] = graph_path
        
        # 4. Execution timeline comparison
        print("⏱️ Generating execution timeline...")
        timeline_path = os.path.join(output_dir, "execution_timeline.png")
        self._plot_execution_timeline(timeline_path)
        output_files['timeline'] = timeline_path
        
        # 5. Generate comparison report
        print("📝 Generating comparison report...")
        report_path = os.path.join(output_dir, "comparison_report.md")
        self._generate_comparison_report(graph, source, bmssp_result, dijkstra_result, report_path)
        output_files['report'] = report_path
        
        return output_files
    
    def _plot_performance_comparison(self, bmssp: AlgorithmResult, dijkstra: AlgorithmResult, save_path: str):
        """Create performance metrics comparison chart."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        algorithms = ['BMSSP', 'Dijkstra']
        
        # Runtime comparison
        runtimes = [bmssp.execution_time * 1000, dijkstra.execution_time * 1000]  # Convert to ms
        bars1 = ax1.bar(algorithms, runtimes, color=['orange', 'blue'], alpha=0.7)
        ax1.set_ylabel('Execution Time (ms)')
        ax1.set_title('Runtime Comparison')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, runtime in zip(bars1, runtimes):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(runtimes)*0.01,
                    f'{runtime:.1f}ms', ha='center', va='bottom')
        
        # Memory usage comparison
        memories = [bmssp.memory_usage, dijkstra.memory_usage]
        bars2 = ax2.bar(algorithms, memories, color=['orange', 'blue'], alpha=0.7)
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Usage Comparison')
        ax2.grid(True, alpha=0.3)
        
        for bar, memory in zip(bars2, memories):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(memories)*0.01,
                    f'{memory:.1f}MB', ha='center', va='bottom')
        
        # Completeness comparison
        completeness = [bmssp.complete_percentage, dijkstra.complete_percentage]
        bars3 = ax3.bar(algorithms, completeness, color=['orange', 'blue'], alpha=0.7)
        ax3.set_ylabel('Reachable Vertices (%)')
        ax3.set_title('Completeness Comparison')
        ax3.set_ylim(0, 110)
        ax3.grid(True, alpha=0.3)
        
        for bar, comp in zip(bars3, completeness):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{comp:.1f}%', ha='center', va='bottom')
        
        # Frontier size comparison
        frontiers = [bmssp.max_frontier_size, dijkstra.max_frontier_size]
        bars4 = ax4.bar(algorithms, frontiers, color=['orange', 'blue'], alpha=0.7)
        ax4.set_ylabel('Max Frontier Size')
        ax4.set_title('Maximum Frontier Size')
        ax4.grid(True, alpha=0.3)
        
        for bar, frontier in zip(bars4, frontiers):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(frontiers)*0.01,
                    f'{frontier}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_distance_comparison(self, bmssp: AlgorithmResult, dijkstra: AlgorithmResult, save_path: str):
        """Compare distance results between algorithms."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Get all vertices
        all_vertices = sorted(set(bmssp.distances.keys()) | set(dijkstra.distances.keys()))
        
        # Distance comparison scatter plot
        bmssp_dists = [bmssp.distances.get(v, float('inf')) for v in all_vertices]
        dijkstra_dists = [dijkstra.distances.get(v, float('inf')) for v in all_vertices]
        
        # Filter out infinite distances for scatter plot
        finite_indices = [i for i in range(len(bmssp_dists)) 
                         if bmssp_dists[i] < float('inf') and dijkstra_dists[i] < float('inf')]
        
        if finite_indices:
            finite_bmssp = [bmssp_dists[i] for i in finite_indices]
            finite_dijkstra = [dijkstra_dists[i] for i in finite_indices]
            
            ax1.scatter(finite_dijkstra, finite_bmssp, alpha=0.7)
            
            # Add diagonal line (perfect agreement)
            max_dist = max(max(finite_dijkstra), max(finite_bmssp))
            ax1.plot([0, max_dist], [0, max_dist], 'r--', alpha=0.5, label='Perfect Agreement')
            
            ax1.set_xlabel('Dijkstra Distance')
            ax1.set_ylabel('BMSSP Distance')
            ax1.set_title('Distance Comparison (Finite Distances)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No overlapping finite distances', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Distance Comparison (No Overlap)')
        
        # Reachability comparison
        bmssp_reachable = set(v for v, d in bmssp.distances.items() if d < float('inf'))
        dijkstra_reachable = set(v for v, d in dijkstra.distances.items() if d < float('inf'))
        
        only_bmssp = bmssp_reachable - dijkstra_reachable
        only_dijkstra = dijkstra_reachable - bmssp_reachable
        both = bmssp_reachable & dijkstra_reachable
        
        reachability_data = [len(both), len(only_bmssp), len(only_dijkstra)]
        labels = ['Both Algorithms', 'Only BMSSP', 'Only Dijkstra']
        colors = ['green', 'orange', 'blue']
        
        ax2.pie(reachability_data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Reachability Overlap')
        
        # Distance distribution histograms
        bmssp_finite = [d for d in bmssp_dists if d < float('inf')]
        dijkstra_finite = [d for d in dijkstra_dists if d < float('inf')]
        
        if bmssp_finite:
            ax3.hist(bmssp_finite, bins=20, alpha=0.7, label='BMSSP', color='orange', density=True)
        if dijkstra_finite:
            ax3.hist(dijkstra_finite, bins=20, alpha=0.7, label='Dijkstra', color='blue', density=True)
        
        ax3.set_xlabel('Distance')
        ax3.set_ylabel('Density')
        ax3.set_title('Distance Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Distance difference analysis
        differences = []
        difference_vertices = []
        
        for v in all_vertices:
            b_dist = bmssp.distances.get(v, float('inf'))
            d_dist = dijkstra.distances.get(v, float('inf'))
            
            if b_dist < float('inf') and d_dist < float('inf'):
                diff = abs(b_dist - d_dist)
                differences.append(diff)
                difference_vertices.append(v)
        
        if differences:
            ax4.hist(differences, bins=20, alpha=0.7, color='purple')
            ax4.set_xlabel('|BMSSP Distance - Dijkstra Distance|')
            ax4.set_ylabel('Count')
            ax4.set_title('Distance Difference Distribution')
            ax4.grid(True, alpha=0.3)
            
            # Add statistics
            mean_diff = np.mean(differences)
            max_diff = max(differences)
            ax4.axvline(mean_diff, color='red', linestyle='--', 
                       label=f'Mean: {mean_diff:.3f}')
            ax4.text(0.7, 0.8, f'Max diff: {max_diff:.3f}', 
                    transform=ax4.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No comparable distances', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('No Distance Differences')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_graph_state_comparison(self, graph: nx.DiGraph, source: str,
                                   bmssp: AlgorithmResult, dijkstra: AlgorithmResult, save_path: str):
        """Compare graph states showing reachability differences."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Consistent layout for both plots
        pos = nx.spring_layout(graph, seed=42, k=2, iterations=50)
        
        # BMSSP graph state
        bmssp_colors = []
        bmssp_sizes = []
        
        for node in graph.nodes():
            dist = bmssp.distances.get(node, float('inf'))
            if node == source:
                color, size = 'red', 400
            elif dist < float('inf'):
                # Color by distance
                max_finite_dist = max(d for d in bmssp.distances.values() if d < float('inf'))
                intensity = 1 - (dist / max_finite_dist) if max_finite_dist > 0 else 1
                color = plt.cm.Blues(0.3 + 0.7 * intensity)
                size = 200 + 100 * intensity
            else:
                color, size = 'lightgray', 100
            
            bmssp_colors.append(color)
            bmssp_sizes.append(size)
        
        nx.draw_networkx_nodes(graph, pos, node_color=bmssp_colors, node_size=bmssp_sizes,
                              ax=ax1, edgecolors='black', linewidths=1)
        nx.draw_networkx_edges(graph, pos, ax=ax1, edge_color='gray', arrows=True, 
                              arrowsize=20, alpha=0.6)
        
        # Labels with distances
        bmssp_labels = {}
        for node in graph.nodes():
            dist = bmssp.distances.get(node, float('inf'))
            if dist == float('inf'):
                bmssp_labels[node] = f"{node}\n∞"
            else:
                bmssp_labels[node] = f"{node}\n{dist:.1f}"
        
        nx.draw_networkx_labels(graph, pos, bmssp_labels, ax=ax1, font_size=8)
        ax1.set_title(f"BMSSP Results\n({bmssp.complete_percentage:.1f}% reachable)", fontsize=14)
        ax1.set_aspect('equal')
        
        # Dijkstra graph state
        dijkstra_colors = []
        dijkstra_sizes = []
        
        for node in graph.nodes():
            dist = dijkstra.distances.get(node, float('inf'))
            if node == source:
                color, size = 'red', 400
            elif dist < float('inf'):
                # Color by distance
                max_finite_dist = max(d for d in dijkstra.distances.values() if d < float('inf'))
                intensity = 1 - (dist / max_finite_dist) if max_finite_dist > 0 else 1
                color = plt.cm.Greens(0.3 + 0.7 * intensity)
                size = 200 + 100 * intensity
            else:
                color, size = 'lightgray', 100
            
            dijkstra_colors.append(color)
            dijkstra_sizes.append(size)
        
        nx.draw_networkx_nodes(graph, pos, node_color=dijkstra_colors, node_size=dijkstra_sizes,
                              ax=ax2, edgecolors='black', linewidths=1)
        nx.draw_networkx_edges(graph, pos, ax=ax2, edge_color='gray', arrows=True,
                              arrowsize=20, alpha=0.6)
        
        # Labels with distances
        dijkstra_labels = {}
        for node in graph.nodes():
            dist = dijkstra.distances.get(node, float('inf'))
            if dist == float('inf'):
                dijkstra_labels[node] = f"{node}\n∞"
            else:
                dijkstra_labels[node] = f"{node}\n{dist:.1f}"
        
        nx.draw_networkx_labels(graph, pos, dijkstra_labels, ax=ax2, font_size=8)
        ax2.set_title(f"Dijkstra Results\n({dijkstra.complete_percentage:.1f}% reachable)", fontsize=14)
        ax2.set_aspect('equal')
        
        # Add edge weights
        edge_labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels, ax=ax1, font_size=6)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels, ax=ax2, font_size=6)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_execution_timeline(self, save_path: str):
        """Plot execution timeline comparison."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # BMSSP execution timeline
        if self.bmssp_visualizer.execution_trace:
            steps = range(len(self.bmssp_visualizer.execution_trace))
            frontier_sizes = [len(state.frontier) for state in self.bmssp_visualizer.execution_trace]
            complete_sizes = [len(state.complete) for state in self.bmssp_visualizer.execution_trace]
            
            ax1.plot(steps, frontier_sizes, 'b-', linewidth=2, label='Frontier Size')
            ax1.plot(steps, complete_sizes, 'g-', linewidth=2, label='Complete Vertices')
            
            # Highlight major phases
            findpivots_steps = [i for i, state in enumerate(self.bmssp_visualizer.execution_trace)
                               if 'FindPivots' in state.step_description]
            for step in findpivots_steps:
                ax1.axvline(x=step, color='orange', alpha=0.5, linestyle=':', label='FindPivots' if step == findpivots_steps[0] else '')
            
            ax1.set_xlabel('Execution Step')
            ax1.set_ylabel('Vertex Count')
            ax1.set_title('BMSSP Execution Timeline')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No BMSSP execution trace available', 
                    ha='center', va='center', transform=ax1.transAxes)
        
        # Dijkstra execution timeline
        if self.dijkstra_tracer.execution_steps:
            steps = [step['step'] for step in self.dijkstra_tracer.execution_steps]
            visited_counts = [step['visited_count'] for step in self.dijkstra_tracer.execution_steps]
            frontier_sizes = [step['frontier_size'] for step in self.dijkstra_tracer.execution_steps]
            
            ax2.plot(steps, visited_counts, 'g-', linewidth=2, label='Visited Vertices')
            ax2.plot(steps, frontier_sizes, 'b-', linewidth=2, label='Frontier Size')
            
            ax2.set_xlabel('Execution Step')
            ax2.set_ylabel('Vertex Count')
            ax2.set_title('Dijkstra Execution Timeline')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No Dijkstra execution trace available', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_comparison_report(self, graph: nx.DiGraph, source: str,
                                  bmssp: AlgorithmResult, dijkstra: AlgorithmResult, report_path: str):
        """Generate comprehensive comparison report."""
        with open(report_path, 'w') as f:
            f.write("# BMSSP vs Dijkstra Comparison Report\n\n")
            f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Graph statistics
            f.write("## Graph Statistics\n\n")
            f.write(f"- **Vertices**: {len(graph.nodes())}\n")
            f.write(f"- **Edges**: {len(graph.edges())}\n")
            f.write(f"- **Density**: {len(graph.edges()) / (len(graph.nodes()) * (len(graph.nodes()) - 1)):.4f}\n")
            f.write(f"- **Source**: {source}\n\n")
            
            # Performance comparison
            f.write("## Performance Comparison\n\n")
            f.write("| Metric | BMSSP | Dijkstra | Difference |\n")
            f.write("|--------|-------|----------|------------|\n")
            
            runtime_diff = (bmssp.execution_time - dijkstra.execution_time) / dijkstra.execution_time * 100
            f.write(f"| Runtime (ms) | {bmssp.execution_time*1000:.1f} | {dijkstra.execution_time*1000:.1f} | {runtime_diff:+.1f}% |\n")
            
            memory_diff = (bmssp.memory_usage - dijkstra.memory_usage) / max(dijkstra.memory_usage, 1) * 100
            f.write(f"| Memory (MB) | {bmssp.memory_usage:.1f} | {dijkstra.memory_usage:.1f} | {memory_diff:+.1f}% |\n")
            
            completeness_diff = bmssp.complete_percentage - dijkstra.complete_percentage
            f.write(f"| Completeness (%) | {bmssp.complete_percentage:.1f} | {dijkstra.complete_percentage:.1f} | {completeness_diff:+.1f}pp |\n")
            
            frontier_diff = (bmssp.max_frontier_size - dijkstra.max_frontier_size) / max(dijkstra.max_frontier_size, 1) * 100
            f.write(f"| Max Frontier | {bmssp.max_frontier_size} | {dijkstra.max_frontier_size} | {frontier_diff:+.1f}% |\n\n")
            
            # Reachability analysis
            bmssp_reachable = set(v for v, d in bmssp.distances.items() if d < float('inf'))
            dijkstra_reachable = set(v for v, d in dijkstra.distances.items() if d < float('inf'))
            
            only_bmssp = bmssp_reachable - dijkstra_reachable
            only_dijkstra = dijkstra_reachable - bmssp_reachable
            both = bmssp_reachable & dijkstra_reachable
            
            f.write("## Reachability Analysis\n\n")
            f.write(f"- **Both algorithms reach**: {len(both)} vertices\n")
            f.write(f"- **Only BMSSP reaches**: {len(only_bmssp)} vertices\n")
            f.write(f"- **Only Dijkstra reaches**: {len(only_dijkstra)} vertices\n\n")
            
            if only_bmssp:
                f.write(f"**Vertices only reachable by BMSSP**: {', '.join(sorted(only_bmssp))}\n\n")
            if only_dijkstra:
                f.write(f"**Vertices only reachable by Dijkstra**: {', '.join(sorted(only_dijkstra))}\n\n")
            
            # Distance comparison
            f.write("## Distance Comparison\n\n")
            f.write("| Vertex | BMSSP Distance | Dijkstra Distance | Difference |\n")
            f.write("|--------|----------------|-------------------|------------|\n")
            
            all_vertices = sorted(set(bmssp.distances.keys()) | set(dijkstra.distances.keys()))
            for vertex in all_vertices:
                b_dist = bmssp.distances.get(vertex, float('inf'))
                d_dist = dijkstra.distances.get(vertex, float('inf'))
                
                if b_dist == float('inf') and d_dist == float('inf'):
                    diff_str = "Both ∞"
                elif b_dist == float('inf'):
                    diff_str = "BMSSP ∞"
                elif d_dist == float('inf'):
                    diff_str = "Dijkstra ∞"
                else:
                    diff = b_dist - d_dist
                    diff_str = f"{diff:+.3f}"
                
                b_str = "∞" if b_dist == float('inf') else f"{b_dist:.3f}"
                d_str = "∞" if d_dist == float('inf') else f"{d_dist:.3f}"
                
                f.write(f"| {vertex} | {b_str} | {d_str} | {diff_str} |\n")
            
            f.write("\n## Algorithm Insights\n\n")
            
            # BMSSP characteristics
            f.write("### BMSSP Characteristics\n")
            n = len(graph.nodes())
            k = max(1, int(math.log(n, 2) ** (1/3)))
            t = max(1, int(math.log(n, 2) ** (2/3)))
            l = math.ceil(math.log(n, 2) / t)
            
            f.write(f"- **Parameters**: k={k}, t={t}, l={l}\n")
            f.write(f"- **Bounded exploration**: Limits search for theoretical efficiency\n")
            f.write(f"- **Frontier reduction**: Uses pivot selection to reduce working set\n")
            f.write(f"- **Recursion levels**: {l} levels of divide-and-conquer\n\n")
            
            # Dijkstra characteristics
            f.write("### Dijkstra Characteristics\n")
            f.write(f"- **Complete exploration**: Guarantees optimal distances to all reachable vertices\n")
            f.write(f"- **Priority queue**: Maintains total order of frontier vertices\n")
            f.write(f"- **Single-source**: Explores from source in distance order\n")
            f.write(f"- **Optimal for completeness**: Finds shortest paths when they exist\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            if runtime_diff < -10:
                f.write("- **BMSSP shows performance advantage** for this graph size/structure\n")
            elif runtime_diff > 50:
                f.write("- **Dijkstra is significantly faster** due to BMSSP overhead on small graphs\n")
            else:
                f.write("- **Similar performance** between algorithms for this graph\n")
            
            if completeness_diff < -10:
                f.write("- **Dijkstra finds significantly more paths** - consider if completeness is required\n")
            elif completeness_diff > 10:
                f.write("- **BMSSP achieves good completeness** while maintaining efficiency benefits\n")
            
            f.write("\n## Visualization Files\n\n")
            f.write("- `performance_comparison.png` - Runtime, memory, and efficiency metrics\n")
            f.write("- `distance_comparison.png` - Distance results and reachability analysis\n") 
            f.write("- `graph_state_comparison.png` - Side-by-side graph visualization\n")
            f.write("- `execution_timeline.png` - Algorithm execution progression\n")

def create_comparison_examples():
    """Create standard comparison examples."""
    
    output_dir = Path("algorithm_comparisons")
    output_dir.mkdir(exist_ok=True)
    
    # Test graphs of different types and sizes
    test_cases = [
        {
            'name': 'small_path',
            'description': 'Small path graph - overhead analysis',
            'graph': create_path_graph(6),
            'source': 'v0'
        },
        {
            'name': 'medium_random',
            'description': 'Medium random graph - balanced comparison',
            'graph': create_random_graph(15),
            'source': 0
        },
        {
            'name': 'layered_structure',
            'description': 'Layered graph - frontier evolution comparison',
            'graph': create_layered_graph(4, 4),
            'source': 'L0_N0'
        },
        {
            'name': 'scale_free',
            'description': 'Scale-free network - hub effect analysis',
            'graph': create_scale_free_graph(20),
            'source': 0
        }
    ]
    
    print("🔬 Creating BMSSP vs Dijkstra Comparison Examples")
    print("=" * 60)
    
    comparator = AlgorithmComparator()
    all_results = {}
    
    for test_case in test_cases:
        print(f"\n📊 Running comparison: {test_case['name']}")
        print(f"   Description: {test_case['description']}")
        
        try:
            # Run algorithm comparison
            bmssp_result, dijkstra_result = comparator.run_comparison(
                test_case['graph'], test_case['source']
            )
            
            # Generate visualizations
            case_dir = output_dir / test_case['name']
            outputs = comparator.create_comparison_visualization(
                test_case['graph'], test_case['source'],
                bmssp_result, dijkstra_result, str(case_dir)
            )
            
            all_results[test_case['name']] = {
                'bmssp': bmssp_result,
                'dijkstra': dijkstra_result,
                'outputs': outputs
            }
            
            print(f"   ✅ Generated {len(outputs)} comparison files")
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
            continue
    
    # Create summary comparison
    create_summary_comparison(output_dir, all_results)
    
    print(f"\n🎯 Algorithm comparison complete!")
    print(f"📁 Results saved to {output_dir}/")
    print(f"📄 See {output_dir}/summary_comparison.md for overview")

def create_summary_comparison(output_dir: Path, all_results: Dict):
    """Create summary comparison across all test cases."""
    
    summary_path = output_dir / "summary_comparison.md"
    
    with open(summary_path, 'w') as f:
        f.write("# BMSSP vs Dijkstra Summary Comparison\n\n")
        f.write("This document summarizes algorithm comparisons across multiple graph types.\n\n")
        
        f.write("## Performance Summary\n\n")
        f.write("| Graph Type | BMSSP Time (ms) | Dijkstra Time (ms) | Speedup | BMSSP Completeness | Dijkstra Completeness |\n")
        f.write("|------------|----------------|-------------------|---------|--------------------|-----------------------|\n")
        
        for name, results in all_results.items():
            bmssp = results['bmssp']
            dijkstra = results['dijkstra']
            speedup = dijkstra.execution_time / bmssp.execution_time
            
            f.write(f"| {name.replace('_', ' ').title()} | ")
            f.write(f"{bmssp.execution_time*1000:.1f} | ")
            f.write(f"{dijkstra.execution_time*1000:.1f} | ")
            f.write(f"{speedup:.2f}x | ")
            f.write(f"{bmssp.complete_percentage:.1f}% | ")
            f.write(f"{dijkstra.complete_percentage:.1f}% |\n")
        
        f.write("\n## Key Findings\n\n")
        
        # Find cases where BMSSP is faster
        faster_cases = [name for name, results in all_results.items()
                       if results['bmssp'].execution_time < results['dijkstra'].execution_time]
        
        if faster_cases:
            f.write(f"### BMSSP Performance Advantages\n")
            f.write(f"BMSSP outperformed Dijkstra on: {', '.join(faster_cases)}\n\n")
        
        # Find cases with significant completeness differences
        completeness_diff_cases = [(name, results['dijkstra'].complete_percentage - results['bmssp'].complete_percentage)
                                  for name, results in all_results.items()]
        
        high_diff_cases = [name for name, diff in completeness_diff_cases if diff > 10]
        
        if high_diff_cases:
            f.write(f"### Completeness Trade-offs\n")
            f.write(f"Significant completeness differences observed in: {', '.join(high_diff_cases)}\n")
            f.write("This demonstrates BMSSP's bounded exploration strategy.\n\n")
        
        f.write("## Individual Analysis\n\n")
        for name in all_results.keys():
            f.write(f"### {name.replace('_', ' ').title()}\n")
            f.write(f"- **Analysis**: `{name}/comparison_report.md`\n")
            f.write(f"- **Visualizations**: `{name}/` directory\n\n")

# Helper functions for creating test graphs
def create_path_graph(n: int) -> nx.DiGraph:
    """Create a simple path graph."""
    G = nx.DiGraph()
    for i in range(n-1):
        G.add_edge(f'v{i}', f'v{i+1}', weight=random.uniform(1, 3))
    return G

def create_random_graph(n: int) -> nx.DiGraph:
    """Create a random graph."""
    G = nx.erdos_renyi_graph(n, 0.3, directed=True, seed=42)
    for u, v in G.edges():
        G[u][v]['weight'] = random.uniform(1, 5)
    return G

def create_layered_graph(layers: int, nodes_per_layer: int) -> nx.DiGraph:
    """Create a layered graph."""
    G = nx.DiGraph()
    
    # Create layers
    for layer in range(layers):
        for node in range(nodes_per_layer):
            G.add_node(f'L{layer}_N{node}')
    
    # Connect layers
    for layer in range(layers - 1):
        for i in range(nodes_per_layer):
            for j in range(min(2, nodes_per_layer)):  # Connect to at most 2 nodes
                G.add_edge(f'L{layer}_N{i}', f'L{layer+1}_N{j}', 
                          weight=random.uniform(1, 4))
    
    return G

def create_scale_free_graph(n: int) -> nx.DiGraph:
    """Create a scale-free graph."""
    G = nx.scale_free_graph(n, alpha=0.4, beta=0.5, gamma=0.1, seed=42)
    DG = nx.DiGraph()
    for u, v in G.edges():
        DG.add_edge(u, v, weight=random.uniform(1, 6))
    return DG

def main():
    """Main function to create algorithm comparisons."""
    import random
    random.seed(42)  # For reproducible results
    
    try:
        create_comparison_examples()
        
        print("\n" + "="*60)
        print("🎯 Next Steps:")
        print("   1. Open algorithm_comparisons/summary_comparison.md for overview")
        print("   2. Examine individual case directories for detailed analysis")
        print("   3. Compare performance patterns across different graph types")
        print("   4. Use insights to understand when BMSSP provides advantages")
        
    except KeyboardInterrupt:
        print("\n🛑 Comparison interrupted by user")
    except Exception as e:
        print(f"\n❌ Comparison failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()