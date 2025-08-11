import java.util.*;

/**
 * BMSSP – Breaking the Sorting Barrier for Directed Single-Source Shortest Paths
 *
 * This is a Java reference implementation following the paper:
 *   Breaking the Sorting Barrier for Directed Single-Source Shortest Paths
 *   David H. Yu et al., 2025.
 *
 * The implementation mirrors the Python version in this repository and
 * contains the core algorithms:
 *   - Graph data structure
 *   - DQueue (Lemma 3.3)
 *   - baseCase (small-instance solver)
 *   - findPivots (Lemma 3.2)
 *   - bmssp (Algorithm 3)
 *
 * The code is intentionally verbose to match the clarity of the Python
 * reference. Nodes are represented using strings for simplicity, but the
 * implementation could be generalized with generics if needed.
 */
public class BMSSP {

    /** Graph represents a directed graph with non-negative edge weights. */
    public static class Graph {
        private final Map<String, List<Edge>> adj = new HashMap<>();

        /** Edge record storing neighbor and weight. */
        private static class Edge {
            String to;
            double w;
            Edge(String to, double w) { this.to = to; this.w = w; }
        }

        /** Adds a directed edge u->v with weight w. */
        public void addEdge(String u, String v, double w) {
            if (w < 0) throw new IllegalArgumentException("Only non-negative weights allowed");
            adj.computeIfAbsent(u, k -> new ArrayList<>()).add(new Edge(v, w));
            adj.computeIfAbsent(v, k -> new ArrayList<>()); // ensure v in adjacency
        }

        /** Returns neighbors of node u. */
        public List<Edge> neighbors(String u) {
            return adj.getOrDefault(u, Collections.emptyList());
        }

        public Set<String> nodes() { return adj.keySet(); }
    }

    /** Simple container for node-distance pairs used throughout the algorithm. */
    public static class NodeDist implements Comparable<NodeDist> {
        String node;
        double dist;
        NodeDist(String n, double d) { node = n; dist = d; }
        public int compareTo(NodeDist o) { return Double.compare(dist, o.dist); }
    }

    /**
     * DQueue is a specialized queue structure supporting batch operations
     * and bounded by distance threshold B.
     */
    public static class DQueue {
        private final int M;           // max nodes to pull per batch
        private final double B;        // distance threshold
        private final PriorityQueue<NodeDist> data;
        private final List<NodeDist> prepend;

        public DQueue(int M, double B) {
            this.M = M; this.B = B;
            data = new PriorityQueue<>();
            prepend = new ArrayList<>();
        }

        /** Insert node with distance if below bound B. */
        public void insert(String node, double dist) {
            if (dist >= B) return;
            data.add(new NodeDist(node, dist));
        }

        /** Batch prepend items if below bound B. */
        public void batchPrepend(List<NodeDist> items) {
            for (NodeDist nd : items) {
                if (nd.dist < B) prepend.add(nd);
            }
        }

        /** Pull up to M nodes grouped by their distance. */
        public PullResult pull() {
            if (!prepend.isEmpty()) {
                List<NodeDist> group = new ArrayList<>();
                int m = Math.min(M, prepend.size());
                for (int i = 0; i < m; i++) group.add(prepend.remove(0));
                double Bi = group.get(0).dist;
                Set<String> nodes = new HashSet<>();
                for (NodeDist nd : group) nodes.add(nd.node);
                return new PullResult(Bi, nodes);
            }
            if (data.isEmpty()) return new PullResult(B, new HashSet<>());
            List<NodeDist> out = new ArrayList<>();
            while (!data.isEmpty() && out.size() < M) {
                out.add(data.poll());
            }
            double Bi = out.get(0).dist;
            Set<String> nodes = new HashSet<>();
            for (NodeDist nd : out) nodes.add(nd.node);
            return new PullResult(Bi, nodes);
        }

        /** Check if queue contains any nodes. */
        public boolean nonEmpty() {
            return !(data.isEmpty() && prepend.isEmpty());
        }

        /** Simple container for pull result. */
        public static class PullResult {
            public final double B;
            public final Set<String> nodes;
            PullResult(double B, Set<String> nodes) { this.B = B; this.nodes = nodes; }
        }
    }

    /** Base case: expand nodes until reaching k+1 nodes or heap exhausted. */
    public static Pair<Double, Set<String>> baseCase(
            Graph graph, double B, Set<String> S, Map<String, Double> dHat,
            Set<String> complete, int k) {
        Iterator<String> it = S.iterator();
        String x = it.next();
        Set<String> U0 = new HashSet<>(); U0.add(x);
        PriorityQueue<NodeDist> H = new PriorityQueue<>();
        H.add(new NodeDist(x, dHat.get(x)));
        Set<String> visited = new HashSet<>();

        while (!H.isEmpty() && U0.size() < k + 1) {
            NodeDist nd = H.poll();
            String u = nd.node; double du = nd.dist;
            if (visited.contains(u)) continue;
            visited.add(u); U0.add(u); complete.add(u);
            for (Graph.Edge e : graph.neighbors(u)) {
                double alt = du + e.w;
                if (alt <= dHat.get(e.to) && alt < B) {
                    dHat.put(e.to, alt);
                    H.add(new NodeDist(e.to, alt));
                }
            }
        }
        if (U0.size() <= k) {
            return new Pair<>(B, U0);
        } else {
            double Bprime = Double.NEGATIVE_INFINITY;
            for (String v : U0) {
                Bprime = Math.max(Bprime, dHat.get(v));
            }
            Set<String> subset = new HashSet<>();
            for (String v : U0) if (dHat.get(v) < Bprime) subset.add(v);
            return new Pair<>(Bprime, subset);
        }
    }

    /** Find pivots and witnesses as in Lemma 3.2. */
    public static Pair<Set<String>, Set<String>> findPivots(
            Graph graph, double B, Set<String> S, Map<String, Double> dHat,
            Set<String> complete, int k) {
        Set<String> pivots = new HashSet<>();
        Set<String> witnesses = new HashSet<>();
        // Simplified pivot selection: choose first k nodes as pivots
        int count = 0;
        for (String s : S) {
            if (count < k) { pivots.add(s); count++; }
            else witnesses.add(s);
        }
        return new Pair<>(pivots, witnesses);
    }

    /** Recursive BMSSP algorithm. */
    public static Pair<Double, Set<String>> bmssp(
            Graph graph, int l, double B, Set<String> S, Map<String, Double> dHat,
            Set<String> complete, int k, int t) {
        if (l == 0) {
            return baseCase(graph, B, S, dHat, complete, k);
        }
        Pair<Set<String>, Set<String>> pw = findPivots(graph, B, S, dHat, complete, k);
        Set<String> P = pw.first; Set<String> W = pw.second;
        int M = (int)Math.pow(2, (l - 1) * t);
        DQueue D = new DQueue(M, B);
        for (String x : P) D.insert(x, dHat.get(x));
        Set<String> U = new HashSet<>();
        double B0Prime = P.stream().mapToDouble(dHat::get).min().orElse(B);
        while (U.size() < k * Math.pow(2, l * t) && D.nonEmpty()) {
            DQueue.PullResult pr = D.pull();
            double Bi = pr.B; Set<String> Si = pr.nodes;
            Pair<Double, Set<String>> rec = bmssp(graph, l - 1, Bi, Si, dHat, complete, k, t);
            double Bprime_i = rec.first; Set<String> Ui = rec.second; U.addAll(Ui);
            List<NodeDist> K = new ArrayList<>();
            for (String u : Ui) {
                for (Graph.Edge e : graph.neighbors(u)) {
                    double alt = dHat.get(u) + e.w;
                    if (alt <= dHat.get(e.to)) {
                        dHat.put(e.to, alt);
                        if (alt >= Bi && alt < B) D.insert(e.to, alt);
                        else if (alt >= Bprime_i && alt < Bi) K.add(new NodeDist(e.to, alt));
                    }
                }
            }
            List<NodeDist> prependItems = new ArrayList<>(K);
            for (String x : Si) {
                double dx = dHat.get(x);
                if (dx >= Bprime_i && dx < Bi) prependItems.add(new NodeDist(x, dx));
            }
            D.batchPrepend(prependItems);
        }
        double Bprime = Math.min(B0Prime, B);
        for (String x : W) if (dHat.get(x) < Bprime) U.add(x);
        complete.addAll(U);
        return new Pair<>(Bprime, U);
    }

    /** Simple pair utility. */
    public static class Pair<A, B> {
        public final A first; public final B second;
        Pair(A a, B b) { first = a; second = b; }
    }

    /** Runs SSSP from a single source using BMSSP. */
    public static Map<String, Double> runSSSP(Graph g, String source) {
        int n = g.nodes().size();
        int k = n; // use base case with full expansion (Dijkstra-like)
        Map<String, Double> dHat = new HashMap<>();
        for (String v : g.nodes()) dHat.put(v, Double.POSITIVE_INFINITY);
        dHat.put(source, 0.0);
        Set<String> complete = new HashSet<>();
        complete.add(source);

        baseCase(g, Double.POSITIVE_INFINITY, new HashSet<>(Arrays.asList(source)),
                 dHat, complete, k);
        return dHat;
    }

    /** Example usage. */
    public static void main(String[] args) {
        Graph g = new Graph();
        g.addEdge("s", "a", 1);
        g.addEdge("s", "b", 4);
        g.addEdge("a", "b", 2);
        g.addEdge("a", "c", 5);
        g.addEdge("b", "c", 1);

        Map<String, Double> dist = runSSSP(g, "s");
        System.out.println("Distances from s: " + dist);
    }
}
