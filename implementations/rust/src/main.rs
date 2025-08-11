use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

#[derive(Clone)]
struct Edge {
    to: String,
    w: f64,
}

struct Graph {
    adj: HashMap<String, Vec<Edge>>,
}

impl Graph {
    fn new() -> Self {
        Self { adj: HashMap::new() }
    }
    fn add_edge(&mut self, u: &str, v: &str, w: f64) {
        if w < 0.0 {
            panic!("Only non-negative weights allowed");
        }
        self.adj
            .entry(u.to_string())
            .or_default()
            .push(Edge { to: v.to_string(), w });
        self.adj.entry(v.to_string()).or_default();
    }
    fn neighbors<'a>(&'a self, u: &str) -> std::slice::Iter<'a, Edge> {
        if let Some(v) = self.adj.get(u) {
            v.iter()
        } else {
            [].iter()
        }
    }
    fn nodes(&self) -> Vec<String> {
        self.adj.keys().cloned().collect()
    }
}

#[derive(Clone)]
struct NodeDist {
    node: String,
    dist: f64,
}

impl PartialEq for NodeDist {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.node == other.node
    }
}

impl Eq for NodeDist {}

impl Ord for NodeDist {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for NodeDist {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct DQueue {
    m: usize,
    b: f64,
    data: BinaryHeap<NodeDist>,
    prepend: Vec<NodeDist>,
}

impl DQueue {
    fn new(m: usize, b: f64) -> Self {
        Self {
            m,
            b,
            data: BinaryHeap::new(),
            prepend: Vec::new(),
        }
    }
    fn insert(&mut self, node: String, dist: f64) {
        if dist < self.b {
            self.data.push(NodeDist { node, dist });
        }
    }
    fn batch_prepend(&mut self, items: Vec<NodeDist>) {
        for nd in items {
            if nd.dist < self.b {
                self.prepend.push(nd);
            }
        }
    }
    fn pull(&mut self) -> PullResult {
        if !self.prepend.is_empty() {
            let m = self.m.min(self.prepend.len());
            let group: Vec<NodeDist> = self.prepend.drain(0..m).collect();
            let bi = group[0].dist;
            let nodes = group.into_iter().map(|nd| nd.node).collect();
            return PullResult { b: bi, nodes };
        }
        if self.data.is_empty() {
            return PullResult {
                b: self.b,
                nodes: HashSet::new(),
            };
        }
        let mut out = Vec::new();
        for _ in 0..self.m {
            if let Some(nd) = self.data.pop() {
                out.push(nd);
            } else {
                break;
            }
        }
        let bi = out[0].dist;
        let nodes = out.into_iter().map(|nd| nd.node).collect();
        PullResult { b: bi, nodes }
    }
    fn non_empty(&self) -> bool {
        !(self.data.is_empty() && self.prepend.is_empty())
    }
}

struct PullResult {
    b: f64,
    nodes: HashSet<String>,
}

fn base_case(
    graph: &Graph,
    b: f64,
    s: &HashSet<String>,
    d_hat: &mut HashMap<String, f64>,
    complete: &mut HashSet<String>,
    k: usize,
) -> (f64, HashSet<String>) {
    let x = s.iter().next().unwrap().clone();
    let mut u0 = HashSet::new();
    u0.insert(x.clone());
    let mut h = BinaryHeap::new();
    h.push(NodeDist {
        node: x.clone(),
        dist: *d_hat.get(&x).unwrap(),
    });
    let mut visited = HashSet::new();
    while let Some(NodeDist { node: u, dist: du }) = h.pop() {
        if visited.contains(&u) {
            continue;
        }
        visited.insert(u.clone());
        u0.insert(u.clone());
        complete.insert(u.clone());
        for e in graph.neighbors(&u) {
            let alt = du + e.w;
            if alt <= *d_hat.get(&e.to).unwrap() && alt < b {
                d_hat.insert(e.to.clone(), alt);
                h.push(NodeDist {
                    node: e.to.clone(),
                    dist: alt,
                });
            }
        }
        if u0.len() >= k + 1 {
            break;
        }
    }
    if u0.len() <= k {
        (b, u0)
    } else {
        let mut b_prime = f64::NEG_INFINITY;
        for v in &u0 {
            b_prime = b_prime.max(*d_hat.get(v).unwrap());
        }
        let subset = u0
            .into_iter()
            .filter(|v| d_hat.get(v).unwrap() < &b_prime)
            .collect();
        (b_prime, subset)
    }
}

fn find_pivots(
    _graph: &Graph,
    _b: f64,
    s: &HashSet<String>,
    _d_hat: &HashMap<String, f64>,
    _complete: &HashSet<String>,
    k: usize,
) -> (HashSet<String>, HashSet<String>) {
    let mut pivots = HashSet::new();
    let mut witnesses = HashSet::new();
    let mut count = 0;
    for x in s {
        if count < k {
            pivots.insert(x.clone());
        } else {
            witnesses.insert(x.clone());
        }
        count += 1;
    }
    (pivots, witnesses)
}

fn bmssp(
    graph: &Graph,
    l: usize,
    b: f64,
    s: &HashSet<String>,
    d_hat: &mut HashMap<String, f64>,
    complete: &mut HashSet<String>,
    k: usize,
    t: usize,
) -> (f64, HashSet<String>) {
    if l == 0 {
        return base_case(graph, b, s, d_hat, complete, k);
    }
    let (p, w) = find_pivots(graph, b, s, d_hat, complete, k);
    let m = 2usize.pow(((l - 1) * t) as u32);
    let mut dq = DQueue::new(m, b);
    for x in &p {
        let dx = *d_hat.get(x).unwrap();
        dq.insert(x.clone(), dx);
    }
    let mut u = HashSet::new();
    let b0_prime = p.iter().map(|x| d_hat.get(x).cloned().unwrap()).fold(b, f64::min);
    while (u.len() as f64) < (k as f64) * (2usize.pow((l * t) as u32) as f64) && dq.non_empty() {
        let PullResult { b: bi, nodes: si } = dq.pull();
        let (b_prime_i, ui) = bmssp(graph, l - 1, bi, &si, d_hat, complete, k, t);
        u.extend(ui.iter().cloned());
        let mut k_vec = Vec::new();
        for node in &ui {
            let du = d_hat.get(node).cloned().unwrap();
            for e in graph.neighbors(node) {
                let alt = du + e.w;
                let cur = d_hat.get(&e.to).cloned().unwrap_or(f64::INFINITY);
                if alt <= cur {
                    d_hat.insert(e.to.clone(), alt);
                    if alt >= bi && alt < b {
                        dq.insert(e.to.clone(), alt);
                    } else if alt >= b_prime_i && alt < bi {
                        k_vec.push(NodeDist {
                            node: e.to.clone(),
                            dist: alt,
                        });
                    }
                }
            }
        }
        let mut prepend_items = k_vec;
        for x in &si {
            let dx = *d_hat.get(x).unwrap();
            if dx >= b_prime_i && dx < bi {
                prepend_items.push(NodeDist {
                    node: x.clone(),
                    dist: dx,
                });
            }
        }
        dq.batch_prepend(prepend_items);
    }
    let mut b_prime = b0_prime.min(b);
    for x in &w {
        if d_hat.get(x).cloned().unwrap_or(f64::INFINITY) < b_prime {
            u.insert(x.clone());
        }
    }
    complete.extend(u.iter().cloned());
    (b_prime, u)
}

fn run_sssp(g: &Graph, source: &str) -> HashMap<String, f64> {
    let n = g.nodes().len();
    let k = n;
    let mut d_hat: HashMap<String, f64> = g
        .nodes()
        .into_iter()
        .map(|v| (v, f64::INFINITY))
        .collect();
    d_hat.insert(source.to_string(), 0.0);
    let mut complete: HashSet<String> = HashSet::new();
    complete.insert(source.to_string());
    let s: HashSet<String> = [source.to_string()].iter().cloned().collect();
    base_case(g, f64::INFINITY, &s, &mut d_hat, &mut complete, k);
    d_hat
}

fn main() {
    let mut g = Graph::new();
    g.add_edge("s", "a", 1.0);
    g.add_edge("s", "b", 4.0);
    g.add_edge("a", "b", 2.0);
    g.add_edge("a", "c", 5.0);
    g.add_edge("b", "c", 1.0);
    let dist = run_sssp(&g, "s");
    println!("Distances from s: {:?}", dist);
}
