"""Deterministic SSSP solver based on the BMSSP algorithm."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .exceptions import AlgorithmError, ConfigError
from .frontier import BlockFrontier, FrontierProtocol, HeapFrontier
from .graph import Float, Graph, Vertex
from .logger import Logger, NoopLogger
from .path import compress_original_path_from_clones, reconstruct_path_basic


@dataclass(frozen=True)
class SSSPResult:
    """Distances and predecessors produced by the solver."""

    distances: List[Float]
    predecessors: List[Optional[Vertex]]


@dataclass(frozen=True)
class SolverMetrics:
    """Performance metrics collected from a solver run."""

    n: int
    m: int
    frontier: str
    transform: bool
    counters: Dict[str, int]
    wall_ms: float
    peak_mib: float | None = None


@dataclass(frozen=True)
class SolverConfig:
    """Configuration knobs for the solver.

    Attributes:
        use_transform: If ``True``, apply the constant-outdegree transform
            internally.
        target_outdeg: Maximum out-degree after the transform (ignored when
            ``use_transform`` is ``False``).
        frontier: ``"block"`` (paper-style) or ``"heap"`` (baseline).
        k_t_auto: If ``True``, compute ``(k, t)`` from ``n`` as in the paper.
            If ``False``, use the provided ``k`` and ``t`` values.
        k: Branching parameter when ``k_t_auto`` is ``False``.
        t: Level depth parameter when ``k_t_auto`` is ``False``.
    """

    use_transform: bool = True
    target_outdeg: int = 4
    frontier: str = "block"  # or "heap"
    k_t_auto: bool = True
    k: int = 0
    t: int = 0


class SSSPSolver:
    """Deterministic SSSP (directed, non-negative) with BMSSP-style recursion."""

    def __init__(
        self,
        G: Graph,
        source: Vertex,
        config: Optional[SolverConfig] = None,
        logger: Logger | None = None,
        sources: Optional[List[Vertex]] = None,
    ) -> None:
        """Initialize the solver.

        Args:
            G: Input graph.
            source: Source vertex identifier.
            config: Optional solver configuration.
            sources: Optional list of additional source vertices.

        Raises:
            AlgorithmError: If any provided source is not a valid vertex id.
        """
        if not (0 <= source < G.n):
            raise AlgorithmError("source must be a valid vertex id.")
        if sources is None:
            sources = [source]
        else:
            for s in sources:
                if not (0 <= s < G.n):
                    raise AlgorithmError("all sources must be valid vertex ids.")
        self._G_orig = G
        self._source_orig = source
        self._sources_orig = list(sources)
        self.cfg = config or SolverConfig()
        self.logger = logger or NoopLogger()

        self.counters = {
            "edges_relaxed": 0,
            "pulls": 0,
            "findpivots_rounds": 0,
            "basecase_pops": 0,
            "iterations_protected": 0,  # Track safety limit hits
        }

        # Optional transform for outdegree
        self._mapping = None  # orig -> [clones]
        self._clone2orig: Optional[List[int]] = None  # clone -> orig
        if self.cfg.use_transform:
            from .transform import constant_outdegree_transform

            G2, mapping = constant_outdegree_transform(G, self.cfg.target_outdeg)
            self.G = G2
            self.sources = [mapping[s][0] for s in self._sources_orig]
            self._mapping = mapping
            # Build reverse mapping for path compression
            clone2orig = [0] * self.G.n
            for u_orig, clones in mapping.items():
                for c in clones:
                    clone2orig[c] = u_orig
            self._clone2orig = clone2orig
        else:
            self.G = G
            self.sources = list(self._sources_orig)

        self.s = self.sources[0]

        # Distances and predecessors in (possibly transformed) space
        self.dhat: List[Float] = [math.inf] * self.G.n
        self.pred: List[Optional[Vertex]] = [None] * self.G.n
        self.complete: List[bool] = [False] * self.G.n
        self.root: List[int] = [-1] * self.G.n
        for s in self.sources:
            self.dhat[s] = 0.0
            self.complete[s] = True
            self.root[s] = s

        # Parameters (k, t, levels) with safety bounds
        n = max(2, self.G.n)
        if self.cfg.k_t_auto:
            log2n = math.log2(n)
            k = max(1, int(round(log2n ** (1.0 / 3.0))))
            t = max(1, int(round(log2n ** (2.0 / 3.0))))
            k = max(1, min(k, t))
            # Cap to reasonable values to prevent runaway algorithms
            k = min(k, 100)
            t = min(t, 20)
        else:
            k = max(1, min(self.cfg.k, 100))  # Safety cap
            t = max(1, min(self.cfg.t, 20))  # Safety cap
        self.k: int = k
        self.t: int = t
        self.L: int = max(1, min(math.ceil(math.log2(n) / t), 10))  # Cap levels

        # Best-clone cache for each original vertex after solve()
        self._best_clone_for_orig: Optional[List[int]] = None
        # Compressed distances in original space (after solve())
        self._distances_original: Optional[List[Float]] = None

    # ---------- utilities -------------------------------------------------

    def _relax(self, u: Vertex, v: Vertex, w: Float) -> bool:
        """Relax edge ``(u, v)``.

        Args:
            u: Tail vertex.
            v: Head vertex.
            w: Edge weight.

        Returns:
            ``True`` if ``v`` improved or tied with its previous distance.
        """
        self.counters["edges_relaxed"] += 1
        cand = self.dhat[u] + w
        if cand <= self.dhat[v]:
            if (
                cand < self.dhat[v]
                or self.pred[v] is None
                or (cand == self.dhat[v] and self.root[u] < self.root[v])
            ):
                self.dhat[v] = cand
                self.pred[v] = u
                self.root[v] = self.root[u]
            return True
        return False

    def _weight(self, u: Vertex, v: Vertex) -> Float:
        """Return ``w(u, v)`` by scanning ``u``'s adjacency list."""
        for vv, w in self.G.adj[u]:
            if vv == v:
                return w
        return 0.0

    # ---------- base case -------------------------------------------------

    def _base_case(self, B: Float, S: Set[Vertex]) -> Tuple[Float, Set[Vertex]]:
        """Explore from a single pivot using a bounded Dijkstra search.

        Args:
            B: Upper bound on distances considered.
            S: Set containing exactly one pivot vertex ``x``.

        Returns:
            A tuple ``(B', U)`` where ``U`` are vertices completed with distance
            less than ``B'``.
        """
        if len(S) != 1:
            raise AlgorithmError("BaseCase expects a singleton set.")
        (x,) = tuple(S)
        if self.dhat[x] == math.inf:
            raise AlgorithmError("BaseCase requires a finite pivot distance.")

        import heapq

        U0: List[Vertex] = []
        seen: Set[Vertex] = set()
        heap: List[Tuple[Float, Vertex]] = [(self.dhat[x], x)]
        in_heap: Set[Vertex] = {x}

        # Safety limits to prevent infinite loops
        iterations = 0
        max_iterations = min(self.k * 1000, self.G.n * 10)

        while heap and len(U0) < self.k + 1 and iterations < max_iterations:
            self.counters["basecase_pops"] += 1
            iterations += 1

            du, u = heapq.heappop(heap)
            in_heap.discard(u)
            if du != self.dhat[u] or u in seen:
                continue
            seen.add(u)
            self.complete[u] = True
            U0.append(u)

            for v, w in self.G.adj[u]:
                if self._relax(u, v, w) and self.dhat[u] + w < B:
                    if v not in in_heap:
                        heapq.heappush(heap, (self.dhat[v], v))
                        in_heap.add(v)

        if iterations >= max_iterations:
            self.counters["iterations_protected"] += 1

        if len(U0) <= self.k:
            return (B, set(U0))
        Bprime = max(self.dhat[v] for v in U0)
        U = {v for v in U0 if self.dhat[v] < Bprime}
        return (Bprime, U)

    # ---------- FindPivots ------------------------------------------------

    def _find_pivots(self, B: Float, S: Set[Vertex]) -> Tuple[Set[Vertex], Set[Vertex]]:
        """Run ``k`` relax rounds from ``S`` to collect candidate pivots.

        Args:
            B: Distance bound.
            S: Current set of vertices.

        Returns:
            A tuple ``(P, W)`` where ``P`` are chosen pivots and ``W`` is the
            set of vertices reached during the relax rounds.
        """
        W: Set[Vertex] = set(S)
        current: Set[Vertex] = set(S)

        # Safety limits for findpivots
        iterations = 0
        max_iterations = min(self.k * len(S) * 100, self.G.n * 10)

        for round_num in range(1, self.k + 1):
            if iterations >= max_iterations:
                self.counters["iterations_protected"] += 1
                break

            self.counters["findpivots_rounds"] += 1
            nxt: Set[Vertex] = set()

            for u in current:
                iterations += 1
                if iterations >= max_iterations:
                    break

                for v, w in self.G.adj[u]:
                    if self._relax(u, v, w) and (self.dhat[u] + w < B):
                        nxt.add(v)

            if not nxt:
                break
            W |= nxt

            # Early termination if W gets too large
            if len(W) > self.k * max(1, len(S)) * 5:  # More generous limit
                return set(S), W
            current = nxt

        # Build pivot tree with safety limits
        children: Dict[Vertex, List[Vertex]] = {u: [] for u in W}
        for v in W:
            p = self.pred[v]
            if p is not None and p in W and self.dhat[p] + self._weight(p, v) == self.dhat[v]:
                children[p].append(v)

        P: Set[Vertex] = set()
        for u in S:
            size = 0
            stack = [u]
            seen: Set[Vertex] = set()
            iterations = 0
            max_tree_iterations = min(self.k * 10, len(W))

            while stack and iterations < max_tree_iterations:
                iterations += 1
                a = stack.pop()
                if a in seen:
                    continue
                seen.add(a)
                size += 1
                stack.extend(children.get(a, ()))
                if size >= self.k:
                    P.add(u)
                    break
        return P, W

    # ---------- BMSSP -----------------------------------------------------

    def _make_frontier(self, level: int, B: Float) -> FrontierProtocol:
        # Cap the frontier size to prevent excessive memory usage
        M = max(1, min(2 ** ((level - 1) * self.t), 10000))
        if self.cfg.frontier == "heap":
            return HeapFrontier(M=M, B=B)
        if self.cfg.frontier == "block":
            return BlockFrontier(M=M, B=B)
        raise ConfigError(f"unknown frontier '{self.cfg.frontier}'")

    def _bmssp(
        self, level: int, B: Float, S: Set[Vertex], depth: int = 0
    ) -> Tuple[Float, Set[Vertex]]:
        # Prevent excessive recursion
        if depth > 50 or level > 20:
            return self._base_case(B, S)

        if level == 0:
            return self._base_case(B, S)

        P, W = self._find_pivots(B, S)
        D = self._make_frontier(level, B)
        for x in P:
            D.insert(x, self.dhat[x])

        U_accum: Set[Vertex] = set()
        cap = min(self.k * max(1, 2 ** (level * self.t)), self.G.n)  # Cap to graph size
        pull_iterations = 0
        max_pull_iterations = min(cap * 10, 1000)  # Safety limit on pulls

        while len(U_accum) < cap and pull_iterations < max_pull_iterations:
            self.counters["pulls"] += 1
            pull_iterations += 1

            S_i, B_i = D.pull()
            if not S_i:
                Bprime = B
                U_accum |= {x for x in W if self.dhat[x] < Bprime}
                for u in U_accum:
                    self.complete[u] = True
                return Bprime, U_accum

            B_i_prime, U_i = self._bmssp(level - 1, B_i, S_i, depth + 1)
            for u in U_i:
                self.complete[u] = True
            U_accum |= U_i

            K_pairs: List[Tuple[Vertex, Float]] = []
            for u in U_i:
                du = self.dhat[u]
                for v, w in self.G.adj[u]:
                    if self._relax(u, v, w):
                        val = du + w
                        if B_i <= val < B:
                            D.insert(v, val)
                        elif B_i_prime <= val < B_i:
                            K_pairs.append((v, val))

            extra_pairs = [(x, self.dhat[x]) for x in S_i if B_i_prime <= self.dhat[x] < B_i]
            if K_pairs or extra_pairs:
                D.batch_prepend(K_pairs + extra_pairs)

            if len(U_accum) >= cap:
                Bprime = B_i_prime
                U_accum |= {x for x in W if self.dhat[x] < Bprime}
                for u in U_accum:
                    self.complete[u] = True
                return Bprime, U_accum

        if pull_iterations >= max_pull_iterations:
            self.counters["iterations_protected"] += 1

        return B, U_accum

    # ---------- public API ------------------------------------------------

    def solve(self) -> SSSPResult:
        """Run the BMSSP algorithm and return distances and predecessors."""
        top_level = self.L
        B = math.inf
        S0 = set(self.sources)
        _Bprime, _U = self._bmssp(top_level, B, S0)

        # If we transformed, compress distances back to original vertices (predecessors omitted).
        if self.cfg.use_transform and self._mapping is not None and self._clone2orig is not None:
            comp: List[Float] = [math.inf] * self._G_orig.n
            best_clone: List[int] = [-1] * self._G_orig.n
            for u_orig, clones in self._mapping.items():
                best = math.inf
                arg = -1
                best_root = math.inf
                for c in clones:
                    r = self.root[c]
                    d = self.dhat[c]
                    if d < best or (d == best and r < best_root):
                        best = d
                        arg = c
                        best_root = r
                comp[u_orig] = best
                best_clone[u_orig] = arg
            # Cache for path reconstruction
            self._best_clone_for_orig = best_clone
            self._distances_original = comp
            return SSSPResult(distances=comp, predecessors=[None] * self._G_orig.n)

        # No transform: distances/preds already in original space
        self._distances_original = self.dhat
        return SSSPResult(distances=self.dhat, predecessors=self.pred)

    def path(self, target_original: Vertex) -> List[Vertex]:
        """Return a path from the source to ``target_original`` in original ids.

        Args:
            target_original: Target vertex identifier in the original graph.

        Returns:
            List of vertex ids from source to target (inclusive). Returns an
            empty list if no path exists. ``solve`` must be called beforehand.
        """
        if self._distances_original is None:
            raise AlgorithmError("Call solve() before requesting paths.")

        # No transform: reconstruct directly in original space
        if not self.cfg.use_transform or self._mapping is None or self._clone2orig is None:
            src = self.root[target_original]
            if src < 0:
                return []
            return reconstruct_path_basic(self.pred, src, target_original)

        # With transform: walk from best clone back to the source clone, compress to original ids
        if target_original < 0 or target_original >= self._G_orig.n:
            raise AlgorithmError("target out of range for original graph.")

        if self._best_clone_for_orig is None:
            raise AlgorithmError("Internal state missing best-clone cache. Call solve() first.")

        start_clone = self._best_clone_for_orig[target_original]
        if start_clone < 0:
            return []  # unreachable
        src_clone = self.root[start_clone]

        # Walk predecessors in clone-space with safety limits
        chain: List[int] = []
        cur: Optional[int] = start_clone
        seen = set()
        iterations = 0
        max_iterations = self.G.n * 2  # Safety limit

        while cur is not None and iterations < max_iterations:
            iterations += 1
            chain.append(cur)
            if cur == src_clone:
                chain.reverse()
                return compress_original_path_from_clones(chain, self._clone2orig)
            if cur in seen:
                break
            seen.add(cur)
            cur = self.pred[cur]

        return []  # unreachable or inconsistent

    # ---------- counters --------------------------------------------------

    def summary(self) -> Dict[str, int]:
        """Return a copy of internal counter values."""
        return dict(self.counters)

    def metrics(self, wall_ms: float, peak_mib: float | None = None) -> SolverMetrics:
        """Return performance metrics for the most recent run.

        Args:
            wall_ms: Wall-clock time spent in :meth:`solve` in milliseconds.
            peak_mib: Optional peak memory usage in MiB.

        Returns:
            A dataclass capturing run parameters and counter values.
        """
        m = sum(len(lst) for lst in self._G_orig.adj)
        return SolverMetrics(
            n=self._G_orig.n,
            m=m,
            frontier=self.cfg.frontier,
            transform=self.cfg.use_transform,
            counters=self.summary(),
            wall_ms=wall_ms,
            peak_mib=peak_mib,
        )
