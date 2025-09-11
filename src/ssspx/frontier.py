"""Frontier data structures used by the solver."""

from __future__ import annotations

import heapq
from bisect import bisect_left
from dataclasses import dataclass
from typing import Dict, Iterable, List, Protocol, Set, Tuple

Vertex = int
Float = float


class FrontierProtocol(Protocol):
    """Protocol for frontier structures consumed by the solver."""

    def insert(self, key: Vertex, value: Float) -> None:
        """Insert or update a key."""
        ...

    def batch_prepend(self, pairs: Iterable[Tuple[Vertex, Float]]) -> None:
        """Prepend key/value pairs."""
        ...

    def pull(self) -> Tuple[Set[Vertex], Float]:
        """Return up to ``M`` keys with the smallest values."""
        ...


@dataclass
class BlockFrontier:
    """Block-based frontier approximating the paper's data structure.

    Args:
        M: Maximum keys returned per ``pull``.
        B: Upper bound separator value.
    """

    M: int
    B: Float

    def __post_init__(self) -> None:
        """Validate parameters and set up internal structures."""
        if self.M <= 0:
            raise ValueError("M must be positive.")
        self._d0: List[List[Tuple[Vertex, Float]]] = []
        self._d1: List[List[Tuple[Vertex, Float]]] = []
        self._d1_bounds: List[Float] = []
        self._best: Dict[Vertex, Float] = {}
        self._current_min: Float = self.B

    # ---- internals ----------------------------------------------------

    def _block_upper(self, block: List[Tuple[Vertex, Float]]) -> Float:
        return max(v for _, v in block)

    def _insert_into_d1_block(self, idx: int, pair: Tuple[Vertex, Float]) -> None:
        self._d1[idx].append(pair)
        if len(self._d1[idx]) > self.M:
            block = self._d1[idx]
            block.sort(key=lambda kv: kv[1])
            mid = len(block) // 2
            left = block[:mid]
            right = block[mid:]
            self._d1[idx] = left
            self._d1.insert(idx + 1, right)
            self._d1_bounds[idx] = self._block_upper(left)
            self._d1_bounds.insert(idx + 1, self._block_upper(right))
        else:
            self._d1_bounds[idx] = max(self._d1_bounds[idx], pair[1])

    def _append_new_d1_block(self, pair: Tuple[Vertex, Float]) -> None:
        self._d1.append([pair])
        self._d1_bounds.append(pair[1])

    # ---- public API ---------------------------------------------------

    def insert(self, key: Vertex, value: Float) -> None:
        """Insert or update a key with the given value.

        Only the smallest value for each key is retained; larger duplicates are
        ignored.

        Args:
            key: Vertex identifier.
            value: Associated value.
        """
        old = self._best.get(key)
        if old is not None and old <= value:
            return
        self._best[key] = value

        if not self._d1_bounds:
            self._append_new_d1_block((key, value))
            self._current_min = min(self._current_min, value)
            return

        idx = bisect_left(self._d1_bounds, value)
        if idx == len(self._d1_bounds):
            self._append_new_d1_block((key, value))
        else:
            self._insert_into_d1_block(idx, (key, value))
        if value < self._current_min:
            self._current_min = value

    def batch_prepend(self, pairs: Iterable[Tuple[Vertex, Float]]) -> None:
        """Prepend pairs whose values are all smaller than any current value.

        The pairs are sorted and chunked into blocks to keep pull locality.

        Args:
            pairs: Iterable of ``(key, value)`` tuples to prepend.
        """
        lst = [(int(k), float(v)) for (k, v) in pairs]
        if not lst:
            return

        # Keep the smallest per key within this batch
        tmp: Dict[Vertex, Float] = {}
        for k, v in lst:
            if (k not in tmp) or (v < tmp[k]):
                tmp[k] = v

        items = sorted(tmp.items(), key=lambda kv: kv[1])

        # Update best table
        for k, v in items:
            old = self._best.get(k)
            if old is None or v < old:
                self._best[k] = v

        cap = max(1, (self.M + 1) // 2)
        new_blocks: List[List[Tuple[Vertex, Float]]] = []
        for i in range(0, len(items), cap):
            new_blocks.append([(k, v) for (k, v) in items[i : i + cap]])

        # Prepend before D1
        self._d0 = new_blocks + self._d0
        self._current_min = min(self._current_min, items[0][1])

    def _consume_block_prefix(
        self,
        blocks: List[List[Tuple[Vertex, Float]]],
        want: int,
        chosen: Dict[Vertex, Float],
        pulled_keys: Set[Vertex],
    ) -> int:
        """Greedily take up to ``want`` keys from the head of ``blocks``."""
        got = 0
        idx_block = 0
        iterations = 0
        max_iterations = len(blocks) * 100  # Safety limit to prevent infinite loops

        while got < want and idx_block < len(blocks) and iterations < max_iterations:
            iterations += 1
            block = blocks[idx_block]
            keep: List[Tuple[Vertex, Float]] = []
            processed_any = False

            for k, v in block:
                bestv = self._best.get(k)
                if bestv is None or v != bestv:
                    continue  # stale
                if k in pulled_keys:
                    keep.append((k, v))
                    continue
                if got < want:
                    chosen[k] = v
                    pulled_keys.add(k)
                    got += 1
                    processed_any = True
                else:
                    keep.append((k, v))

            blocks[idx_block] = keep
            if not blocks[idx_block]:
                blocks.pop(idx_block)
                # Don't increment idx_block since we removed an element
            else:
                idx_block += 1

            # If we didn't process anything useful, break to avoid infinite loop
            if not processed_any and not keep:
                break

        return got

    def pull(self) -> Tuple[Set[Vertex], Float]:
        """Return up to ``M`` keys with the smallest values and a separator.

        Returns ``(S, x)`` where ``S`` is the set of selected keys and ``x`` is
        a value separating the selected keys from the remaining ones.

        Returns:
            A tuple ``(S, x)`` as described above. If the frontier is empty,
            ``S`` is empty and ``x`` equals ``B``.
        """
        chosen: Dict[Vertex, Float] = {}
        pulled_keys: Set[Vertex] = set()

        got = self._consume_block_prefix(self._d0, self.M, chosen, pulled_keys)
        if got < self.M:
            got += self._consume_block_prefix(self._d1, self.M - got, chosen, pulled_keys)
            new_bounds: List[Float] = []
            for blk in self._d1:
                if blk:
                    new_bounds.append(self._block_upper(blk))
            self._d1_bounds = new_bounds

        s_prime = set(chosen.keys())
        if not s_prime:
            return set(), self.B

        if not self._d0 and not self._d1:
            x = self.B
        else:
            next_candidates: List[Float] = []
            if self._d0 and self._d0[0]:
                next_candidates.append(min(v for _, v in self._d0[0]))
            if self._d1_bounds:
                next_candidates.append(self._d1_bounds[0])
            x = min(next_candidates) if next_candidates else self.B

        # Update cached min
        if self._d0 and self._d0[0]:
            self._current_min = min(v for _, v in self._d0[0])
        elif self._d1_bounds:
            self._current_min = self._d1_bounds[0]
        else:
            self._current_min = self.B

        return s_prime, x


class HeapFrontier:
    """Simpler frontier based on a binary heap.

    It matches the :class:`FrontierProtocol` semantics but is slower than
    :class:`BlockFrontier` and mostly useful for profiling or as a baseline.
    """

    def __init__(self, M: int, B: Float) -> None:
        """Initialize the frontier.

        Args:
            M: Maximum keys returned per ``pull``.
            B: Upper bound separator value.

        Raises:
            ValueError: If ``M`` is not positive.
        """
        if M <= 0:
            raise ValueError("M must be positive.")
        self.M = int(M)
        self.B = float(B)
        self._best: Dict[Vertex, Float] = {}
        self._heap: List[Tuple[Float, Vertex]] = []

    def insert(self, key: Vertex, value: Float) -> None:
        """Insert or update a key with the given value."""
        old = self._best.get(key)
        if old is not None and old <= value:
            return
        self._best[key] = value
        heapq.heappush(self._heap, (value, key))

    def batch_prepend(self, pairs: Iterable[Tuple[Vertex, Float]]) -> None:
        """Insert all given pairs into the frontier."""
        for k, v in pairs:
            self.insert(k, v)

    def pull(self) -> Tuple[Set[Vertex], Float]:
        """Return up to ``M`` keys with the smallest values."""
        s: Set[Vertex] = set()
        iterations = 0
        max_iterations = len(self._heap) * 2  # Safety limit

        while self._heap and len(s) < self.M and iterations < max_iterations:
            iterations += 1
            val, key = heapq.heappop(self._heap)
            if self._best.get(key) != val:
                continue  # stale
            s.add(key)

        if not s:
            return set(), self.B
        x = self._heap[0][0] if self._heap else self.B
        return s, float(x)
