"""Graph input/output helpers."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from .deprecation import warn_once
from .exceptions import GraphFormatError
from .graph import Graph

EdgeList = List[Tuple[int, int, float]]


def _iter_edges(G: Graph) -> Iterable[Tuple[int, int, float]]:
    """Yield all edges ``(u, v, w)`` from ``G``."""
    for u in range(G.n):
        for v, w in G.adj[u]:
            yield u, v, w


def _read_csv(path: Path) -> Tuple[int, EdgeList]:
    """Parse a CSV/TSV edges file."""
    edges: EdgeList = []
    max_id = -1
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            row = raw.strip()
            if not row or row.startswith("#"):
                continue
            parts = row.replace("\t", ",").split(",")
            if len(parts) < 3:
                continue
            try:
                u = int(parts[0].strip())
                v = int(parts[1].strip())
                w = float(parts[2].strip())
            except Exception:
                continue
            edges.append((u, v, w))
            max_id = max(max_id, u, v)
    if max_id < 0:
        raise GraphFormatError("no edges parsed from file")
    return max_id + 1, edges


def _write_csv(path: Path, G: Graph) -> None:
    """Write ``G`` as a CSV edges file."""
    with path.open("w", encoding="utf-8") as fh:
        for u, v, w in _iter_edges(G):
            fh.write(f"{u},{v},{w}\n")


def _read_jsonl(path: Path) -> Tuple[int, EdgeList]:
    """Parse a JSON Lines edges file."""
    edges: EdgeList = []
    max_id = -1
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            row = raw.strip()
            if not row:
                continue
            obj = json.loads(row)
            u = int(obj["u"])
            v = int(obj["v"])
            w = float(obj["w"])
            edges.append((u, v, w))
            max_id = max(max_id, u, v)
    if max_id < 0:
        raise GraphFormatError("no edges parsed from file")
    return max_id + 1, edges


def _write_jsonl(path: Path, G: Graph) -> None:
    """Write ``G`` as a JSON Lines edges file."""
    with path.open("w", encoding="utf-8") as fh:
        for u, v, w in _iter_edges(G):
            fh.write(json.dumps({"u": u, "v": v, "w": w}) + "\n")


def _read_mtx(path: Path) -> Tuple[int, EdgeList]:
    """Parse a MatrixMarket edges file."""
    edges: EdgeList = []
    it = path.open("r", encoding="utf-8")
    with it as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("%"):
                continue
            dims = line.split()
            nrows, ncols, _ = map(int, dims)
            break
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            u = int(parts[0]) - 1
            v = int(parts[1]) - 1
            w = float(parts[2])
            edges.append((u, v, w))
    n = max(nrows, ncols)
    return n, edges


def _write_mtx(path: Path, G: Graph) -> None:
    """Write ``G`` as a MatrixMarket edges file."""
    edges = list(_iter_edges(G))
    with path.open("w", encoding="utf-8") as fh:
        fh.write("%%MatrixMarket matrix coordinate real general\n")
        fh.write(f"{G.n} {G.n} {len(edges)}\n")
        for u, v, w in edges:
            fh.write(f"{u+1} {v+1} {w}\n")


def _read_graphml(path: Path) -> Tuple[int, EdgeList]:
    """
    Parse a GraphML file and extract the edge list.
    
    Args:
        path (Path): Path to the GraphML file.
    Returns:
        Tuple[int, EdgeList]: A tuple containing the number of nodes (as max node id + 1)
        and a list of edges, where each edge is represented as a tuple (u, v, w) with
        integer node IDs and float weights.
    Raises:
        GraphFormatError: If no edges are parsed from the file.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    ns = "{http://graphml.graphdrawing.org/xmlns}"
    edges: EdgeList = []
    max_id = -1
    for edge in root.findall(f".//{ns}edge"):
        u_str = edge.attrib.get("source", "")
        v_str = edge.attrib.get("target", "")      
        # Convert string IDs to integers
        if u_str.startswith("n"):
            u = int(u_str[1:])
        else:
            u = int(u_str)
        if v_str.startswith("n"):
            v = int(v_str[1:])
        else:
            v = int(v_str)    
        w_attr = edge.attrib.get("weight")
        if w_attr is None:
            data = edge.find(f"{ns}data[@key='w']")
            w = float(data.text) if (data is not None and data.text is not None) else 1.0
        else:
            w = float(w_attr)
        edges.append((u, v, w))  # Now u and v are definitely integers
        max_id = max(max_id, u, v)
    if max_id < 0:
        raise GraphFormatError("no edges parsed from file")
    return max_id + 1, edges


def _write_graphml(path: Path, G: Graph) -> None:
    """Write ``G`` as a GraphML edges file."""
    lines: List[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">')
    lines.append('  <graph id="G" edgedefault="directed">')
    for i in range(G.n):
        lines.append(f'    <node id="n{i}"/>')
    for u, v, w in _iter_edges(G):
        lines.append(f'    <edge source="n{u}" target="n{v}" weight="{w}"/>')
    lines.append("  </graph>")
    lines.append("</graphml>")
    path.write_text("\n".join(lines), encoding="utf-8")


_FMT_READERS = {
    "csv": _read_csv,
    "jsonl": _read_jsonl,
    "mtx": _read_mtx,
    "graphml": _read_graphml,
}

_FMT_WRITERS = {
    "csv": _write_csv,
    "jsonl": _write_jsonl,
    "mtx": _write_mtx,
    "graphml": _write_graphml,
}


def _detect_format(path: Path) -> Optional[str]:
    """Infer graph format from ``path`` extension."""
    ext = path.suffix.lower()
    if ext in {".csv", ".tsv"}:
        return "csv"
    if ext in {".jsonl", ".json"}:
        return "jsonl"
    if ext == ".mtx":
        return "mtx"
    if ext == ".graphml":
        return "graphml"
    return None


def read_graph(path: str, fmt: Optional[str] = None) -> Graph:
    """Read a graph from ``path`` in the given format."""
    p = Path(path)
    fmt = fmt or _detect_format(p)
    if fmt is None or fmt not in _FMT_READERS:
        raise GraphFormatError("unknown graph format")
    n, edges = _FMT_READERS[fmt](p)
    return Graph.from_edges(n, edges)


def write_graph(G: Graph, path: str, fmt: Optional[str] = None) -> None:
    """Write ``G`` to ``path`` in the given format."""
    p = Path(path)
    fmt = fmt or _detect_format(p)
    if fmt is None or fmt not in _FMT_WRITERS:
        raise GraphFormatError("unknown graph format")
    _FMT_WRITERS[fmt](p, G)


def load_graph(path: str, fmt: Optional[str] = None) -> Graph:
    """Deprecated alias for :func:`read_graph`."""
    warn_once(
        "load_graph is deprecated; use read_graph",
        since="0.1.0",
        remove_in="0.2.0",
    )
    return read_graph(path, fmt=fmt)
