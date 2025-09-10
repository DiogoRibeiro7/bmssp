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
    """Iterate over all edges in the given graph.

    Args:
        G: The graph object containing adjacency information.

    Yields:
        A tuple representing an edge in the graph, where the first element
        is the source node index (u), the second element is the destination
        node index (v), and the third element is the edge weight (w).
    """
    for u in range(G.n):
        for v, w in G.adj[u]:
            yield u, v, w


def _read_csv(path: Path) -> Tuple[int, EdgeList]:
    """Read a CSV file containing edge data and return the number of nodes and edge list.

    Each line in the file should contain at least three columns: source node,
    target node, and edge weight. Lines starting with '#' or empty lines are
    ignored. Columns can be separated by commas or tabs.

    Args:
        path: The path to the CSV file.

    Returns:
        A tuple containing the number of nodes (max node id + 1) and a list
        of edges, where each edge is represented as a tuple (u, v, w).

    Raises:
        GraphFormatError: If no edges are parsed from the file.
    """
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
    """Write the edges of a graph to a CSV file.

    Each row in the CSV file represents an edge in the graph, with columns for
    the source node, target node, and edge weight.

    Args:
        path: The file path where the CSV will be written.
        G: The graph whose edges will be written to the CSV file.
    """
    with path.open("w", encoding="utf-8") as fh:
        for u, v, w in _iter_edges(G):
            fh.write(f"{u},{v},{w}\n")


def _read_jsonl(path: Path) -> Tuple[int, EdgeList]:
    """Read a JSON Lines (JSONL) file containing graph edges.

    Each line in the file should be a JSON object with keys "u", "v", and "w",
    representing the source node, target node, and edge weight, respectively.

    Args:
        path: Path to the JSONL file.

    Returns:
        A tuple containing the number of nodes (max node id + 1) and a list
        of edges as (u, v, w) tuples.

    Raises:
        GraphFormatError: If no edges are parsed from the file.
    """
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
    """Write the edges of a graph to a file in JSON Lines (JSONL) format.

    Each line in the output file represents an edge as a JSON object
    with keys 'u', 'v', and 'w', corresponding to the source node, target node,
    and edge weight, respectively.

    Args:
        path: The file path where the JSONL data will be written.
        G: The graph object containing the edges to be serialized.
    """
    with path.open("w", encoding="utf-8") as fh:
        for u, v, w in _iter_edges(G):
            fh.write(json.dumps({"u": u, "v": v, "w": w}) + "\n")


def _read_mtx(path: Path) -> Tuple[int, EdgeList]:
    """Read a Matrix Market (.mtx) file and extract the edge list.

    Args:
        path: Path to the Matrix Market file.

    Returns:
        A tuple containing the number of nodes (n) and the edge list.
        The edge list is a list of tuples (u, v, w), where u and v
        are zero-based node indices, and w is the edge weight.

    Notes:
        Lines starting with '%' are treated as comments and skipped.
        Assumes the file contains at least three columns: source, target,
        and weight. Node indices in the file are assumed to be 1-based
        and are converted to 0-based.
    """
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
    """Write the given graph to a Matrix Market (.mtx) file.

    The output file will contain the graph's adjacency matrix in coordinate
    format, where each line represents an edge with its source node, target
    node, and weight. Node indices are written as 1-based (Matrix Market
    convention).

    Args:
        path: The file path where the Matrix Market file will be written.
        G: The graph object containing nodes and weighted edges.
    """
    edges = list(_iter_edges(G))
    with path.open("w", encoding="utf-8") as fh:
        fh.write("%%MatrixMarket matrix coordinate real general\n")
        fh.write(f"{G.n} {G.n} {len(edges)}\n")
        for u, v, w in edges:
            fh.write(f"{u+1} {v+1} {w}\n")


def _read_graphml(path: Path) -> Tuple[int, EdgeList]:
    """Parse a GraphML file and extract the edge list.

    Args:
        path: Path to the GraphML file.

    Returns:
        A tuple containing the number of nodes (as max node id + 1) and a
        list of edges, where each edge is represented as a tuple (u, v, w)
        with integer node IDs and float weights.

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
    """Write the given graph to a GraphML file.

    Args:
        path: The file path where the GraphML output will be written.
        G: The graph object to serialize, expected to have an attribute `n`
           for the number of nodes.

    Notes:
        The function assumes the graph is directed. Each node is assigned an
        ID in the format "n{i}". Edges are written with source, target, and
        weight attributes. The helper function `_iter_edges(G)` should yield
        tuples of (u, v, w) for each edge.
    """
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
    """Detect the file format based on the file extension.

    Args:
        path: The path to the file whose format is to be detected.

    Returns:
        The detected format as a string ("csv", "jsonl", "mtx", "graphml"),
        or None if the format is not recognized.
    """
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
    """Read a graph from a file in the specified format.

    Args:
        path: The path to the graph file.
        fmt: The format of the graph file. If None, the format is auto-detected.

    Returns:
        The graph object constructed from the file.

    Raises:
        GraphFormatError: If the graph format is unknown or unsupported.
    """
    p = Path(path)
    fmt = fmt or _detect_format(p)
    if fmt is None or fmt not in _FMT_READERS:
        raise GraphFormatError("unknown graph format")
    n, edges = _FMT_READERS[fmt](p)
    return Graph.from_edges(n, edges)


def write_graph(G: Graph, path: str, fmt: Optional[str] = None) -> None:
    """Write a graph to a file in the specified format.

    Args:
        G: The graph object to be written.
        path: The file path where the graph will be saved.
        fmt: The format to use for writing the graph. If None, the format is
             auto-detected from the file extension.

    Raises:
        GraphFormatError: If the format is unknown or unsupported.
    """
    p = Path(path)
    fmt = fmt or _detect_format(p)
    if fmt is None or fmt not in _FMT_WRITERS:
        raise GraphFormatError("unknown graph format")
    _FMT_WRITERS[fmt](p, G)


def load_graph(path: str, fmt: Optional[str] = None) -> Graph:
    """Load a graph from the specified file path.

    This function is deprecated; use `read_graph` instead.

    Args:
        path: The path to the graph file.
        fmt: The format of the graph file. Defaults to None.

    Returns:
        The loaded graph object.

    Deprecated:
        Since version 0.1.0. Will be removed in version 0.2.0. Use `read_graph` instead.
    """
    warn_once(
        "load_graph is deprecated; use read_graph",
        since="0.1.0",
        remove_in="0.2.0",
    )
    return read_graph(path, fmt=fmt)
