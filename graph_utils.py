"""Graph construction and graph-derived metrics."""

from __future__ import annotations

from typing import Dict, List

import networkx as nx
import numpy as np


def build_graph_from_adjacency(
    adjacency: np.ndarray,
    threshold: float = 0.1,
    node_names: List[str] | None = None,
) -> nx.Graph:
    """Build a weighted undirected graph from an adjacency matrix.

    Edges with weight below threshold are ignored.
    """
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Adjacency must be a square 2D matrix.")

    n = adjacency.shape[0]
    names = node_names or [f"Person {i + 1}" for i in range(n)]
    if len(names) != n:
        raise ValueError("node_names length must match matrix dimension.")

    g = nx.Graph()
    for idx, name in enumerate(names):
        g.add_node(idx, label=name)

    for i in range(n):
        for j in range(i + 1, n):
            w = float(adjacency[i, j])
            if w >= threshold:
                g.add_edge(i, j, weight=w)

    return g


def compute_weighted_degrees(graph: nx.Graph) -> Dict[int, float]:
    """Return weighted degree by node."""
    return dict(graph.degree(weight="weight"))


def compute_node_sizes(
    weighted_degrees: Dict[int, float],
    min_size: float = 18.0,
    max_size: float = 56.0,
) -> Dict[int, float]:
    """Map weighted degree values into display node sizes."""
    if not weighted_degrees:
        return {}

    values = np.array(list(weighted_degrees.values()), dtype=float)
    v_min = float(np.min(values))
    v_max = float(np.max(values))

    if np.isclose(v_max, v_min):
        return {k: (min_size + max_size) / 2.0 for k in weighted_degrees}

    sizes = {}
    for node, val in weighted_degrees.items():
        norm = (val - v_min) / (v_max - v_min)
        sizes[node] = min_size + norm * (max_size - min_size)
    return sizes


def get_stable_layout(
    adjacency_tensor: np.ndarray,
    spring_seed: int = 42,
    k: float | None = None,
) -> Dict[int, np.ndarray]:
    """Compute a single stable spring layout using average adjacency over time."""
    if adjacency_tensor.ndim != 3:
        raise ValueError("Expected adjacency tensor with shape (T, N, N).")

    avg_matrix = np.mean(adjacency_tensor, axis=0)
    n = avg_matrix.shape[0]

    layout_graph = nx.Graph()
    layout_graph.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            w = float(avg_matrix[i, j])
            if w > 1e-9:
                layout_graph.add_edge(i, j, weight=w)

    return nx.spring_layout(layout_graph, seed=spring_seed, weight="weight", k=k)


def summarize_graph(graph: nx.Graph, node_names: List[str]) -> dict:
    """Compute high-level graph summary stats for UI display."""
    edges = list(graph.edges(data=True))
    if edges:
        weights = [float(data.get("weight", 0.0)) for _, _, data in edges]
        max_idx = int(np.argmax(weights))
        u, v, data = edges[max_idx]
        strongest_connection = f"{node_names[u]} <-> {node_names[v]} ({data.get('weight', 0.0):.3f})"
        avg_weight = float(np.mean(weights))
    else:
        strongest_connection = "No active edges"
        avg_weight = 0.0

    weighted_deg = compute_weighted_degrees(graph)
    if weighted_deg:
        top_node = max(weighted_deg.items(), key=lambda x: x[1])
        most_connected = f"{node_names[top_node[0]]}"
    else:
        most_connected = "No connected nodes"

    return {
        "num_active_edges": graph.number_of_edges(),
        "average_weight": avg_weight,
        "strongest_connection": strongest_connection,
        "most_connected_node": most_connected,
    }
