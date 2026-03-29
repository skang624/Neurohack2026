"""Plotly-based graph and matrix visualization helpers."""

from __future__ import annotations

from typing import Dict, List

import networkx as nx
import numpy as np
import plotly.graph_objects as go


def _weight_to_color(weight: float, alpha: float = 0.65) -> str:
    """Map an edge weight in [0, 1] to an RGBA orange color."""
    v = min(1.0, max(0.0, weight))
    r = int(255 - (1 - v) * 25)
    g = int(140 + (1 - v) * 55)
    b = int(40 + (1 - v) * 35)
    return f"rgba({r},{g},{b},{alpha})"


def plot_graph_plotly(
    graph: nx.Graph,
    positions: Dict[int, np.ndarray],
    node_names: List[str],
    node_sizes: Dict[int, float],
    weighted_degrees: Dict[int, float],
    show_labels: bool = True,
    color_edges_by_weight: bool = True,
    color_nodes_by_degree: bool = True,
    title: str = "Dynamic People Graph",
    focus_node: int | None = None,
    square_map: bool = False,
) -> go.Figure:
    """Render the weighted graph in Plotly."""
    focus_neighbors = set(graph.neighbors(focus_node)) if focus_node is not None and focus_node in graph else set()
    focus_group = focus_neighbors | ({focus_node} if focus_node is not None and focus_node in graph else set())

    edge_traces = []
    for u, v, data in graph.edges(data=True):
        w = float(data.get("weight", 0.0))
        x0, y0 = positions[u]
        x1, y1 = positions[v]
        is_focus_edge = focus_node is None or (u in focus_group and v in focus_group)
        if is_focus_edge:
            color = _weight_to_color(w) if color_edges_by_weight else "rgba(120,120,160,0.45)"
            width = 0.6 + 4.2 * w
        else:
            color = "rgba(140,150,170,0.10)"
            width = 0.35 + 0.7 * w
        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=width, color=color),
                hoverinfo="text",
                text=[f"{node_names[u]} <-> {node_names[v]}<br>Weight: {w:.3f}", "", ""],
                showlegend=False,
            )
        )

    node_x, node_y, node_text, marker_sizes, marker_colors = [], [], [], [], []

    for node in graph.nodes():
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)

        neighbors = list(graph.neighbors(node))
        neighbor_names = [node_names[n] for n in neighbors]
        neighbor_str = ", ".join(neighbor_names[:8]) if neighbor_names else "None"
        if len(neighbor_names) > 8:
            neighbor_str += ", ..."

        deg = weighted_degrees.get(node, 0.0)
        node_text.append(f"Node: {node_names[node]}<br>Neighbors: {neighbor_str}")
        marker_sizes.append(node_sizes.get(node, 28.0))
        marker_colors.append(deg)

    if color_nodes_by_degree and marker_colors:
        colorbar = dict(title="Connectivity", thickness=12, x=1.02)
        marker = dict(
            size=marker_sizes,
            color=marker_colors,
            colorscale="YlGnBu",
            reversescale=False,
            line=dict(width=1.2, color="rgba(255,255,255,0.95)"),
            opacity=0.95,
            colorbar=colorbar,
        )
    else:
        marker = dict(
            size=marker_sizes,
            color="rgba(37,99,235,0.85)",
            line=dict(width=1.2, color="rgba(255,255,255,0.95)"),
            opacity=0.95,
        )

    if focus_node is not None and focus_node in graph:
        focus_colors = []
        for node in graph.nodes():
            if node == focus_node:
                focus_colors.append("rgba(239,68,68,0.96)")
            elif node in focus_group:
                focus_colors.append("rgba(37,99,235,0.92)")
            else:
                focus_colors.append("rgba(175,186,205,0.25)")
        marker["color"] = focus_colors
        marker.pop("colorbar", None)
        marker.pop("colorscale", None)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text" if show_labels else "markers",
        text=node_names if show_labels else None,
        textposition="top center",
        textfont=dict(size=12, color="rgba(25,30,45,0.95)"),
        hoverinfo="text",
        hovertext=node_text,
        marker=marker,
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title=title,
        title_x=0.02,
        paper_bgcolor="rgba(245,248,255,1)",
        plot_bgcolor="rgba(245,248,255,1)",
        margin=dict(l=10, r=10, t=48, b=18),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode="closest",
    )
    if square_map:
        fig.update_layout(height=760)
        fig.update_xaxes(scaleanchor="y", scaleratio=1, constrain="domain")
        fig.update_yaxes(constrain="domain")
    return fig


def plot_matrix_heatmap(matrix: np.ndarray, node_names: List[str], title: str = "Adjacency Matrix") -> go.Figure:
    """Render adjacency matrix heatmap."""
    fig = go.Figure(
        data=
        [
            go.Heatmap(
                z=matrix,
                x=node_names,
                y=node_names,
                colorscale="YlGnBu",
                zmin=0,
                zmax=1,
                colorbar=dict(title="Weight"),
                hovertemplate="From %{y}<br>To %{x}<br>Weight %{z:.3f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=title,
        title_x=0.02,
        paper_bgcolor="rgba(245,248,255,1)",
        plot_bgcolor="rgba(245,248,255,1)",
        margin=dict(l=10, r=10, t=46, b=10),
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed"),
    )
    return fig
