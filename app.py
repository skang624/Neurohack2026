"""Streamlit app for dynamic graph visualization from adjacency matrices."""

from __future__ import annotations

import base64
import io
import os
from datetime import datetime
from typing import List

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from data_utils import (
    load_adjacency_file,
    load_coordinates_file,
    load_local_brain_dataset,
)
from graph_utils import (
    build_graph_from_adjacency,
    compute_node_sizes,
    compute_weighted_degrees,
    get_stable_layout,
    summarize_graph,
)
from visualization import plot_graph_plotly, plot_matrix_heatmap

st.set_page_config(page_title="Dynamic Graph Demo", layout="wide")


def _load_neuron_stage_image(image_root: str, person: str, stage: str) -> bytes:
    """Load person/stage neuron image from local directory using filename heuristics."""
    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"Image directory not found: {image_root}")

    valid_ext = (".jpg", ".jpg", ".jpeg", ".webp")
    files = [f for f in os.listdir(image_root) if f.lower().endswith(valid_ext)]
    if not files:
        raise FileNotFoundError(f"No image files found in: {image_root}")

    person = person.lower().strip()
    stage = stage.lower().strip()
    person_files = [fname for fname in files if person in fname.lower()]
    if not person_files:
        raise FileNotFoundError(f"No images found for person '{person}' in: {image_root}")

    preferred = []
    for fname in person_files:
        low = fname.lower()
        if stage == "early" and ("early" in low or "before" in low):
            preferred.append(fname)
        if stage == "after" and ("after" in low or "moving" in low or "post" in low):
            preferred.append(fname)

    # Fallback: still restrict to this person's files only.
    if not preferred:
        ordered = sorted(person_files)
        if len(ordered) >= 2:
            preferred = [ordered[0]] if stage == "early" else [ordered[-1]]
        else:
            preferred = [ordered[0]]

    chosen = os.path.join(image_root, sorted(preferred)[0])
    with open(chosen, "rb") as f:
        return f.read()


def _load_exact_image_file(image_root: str, filename: str) -> bytes:
    """Load an exact image file from the local image directory."""
    path = os.path.join(image_root, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    with open(path, "rb") as f:
        return f.read()


def _render_neuron_image_card(title: str, subtitle: str, image_bytes: bytes) -> None:
    """Render image in a fixed white card while preserving aspect ratio."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data_url = base64.b64encode(buf.getvalue()).decode("utf-8")
    st.markdown(
        f"""
        <div style="
          border-radius: 14px;
          border: 1px solid rgba(120, 145, 200, 0.28);
          background: rgba(255,255,255,0.95);
          padding: 12px 12px 14px 12px;
          min-height: 430px;
          box-shadow: 0 8px 18px rgba(30,55,110,0.10);
        ">
          <div style="font-size:30px; font-weight:800; color:rgba(22,34,64,0.96); margin-bottom:4px;">{title}</div>
          <div style="font-size:18px; font-weight:600; color:rgba(68,83,118,0.90); margin-bottom:8px;">{subtitle}</div>
          <div style="
            background:#ffffff;
            border:1px solid rgba(160,175,210,0.25);
            border-radius:12px;
            height:340px;
            display:flex;
            align-items:center;
            justify-content:center;
            overflow:hidden;
          ">
            <img src="data:image/png;base64,{data_url}" style="max-width:96%; max-height:96%; object-fit:contain;" />
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _build_social_neuron_prompt(
    node_idx: int,
    node_names: List[str],
    matrix: np.ndarray,
    frame_label: str,
    components_label: str,
) -> str:
    """Create a neuron-only prompt from highlighted-node connectivity rules."""
    m = np.asarray(matrix, dtype=float)
    row = m[node_idx].copy()
    row[node_idx] = 0.0

    # Active edges around the highlighted node.
    active = row > 0.05
    edge_count = int(np.sum(active))
    edge_ratio = edge_count / max(1, len(row) - 1)

    # Rule 1) Dendrite count from average edge count (mapped to 3..18 branches).
    dendrite_count = int(np.clip(round(3 + 15 * edge_ratio), 3, 18))

    # Rule 2) Dendrite thickness from average edge strength.
    avg_edge_strength = float(np.mean(row[active])) if np.any(active) else 0.0
    dendrite_thickness = float(np.clip(0.6 + 4.6 * avg_edge_strength, 0.6, 5.2))

    # Rule 3) Axon length from average node distance (inverse of edge weights as proxy distance).
    if np.any(active):
        dist_proxy = 1.0 / np.clip(row[active], 1e-4, 1.0)
        avg_distance = float(np.mean(dist_proxy))
    else:
        avg_distance = 12.0
    axon_length_level = (
        "short"
        if avg_distance < 3.0
        else ("medium" if avg_distance < 8.0 else "long")
    )

    order = np.argsort(-row)
    top = [int(i) for i in order if int(i) != node_idx and row[int(i)] > 0][:4]
    top_links = ", ".join(f"{node_names[i]} ({row[i]:.2f})" for i in top) if top else "none"

    return (
        "Generate a clean scientific neuron illustration ONLY. "
        f"Context: {node_names[node_idx]}, frame {frame_label}, mode {components_label}. "
        f"Neuron morphology rules: soma is central; dendrite count = {dendrite_count}; "
        f"dendrite thickness = {dendrite_thickness:.2f}px equivalent; "
        f"axon length should be {axon_length_level} based on average node distance ({avg_distance:.2f}). "
        f"Strongest synaptic hints: {top_links}. "
        "Render a single isolated neuron with visible soma, dendrites, one axon, and axon terminals. "
        "No human face, no body, no portrait, no characters, no text, no logo, no watermark. "
        "Background must be plain white or transparent-looking very light neutral background. "
        "Minimal composition, high anatomical clarity, soft realistic shading, centered subject."
    )


@st.cache_resource(show_spinner=False)
def _cached_layout(adjacency_tensor: np.ndarray, spring_seed: int):
    """Cache the stable layout to avoid node jumping and expensive recompute."""
    return get_stable_layout(adjacency_tensor, spring_seed=spring_seed)


@st.cache_data(show_spinner=False)
def _default_node_names(n_nodes: int, seed: int = 17) -> List[str]:
    """Generate deterministic but human-like default names."""
    pool = [
        "Alex", "Mina", "Daniel", "Sofia", "Leo", "Hana", "Noah", "Yuna", "Ethan", "Jisoo",
        "Ava", "Liam", "Emma", "Oliver", "Amelia", "Mason", "Isla", "Lucas", "Mia", "Henry",
        "Aiden", "Nora", "Elijah", "Aria", "James", "Chloe", "Logan", "Grace", "Evelyn", "Harper",
        "Asher", "Ella", "Wyatt", "Zoey", "Samuel", "Luna", "Jacob", "Sage", "Thea", "Ezra",
        "Iris", "Julian", "Ruby", "Owen", "Hazel", "Mila", "Kai", "Nina", "Aaron", "Clara",
    ]
    rng = np.random.default_rng(seed)
    if n_nodes <= len(pool):
        return rng.choice(pool, size=n_nodes, replace=False).tolist()

    out = pool.copy()
    idx = 1
    while len(out) < n_nodes:
        out.append(f"Guest {idx}")
        idx += 1
    rng.shuffle(out)
    return out[:n_nodes]


def _safe_node_names(raw_text: str, n_nodes: int, fallback_names: List[str]) -> List[str]:
    """Parse comma-separated node names and fill missing names."""
    if not raw_text.strip():
        return fallback_names

    pieces = [p.strip() for p in raw_text.split(",")]
    names = []
    for i in range(n_nodes):
        if i < len(pieces) and pieces[i]:
            names.append(pieces[i])
        else:
            names.append(fallback_names[i])
    return names


def _positions_from_coordinates(coords: np.ndarray) -> dict:
    """Convert (N,2) coordinates into normalized plot positions."""
    arr = np.asarray(coords, dtype=float)
    x = arr[:, 0]
    y = arr[:, 1]
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    xr = max(1e-9, x_max - x_min)
    yr = max(1e-9, y_max - y_min)

    # Normalize to roughly [-1, 1] and flip y for more natural top-down display.
    xn = ((x - x_min) / xr) * 2.0 - 1.0
    yn = -(((y - y_min) / yr) * 2.0 - 1.0)
    return {i: np.array([float(xn[i]), float(yn[i])]) for i in range(arr.shape[0])}


def _human_interpretation(summary: dict, time_idx: int) -> str:
    """Generate a short human-readable interpretation line."""
    if summary["num_active_edges"] == 0:
        return f"At time step {time_idx}, no relationships pass the threshold."
    return (
        f"At time step {time_idx}, {summary['most_connected_node']} is currently the most connected, "
        f"and the strongest link is {summary['strongest_connection']}."
    )


@st.cache_data(show_spinner=False)
def _build_month_labels(num_steps: int, start_year: int = 2021, start_month: int = 1) -> List[str]:
    """Return month labels like Jan 2021, Feb 2021, ..."""
    labels: List[str] = []
    y = start_year
    m = start_month
    for _ in range(num_steps):
        labels.append(datetime(y, m, 1).strftime("%b %Y"))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return labels


@st.cache_data(show_spinner=False)
def _load_world_map_image(image_path: str) -> tuple[Image.Image, int, int]:
    """Load image and return (PIL RGBA image, width, height)."""
    with Image.open(image_path) as img:
        rgba = img.convert("RGBA")
        arr = np.array(rgba)

    # Match transparent regions to surrounding app/plot background tone.
    out = np.full((arr.shape[0], arr.shape[1], 3), [245, 248, 255], dtype=np.uint8)

    # Make map clearly visible on bright backgrounds.
    mask = arr[:, :, 3] > 0
    out[mask, 0] = 66
    out[mask, 1] = 98
    out[mask, 2] = 152

    out_img = Image.fromarray(out, mode="RGB").convert("RGBA")
    w, h = out_img.size
    return out_img, w, h


@st.cache_data(show_spinner=False)
def _load_world_demo_data(
    num_steps: int, random_seed: int = 11
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Build demo tensors: total, structural, functional, coords, names."""
    t_max = max(6, int(num_steps))
    num_nodes = 12
    rng = np.random.default_rng(random_seed)

    us_idx = list(range(6))
    kr_idx = list(range(6, 12))
    sehong_idx = kr_idx[2]  # keep stable for demo narrative

    # Names: random roster, one fixed Sehong.
    name_pool = [
        "Amy", "Brian", "Chloe", "Daniel", "Emma", "Felix", "Grace", "Hannah", "Ian", "Julia",
        "Kevin", "Liam", "Mia", "Noah", "Olivia", "Peter", "Quinn", "Ryan", "Sophie", "Tyler",
        "Uma", "Victor", "Wendy", "Xavier", "Yuna", "Zane",
    ]
    guaranteed = ["Amy", "Peter"]
    candidates = [n for n in name_pool if n not in guaranteed]
    selected_rest = rng.choice(candidates, size=num_nodes - 1 - len(guaranteed), replace=False).tolist()
    selected = guaranteed + selected_rest
    rng.shuffle(selected)
    names: List[str] = []
    src_i = 0
    for i in range(num_nodes):
        if i == sehong_idx:
            names.append("Sehong")
        else:
            names.append(selected[src_i])
            src_i += 1

    # Keep key narrative nodes in the US cluster for clearer demo dynamics.
    for target_name, target_pos in [("Amy", us_idx[0]), ("Peter", us_idx[1])]:
        cur = names.index(target_name)
        if cur != target_pos and target_pos != sehong_idx:
            names[cur], names[target_pos] = names[target_pos], names[cur]

    # Pixel anchors on the provided map.
    us_anchor = np.array([380.0, 245.0])
    kr_anchor = np.array([1290.0, 250.0])
    base_coords = np.zeros((num_nodes, 2), dtype=float)
    for i in us_idx:
        base_coords[i] = us_anchor + rng.normal(loc=[0.0, 0.0], scale=[85.0, 60.0], size=2)
    for i in kr_idx:
        base_coords[i] = kr_anchor + rng.normal(loc=[0.0, 0.0], scale=[55.0, 40.0], size=2)

    # Time-varying coordinates: only Sehong migrates KR -> US.
    coords_t = np.repeat(base_coords[None, :, :], t_max, axis=0)
    # Fixed migration window: Jun 2022 ~ Aug 2022 on a Jan 2019 timeline.
    start_move = min(t_max - 1, (2022 - 2019) * 12 + (6 - 1))
    end_move = min(t_max - 1, (2022 - 2019) * 12 + (8 - 1))
    sehong_start = base_coords[sehong_idx].copy()
    sehong_end = us_anchor + rng.normal(loc=[0.0, 0.0], scale=[45.0, 35.0], size=2)
    for t in range(t_max):
        if t <= start_move:
            coords_t[t, sehong_idx] = sehong_start
        elif t >= end_move:
            coords_t[t, sehong_idx] = sehong_end
        else:
            alpha = (t - start_move) / max(1, (end_move - start_move))
            coords_t[t, sehong_idx] = (1 - alpha) * sehong_start + alpha * sehong_end

    coords_t[:, :, 0] = np.clip(coords_t[:, :, 0], 35.0, 1565.0)
    coords_t[:, :, 1] = np.clip(coords_t[:, :, 1], 35.0, 678.0)

    # Functional base matrix with region preference (shared interests / habits).
    base_func = np.zeros((num_nodes, num_nodes), dtype=float)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            same_region = (i in us_idx and j in us_idx) or (i in kr_idx and j in kr_idx)
            if same_region:
                w0 = rng.uniform(0.45, 0.86)
            else:
                w0 = rng.uniform(0.08, 0.28)
            base_func[i, j] = base_func[j, i] = w0

    # Sehong starts functionally KR-heavy and US-light.
    for j in us_idx:
        base_func[sehong_idx, j] = base_func[j, sehong_idx] = rng.uniform(0.03, 0.12)
    for j in kr_idx:
        if j != sehong_idx:
            base_func[sehong_idx, j] = base_func[j, sehong_idx] = rng.uniform(0.5, 0.85)

    # Functional connectivity with migration-driven preference shifts.
    functional = np.zeros((t_max, num_nodes, num_nodes), dtype=float)
    phase = rng.uniform(0, 2 * np.pi, size=(num_nodes, num_nodes))
    freq = rng.uniform(0.4, 1.4, size=(num_nodes, num_nodes))
    amy_idx = names.index("Amy")
    peter_idx = names.index("Peter")
    # Keep only a subset of US ties strongly strengthened after migration.
    strong_us_names = {"Amy", "Peter"}
    strong_us_idx = {names.index(n) for n in strong_us_names if n in names}
    weak_us_idx = [j for j in us_idx if j not in strong_us_idx]
    sehong_us_target_strong = {j: float(rng.uniform(0.42, 0.60)) for j in strong_us_idx}
    sehong_us_target_weak = {j: float(rng.uniform(0.05, 0.18)) for j in weak_us_idx}
    sehong_kr_target = {j: float(rng.uniform(0.34, 0.55)) for j in kr_idx if j != sehong_idx}
    sehong_special_start = {
        amy_idx: 0.0,
        peter_idx: 0.0,
    }
    sehong_special_target = {
        amy_idx: float(rng.uniform(0.78, 0.82)),
        peter_idx: float(rng.uniform(0.78, 0.82)),
    }
    weak_contact_names = ["Kevin", "Mia", "Ryan", "Quinn"]
    weak_contact_idx = [names.index(n) for n in weak_contact_names if n in names and names.index(n) != sehong_idx]
    special_full_idx = min(t_max - 1, end_move + 18)
    for t in range(t_max):
        tau = 2 * np.pi * (t / max(1, t_max - 1))
        m = base_func.copy()
        dyn = 0.05 * np.sin(freq * tau + phase)
        m += np.triu(dyn, 1)
        m = np.triu(m, 1)
        m = m + m.T

        if t <= start_move:
            travel_alpha = 0.0
        elif t >= end_move:
            travel_alpha = 1.0
        else:
            travel_alpha = (t - start_move) / max(1, (end_move - start_move))

        # As Sehong arrives in US, US links become much stronger and KR links soften.
        for j in us_idx:
            old = float(m[sehong_idx, j])
            if j in sehong_us_target_strong:
                target = sehong_us_target_strong[j]
            else:
                target = sehong_us_target_weak[j]
            # Non-key US ties should stay limited after migration.
            if j in sehong_us_target_strong:
                blend = travel_alpha
            else:
                blend = min(1.0, 1.15 * travel_alpha)
            new = (1.0 - blend) * old + blend * target
            m[sehong_idx, j] = m[j, sehong_idx] = new
        for j in kr_idx:
            if j == sehong_idx:
                continue
            old = float(m[sehong_idx, j])
            target = sehong_kr_target[j]
            new = (1.0 - travel_alpha) * old + travel_alpha * target
            m[sehong_idx, j] = m[j, sehong_idx] = new

        # Explicitly highlight new close ties after moving abroad.
        for j, target in sehong_special_target.items():
            if t <= start_move:
                special_alpha = 0.0
            elif t >= special_full_idx:
                special_alpha = 1.0
            else:
                special_alpha = (t - start_move) / max(1, (special_full_idx - start_move))
            smooth_alpha = special_alpha * special_alpha * (3.0 - 2.0 * special_alpha)
            start_val = sehong_special_start[j]
            new = (1.0 - smooth_alpha) * start_val + smooth_alpha * target
            m[sehong_idx, j] = m[j, sehong_idx] = new

        # Keep selected ties nearly disconnected across the timeline.
        for j in weak_contact_idx:
            m[sehong_idx, j] = m[j, sehong_idx] = min(float(m[sehong_idx, j]), 0.03)

        m = np.clip(m, 0.0, 1.0)
        np.fill_diagonal(m, 0.0)
        functional[t] = m

    # Structural connectivity is distance-driven (closer => stronger).
    structural = np.zeros((t_max, num_nodes, num_nodes), dtype=float)
    sigma = 340.0
    for t in range(t_max):
        c = coords_t[t]
        diff = c[:, None, :] - c[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        s = np.exp(-((dist / sigma) ** 2))
        s = np.clip(s, 0.0, 1.0)
        np.fill_diagonal(s, 0.0)
        s = (s + s.T) * 0.5
        for j in weak_contact_idx:
            s[sehong_idx, j] = s[j, sehong_idx] = min(float(s[sehong_idx, j]), 0.03)
        structural[t] = s

    # Total connectivity: sum-like blend of both factors, kept in [0,1].
    total = np.clip(0.5 * structural + 0.5 * functional, 0.0, 1.0)
    return total, structural, functional, coords_t, names


def _positions_from_pixels(coords: np.ndarray) -> dict:
    """Convert (N,2) pixel coordinates directly to plot positions."""
    arr = np.asarray(coords, dtype=float)
    return {i: np.array([float(arr[i, 0]), float(arr[i, 1])]) for i in range(arr.shape[0])}


def _apply_world_map_overlay(fig: go.Figure, image_obj: Image.Image, width: int, height: int) -> None:
    """Overlay world map image while preserving its native aspect ratio."""
    fig.add_layout_image(
        dict(
            source=image_obj,
            xref="x",
            yref="y",
            x=0,
            y=height,
            sizex=width,
            sizey=height,
            sizing="contain",
            opacity=1.0,
            layer="below",
        )
    )
    fig.update_xaxes(range=[0, width], constrain="domain")
    fig.update_yaxes(range=[height, 0], scaleanchor="x", scaleratio=1, constrain="domain")
    fig.update_layout(
        height=620,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(245,248,255,1)",
    )


def _edge_color(weight: float, alpha: float = 0.62) -> str:
    """Map edge weight to RGBA orange color."""
    v = min(1.0, max(0.0, float(weight)))
    r = int(255 - (1 - v) * 25)
    g = int(140 + (1 - v) * 55)
    b = int(40 + (1 - v) * 35)
    return f"rgba({r},{g},{b},{alpha})"


def _build_demo_fixed_animation(
    adjacency_tensor: np.ndarray,
    coords_t: np.ndarray,
    node_names: List[str],
    edge_threshold: float,
    show_labels: bool,
    frame_duration_ms: int,
    start_idx: int,
    focus_node: int | None,
    frame_labels: List[str],
    map_img: Image.Image,
    map_w: int,
    map_h: int,
    heatmap_title: str,
) -> go.Figure:
    """Stable demo animation with synchronized map + heatmap in one figure."""
    t_max, n_nodes, _ = adjacency_tensor.shape
    edge_pairs = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]

    def frame_data(t: int):
        mat = adjacency_tensor[t]
        pos = coords_t[t]
        g = build_graph_from_adjacency(mat, threshold=edge_threshold, node_names=node_names)
        deg = compute_weighted_degrees(g)
        sizes = compute_node_sizes(deg)

        focus_neighbors = set(g.neighbors(focus_node)) if focus_node is not None and focus_node in g else set()
        focus_group = focus_neighbors | ({focus_node} if focus_node is not None and focus_node in g else set())

        traces = []
        for u, v in edge_pairs:
            w = float(mat[u, v])
            active = w >= edge_threshold
            x0, y0 = float(pos[u, 0]), float(pos[u, 1])
            x1, y1 = float(pos[v, 0]), float(pos[v, 1])

            if not active:
                color = "rgba(0,0,0,0)"
                width = 0.1
            else:
                if focus_node is None or (u in focus_group and v in focus_group):
                    color = _edge_color(w)
                    width = 0.6 + 4.2 * w
                else:
                    color = "rgba(140,150,170,0.10)"
                    width = 0.35 + 0.7 * w

            traces.append(
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

        node_x = [float(pos[i, 0]) for i in range(n_nodes)]
        node_y = [float(pos[i, 1]) for i in range(n_nodes)]
        node_text, node_sizes, node_colors = [], [], []
        for i in range(n_nodes):
            nbrs = [node_names[n] for n in g.neighbors(i)]
            nbr_s = ", ".join(nbrs[:8]) if nbrs else "None"
            if len(nbrs) > 8:
                nbr_s += ", ..."
            node_text.append(f"Node: {node_names[i]}<br>Neighbors: {nbr_s}")
            node_sizes.append(sizes.get(i, 22.0))
            if focus_node is None:
                node_colors.append("rgba(37,99,235,0.88)")
            else:
                if i == focus_node:
                    node_colors.append("rgba(239,68,68,0.96)")
                elif i in focus_group:
                    node_colors.append("rgba(37,99,235,0.92)")
                else:
                    node_colors.append("rgba(175,186,205,0.25)")

        traces.append(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text" if show_labels else "markers",
                text=node_names if show_labels else None,
                textposition="top center",
                textfont=dict(size=12, color="rgba(25,30,45,0.95)"),
                hoverinfo="text",
                hovertext=node_text,
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    line=dict(width=1.2, color="rgba(255,255,255,0.95)"),
                    opacity=0.95,
                ),
                showlegend=False,
            )
        )
        traces.append(
            go.Heatmap(
                z=mat,
                x=node_names,
                y=node_names,
                xaxis="x2",
                yaxis="y2",
                colorscale="YlGnBu",
                zmin=0,
                zmax=1,
                colorbar=dict(title="Weight", x=0.995, y=0.5, len=0.86, thickness=10),
                hovertemplate="From %{y}<br>To %{x}<br>Weight %{z:.3f}<extra></extra>",
            )
        )
        return traces

    initial_data = frame_data(start_idx)
    total_traces = len(initial_data)
    fig = go.Figure(
        data=initial_data,
        frames=[
            go.Frame(
                name=str(t),
                data=frame_data(t),
                traces=list(range(total_traces)),
            )
            for t in range(t_max)
        ],
    )
    fig.add_layout_image(
        dict(
            source=map_img,
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=map_w,
            sizey=map_h,
            sizing="contain",
            opacity=0.95,
            layer="below",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(245,248,255,1)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=20),
        xaxis=dict(
            domain=[0.0, 0.66],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[0, map_w],
            constrain="domain",
        ),
        yaxis=dict(
            domain=[0.10, 1.0],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[map_h, 0],
            scaleanchor="x",
            scaleratio=1,
        ),
        xaxis2=dict(
            domain=[0.70, 0.98],
            side="top",
            tickangle=30,
            tickfont=dict(size=10, color="rgba(70,85,120,0.9)"),
        ),
        yaxis2=dict(
            domain=[0.10, 1.0],
            autorange="reversed",
            side="right",
            tickfont=dict(size=10, color="rgba(70,85,120,0.9)"),
        ),
        height=620,
        hovermode="closest",
        annotations=[
            dict(
                text="<b>Connectivity Map</b>",
                x=0.0,
                y=1.08,
                xref="paper",
                yref="paper",
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                font=dict(size=18, color="rgba(24,36,64,0.95)"),
            ),
            dict(
                text=f"<b>{heatmap_title}</b>",
                x=0.70,
                y=1.08,
                xref="paper",
                yref="paper",
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                font=dict(size=18, color="rgba(24,36,64,0.95)"),
            ),
        ],
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": 1.08,
                "xanchor": "left",
                "yanchor": "bottom",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": frame_duration_ms, "redraw": True},
                                "transition": {"duration": int(frame_duration_ms * 0.9), "easing": "cubic-in-out"},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": start_idx,
                "x": 0.0,
                "y": 0.02,
                "len": 0.98,
                "xanchor": "left",
                "yanchor": "bottom",
                "currentvalue": {"prefix": "Frame: "},
                "steps": [
                    {
                        "label": frame_labels[t] if t < len(frame_labels) else str(t),
                        "method": "animate",
                        "args": [[str(t)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                    }
                    for t in range(t_max)
                ],
            }
        ],
    )
    return fig


def _inject_styles() -> None:
    """Apply a polished visual style to the app."""
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at 10% 20%, #f5f9ff 0%, #e6eefc 45%, #dbe7ff 100%);
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1.5rem;
        }
        .top-banner {
            border-radius: 18px;
            padding: 18px 22px;
            background: linear-gradient(135deg, rgba(9,35,82,0.95), rgba(28,64,145,0.9));
            color: #f7fbff;
            box-shadow: 0 10px 28px rgba(24, 49, 102, 0.25);
            margin-bottom: 0.8rem;
        }
        .metric-card {
            border-radius: 14px;
            padding: 8px 10px;
            background: rgba(255,255,255,0.72);
            border: 1px solid rgba(109, 139, 199, 0.25);
            box-shadow: 0 6px 16px rgba(26, 56, 120, 0.12);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _build_animated_graph_figure(
    adjacency_tensor: np.ndarray,
    positions: dict,
    positions_by_time: np.ndarray | None,
    node_names: List[str],
    edge_threshold: float,
    show_labels: bool,
    color_edges_by_weight: bool,
    color_nodes_by_degree: bool,
    frame_duration_ms: int,
    start_idx: int,
    graph_title_prefix: str,
    focus_node: int | None,
    square_map: bool,
    frame_labels: List[str] | None = None,
) -> go.Figure:
    """Create Plotly frame animation for smooth graph playback (no Streamlit reruns)."""
    t_max = adjacency_tensor.shape[0]

    frames = []
    snapshots = []
    for t in range(t_max):
        matrix = adjacency_tensor[t]
        pos_t = positions if positions_by_time is None else _positions_from_pixels(positions_by_time[t])
        graph = build_graph_from_adjacency(matrix, threshold=edge_threshold, node_names=node_names)
        weighted_degrees = compute_weighted_degrees(graph)
        node_sizes = compute_node_sizes(weighted_degrees)

        snap = plot_graph_plotly(
            graph=graph,
            positions=pos_t,
            node_names=node_names,
            node_sizes=node_sizes,
            weighted_degrees=weighted_degrees,
            show_labels=show_labels,
            color_edges_by_weight=color_edges_by_weight,
            color_nodes_by_degree=color_nodes_by_degree,
            title=f"{graph_title_prefix}",
            focus_node=focus_node,
            square_map=square_map,
        )
        snapshots.append(snap)
        frames.append(go.Frame(name=str(t), data=snap.data, layout=go.Layout(title=snap.layout.title)))

    initial = snapshots[start_idx] if snapshots else go.Figure()
    fig = go.Figure(data=initial.data, layout=initial.layout, frames=frames)

    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": 1.08,
                "xanchor": "left",
                "yanchor": "bottom",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": frame_duration_ms, "redraw": False},
                                "transition": {"duration": int(frame_duration_ms * 0.9), "easing": "cubic-in-out"},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": start_idx,
                "x": 0.0,
                "y": -0.04,
                "len": 1.0,
                "xanchor": "left",
                "yanchor": "top",
                "currentvalue": {"prefix": "Frame: "},
                "steps": [
                    {
                        "label": frame_labels[t] if frame_labels and t < len(frame_labels) else str(t),
                        "method": "animate",
                        "args": [[str(t)], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}],
                    }
                    for t in range(t_max)
                ],
            }
        ],
    )
    return fig


def _build_animated_heatmap_figure(
    adjacency_tensor: np.ndarray,
    node_names: List[str],
    frame_duration_ms: int,
    start_idx: int,
    frame_labels: List[str] | None = None,
) -> go.Figure:
    """Animated adjacency heatmap across time."""
    t_max = adjacency_tensor.shape[0]

    def heat(t: int) -> go.Heatmap:
        return go.Heatmap(
            z=adjacency_tensor[t],
            x=node_names,
            y=node_names,
            colorscale="YlGnBu",
            zmin=0,
            zmax=1,
            colorbar=dict(title="Weight"),
            hovertemplate="From %{y}<br>To %{x}<br>Weight %{z:.3f}<extra></extra>",
        )

    fig = go.Figure(
        data=[heat(start_idx)],
        frames=[go.Frame(name=str(t), data=[heat(t)]) for t in range(t_max)],
    )
    fig.update_layout(
        title=None,
        paper_bgcolor="rgba(245,248,255,1)",
        plot_bgcolor="rgba(245,248,255,1)",
        margin=dict(l=10, r=10, t=10, b=20),
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed"),
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": 1.08,
                "xanchor": "left",
                "yanchor": "bottom",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": frame_duration_ms, "redraw": False},
                                "transition": {"duration": int(frame_duration_ms * 0.9), "easing": "cubic-in-out"},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": start_idx,
                "x": 0.0,
                "y": -0.04,
                "len": 1.0,
                "xanchor": "left",
                "yanchor": "top",
                "currentvalue": {"prefix": "Frame: "},
                "steps": [
                    {
                        "label": frame_labels[t] if frame_labels and t < len(frame_labels) else str(t),
                        "method": "animate",
                        "args": [[str(t)], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}],
                    }
                    for t in range(t_max)
                ],
            }
        ],
    )
    return fig


def main() -> None:
    """App entrypoint."""
    _inject_styles()

    st.markdown(
        """
        <div class="top-banner">
          <h2 style="margin:0 0 6px 0;">Social Connectome</h2>
          <div style="font-size:15px; opacity:0.95;">
            We turn human relationships into a social brain, inspired by neural connection.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Controls")

        mode = st.radio("Data source", ["Upload", "Brain ROI (Functional Connectivity Only)", "Demo"], index=2)

        symmetrize = st.checkbox("Auto-symmetrize uploaded matrices", value=True)
        spring_seed = 42
        edge_threshold = st.slider("Edge threshold", min_value=0.0, max_value=1.0, value=0.15, step=0.01)
        show_labels = st.checkbox("Show node labels", value=True)
        color_edges = st.checkbox("Color edges by weight", value=True)

        smooth_animation = st.checkbox("Smooth animation mode", value=True)
        animation_speed = st.slider("Animation speed (sec/frame)", 0.05, 1.0, 0.20, 0.05)

        adjacency_tensor = None
        structural_tensor = None
        functional_tensor = None
        selected_components: List[str] = []
        upload_warning = None
        uploaded_coords = None
        demo_coords_t = None
        frame_labels = None
        world_map_img = None
        world_map_size = None

        if mode == "Upload":
            uploaded = st.file_uploader("Upload adjacency file", type=["csv", "npy"])
            uploaded_coords = st.file_uploader(
                "Optional node coordinates file (N,2 or 2,N in csv/npy)",
                type=["csv", "npy"],
            )
            if uploaded_coords is None:
                spring_seed = st.number_input("Spring layout seed", min_value=0, value=42, step=1)
            if uploaded is not None:
                try:
                    adjacency_tensor, upload_warning = load_adjacency_file(uploaded, symmetrize=symmetrize)
                except Exception as exc:
                    st.error(f"Could not parse file: {exc}")
            else:
                st.info("Upload a `.csv` or `.npy` to start.")

        elif mode == "Brain ROI (Functional Connectivity Only)":
            local_data_dir = st.text_input("Local data directory", value="/home/acrl/research/Neurohack/data")
            try:
                adjacency_tensor, uploaded_coords = load_local_brain_dataset(local_data_dir, symmetrize=symmetrize)
                adjacency_tensor = adjacency_tensor[:, :50, :50]
                uploaded_coords = uploaded_coords[:50]
                st.success("Loaded ROI.csv and ROI_XY.csv.")
            except Exception as exc:
                st.error(f"Could not load local brain data: {exc}")

        else:
            demo_steps = (2026 - 2019 + 1) * 12  # Jan 2019 ~ Dec 2026
            frame_labels = _build_month_labels(demo_steps, start_year=2019, start_month=1)
            total_tensor, structural_tensor, functional_tensor, demo_coords_t, default_demo_names = _load_world_demo_data(demo_steps)
            show_structural = st.checkbox("Show structural connectivity", value=True)
            show_functional = st.checkbox("Show functional connectivity", value=True)
            selected_components = []
            if show_structural:
                selected_components.append("Structural")
            if show_functional:
                selected_components.append("Functional")
            if not selected_components:
                selected_components = ["Structural"]
                st.caption("At least one component is required. Falling back to Structural.")

            if selected_components == ["Structural"]:
                adjacency_tensor = structural_tensor
            elif selected_components == ["Functional"]:
                adjacency_tensor = functional_tensor
            else:
                adjacency_tensor = np.clip(0.5 * structural_tensor + 0.5 * functional_tensor, 0.0, 1.0)
            world_map_path = "/home/acrl/Downloads/world-map-png-35423.png"
            if os.path.exists(world_map_path):
                world_map_img, w, h = _load_world_map_image(world_map_path)
                world_map_size = (w, h)
            else:
                st.warning(f"World map image not found: {world_map_path}")

    if adjacency_tensor is None:
        st.stop()

    if upload_warning:
        st.warning(upload_warning)

    t_max, n_nodes, _ = adjacency_tensor.shape

    if t_max > 1:
        if mode == "Demo" and frame_labels is not None:
            selected_label = st.select_slider(
                "Time frame",
                options=frame_labels,
                value=frame_labels[0],
            )
            time_idx = frame_labels.index(selected_label)
        else:
            time_idx = st.slider(
                "Snapshot time step",
                min_value=0,
                max_value=t_max - 1,
                value=0,
                step=1,
            )
    else:
        time_idx = 0
        st.caption("Single-step dataset detected (T=1).")

    if mode == "Brain ROI (Functional Connectivity Only)":
        default_names = [f"neuron{i + 1}" for i in range(n_nodes)]
    elif mode == "Demo":
        default_names = default_demo_names
    else:
        default_names = _default_node_names(n_nodes)
    node_names = default_names

    focus_options = ["None"] + node_names
    focus_label = st.selectbox("Highlight node", options=focus_options, index=0)
    focus_node_idx = None if focus_label == "None" else node_names.index(focus_label)

    layout = _cached_layout(adjacency_tensor, int(spring_seed))
    if mode == "Upload" and uploaded_coords is not None:
        try:
            coords = load_coordinates_file(uploaded_coords, expected_nodes=n_nodes)
            layout = _positions_from_coordinates(coords)
            st.caption("Using uploaded node coordinates for fixed spatial placement.")
        except Exception as exc:
            st.warning(f"Coordinate file ignored: {exc}")
    elif mode == "Brain ROI (Functional Connectivity Only)" and uploaded_coords is not None:
        layout = _positions_from_coordinates(uploaded_coords)
    elif mode == "Demo" and demo_coords_t is not None:
        layout = _positions_from_pixels(demo_coords_t[time_idx])

    current_matrix = adjacency_tensor[time_idx]
    graph = build_graph_from_adjacency(current_matrix, threshold=edge_threshold, node_names=node_names)
    weighted_degrees = compute_weighted_degrees(graph)
    node_sizes = compute_node_sizes(weighted_degrees)

    graph_title_prefix = "Connectivity Map"
    square_map = mode == "Brain ROI (Functional Connectivity Only)"
    static_graph_title = f"{graph_title_prefix}"
    fig_graph_static = plot_graph_plotly(
        graph=graph,
        positions=layout,
        node_names=node_names,
        node_sizes=node_sizes,
        weighted_degrees=weighted_degrees,
        show_labels=show_labels,
        color_edges_by_weight=color_edges,
        color_nodes_by_degree=False,
        title="",
        focus_node=focus_node_idx,
        square_map=square_map,
    )

    fig_matrix = plot_matrix_heatmap(current_matrix, node_names, title="")
    if mode == "Demo" and world_map_img is not None and world_map_size is not None:
        _apply_world_map_overlay(fig_graph_static, world_map_img, world_map_size[0], world_map_size[1])

    summary = summarize_graph(graph, node_names)

    selected_strength = None
    if focus_node_idx is not None and focus_node_idx in graph:
        edge_weights = [float(data.get("weight", 0.0)) for _, _, data in graph.edges(focus_node_idx, data=True)]
        selected_strength = float(np.mean(edge_weights)) if edge_weights else 0.0

    stats_cols = st.columns(5)
    stats_cols[0].metric("Active edges", summary["num_active_edges"])
    stats_cols[1].metric("Avg Connection Strength", f"{summary['average_weight']:.3f}")
    stats_cols[2].metric("Strongest connection", summary["strongest_connection"])
    stats_cols[3].metric("Most connected node", summary["most_connected_node"])
    stats_cols[4].metric(
        "Selected Avg Connection Strength",
        f"{selected_strength:.3f}" if selected_strength is not None else "-",
    )

    st.info(_human_interpretation(summary, time_idx))
    if mode == "Demo":
        comp_label = " + ".join(selected_components) if selected_components else "Structural"
        st.caption(f"Current score view: {comp_label} connectivity")
    else:
        comp_label = "Connectivity"

    if mode == "Demo" and demo_coords_t is not None and world_map_img is not None and world_map_size is not None:
        heatmap_title = "Adjacency Heatmap"
        if selected_components:
            heatmap_title = f"Adjacency Heatmap ({' + '.join(selected_components)})"
        demo_anim = _build_demo_fixed_animation(
            adjacency_tensor=adjacency_tensor,
            coords_t=demo_coords_t,
            node_names=node_names,
            edge_threshold=edge_threshold,
            show_labels=show_labels,
            frame_duration_ms=int(animation_speed * 1000),
            start_idx=time_idx,
            focus_node=focus_node_idx,
            frame_labels=frame_labels if frame_labels else [str(i) for i in range(t_max)],
            map_img=world_map_img,
            map_w=world_map_size[0],
            map_h=world_map_size[1],
            heatmap_title=heatmap_title,
        )
        st.plotly_chart(demo_anim, use_container_width=True)
    else:
        col_left, col_right = st.columns([1.7, 1.1])
        with col_left:
            st.markdown(f"<div><strong>{static_graph_title}</strong></div>", unsafe_allow_html=True)
            if smooth_animation and t_max > 1:
                frame_duration_ms = int(animation_speed * 1000)
                animated_fig = _build_animated_graph_figure(
                    adjacency_tensor=adjacency_tensor,
                    positions=layout,
                    positions_by_time=demo_coords_t if mode == "Demo" else None,
                    node_names=node_names,
                    edge_threshold=edge_threshold,
                    show_labels=show_labels,
                    color_edges_by_weight=color_edges,
                    color_nodes_by_degree=False,
                    frame_duration_ms=frame_duration_ms,
                    start_idx=time_idx,
                    graph_title_prefix="",
                    focus_node=focus_node_idx,
                    square_map=square_map,
                    frame_labels=frame_labels if mode == "Demo" else None,
                )
                animated_fig.update_layout(title=None)
                if mode == "Demo" and world_map_img is not None and world_map_size is not None:
                    _apply_world_map_overlay(animated_fig, world_map_img, world_map_size[0], world_map_size[1])
                st.plotly_chart(animated_fig, use_container_width=True)
            else:
                st.plotly_chart(fig_graph_static, use_container_width=True)

        with col_right:
            st.markdown("<div><strong>Adjacency Heatmap</strong></div>", unsafe_allow_html=True)
            if smooth_animation and t_max > 1:
                heat_anim = _build_animated_heatmap_figure(
                    adjacency_tensor=adjacency_tensor,
                    node_names=node_names,
                    frame_duration_ms=int(animation_speed * 1000),
                    start_idx=time_idx,
                    frame_labels=frame_labels if mode == "Demo" else None,
                )
                st.plotly_chart(heat_anim, use_container_width=True)
            else:
                st.plotly_chart(fig_matrix, use_container_width=True)

    st.subheader("Visuazlize Social Neuron")
    image_root = "/home/acrl/research/Neurohack/image"

    if focus_node_idx is None:
        st.caption("Choose a `Highlight node` first to enable neuron visualization.")
        st.session_state["show_neuron_panel"] = False
    else:
        if st.button("Visuazlize Social Neuron", type="primary"):
            st.session_state["show_neuron_panel"] = True

    if st.session_state.get("show_neuron_panel", False):
        friend_choice = st.selectbox("This Person May Match Your Vibe", options=["Mia", "Quinn"], index=0, key="friend_choice")
        friend_image_file = "energetic.jpg" if friend_choice == "Mia" else "calm.jpg"

        col_neuron_left, col_neuron_right = st.columns(2)
        with col_neuron_left:
            try:
                sehong_img = _load_exact_image_file(image_root=image_root, filename="chill.jpg")
                _render_neuron_image_card(
                    title="Your Neuron",
                    subtitle="Chill",
                    image_bytes=sehong_img,
                )
            except Exception as exc:
                st.error(f"Could not load Sehong image: {exc}")

        with col_neuron_right:
            try:
                friend_img = _load_exact_image_file(image_root=image_root, filename=friend_image_file)
                _render_neuron_image_card(
                    title=f"This Person May Match Your Vibe: <span style='color:#1d4ed8;'>{friend_choice}</span>",
                    subtitle=f"You may vibe well with {friend_choice}. {'Energetic' if friend_choice == 'Mia' else 'Calm'} style.",
                    image_bytes=friend_img,
                )
            except Exception as exc:
                st.error(f"Could not load {friend_choice} image: {exc}")

        st.markdown(f"### Hear Sehong and {friend_choice} Harmony")
        if st.button("Generate Matching Vibe Music with ElevenLabs", key="generate_vibe_music_btn"):
            friend_choice = st.session_state.get("friend_choice", "Mia")
            music_path = (
                "/home/acrl/research/Neurohack/music/chill_energetic.mp3"
                if friend_choice == "Mia"
                else "/home/acrl/research/Neurohack/music/chill_calm.mp3"
            )
            st.session_state["vibe_music_path"] = music_path

        vibe_music_path = st.session_state.get("vibe_music_path")
        if vibe_music_path:
            if os.path.exists(vibe_music_path):
                with open(vibe_music_path, "rb") as f:
                    st.audio(f.read(), format="audio/mp3")
                st.markdown("**(Generated with EelvenLabs)**")
            else:
                st.error(f"Music file not found: {vibe_music_path}")

    st.subheader("Export")
    html_blob = fig_graph_static.to_html(full_html=True, include_plotlyjs="cdn")
    st.download_button(
        "Download current graph as HTML",
        data=html_blob,
        file_name=f"dynamic_graph_t{time_idx}.html",
        mime="text/html",
    )

    try:
        png_bytes = fig_graph_static.to_image(format="png", scale=2)
        st.download_button(
            "Download current graph as PNG",
            data=png_bytes,
            file_name=f"dynamic_graph_t{time_idx}.jpg",
            mime="image/png",
        )
    except Exception:
        st.caption("PNG export requires the optional `kaleido` package.")


if __name__ == "__main__":
    main()
