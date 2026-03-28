"""Data loading and synthetic dynamic adjacency generation utilities."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class ValidationResult:
    """Container for matrix validation details."""

    data: np.ndarray
    warning: Optional[str] = None


def _ensure_symmetric(matrix: np.ndarray) -> np.ndarray:
    """Return a symmetrized matrix with zero diagonal and clipped weights."""
    sym = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(sym, 0.0)
    return np.clip(sym, 0.0, 1.0)


def _load_csv_matrix_from_bytes(content: bytes) -> np.ndarray:
    """Load a CSV matrix robustly, handling UTF-8 BOM and empty lines."""
    text = content.decode("utf-8-sig")
    arr = np.genfromtxt(io.StringIO(text), delimiter=",", dtype=float)
    if arr.ndim == 1:
        arr = np.atleast_2d(arr)
    return arr


def generate_dynamic_adjacency(
    num_nodes: int = 10,
    edge_density: float = 0.35,
    dynamic_fraction: float = 0.5,
    noise_level: float = 0.03,
    random_seed: int = 42,
    num_time_steps: int = 60,
) -> np.ndarray:
    """Generate a smooth, symmetric dynamic adjacency tensor of shape (T, N, N).

    Some edges are static and some vary with sinusoidal drift over time.
    """
    rng = np.random.default_rng(random_seed)
    n = max(2, int(num_nodes))
    t = max(1, int(num_time_steps))

    upper_mask = np.triu(rng.random((n, n)) < edge_density, k=1)

    base_weights = rng.uniform(0.2, 0.95, size=(n, n))
    base_weights = np.where(upper_mask, base_weights, 0.0)

    dynamic_mask = np.triu(rng.random((n, n)) < dynamic_fraction, k=1)
    dynamic_mask = dynamic_mask & upper_mask

    amplitude = rng.uniform(0.05, 0.35, size=(n, n)) * dynamic_mask
    frequency = rng.uniform(0.6, 2.0, size=(n, n))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(n, n))

    timeline = np.linspace(0.0, 2.0 * np.pi, t)
    output = np.zeros((t, n, n), dtype=float)

    for idx, tau in enumerate(timeline):
        dynamic_component = amplitude * np.sin(frequency * tau + phase)
        noise = rng.normal(0.0, noise_level, size=(n, n)) * upper_mask

        upper = base_weights + dynamic_component + noise
        upper = np.where(upper_mask, upper, 0.0)
        upper = np.clip(upper, 0.0, 1.0)

        mat = upper + upper.T
        np.fill_diagonal(mat, 0.0)
        output[idx] = mat

    return output


def _validate_2d_matrix(matrix: np.ndarray, symmetrize: bool) -> ValidationResult:
    """Validate and normalize a single adjacency matrix."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Uploaded matrix must be square with shape (N, N).")

    warning = None
    matrix = np.asarray(matrix, dtype=float)

    if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
        raise ValueError("Matrix contains NaN or infinite values.")

    if not np.allclose(matrix, matrix.T, atol=1e-6):
        if symmetrize:
            matrix = _ensure_symmetric(matrix)
            warning = "Matrix was not symmetric; it was symmetrized automatically."
        else:
            warning = "Matrix appears non-symmetric; visualization may be less meaningful."

    np.fill_diagonal(matrix, 0.0)
    matrix = np.clip(matrix, 0.0, 1.0)
    return ValidationResult(data=matrix, warning=warning)


def load_adjacency_file(uploaded_file, symmetrize: bool = True) -> Tuple[np.ndarray, Optional[str]]:
    """Load uploaded CSV or NPY adjacency data.

    Returns:
        A tensor of shape (T, N, N) and an optional warning string.
    """
    if uploaded_file is None:
        raise ValueError("No file provided.")

    filename = uploaded_file.name.lower()
    content = uploaded_file.getvalue()

    if filename.endswith(".csv"):
        matrix = _load_csv_matrix_from_bytes(content)
        result = _validate_2d_matrix(matrix, symmetrize=symmetrize)
        return result.data[None, :, :], result.warning

    if filename.endswith(".npy"):
        raw = np.load(io.BytesIO(content), allow_pickle=False)
        if raw.ndim == 2:
            result = _validate_2d_matrix(raw, symmetrize=symmetrize)
            return result.data[None, :, :], result.warning
        if raw.ndim == 3:
            if raw.shape[1] != raw.shape[2]:
                raise ValueError("For 3D arrays, expected shape (T, N, N).")
            warnings = []
            fixed = []
            for i in range(raw.shape[0]):
                result = _validate_2d_matrix(raw[i], symmetrize=symmetrize)
                fixed.append(result.data)
                if result.warning:
                    warnings.append(f"t={i}: {result.warning}")
            warning = " | ".join(warnings) if warnings else None
            return np.stack(fixed, axis=0), warning
        raise ValueError(".npy file must contain shape (N, N) or (T, N, N).")

    raise ValueError("Unsupported file type. Upload a .csv or .npy file.")


def load_coordinates_file(uploaded_file, expected_nodes: int) -> np.ndarray:
    """Load node coordinate file from CSV/NPY and return shape (N, 2)."""
    if uploaded_file is None:
        raise ValueError("No coordinate file provided.")

    filename = uploaded_file.name.lower()
    content = uploaded_file.getvalue()

    if filename.endswith(".csv"):
        coords = _load_csv_matrix_from_bytes(content)
    elif filename.endswith(".npy"):
        coords = np.load(io.BytesIO(content), allow_pickle=False)
    else:
        raise ValueError("Unsupported coordinate file type. Upload .csv or .npy.")

    coords = np.asarray(coords, dtype=float)
    if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
        raise ValueError("Coordinate file contains NaN or infinite values.")

    if coords.ndim != 2:
        raise ValueError("Coordinate file must be 2D: either (N,2) or (2,N).")

    if coords.shape == (2, expected_nodes):
        coords = coords.T
    elif coords.shape != (expected_nodes, 2):
        raise ValueError(
            f"Coordinate shape mismatch. Expected (N,2) or (2,N) with N={expected_nodes}, "
            f"but got {coords.shape}."
        )

    return coords


def load_local_brain_dataset(data_dir: str, symmetrize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Load local brain ROI adjacency + XY coordinates from disk.

    Expected files:
      - ROI.csv
      - ROI_XY.csv
    """
    roi_path = os.path.join(data_dir, "ROI.csv")
    xy_path = os.path.join(data_dir, "ROI_XY.csv")

    if not os.path.exists(roi_path):
        raise FileNotFoundError(f"Missing adjacency file: {roi_path}")
    if not os.path.exists(xy_path):
        raise FileNotFoundError(f"Missing coordinate file: {xy_path}")

    with open(roi_path, "rb") as f:
        roi_mat = _load_csv_matrix_from_bytes(f.read())
    roi = _validate_2d_matrix(roi_mat, symmetrize=symmetrize).data[None, :, :]

    with open(xy_path, "rb") as f:
        xy_mat = _load_csv_matrix_from_bytes(f.read())
    coords = np.asarray(xy_mat, dtype=float)
    n = roi.shape[1]
    if coords.shape == (2, n):
        coords = coords.T
    if coords.shape != (n, 2):
        raise ValueError(f"ROI_XY.csv must have shape (N,2) or (2,N). Found {coords.shape}, expected N={n}.")

    return roi, coords
