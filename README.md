# Dynamic Graph Visualizer (Streamlit)

A polished Python demo app that visualizes dynamic weighted graphs from adjacency matrices.

## Features
- Synthetic dynamic adjacency generation: symmetric, weighted, time-varying, noise-aware.
- File upload support for `.csv` and `.npy` with robust validation.
- Interactive graph visualization with NetworkX + Plotly.
- Stable layout across time (spring layout computed once).
- Node size by weighted degree and edge width by weight.
- Heatmap view of the current adjacency matrix.
- Summary stats and human-readable insight text.
- Animation playback mode and adjustable speed.
- Export graph as HTML and PNG.

## Project structure
- `app.py` - Streamlit UI and app orchestration.
- `data_utils.py` - synthetic data generation and file loading.
- `graph_utils.py` - graph construction, metrics, and summary logic.
- `visualization.py` - Plotly graph and heatmap rendering helpers.

## Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the app:
   ```bash
   streamlit run app.py
   ```

## Upload formats
- `.csv`: one square adjacency matrix `(N, N)`.
- `.npy`: either `(N, N)` or `(T, N, N)`.

If a single matrix is uploaded, the app treats it as static over time.

## Notes
- Matrices are clipped to `[0, 1]` and diagonal entries are forced to zero.
- Optional auto-symmetrization is available for uploaded data.
