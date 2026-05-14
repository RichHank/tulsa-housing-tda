import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/data")
from fetch_data import synthesize_tulsa_manifold
import umap
from ripser import ripser

# 1. Fetch data
df = synthesize_tulsa_manifold()
df = df.sort_values('Date').reset_index(drop=True)

# 2. Time-Delay Embedding (Takens' Theorem)
series = df[['price', 'mortgage', 'unemployment']].values
series = (series - series.mean(axis=0)) / series.std(axis=0)

delay = 2 # 2 quarters delay
dim = 3 # embedding dimension
n = len(series)

embedded = np.zeros((n - (dim - 1) * delay, series.shape[1] * dim))
dates = df['Date'].iloc[(dim - 1) * delay:].values
hpi = df['price'].iloc[(dim - 1) * delay:].values

for i in range(dim):
    embedded[:, i*series.shape[1]:(i+1)*series.shape[1]] = series[i*delay : i*delay + embedded.shape[0]]

# 3. UMAP dimensionality reduction on the attractor
reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
attractor_3d = reducer.fit_transform(embedded)

# 4. Persistent Homology using Ripser
diagrams = ripser(attractor_3d, maxdim=1)['dgms']

# 5. Build Advanced Plotly Dashboard
fig = make_subplots(
    rows=2, cols=2,
    specs=[[{"type": "scatter3d", "rowspan": 2}, {"type": "scatter"}],
           [None, {"type": "scatter"}]],
    subplot_titles=(
        "Takens' Time-Delay Attractor (3D UMAP Manifold)",
        "Persistent Homology: Betti-1 (Cyclical Market Features)",
        "Tulsa HPI Mapped to Attractor Path"
    )
)

# Plot 1: 3D Attractor
fig.add_trace(go.Scatter3d(
    x=attractor_3d[:, 0], y=attractor_3d[:, 1], z=attractor_3d[:, 2],
    mode='lines+markers',
    marker=dict(size=5, color=hpi, colorscale='Viridis', showscale=True, colorbar=dict(x=0.45, title="HPI")),
    line=dict(color='rgba(255, 255, 255, 0.3)', width=1),
    text=[str(pd.to_datetime(d).date()) for d in dates],
    hovertemplate="Date: %{text}<br>HPI: %{marker.color:.1f}<extra></extra>"
), row=1, col=1)

# Plot 2: Persistence Diagram
H1 = diagrams[1]
if len(H1) > 0:
    H1_clean = np.array([pt for pt in H1 if pt[1] != np.inf and pt[1] < 1e5])
    if len(H1_clean) > 0:
        fig.add_trace(go.Scatter(
            x=H1_clean[:, 0], y=H1_clean[:, 1],
            mode='markers',
            marker=dict(size=12, color='#00ffcc', line=dict(width=1, color='white'), opacity=0.8),
            name="H1 (Cycles)",
            hovertemplate="Birth: %{x:.2f}<br>Death: %{y:.2f}<br>Persistence: %{text:.2f}<extra></extra>",
            text=(H1_clean[:,1] - H1_clean[:,0])
        ), row=1, col=2)
        
        max_val = max(np.max(H1_clean[:,0]), np.max(H1_clean[:,1])) * 1.1
        fig.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode='lines', line=dict(color='rgba(255,255,255,0.3)', dash='dash'), showlegend=False
        ), row=1, col=2)

# Plot 3: HPI mapping
fig.add_trace(go.Scatter(
    x=dates, y=hpi,
    mode='lines+markers',
    marker=dict(color=attractor_3d[:,0], colorscale='Viridis', size=6),
    line=dict(color='rgba(255,255,255,0.2)'),
    showlegend=False,
    hovertemplate="Date: %{x}<br>HPI: %{y:.1f}<extra></extra>"
), row=2, col=2)

fig.update_layout(
    title=dict(text="<b>DOCTORAL TDA: DYNAMICAL SYSTEMS & PERSISTENCE</b><br><span style='font-size:14px; color:#a0a0a0;'>Time-Delay Embedding (Takens) & Persistent Homology (Vietoris-Rips)</span>", font=dict(size=24, color="white", family="Arial Black")),
    template="plotly_dark", paper_bgcolor="#050505", plot_bgcolor="#050505", height=900,
    margin=dict(l=40, r=40, t=100, b=40)
)
fig.update_xaxes(title_text="Birth Threshold (ε)", row=1, col=2, gridcolor="#222")
fig.update_yaxes(title_text="Death Threshold (ε)", row=1, col=2, gridcolor="#222")
fig.update_xaxes(title_text="Date", row=2, col=2, gridcolor="#222")
fig.update_yaxes(title_text="Tulsa HPI", row=2, col=2, gridcolor="#222")
fig.update_scenes(xaxis_backgroundcolor="#050505", yaxis_backgroundcolor="#050505", zaxis_backgroundcolor="#050505")

html_out = "tulsa_doctoral_tda.html"
fig.write_html(html_out)
print("Doctoral TDA HTML generated.")
