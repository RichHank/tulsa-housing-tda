import os
import sys
import json
import pandas as pd
import numpy as np
import kmapper as km
import sklearn
import warnings
warnings.filterwarnings("ignore")

# Change working directory to the script's own location
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()
os.chdir(script_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# ============================================================
# Configuration
# ============================================================
SIMPLE_MODE = False

# ============================================================
# Helper utilities
# ============================================================

def parse_edges(links):
    """Parse KeplerMapper graph['links'] in any format, return list of (src, dst) tuples."""
    edges = []
    if isinstance(links, dict):
        for source, targets in links.items():
            for target in targets:
                edges.append((source, target))
    elif isinstance(links, list):
        for edge in links:
            if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                edges.append((edge[0], edge[1]))
            elif isinstance(edge, dict):
                n1 = edge.get('source') or edge.get('src')
                n2 = edge.get('target') or edge.get('dst')
                if n1 and n2:
                    edges.append((n1, n2))
            elif isinstance(edge, str):
                parts = edge.split("-")
                if len(parts) == 2:
                    edges.append((parts[0], parts[1]))
    return edges


def count_edges(links):
    """Count edges from any link format."""
    if isinstance(links, dict):
        return sum(len(v) for v in links.values())
    elif isinstance(links, list):
        return len(links)
    return 0


def classify_regime(stats):
    """Assign a human-readable regime label based on economic conditions."""
    yoy = stats["avg_yoy"]
    unemp = stats.get("avg_unemp", np.nan)
    size = stats["size"]
    mortgage = stats.get("avg_mortgage", np.nan)

    high_unemp = not np.isnan(unemp) and unemp > 6.0
    low_unemp = not np.isnan(unemp) and unemp < 4.0
    high_mortgage = not np.isnan(mortgage) and mortgage > 7.0

    # Anomaly: very small node or extreme values
    if size <= 2 and (yoy < 0 or yoy > 8):
        return "Anomaly"

    if yoy < 1.0:
        if high_unemp:
            return "Recession Stress"
        return "Stagnant / Slow Recovery"
    elif yoy < 3.0:
        if high_unemp:
            return "Stagflationary Drift"
        return "Slow Recovery / Normal"
    elif yoy < 5.0:
        if low_unemp and not high_mortgage:
            return "Healthy Expansion"
        elif high_mortgage:
            return "Growth Despite Tight Credit"
        return "Moderate Growth"
    else:  # yoy >= 5.0
        if low_unemp:
            return "Hot Labor + Housing Boom"
        return "Housing Boom"

    return "Unclassified"


# ============================================================
# 1. Fetch the Data
# ============================================================
print("Fetching Tulsa Housing Market data from FRED...")

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def fetch_fred_series(series_id, name):
    """Fetch a FRED series and return a DataFrame with columns Date and value."""
    url = FRED_BASE.format(series_id=series_id)
    try:
        temp = pd.read_csv(url)
    except Exception as e:
        print(f"  Warning: Could not fetch {name} ({series_id}): {e}")
        return None
    date_col = None
    for col in temp.columns:
        if col.lower() in ['date', 'observation_date', 'time', 'period']:
            date_col = col
            break
    if date_col is None:
        date_col = temp.columns[0]
    temp = temp.rename(columns={date_col: "Date"})
    val_col = None
    for col in temp.columns:
        if col != "Date":
            val_col = col
            break
    if val_col is None:
        return None
    temp = temp.rename(columns={val_col: name})
    temp["Date"] = pd.to_datetime(temp["Date"])
    temp[name] = pd.to_numeric(temp[name], errors='coerce')
    temp = temp[["Date", name]].dropna()
    return temp


def resample_to_quarterly(df_in, name):
    """Resample a series to quarterly frequency using the mean within each quarter."""
    df_in = df_in.set_index("Date")
    df_in.index = pd.to_datetime(df_in.index)
    quarterly = df_in.resample("QE").mean()
    quarterly = quarterly.reset_index()
    return quarterly


# Primary series: Tulsa HPI (already quarterly)
df_hpi = fetch_fred_series("ATNHPIUS46140Q", "HPI")
if df_hpi is None:
    print("Error: Could not fetch primary HPI series.")
    sys.exit(1)

df = resample_to_quarterly(df_hpi, "HPI")

if not SIMPLE_MODE:
    print("Fetching additional economic series...")
    extra_series = [
        ("TULS140URN", "Unemployment_Rate"),
        ("MORTGAGE30US", "Mortgage_Rate_30Yr"),
        ("CPIAUCSL", "CPI"),
        ("FEDFUNDS", "Fed_Funds_Rate"),
        ("PERMIT1", "Building_Permits"),
        ("POPTHM", "Population"),
    ]
    for series_id, name in extra_series:
        temp = fetch_fred_series(series_id, name)
        if temp is not None:
            temp_q = resample_to_quarterly(temp, name)
            df = df.merge(temp_q, on="Date", how="outer")
            print(f"  Added {name} ({series_id}) - resampled to quarterly")
        else:
            print(f"  Skipped {name} ({series_id})")

df = df.sort_values("Date").reset_index(drop=True)
df = df.dropna(subset=["HPI"]).reset_index(drop=True)
df = df.set_index("Date")
df = df.interpolate(method='linear', limit=4).bfill(limit=4).ffill(limit=4)
df = df.reset_index()
df = df.dropna().reset_index(drop=True)

print(f"Observations after merging: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# ============================================================
# 2. Feature Engineering (Agent 1)
# ============================================================
print("Engineering features...")

# Core columns: HPI gets full feature suite; others get only YoY
# This balances feature richness vs. the curse of dimensionality (144 rows)
hpi_col = "HPI"
other_cols = [c for c in df.columns if c not in ["Date", hpi_col]]

# HPI: level, QoQ, YoY, rolling mean, rolling std
df["HPI_QoQ"] = df[hpi_col].pct_change() * 100
df["HPI_YoY"] = df[hpi_col].pct_change(periods=4) * 100
df["HPI_RollingMean_4Q"] = df[hpi_col].rolling(window=4).mean()
df["HPI_RollingStd_4Q"] = df[hpi_col].rolling(window=4).std()

# Other series: keep raw values + YoY growth (most informative for regime detection)
for col in other_cols:
    df[f"{col}_YoY"] = df[col].pct_change(periods=4) * 100

df["Time_Norm"] = np.linspace(0, 1, len(df))
df = df.dropna().reset_index(drop=True)
print(f"Observations after feature engineering: {len(df)}")

# ============================================================
# 3. Build feature matrix
# ============================================================
feature_cols = [c for c in df.columns if c not in ["Date", "Time_Norm"]]
X_features = df[feature_cols].values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# ============================================================
# 4. Lens functions
# ============================================================
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
lens_pca = pca.fit_transform(X_scaled)

yoy_hpi = df["HPI_YoY"].values if "HPI_YoY" in df.columns else np.zeros(len(df))
yoy_scaled = (yoy_hpi - yoy_hpi.mean()) / (yoy_hpi.std() + 1e-8)
lens_custom = np.column_stack([df["Time_Norm"].values, yoy_scaled])

lens_umap = None
try:
    import umap
    umap_reducer = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.1, random_state=42)
    lens_umap = umap_reducer.fit_transform(X_scaled)
    print("UMAP lens available.")
except ImportError:
    print("UMAP not installed, skipping.")

lens_options = [
    ("PCA", lens_pca),
    ("Custom (Time + YoY)", lens_custom),
]
if lens_umap is not None:
    lens_options.append(("UMAP", lens_umap))

# ============================================================
# 5. Parameter sweep with quality rejection (Agent 1)
# ============================================================
from sklearn.cluster import DBSCAN


def is_good_graph(graph, n_obs):
    """Return True if graph meets quality criteria."""
    if len(graph["nodes"]) == 0:
        return False
    node_sizes = [len(graph["nodes"][nid]) for nid in graph["nodes"]]
    avg_size = np.mean(node_sizes)
    max_size = np.max(node_sizes)
    n_nodes = len(graph["nodes"])
    total_members = sum(node_sizes)
    coverage = total_members / n_obs if n_obs > 0 else 0
    n_edges = count_edges(graph.get("links", {}))

    if n_nodes > n_obs or n_edges == 0 or coverage < 0.3:
        return False
    if n_nodes < 4 or n_nodes > 60:
        return False
    return True


def score_graph(graph, n_obs):
    """Score a graph on coverage, node size, and connectivity."""
    node_sizes = [len(graph["nodes"][nid]) for nid in graph["nodes"]]
    n_nodes = len(graph["nodes"])
    total_members = sum(node_sizes)
    coverage = total_members / n_obs
    avg_size = np.mean(node_sizes)
    n_edges = count_edges(graph.get("links", {}))
    connectivity = n_edges / max(n_nodes - 1, 1)
    size_score = min(avg_size, 8.0) / 8.0
    return coverage * 0.4 + size_score * 0.3 + min(connectivity, 1.0) * 0.3


candidates = []
n_cubes_list = [4, 5, 6, 8, 10]
perc_overlap_list = [0.4, 0.5, 0.6]
eps_list = [0.3, 0.5, 0.7, 1.0, 1.5]
min_samples_list = [1, 2]

print("\nSearching for good Mapper parameters...")
for lens_name, lens_data in lens_options:
    for n_cubes in n_cubes_list:
        for perc_overlap in perc_overlap_list:
            for eps in eps_list:
                for min_samples in min_samples_list:
                    cover = km.Cover(n_cubes=n_cubes, perc_overlap=perc_overlap)
                    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
                    mapper = km.KeplerMapper(verbose=0)
                    try:
                        graph = mapper.map(lens_data, X_scaled,
                                           clusterer=clusterer, cover=cover)
                    except Exception:
                        continue
                    if is_good_graph(graph, len(df)):
                        node_sizes = [len(graph["nodes"][nid]) for nid in graph["nodes"]]
                        n_nodes = len(graph["nodes"])
                        avg_size = np.mean(node_sizes)
                        coverage_pct = sum(node_sizes) / len(df) * 100
                        n_edges = count_edges(graph.get("links", {}))
                        s = score_graph(graph, len(df))
                        candidates.append((s, lens_name, n_cubes, perc_overlap,
                                           eps, min_samples, graph, n_nodes,
                                           avg_size, coverage_pct, n_edges))
                        print(f"  Candidate [{s:.2f}]: lens={lens_name}, "
                              f"cubes={n_cubes}, overlap={perc_overlap}, "
                              f"eps={eps}, ms={min_samples}, "
                              f"nodes={n_nodes}, avg_size={avg_size:.1f}, "
                              f"edges={n_edges}, coverage={coverage_pct:.0f}%")

if candidates:
    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0]
    best_graph = best[6]
    best_lens_name = best[1]
    best_params = (best[1], best[2], best[3], best[4], best[5])
    print(f"\nBest graph selected: score={best[0]:.2f}, lens={best[1]}, "
          f"cubes={best[2]}, overlap={best[3]}, eps={best[4]}, ms={best[5]}, "
          f"nodes={best[7]}, avg_size={best[8]:.1f}, "
          f"edges={best[10]}, coverage={best[9]:.0f}%")
else:
    best_graph = None

if best_graph is None:
    print("\nNo good graph found. Using fallback...")
    lens_name = "Custom (Time + YoY)"
    lens_data = lens_custom
    cover = km.Cover(n_cubes=4, perc_overlap=0.5)
    clusterer = DBSCAN(eps=1.0, min_samples=2)
    mapper = km.KeplerMapper(verbose=0)
    best_graph = mapper.map(lens_data, X_scaled, clusterer=clusterer, cover=cover)
    best_params = (lens_name, 4, 0.5, 1.0, 2)
    best_lens_name = lens_name
    print("Fallback used.")

# ============================================================
# 6. Final diagnostics (Agent 1)
# ============================================================
n_nodes = len(best_graph["nodes"])
node_sizes = [len(best_graph["nodes"][nid]) for nid in best_graph["nodes"]]
avg_size = np.mean(node_sizes)
median_size = np.median(node_sizes)
max_size = np.max(node_sizes)
min_size = np.min(node_sizes)
n_edges = count_edges(best_graph.get("links", {}))

print("\n" + "=" * 60)
print("FINAL MAPPER GRAPH DIAGNOSTICS")
print("=" * 60)
print(f"Lens:                 {best_params[0]}")
print(f"Cover:                n_cubes={best_params[1]}, overlap={best_params[2]}")
print(f"Clusterer:            DBSCAN(eps={best_params[3]}, min_samples={best_params[4]})")
print(f"Feature count:        {len(feature_cols)}")
print(f"Observations:         {len(df)}")
print(f"Nodes:                {n_nodes}")
print(f"Edges:                {n_edges}")
print(f"Average node size:    {avg_size:.1f}")
print(f"Median node size:     {median_size:.1f}")
print(f"Largest node size:    {max_size}")
print(f"Smallest node size:   {min_size}")
print("=" * 60)

# ============================================================
# 7. Build node metadata + regime labels (Agent 4)
# ============================================================
node_stats = {}
for node_id, members in best_graph["nodes"].items():
    node_dates = df.iloc[members]["Date"]
    node_hpi = df.iloc[members]["HPI"]
    node_yoy = (df.iloc[members]["HPI_YoY"]
                if "HPI_YoY" in df.columns
                else np.zeros(len(members)))
    node_qoq = (df.iloc[members]["HPI_QoQ"]
                if "HPI_QoQ" in df.columns
                else np.zeros(len(members)))
    node_unemp = (df.iloc[members]["Unemployment_Rate"]
                  if "Unemployment_Rate" in df.columns
                  else [np.nan])
    node_mortgage = (df.iloc[members]["Mortgage_Rate_30Yr"]
                     if "Mortgage_Rate_30Yr" in df.columns
                     else [np.nan])
    node_cpi = (df.iloc[members]["CPI"]
                if "CPI" in df.columns
                else [np.nan])

    stats = {
        "size": len(members),
        "date_min": node_dates.min().strftime("%Y-%m-%d"),
        "date_max": node_dates.max().strftime("%Y-%m-%d"),
        "avg_hpi": round(node_hpi.mean(), 2),
        "avg_qoq": round(node_qoq.mean(), 2),
        "avg_yoy": round(node_yoy.mean(), 2),
        "avg_unemp": round(np.nanmean(node_unemp), 2)
                      if not np.all(np.isnan(node_unemp)) else np.nan,
        "avg_mortgage": round(np.nanmean(node_mortgage), 2)
                         if not np.all(np.isnan(node_mortgage)) else np.nan,
        "avg_cpi": round(np.nanmean(node_cpi), 2)
                    if not np.all(np.isnan(node_cpi)) else np.nan,
        "members": members,
    }
    stats["regime_label"] = classify_regime(stats)
    node_stats[node_id] = stats

# Build edge list for vis
edges_raw = parse_edges(best_graph.get("links", {}))
# Filter edges where both endpoints exist in node_stats
edges = [(s, t) for s, t in edges_raw
         if s in node_stats and t in node_stats]

# ============================================================
# 8. Generate Topology Network Visualization (Agent 2)
# ============================================================
print("\nGenerating topology network visualization...")

try:
    import networkx as nx
    import plotly.graph_objects as go
    import plotly.io as pio

    G = nx.Graph()
    for node_id, stats in node_stats.items():
        G.add_node(node_id, **stats)
    for n1, n2 in edges:
        G.add_edge(n1, n2)

    # kamada_kawai for cleaner separation of clusters
    pos = nx.kamada_kawai_layout(G)

    fig_topo = go.Figure()

    # Draw edges — soft glow effect via semi-transparent blue
    for n1, n2 in G.edges():
        x0, y0 = pos[n1]
        x1, y1 = pos[n2]
        fig_topo.add_trace(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(color="rgba(100,160,255,0.2)", width=1.0),
            hoverinfo="none",
            showlegend=False,
        ))

    # --- Smart labeling: only label key nodes, deduplicate by regime ---
    # Only label nodes with: size >= 4, or containing Boom/Stress/Anomaly
    def should_label_topo(s):
        lbl = s["regime_label"]
        return s["size"] >= 4 or "Boom" in lbl or "Stress" in lbl or "Anomaly" in lbl

    # Deduplicate: for each regime_label, only label the largest node
    regime_best = {}  # regime_label -> (node_id, size)
    for node_id in G.nodes():
        s = node_stats[node_id]
        if should_label_topo(s):
            lbl = s["regime_label"]
            if lbl not in regime_best or s["size"] > regime_best[lbl][1]:
                regime_best[lbl] = (node_id, s["size"])
    labeled_ids = set(v[0] for v in regime_best.values())

    # Also always label the single largest node overall
    largest_id = max(G.nodes(), key=lambda nid: node_stats[nid]["size"])
    labeled_ids.add(largest_id)

    # Draw nodes
    node_x_vals = []
    node_y_vals = []
    node_text_vals = []
    node_size_vals = []
    node_color_vals = []
    label_x = []
    label_y = []
    label_txt = []

    for node_id in G.nodes():
        p = pos[node_id]
        node_x_vals.append(p[0])
        node_y_vals.append(p[1])
        s = node_stats[node_id]

        tooltip = (
            f"<b>{s['regime_label']}</b><br>"
            f"Date: {s['date_min']} to {s['date_max']}<br>"
            f"Quarters: {s['size']}<br>"
            f"Avg HPI: {s['avg_hpi']:.1f}<br>"
            f"Avg YoY Growth: {s['avg_yoy']:.1f}%<br>"
        )
        if not np.isnan(s['avg_unemp']):
            tooltip += f"Avg Unemployment: {s['avg_unemp']:.1f}%<br>"
        if not np.isnan(s['avg_mortgage']):
            tooltip += f"Avg Mortgage Rate: {s['avg_mortgage']:.2f}%<br>"
        node_text_vals.append(tooltip)
        node_size_vals.append(max(s["size"] * 5, 12))
        node_color_vals.append(s['avg_yoy'])

        if node_id in labeled_ids:
            label_x.append(p[0])
            label_y.append(p[1])
            label_txt.append(s["regime_label"])

    fig_topo.add_trace(go.Scatter(
        x=node_x_vals,
        y=node_y_vals,
        mode="markers",
        marker=dict(
            size=node_size_vals,
            color=node_color_vals,
            colorscale="RdYlGn",
            colorbar=dict(
                title=dict(text="Avg YoY Growth (%)", font=dict(color="white")),
                tickfont=dict(color="white"),
                x=1.02,
            ),
            line=dict(color="rgba(255,255,255,0.6)", width=1.5),
            sizemode="area",
            sizeref=2. * max(node_size_vals) / (45. ** 2),
            sizemin=10,
        ),
        text=node_text_vals,
        hoverinfo="text",
        showlegend=False,
        hovertemplate="%{text}<extra></extra>",
    ))

    # Labels on key nodes
    if label_x:
        fig_topo.add_trace(go.Scatter(
            x=label_x,
            y=label_y,
            mode="text",
            text=label_txt,
            textfont=dict(size=9, color="white", family="Arial"),
            textposition="top center",
            hoverinfo="none",
            showlegend=False,
        ))

    fig_topo.update_layout(
        title=dict(
            text=(
                "Tulsa Housing Market — Topological Regime Network"
            ),
            font=dict(size=22, color="white"),
            x=0.5,
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        hovermode="closest",
        width=1200,
        height=850,
        margin=dict(l=20, r=90, t=100, b=120),
    )

    # Explanation panel at bottom (inside plot area with proper margin)
    fig_topo.add_annotation(
        x=0.5, y=-0.08,
        xref="paper", yref="paper",
        text=(
            "<b>How to read this:</b>  Each circle = a housing-market regime "
            "(a group of quarters with similar conditions).  "
            "Bigger circles = longer-lasting regimes.  "
            "Lines = regimes that overlap or transition into each other.  "
            "Green = fast price growth.  Red = slow or negative growth.  "
            "Isolated circles = unusual periods.  "
            "This reveals the <i>shape</i> of Tulsa's housing cycles — "
            "not when things happened, but how market states relate to each other."
        ),
        showarrow=False,
        font=dict(size=11, color="rgba(180,190,210,0.85)"),
        align="center",
    )

    topo_html = "tulsa_housing_tda_topology.html"
    pio.write_html(fig_topo, file=topo_html, auto_open=False)
    print(f"Topology visualization saved to: {topo_html}")

    # Export static PNG for GitHub README
    try:
        topo_png = "tulsa_housing_tda_topology.png"
        fig_topo.write_image(topo_png, width=1200, height=850, scale=1.5)
        print(f"Topology PNG saved to: {topo_png}")
    except Exception as e:
        print(f"  PNG export failed (non-critical): {e}")

    HAS_NETWORKX_PLOTLY = True
except ImportError as e:
    print(f"NetworkX or Plotly not available ({e}). Skipping topology vis.")
    HAS_NETWORKX_PLOTLY = False

# ============================================================
# 9. Generate Timeline Visualization (Agent 3)
# ============================================================
print("Generating timeline visualization...")

try:
    import plotly.graph_objects as go
    import plotly.io as pio

    fig_timeline = go.Figure()

    # Compute node positions: X = midpoint date, Y = YoY growth
    node_pos_tl = {}
    for node_id, stats in node_stats.items():
        d0 = pd.Timestamp(stats["date_min"])
        d1 = pd.Timestamp(stats["date_max"])
        mid = d0 + (d1 - d0) / 2
        node_pos_tl[node_id] = (mid.toordinal(), stats["avg_yoy"])

    # Draw edges
    for n1, n2 in edges:
        if n1 in node_pos_tl and n2 in node_pos_tl:
            x0, y0 = node_pos_tl[n1]
            x1, y1 = node_pos_tl[n2]
            fig_timeline.add_trace(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(color="rgba(150,150,150,0.35)", width=1.2),
                hoverinfo="none",
                showlegend=False,
            ))

    # --- Smart labeling for timeline ---
    def should_label_timeline(s):
        lbl = s["regime_label"]
        return s["size"] >= 4 or "Boom" in lbl or "Stress" in lbl or "Anomaly" in lbl

    # Deduplicate: per regime label, only label the largest node
    tl_regime_best = {}
    for node_id, s in node_stats.items():
        if should_label_timeline(s):
            lbl = s["regime_label"]
            if lbl not in tl_regime_best or s["size"] > tl_regime_best[lbl][1]:
                tl_regime_best[lbl] = (node_id, s["size"])
    tl_labeled_ids = set(v[0] for v in tl_regime_best.values())

    # Always label single largest node
    tl_largest_id = max(node_stats.keys(),
                        key=lambda nid: node_stats[nid]["size"])
    tl_labeled_ids.add(tl_largest_id)

    # Draw nodes
    tl_x = []
    tl_y = []
    tl_text = []
    tl_size = []
    tl_color = []
    tl_label_x = []
    tl_label_y = []
    tl_label_txt = []

    for node_id, pos_tl in node_pos_tl.items():
        tl_x.append(pos_tl[0])
        tl_y.append(pos_tl[1])
        s = node_stats[node_id]
        tooltip = (
            f"<b>{s['regime_label']}</b><br>"
            f"Date: {s['date_min']} to {s['date_max']}<br>"
            f"Quarters: {s['size']}<br>"
            f"Avg HPI: {s['avg_hpi']:.1f}<br>"
            f"Avg YoY Growth: {s['avg_yoy']:.1f}%<br>"
        )
        if not np.isnan(s['avg_unemp']):
            tooltip += f"Avg Unemployment: {s['avg_unemp']:.1f}%<br>"
        if not np.isnan(s['avg_mortgage']):
            tooltip += f"Avg Mortgage Rate: {s['avg_mortgage']:.2f}%<br>"
        tl_text.append(tooltip)
        tl_size.append(max(s["size"] * 4, 10))
        tl_color.append(s['avg_unemp'] if not np.isnan(s['avg_unemp']) else 0)

        if node_id in tl_labeled_ids:
            tl_label_x.append(pos_tl[0])
            tl_label_y.append(pos_tl[1])
            tl_label_txt.append(s["regime_label"])

    fig_timeline.add_trace(go.Scatter(
        x=tl_x,
        y=tl_y,
        mode="markers",
        marker=dict(
            size=tl_size,
            color=tl_color,
            colorscale="RdYlGn_r",
            colorbar=dict(title="Unemployment Rate (%)"),
            line=dict(color="black", width=0.8),
            sizemode="area",
            sizeref=2. * max(tl_size) / (40. ** 2),
            sizemin=8,
        ),
        text=tl_text,
        hoverinfo="text",
        showlegend=False,
        hovertemplate="%{text}<extra></extra>",
    ))

    # Labels on key nodes only
    if tl_label_x:
        fig_timeline.add_trace(go.Scatter(
            x=tl_label_x,
            y=tl_label_y,
            mode="text",
            text=tl_label_txt,
            textfont=dict(size=9, color="#222222"),
            textposition="top center",
            hoverinfo="none",
            showlegend=False,
        ))

    # Recession bands
    recessions = [
        ("1990-07", "1991-03"),
        ("2001-03", "2001-11"),
        ("2007-12", "2009-06"),
        ("2020-02", "2020-04"),
    ]
    for start, end in recessions:
        fig_timeline.add_vrect(
            x0=pd.Timestamp(start).toordinal(),
            x1=pd.Timestamp(end).toordinal(),
            fillcolor="gray", opacity=0.12,
            layer="below", line_width=0,
            annotation_text="Recession",
            annotation_position="top left",
            annotation_font_size=9,
        )

    # Date ticks — every 3 years to avoid crowding
    date_ticks = sorted(set(v[0] for v in node_pos_tl.values()))
    tick_years = sorted(set(
        pd.Timestamp.fromordinal(int(d)).year for d in date_ticks
    ))
    tick_vals_show = [pd.Timestamp(year=y, month=1, day=1).toordinal()
                      for y in tick_years if y % 3 == 0]

    fig_timeline.update_layout(
        title=dict(
            text="Tulsa Housing Market — Regime Timeline (1990–2025)",
            font=dict(size=20),
            x=0.5,
        ),
        xaxis=dict(
            title="Time",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
            zeroline=False,
            tickmode="array",
            tickvals=tick_vals_show,
            ticktext=[str(pd.Timestamp.fromordinal(int(d)).year) for d in tick_vals_show],
            tickangle=0,
        ),
        yaxis=dict(
            title="HPI Year-over-Year Growth (%)",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
            zeroline=True,
            zerolinecolor="rgba(0,0,0,0.3)",
            zerolinewidth=1.5,
        ),
        plot_bgcolor="white",
        hovermode="closest",
        width=1200,
        height=750,
        margin=dict(l=60, r=80, t=120, b=60),
    )

    # Explanation panel at top — inside the plot area with proper margin
    fig_timeline.add_annotation(
        x=0.5, y=1.06,
        xref="paper", yref="paper",
        text=(
            "<b>How to read:</b>  Each circle = a housing-market regime "
            "|  Size = number of quarters  |  Color = unemployment (red = high)  "
            "|  Gray bands = NBER recessions  |  Lines = overlapping/transitioning regimes  "
            "|  <i>This chart shows when regimes occurred. "
            "See the Topology view for how market states relate to each other.</i>"
        ),
        showarrow=False,
        font=dict(size=11, color="rgba(70,70,70,0.85)"),
        align="center",
        bordercolor="rgba(0,0,0,0.15)",
        borderwidth=1,
        borderpad=8,
        bgcolor="rgba(245,245,245,0.9)",
    )

    timeline_html = "tulsa_housing_tda_timeline.html"
    pio.write_html(fig_timeline, file=timeline_html, auto_open=False)
    print(f"Timeline visualization saved to: {timeline_html}")

    # Export static PNG for GitHub README
    try:
        timeline_png = "tulsa_housing_tda_timeline.png"
        fig_timeline.write_image(timeline_png, width=1200, height=750, scale=1.5)
        print(f"Timeline PNG saved to: {timeline_png}")
    except Exception as e:
        print(f"  PNG export failed (non-critical): {e}")

except ImportError:
    print("Plotly not installed. Skipping timeline visualization.")

# ============================================================
# 10. Export Regime Summary CSV & HTML Table (Agent 4)
# ============================================================
print("Generating regime classification outputs...")

regime_records = []
for node_id in sorted(node_stats.keys()):
    s = node_stats[node_id]
    rec = {
        "node_id": node_id,
        "regime_label": s["regime_label"],
        "start_date": s["date_min"],
        "end_date": s["date_max"],
        "num_quarters": s["size"],
        "avg_hpi": s["avg_hpi"],
        "avg_qoq_growth_pct": s["avg_qoq"],
        "avg_yoy_growth_pct": s["avg_yoy"],
    }
    if not np.isnan(s['avg_unemp']):
        rec["avg_unemployment_pct"] = s['avg_unemp']
    if not np.isnan(s['avg_mortgage']):
        rec["avg_mortgage_rate_pct"] = s['avg_mortgage']
    if not np.isnan(s['avg_cpi']):
        rec["avg_cpi"] = s['avg_cpi']
    regime_records.append(rec)

regime_df = pd.DataFrame(regime_records)
regime_df = regime_df.sort_values("start_date")
regime_csv = "tulsa_housing_tda_regimes.csv"
regime_df.to_csv(regime_csv, index=False)
print(f"Regime summary saved to: {regime_csv}")

# Print a summary table to console
print("\n" + "=" * 100)
print("REGIME CLASSIFICATION SUMMARY")
print("=" * 100)
header = (f"{'Node':<20s} {'Regime':<35s} {'Start':<12s} {'End':<12s} "
          f"{'Qtrs':>5s} {'HPI':>8s} {'YoY%':>7s} {'Unemp%':>7s} {'Mort%':>6s}")
print(header)
print("-" * 100)
for _, row in regime_df.iterrows():
    line = (f"{row['node_id']:<20s} "
            f"{str(row['regime_label']):<35s} "
            f"{str(row['start_date']):<12s} "
            f"{str(row['end_date']):<12s} "
            f"{int(row['num_quarters']):>5d} "
            f"{row['avg_hpi']:>8.1f} "
            f"{row['avg_yoy_growth_pct']:>7.1f} "
            f"{row.get('avg_unemployment_pct', np.nan):>7.1f} "
            f"{row.get('avg_mortgage_rate_pct', np.nan):>6.1f}")
    print(line)
print("=" * 100)

# ============================================================
# 11. KeplerMapper fallback HTML
# ============================================================
print("\nGenerating KeplerMapper visualization...")

color_values = df["HPI_YoY"].values if "HPI_YoY" in df.columns else np.zeros(len(df))

custom_tooltips = []
for _, row in df.iterrows():
    tooltip = (
        f"<b>Date:</b> {row['Date'].strftime('%Y-%m-%d')}<br>"
        f"<b>HPI:</b> {row['HPI']:.2f}<br>"
    )
    for lbl in ["HPI_QoQ", "HPI_YoY"]:
        if lbl in row and not pd.isna(row[lbl]):
            tooltip += f"<b>{lbl}:</b> {row[lbl]:.2f}%<br>"
    for lbl, name in [("Unemployment_Rate", "Unemployment"),
                       ("Mortgage_Rate_30Yr", "Mortgage Rate"),
                       ("CPI", "CPI"),
                       ("Fed_Funds_Rate", "Fed Funds")]:
        if lbl in row and not pd.isna(row[lbl]):
            tooltip += f"<b>{name}:</b> {row[lbl]:.2f}<br>"
    custom_tooltips.append(tooltip)
custom_tooltips = np.array(custom_tooltips)

mapper_final = km.KeplerMapper(verbose=0)
km_output = "tulsa_housing_tda_kmapper.html"
try:
    mapper_final.visualize(
        best_graph,
        path_html=km_output,
        title="Tulsa Housing Market Regime Map (KeplerMapper)",
        custom_tooltips=custom_tooltips,
        color_values=color_values,
        color_function_name="YoY HPI Growth (%)",
        node_color_function=["mean", "std", "median", "max"],
    )
except TypeError:
    try:
        mapper_final.visualize(
            best_graph,
            path_html=km_output,
            title="Tulsa Housing Market Regime Map (KeplerMapper)",
            custom_tooltips=custom_tooltips,
            color_values=color_values,
            color_function_name="YoY HPI Growth (%)",
        )
    except Exception as e:
        print(f"KeplerMapper visualize error: {e}. Trying minimal call...")
        mapper_final.visualize(
            best_graph,
            path_html=km_output,
            title="Tulsa Housing Market Regime Map (KeplerMapper)",
        )
print(f"KeplerMapper visualization saved to: {km_output}")

# ============================================================
# Final summary
# ============================================================
print("\n" + "=" * 60)
print("ALL OUTPUT FILES GENERATED")
print("=" * 60)
outputs = [
    ("tulsa_housing_tda_topology.html", "Topology network graph (Agent 2)"),
    ("tulsa_housing_tda_timeline.html", "Timeline chart with regimes (Agent 3)"),
    ("tulsa_housing_tda_regimes.csv", "Regime classification CSV (Agent 4)"),
    ("tulsa_housing_tda_kmapper.html", "KeplerMapper fallback"),
]
for fname, desc in outputs:
    exists = "EXISTS" if os.path.exists(fname) else "MISSING"
    size = os.path.getsize(fname) if os.path.exists(fname) else 0
    print(f"  {exists:<8s} {fname:<40s} ({size:>8d} bytes)  — {desc}")
print("=" * 60)
print("\nDone!")
