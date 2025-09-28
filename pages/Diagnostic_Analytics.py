import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.title("🩺 Diagnostic Analytics")
st.caption("This page shows diagnostic analysis of the dataset that the project dashboard utilized and trained on")

ART = Path("artifacts/diagnostic")

# --- heatmap image ---
st.subheader("Correlation Heatmap")
st.markdown("This heatmap shows pairwise correlations among all encoded features in the dataset. "
            "It provides a quick overview of linear relationships between variables.")
hm_path = ART / "correlation_heatmap.png"
if hm_path.exists():
    st.image(str(hm_path), width=800)
else:
    st.info("Missing: artifacts/diagnostic/correlation_heatmap.png")
st.markdown("**Analysis:** Strong positive or negative correlations may indicate multicollinearity or "
            "potentially important relationships for predicting stroke.")

# --- correlation matrix table ---
st.subheader("Correlation Matrix")
st.markdown("The full correlation matrix with numeric values for each feature pair is shown below. "
            "This is useful for precise interpretation of variable associations.")
cm_path = ART / "correlation_matrix.csv"
if cm_path.exists():
    st.dataframe(pd.read_csv(cm_path, index_col=0).round(2), use_container_width=True)
else:
    st.info("Run the notebook to create: artifacts/diagnostic/correlation_matrix.csv")
st.markdown("**Analysis:** Correlation values closer to +1 or -1 suggest strong associations, "
            "while values near 0 suggest little to no linear relationship.")

# --- scatter matrix (prefer HTML; fallback to PNG if you exported it) ---
st.subheader("Pair Plot (Age, Glucose, BMI)")
st.markdown("This pair plot shows scatterplots for Age, Glucose, and BMI, allowing us to visually explore "
            "their distributions and relationships.")
html_path = ART / "scatter_matrix.html"
png_path  = ART / "scatter_matrix.png"

if html_path.exists():
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    st.components.v1.html(html, height=820, scrolling=True)
elif png_path.exists():
    st.image(str(png_path), use_container_width=True)
else:
    st.info("Missing: scatter_matrix.html / scatter_matrix.png")
st.markdown("**Analysis:** We can observe clustering trends, potential outliers, and differences in distributions "
            "for each variable pair.")

# --- lmplot images ---
st.subheader("Scatter Plots with Regression Lines")
st.markdown("These scatter plots show regression lines for selected variable pairs, split by stroke outcome, "
            "to illustrate potential associations with stroke risk.")

c1, c2, c3 = st.columns(3)

with c1:
    p = ART / "lm_age_glucose.png"
    if p.exists():
        st.image(str(p), caption="Age vs Glucose (by Stroke)", use_container_width=True)
    else:
        st.info("Missing: lm_age_glucose.png")
    st.markdown("**Analysis:** Glucose tends to increase with age; the regression line highlights this trend.")

with c2:
    p = ART / "lm_age_bmi.png"
    if p.exists():
        st.image(str(p), caption="Age vs BMI (by Stroke)", use_container_width=True)
    else:
        st.info("Missing: lm_age_bmi.png")
    st.markdown("**Analysis:** BMI distribution across age groups shows variability, "
                "but no strong linear association is evident.")

with c3:
    p = ART / "lm_bmi_glucose.png"
    if p.exists():
        st.image(str(p), caption="BMI vs Glucose (by Stroke)", use_container_width=True)
    else:
        st.info("Missing: lm_bmi_glucose.png")
    st.markdown("**Analysis:** There is a weak correlation between BMI and glucose; "
                "extreme values may influence overall patterns.")

# =====================================================================
# Unsupervised Clustering (interactive)
# =====================================================================
st.subheader("Clustering Analytics")
st.markdown(
    "All available features are used (numeric + one-hot encoded for categoricals). "
    "Choose **K** to explore natural groupings; plot shows a 2D PCA projection."
)

# ---- load the same dataset you used above (adjust path if needed) ----
@st.cache_data
def _load_processed_for_cluster():
    return pd.read_csv("./jupyter-notebooks/processed_data.csv")

raw = _load_processed_for_cluster()

# --- Prepare X (features) and y (optional Stroke for plot/summary) ---
stroke_col = "Stroke" if "Stroke" in raw.columns else None

# use all columns except obvious non-feature columns
drop_cols = set()
if stroke_col:
    drop_cols.add(stroke_col)
for c in raw.columns:
    if c.lower() in {"id", "patient_id", "index"}:
        drop_cols.add(c)

X_all = raw.drop(columns=list(drop_cols), errors="ignore").copy()

# One-hot encode any non-numeric columns (just in case)
non_num = X_all.select_dtypes(exclude=["number", "bool"]).columns.tolist()
if non_num:
    X_all = pd.get_dummies(X_all, columns=non_num, drop_first=True)

# replace inf/NaN, drop rows with any remaining NaN
X_all = X_all.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
if X_all.empty or X_all.shape[1] < 2:
    st.info("Not enough usable features after preprocessing to run clustering.")
else:
    # align stroke labels to the same rows after dropna
    if stroke_col:
        y_stroke = raw.loc[X_all.index, stroke_col].astype(int)
    else:
        y_stroke = None

    # --- UI: only K slider ---
    k = st.slider("K (number of clusters)", min_value=2, max_value=12, value=4, step=1)

    # --- Standardize, fit KMeans ---
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_all)

    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(X_std)

    # --- Silhouette score ---
    sil = silhouette_score(X_std, labels) if len(X_std) > k else np.nan

    # --- PCA (2D) for plotting ---
    pca = PCA(n_components=2, random_state=42)
    XY = pca.fit_transform(X_std)
    plot_df = pd.DataFrame(XY, columns=["PC1", "PC2"], index=X_all.index)
    plot_df["Cluster"] = labels.astype(int).astype(str)

    # Optional: show Stroke as shape, but keep legend clean
    stroke_label_col = None
    if y_stroke is not None:
        stroke_label_col = "StrokeLabel"
        plot_df[stroke_label_col] = y_stroke.map({0: "No Stroke", 1: "Stroke"}).astype(str)
        # Make Stroke points bigger; others small
        plot_df["PointSize"] = np.where(plot_df[stroke_label_col] == "Stroke", 14, 6)
    else:
        plot_df["PointSize"] = 4

    # -------- Fixed color map: 0=blue, 1=red; others get fallbacks --------
    base_map = {"0": "#1f77b4", "1": "#d62728"}  # blue, red
    fallback = ["#2ca02c", "#9467bd", "#ff7f0e", "#8c564b",
                "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    cluster_vals = sorted(plot_df["Cluster"].unique(), key=lambda x: int(x))
    color_map = {}
    fb_i = 0
    for v in cluster_vals:
        if v in base_map:
            color_map[v] = base_map[v]
        else:
            color_map[v] = fallback[fb_i % len(fallback)]
            fb_i += 1

    # --- Scatter (color=Cluster; symbol=Stroke; size only larger for Stroke) ---
    fig_scatter = px.scatter(
        plot_df, x="PC1", y="PC2",
        color="Cluster",
        symbol=stroke_label_col if stroke_label_col else None,
        size="PointSize",
        size_max=14,
        opacity=0.6,
        title=f"PCA Projection (K={k}, all features)"
    )
    fig_scatter.update_layout(height=420, margin=dict(t=50, b=10, l=10, r=10), legend_title_text="Cluster")

    # Collapse legend to one entry per cluster
    seen = set()
    for tr in fig_scatter.data:
        name = tr.name.split(",")[0] if "," in tr.name else tr.name
        val = name.split("=")[-1].strip()
        tr.name = f"Cluster {val}"
        tr.legendgroup = f"cluster_{val}"
        tr.showlegend = val not in seen
        seen.add(val)

    # Separate mini legend for Stroke shapes
    if stroke_label_col:
        fig_scatter.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name="Stroke",
            marker=dict(symbol="diamond", size=10, color="#666"),
            showlegend=True, legendgroup="stroke"
        ))
        fig_scatter.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name="No Stroke",
            marker=dict(symbol="circle", size=10, color="#666"),
            showlegend=True, legendgroup="stroke"
        ))

    st.plotly_chart(fig_scatter, use_container_width=True)

    # --- KPI row ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Features used", f"{X_all.shape[1]}")
    m2.metric("Rows used", f"{X_all.shape[0]}")
    m3.metric("K (clusters)", f"{k}")
    m4.metric("Silhouette score", f"{sil:.3f}" if np.isfinite(sil) else "—")

    # --- Cluster profiles (means of original feature space) ---
    prof_df = (
        pd.concat([pd.DataFrame(X_all.reset_index(drop=True)), pd.Series(labels, name="Cluster")], axis=1)
        .groupby("Cluster")
        .mean()
        .reset_index()
    )

    # Add stroke rate per cluster if available
    if y_stroke is not None:
        sr = pd.DataFrame({"Cluster": labels, "Stroke": y_stroke.reset_index(drop=True)})
        stroke_rate = sr.groupby("Cluster")["Stroke"].mean().mul(100).rename("Stroke Rate (%)").reset_index()
        prof_df = prof_df.merge(stroke_rate, on="Cluster", how="left")

    st.markdown("**Cluster Profiles** (feature means; Stroke Rate if available). Scroll horizontally for wide tables.")
    st.dataframe(prof_df.round(2), use_container_width=True)

    # Optional elbow curve (hidden by default)
    with st.expander("Show elbow curve (SSE)"):
        Ks = list(range(2, min(13, max(3, X_all.shape[0]-1))))
        sse = []
        for kk in Ks:
            km_kk = KMeans(n_clusters=kk, n_init="auto", random_state=42).fit(X_std)
            sse.append(km_kk.inertia_)
        fig_elbow = px.line(x=Ks, y=sse, markers=True,
                            labels={"x": "K", "y": "SSE (inertia)"},
                            title="Elbow Curve")
        fig_elbow.update_layout(height=300, margin=dict(t=40, b=0, l=10, r=10))
        st.plotly_chart(fig_elbow, use_container_width=True)