#Diagnostic page

import streamlit as st

PAGE_ID = "diagnostic-page"
st.markdown(f"<div id='{PAGE_ID}'>", unsafe_allow_html=True)
st.markdown(f"""
<style>
#{PAGE_ID} div[role='radiogroup'] label {{
  background:#fff; border:2px solid #cbd5e1; border-radius:10px; padding:6px 16px; margin:4px; color:#1e293b; font-weight:600; transition:all .25s;
}}
#{PAGE_ID} div[role='radiogroup'] label:hover {{ background:#e2e8f0; border-color:#94a3b8; }}
#{PAGE_ID} div[role='radiogroup'] label:has(input:checked) {{
  background:#2563eb !important; color:#fff !important; border-color:#1e40af !important; box-shadow:0 0 4px rgba(37,99,235,.6);
}}
#{PAGE_ID} div[role='radiogroup'] {{ display:flex; gap:6px; flex-wrap:wrap; }}
</style>
""", unsafe_allow_html=True)

from utils.ui_safety import begin_page
begin_page("Diagnostic Analytics 🩺")

# Clearning up the unnecessaery data
if 'rp_input' in st.session_state:
    del st.session_state['rp_input']

import numpy as np
import pandas as pd
from pathlib import Path
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import re

#st.title("🩺 Diagnostic Analytics")
st.caption("Diagnostic analysis of the dataset used in this project.")

ART = Path("artifacts/diagnostic")
DATA_PATH = "./jupyter-notebooks/processed_data.csv"
TARGET = "Stroke"  # exact name only

# ----------------------------
# Data loader (cached once)
# ----------------------------
@st.cache_data
def load_processed(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.reset_index(drop=True)
    df.columns = [c.strip() for c in df.columns]
    # 🔽 remove any auto-saved index columns like "Unnamed: 0", "unnamed: 0", etc.
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed", case=False)]
    # Treat bools as numeric for correlation (cast to float)
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols):
        df[bool_cols] = df[bool_cols].astype(float)
    return df

df = load_processed(DATA_PATH)

# ----------------------------
# Utilities
# ----------------------------
def id_like_columns(columns):
    """Columns to exclude everywhere: 'id', 'patient_id', 'index', and *_id (case-insensitive)."""
    out = set()
    for c in columns:
        lc = c.lower()
        if lc in {"id", "patient_id", "index"} or lc.endswith("_id"):
            out.add(c)
    return out

# =========================================
# Interactive Top-K correlation heatmap
# =========================================
st.subheader(f"Correlation with {TARGET} — Top-K Heatmap")

if TARGET not in df.columns:
    st.error(f"Required target column **'{TARGET}'** not found in {DATA_PATH}.")
else:
    y = pd.to_numeric(df[TARGET], errors="coerce")
    if not y.notna().any():
        st.error(f"Target column **'{TARGET}'** must be numeric (0/1).")
    else:
        top_k = st.slider(
            "Top-K features", min_value=3, max_value=15, value=7, step=1,
            key="diag_topk_linked"
        )

        drop_ids = id_like_columns(df.columns)
        num_like = df.select_dtypes(include=[np.number]).copy()
        X_num = (
            num_like
            .drop(columns=[TARGET], errors="ignore")
            .drop(columns=list(drop_ids), errors="ignore")
        )

        if X_num.shape[1] == 0:
            st.info("No numeric features available after dropping ID-like columns.")
        else:
            valid_idx = y.dropna().index
            X_corr = X_num.loc[valid_idx]
            y_corr = y.loc[valid_idx]

            sr = X_corr.corrwith(y_corr, method="pearson").dropna()
            if sr.empty:
                st.info("No valid correlations could be computed with the target.")
            else:
                sel = sr.abs().sort_values(ascending=False).head(top_k)
                top_feats = sel.index.tolist()

                sub_df = pd.concat([y_corr.rename(TARGET), X_corr[top_feats]], axis=1)
                sub_corr = sub_df.corr()

                # ------- Build interactive lower-tri heatmap with Plotly -------
                # mask upper triangle
                mat = sub_corr.values.astype(float).copy()
                iu = np.triu_indices_from(mat, k=1)
                mat[iu] = np.nan

                # optional bounds to match your previous seaborn style
                vmin, vmax = -0.6, 0.6
                # height scales with matrix size (keeps it compact but readable)
                h = int(np.clip(160 + 32 * (len(top_feats) + 1), 280, 700))

                fig_heat = go.Figure(
                    data=go.Heatmap(
                        z=mat,
                        x=sub_corr.columns,
                        y=sub_corr.index,
                        colorscale="RdBu",
                        zmin=vmin, zmax=vmax, zmid=0,
                        colorbar=dict(title="corr", len=0.8),
                        hovertemplate=(
                            "<b>%{y}</b> vs <b>%{x}</b><br>"
                            "corr = %{z:.2f}<extra></extra>"
                        ),
                    )
                )
                # mimic seaborn orientation (target on top-left)
                fig_heat.update_yaxes(autorange="reversed")
                fig_heat.update_layout(
                    title=f"Top {len(top_feats)} correlations with {TARGET}",
                    height=h,
                    margin=dict(t=48, b=20, l=60, r=20),
                    xaxis=dict(tickangle=45),
                )
                st.plotly_chart(fig_heat, use_container_width=True, key="corr_heatmap")

st.markdown(
    """
    <div style="
        background-color:#f7fbff;
        border-left:4px solid #1f77b4;
        padding:12px 14px;
        border-radius:8px;
        margin-top:15px;
        margin-bottom:25px;">
        <b>Key Findings (for quick understanding)</b><br><br>
        • <b>Age</b> shows the strongest positive correlation with stroke — older patients tend to have higher stroke risk.<br>
        • <b>Heart Disease</b>, <b>Glucose</b>, and <b>Hypertension</b> are also positively linked, though less strongly, meaning higher values or presence of these conditions increase stroke likelihood.<br>
        • <b>BMI</b> shows a weak negative correlation, indicating that within this dataset, BMI has little or no consistent effect on stroke risk.<br>
        • Overall, age and cardiovascular-related conditions remain the most influential variables in the dataset.
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# Pair plot (Age, Glucose, BMI)
# =========================
st.subheader("Pair Plot (Age, Glucose, BMI)")
st.markdown("Scatterplot matrix to inspect distributions and bivariate patterns.")

html_path = ART / "scatter_matrix.html"
png_path  = ART / "scatter_matrix.png"
if html_path.exists():
    html = html_path.read_text(encoding="utf-8")
    st.components.v1.html(html, height=580, scrolling=False)   # full page width, short height
elif png_path.exists():
    st.image(str(png_path), use_container_width=True)
else:
    st.info("Missing: artifacts/diagnostic/scatter_matrix.html (or .png)")

st.markdown(
    """
    <div style="
        background-color:#f7fbff;
        border-left:4px solid #1f77b4;
        padding:12px 14px;
        border-radius:8px;
        margin-top:15px;
        margin-bottom:25px;">
        <b>Key Findings (for quick understanding)</b><br><br>
        • <b>Age</b> shows a clear upward trend, indicating that most values rise gradually with age distribution.<br>
        • <b>Glucose</b> and <b>BMI</b> appear widely scattered without a strong visible pattern, suggesting only weak relationships between these variables.<br>
        • A few extreme points in glucose and BMI may represent outliers or patients with special metabolic conditions.<br>
        • Overall, the variables are mostly independent, meaning each provides distinct information for understanding stroke risk.
    </div>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
# Clustering by K=2 (aligned to Stroke) — PCA scatter + KPIs only
# =====================================================================
st.subheader("Clustering Analytics (K=2)")
st.markdown("Clusters the full feature space into **2 groups** and aligns clusters to **Stroke** by majority class.")

if TARGET not in df.columns:
    st.stop()

drop_cols = id_like_columns(df.columns)
drop_cols.add(TARGET)
X_all = df.drop(columns=list(drop_cols), errors="ignore").copy()

# One-hot any non-numeric columns (defensive; booleans already numeric)
non_num = X_all.select_dtypes(exclude=[np.number]).columns.tolist()
if non_num:
    X_all = pd.get_dummies(X_all, columns=non_num, drop_first=True)

# Clean
X_all = X_all.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
if X_all.empty or X_all.shape[1] < 2:
    st.info("Not enough usable features after preprocessing to run clustering.")
else:
    y_stroke = pd.to_numeric(df.loc[X_all.index, TARGET], errors="coerce").fillna(0).astype(int)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_all)

    k = 2
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    raw_labels = km.fit_predict(X_std)

    rates = []
    for cl in range(k):
        mask = (raw_labels == cl)
        rates.append(y_stroke[mask].mean() if mask.any() else 0.0)
    stroke_like = int(np.argmax(rates))
    remap = {1 - stroke_like: 0, stroke_like: 1}
    aligned_labels = np.vectorize(remap.get)(raw_labels)
    legend_map = {0: "No-stroke–like", 1: "Stroke–like"}

    sil = silhouette_score(X_std, raw_labels) if len(X_std) > k else np.nan

    pca = PCA(n_components=2, random_state=42)
    XY = pca.fit_transform(X_std)
    plot_df = pd.DataFrame(XY, columns=["PC1", "PC2"], index=X_all.index)
    plot_df["Cluster"] = aligned_labels
    plot_df["Stroke"] = y_stroke.map({0: "No Stroke", 1: "Stroke"})

    fig_scatter = px.scatter(
        plot_df, x="PC1", y="PC2",
        color=plot_df["Cluster"].map(legend_map),
        color_discrete_map={legend_map[i]: ["#1f77b4", "#d62728"][i] for i in [0, 1]},
        symbol="Stroke", opacity=0.70,
        title="PCA Projection (K=2, clusters aligned to Stroke)"
    )
    fig_scatter.update_layout(height=360, margin=dict(t=36, b=10, l=10, r=10), legend_title_text="Cluster")
    st.plotly_chart(fig_scatter, use_container_width=True, key='clustering')

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Features used", f"{X_all.shape[1]}")
    m2.metric("Rows used", f"{X_all.shape[0]}")
    m3.metric("K (clusters)", "2")
    m4.metric("Silhouette score", f"{sil:.3f}" if np.isfinite(sil) else "—")

st.markdown(
    """
    <div style="
        background-color:#f7fbff;
        border-left:4px solid #1f77b4;
        padding:12px 14px;
        border-radius:8px;
        margin-top:15px;
        margin-bottom:25px;">
        <b>Key Findings (for quick understanding)</b><br><br>
        • The model automatically grouped patients into <b>two main clusters</b>: one “<b>stroke-like</b>” (red) and one “<b>no-stroke-like</b>” (blue).<br>
        • Most stroke cases appear inside the red cluster, meaning their clinical profiles share similar patterns.<br>
        • A few overlaps exist where some non-stroke patients fall into the stroke-like cluster — this indicates borderline or at-risk individuals.<br>
        • The <b>silhouette score of 0.174</b> shows modest separation — clusters are distinguishable but not completely distinct.<br>
        • Overall, clustering highlights two broad risk groups that can guide early screening or targeted intervention strategies.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)