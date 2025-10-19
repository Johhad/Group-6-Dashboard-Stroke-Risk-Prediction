# pages/Diagnostic_Analytics.py
import streamlit as st

PAGE_ID = "diagnostic-page"
st.markdown(f"<div id='{PAGE_ID}'>", unsafe_allow_html=True)

# --- Slider & radio styling: higher contrast on light backgrounds ---
st.markdown(
    f"""
<style>
/* Top-K slider: thicker rail, vivid track/handle, visible focus ring */
.stSlider > div[data-baseweb="slider"] .rc-slider-rail {{
  background-color: #e2e8f0;  /* light gray */
  height: 8px;
}}
.stSlider > div[data-baseweb="slider"] .rc-slider-track {{
  background-color: #2563eb;  /* blue */
  height: 8px;
}}
.stSlider > div[data-baseweb="slider"] .rc-slider-handle {{
  border: 2px solid #1d4ed8;
  background-color: #ffffff;
  box-shadow: 0 0 0 5px rgba(37,99,235,.18);
  width: 18px; height: 18px; margin-top: -5px;
}}
/* Hover/focus */
.stSlider > div[data-baseweb="slider"] .rc-slider-handle:hover {{
  box-shadow: 0 0 0 6px rgba(37,99,235,.25);
  border-color: #1d4ed8;
}}
/* Label spacing */
.block-container .stSlider label {{
  font-weight: 600; color: #0f172a; margin-bottom: 6px;
}}
/* Keep your radio-pill look (scoped to this page) */
#{PAGE_ID} div[role='radiogroup'] label {{
  background:#fff; border:2px solid #cbd5e1; border-radius:10px; padding:6px 16px; margin:4px; color:#1e293b; font-weight:600; transition:all .25s;
}}
#{PAGE_ID} div[role='radiogroup'] label:hover {{ background:#e2e8f0; border-color:#94a3b8; }}
#{PAGE_ID} div[role='radiogroup'] label:has(input:checked) {{
  background:#2563eb !important; color:#fff !important; border-color:#1e40af !important; box-shadow:0 0 4px rgba(37,99,235,.6);
}}
#{PAGE_ID} div[role='radiogroup'] {{ display:flex; gap:6px; flex-wrap:wrap; }}
</style>
""",
    unsafe_allow_html=True,
)

# Page header
try:
    from utils.ui_safety import begin_page
    begin_page("Diagnostic Analytics 🩺")
except Exception:
    st.title("Diagnostic Analytics 🩺")

# Clear previous patient state so pages don't leak state into each other
if "rp_input" in st.session_state:
    del st.session_state["rp_input"]

# --- Imports ---
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.caption("Diagnostic analysis of the dataset used in this project.")

ART = Path("artifacts/diagnostic")
DATA_PATH = "./jupyter-notebooks/processed_data.csv"
TARGET = "Stroke"  # exact column name

# ----------------------------
# Data loader
# ----------------------------
@st.cache_data
def load_processed(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.reset_index(drop=True)
    df.columns = [c.strip() for c in df.columns]
    # remove auto-saved index columns like "Unnamed: 0"
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed", case=False)]
    # booleans to float so they are numeric for corr
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols):
        df[bool_cols] = df[bool_cols].astype(float)
    return df

df = load_processed(DATA_PATH)

# ----------------------------
# Utilities
# ----------------------------
def id_like_columns(columns):
    """Exclude obvious ID-like columns."""
    out = set()
    for c in columns:
        lc = c.lower()
        if lc in {"id", "patient_id", "index"} or lc.endswith("_id"):
            out.add(c)
    return out

# =========================================
# Seaborn Top-K correlation heatmap (always one-hot)
# =========================================
st.subheader(f"Correlation with {TARGET} — Top-K Heatmap")

if TARGET not in df.columns:
    st.error(f"Required target column **'{TARGET}'** not found in {DATA_PATH}.")
else:
    y_all = pd.to_numeric(df[TARGET], errors="coerce")
    if not y_all.notna().any():
        st.error(f"Target column **'{TARGET}'** must be numeric (0/1).")
    else:
        drop_ids = id_like_columns(df.columns)
        base = df.drop(columns=[TARGET], errors="ignore")
        base = base.drop(columns=list(drop_ids), errors="ignore")

        # ✅ ALWAYS include one-hot encoded categoricals (no checkbox, no min-rows)
        non_num_cols = base.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_num_cols:
            X_cand = pd.get_dummies(base, columns=non_num_cols, drop_first=True, dummy_na=False)
        else:
            X_cand = base.select_dtypes(include=[np.number]).copy()

        # Align rows with non-missing target
        valid_idx = y_all.dropna().index
        X_cand = X_cand.loc[valid_idx]
        y = y_all.loc[valid_idx]

        # Defensive cleanup
        X_cand = X_cand.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        y = y.loc[X_cand.index]

        available = X_cand.shape[1]
        if available == 0:
            st.info("No usable features found after preprocessing.")
        else:
            max_k = int(min(15, available))
            default_k = int(min(7, max_k))
            top_k = st.slider(
                "Top-K features",
                min_value=3, max_value=max_k,
                value=default_k, step=1,
                key="diag_topk_linked",
                help="Select how many of the strongest-correlated features to include."
            )

            sr = X_cand.corrwith(y, method="pearson").dropna()
            if sr.empty:
                st.info("No valid correlations could be computed with the target.")
            else:
                sel = sr.abs().sort_values(ascending=False).head(top_k)
                top_feats = sel.index.tolist()

                sub_df = pd.concat([y.rename(TARGET), X_cand[top_feats]], axis=1)
                sub_corr = sub_df.corr()

                # --- Original seaborn heatmap (lower triangle, annotated) ---
                plt.rcParams.update({
                    "font.size": 9,
                    "axes.titlesize": 11,
                    "axes.labelsize": 9,
                })
                fig_w = 8.5
                fig_h = min(4.8, 2.0 + 0.22 * (len(top_feats) + 1))
                fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=110)
                mask = np.triu(np.ones_like(sub_corr, dtype=bool), k=1)
                sns.heatmap(
                    sub_corr, mask=mask, annot=True, fmt=".2f",
                    cmap="coolwarm", vmin=-0.6, vmax=0.6,
                    square=False, cbar_kws={"shrink": 0.75},
                    ax=ax, annot_kws={"size": 8}
                )
                ax.set_title(f"Top {len(top_feats)} correlations with {TARGET}", pad=8)
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True, use_container_width=True)
                plt.close(fig)

st.markdown(
    """
    <div style="
        background-color:#f7fbff;
        border-left:4px solid #1f77b4;
        padding:12px 14px;
        border-radius:8px;
        margin-top:15px;
        margin-bottom:25px;">
        <b>Key Findings</b><br><br>
        Age shows the strongest positive correlation with stroke, indicating risk rises with older age.<br>
        Cardiometabolic factors—heart disease, glucose, and hypertension—have moderate positive associations and jointly drive risk.<br>
        Demographic/lifestyle effects are smaller; for example, underage work type lowers risk while former smoking shows a weak positive link.
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# Pair plot (Age, Glucose, BMI)
# =========================
st.subheader("Pair Plot (Age, Glucose, BMI)")
st.markdown("Scatterplot matrix to inspect distributions and bivariate patterns.")

ART = Path("artifacts/diagnostic")
html_path = ART / "scatter_matrix.html"
png_path  = ART / "scatter_matrix.png"
if html_path.exists():
    html = html_path.read_text(encoding="utf-8")
    st.components.v1.html(html, height=580, scrolling=False)
elif png_path.exists():
    st.image(str(png_path), use_container_width=True)
else:
    st.info("Missing: artifacts/diagnostic/scatter_matrix.html (or .png)")

# =====================================================================
# Clustering by K=2 (aligned to Stroke) — PCA scatter + KPIs
# =====================================================================
st.subheader("Clustering Analytics (K=2)")
st.markdown("Clusters the feature space into **2 groups** and aligns clusters to **Stroke** by majority class.")

if "Stroke" not in df.columns:
    st.stop()

drop_cols = id_like_columns(df.columns)
drop_cols.add("Stroke")
X_all = df.drop(columns=list(drop_cols), errors="ignore").copy()

# One-hot categoricals (always)
non_num = X_all.select_dtypes(exclude=[np.number]).columns.tolist()
if non_num:
    X_all = pd.get_dummies(X_all, columns=non_num, drop_first=True)

# Clean
X_all = X_all.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
if X_all.empty or X_all.shape[1] < 2:
    st.info("Not enough usable features after preprocessing to run clustering.")
else:
    y_stroke = pd.to_numeric(df.loc[X_all.index, "Stroke"], errors="coerce").fillna(0).astype(int)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_all)

    k = 2
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    raw_labels = km.fit_predict(X_std)

    # Align clusters to stroke by majority vote
    rates = [y_stroke[raw_labels == i].mean() if np.any(raw_labels == i) else 0 for i in range(k)]
    stroke_like = int(np.argmax(rates))
    aligned_labels = np.where(raw_labels == stroke_like, 1, 0)
    legend_map = {0: "No-stroke–like", 1: "Stroke–like"}

    sil = silhouette_score(X_std, raw_labels) if len(X_std) > k else np.nan

    # PCA projection for plot
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
    st.plotly_chart(fig_scatter, use_container_width=True, key="clustering")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Features used", f"{X_all.shape[1]}")
    m2.metric("Rows used", f"{X_all.shape[0]}")
    m3.metric("K (clusters)", "2")
    m4.metric("Silhouette score", f"{sil:.3f}" if np.isfinite(sil) else "—")

st.markdown("</div>", unsafe_allow_html=True)