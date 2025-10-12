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
# Clustering by K=2 (aligned to Stroke), non-interactive
# =====================================================================
st.subheader("Clustering Analytics (K=2)")
st.markdown(
    "All available features (numeric + one-hot encoded categoricals) are clustered into **2 groups**. "
    "Clusters are aligned to the Stroke label by majority class to aid interpretation."
)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

@st.cache_data
def _load_processed_for_cluster():
    return pd.read_csv("./jupyter-notebooks/processed_data.csv")

raw = _load_processed_for_cluster()

# --- Prepare X (features) and y (optional Stroke for alignment) ---
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

    # --- Standardize, fit KMeans with K=2 ---
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_all)

    k = 2
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    raw_labels = km.fit_predict(X_std)

    # --- Optionally align clusters to Stroke by majority ---
    # Map the cluster with higher Stroke prevalence to "Stroke-like"
    label_names = [f"Cluster {i}" for i in range(k)]
    aligned_labels = raw_labels.copy()
    legend_map = {0: "Cluster 0", 1: "Cluster 1"}
    if y_stroke is not None:
        rates = []
        for cl in range(k):
            mask = (raw_labels == cl)
            rate = y_stroke[mask].mean() if mask.any() else 0.0
            rates.append(rate)
        # cluster with larger stroke rate becomes 1 (Stroke-like)
        stroke_like = int(np.argmax(rates))
        no_stroke_like = 1 - stroke_like
        remap = {no_stroke_like: 0, stroke_like: 1}
        aligned_labels = np.vectorize(remap.get)(raw_labels)
        label_names = ["No-stroke–like", "Stroke–like"]
        legend_map = {0: "No-stroke–like", 1: "Stroke–like"}

    # --- Silhouette score ---
    sil = silhouette_score(X_std, raw_labels) if len(X_std) > k else np.nan

    # --- PCA (2D) for plotting ---
    pca = PCA(n_components=2, random_state=42)
    XY = pca.fit_transform(X_std)
    plot_df = pd.DataFrame(XY, columns=["PC1", "PC2"], index=X_all.index)
    plot_df["Cluster"] = aligned_labels  # 0/1 after alignment if y available

    # Optional: add Stroke labels for shape reference
    if y_stroke is not None:
        plot_df["Stroke"] = y_stroke.map({0: "No Stroke", 1: "Stroke"})

    # Colors fixed: 0=blue, 1=red
    color_map = {0: "#1f77b4", 1: "#d62728"}

    # --- Scatter (color by aligned cluster; symbol by true Stroke if available) ---
    fig_scatter = px.scatter(
        plot_df,
        x="PC1", y="PC2",
        color=plot_df["Cluster"].map(legend_map),
        color_discrete_map={legend_map[i]: color_map[i] for i in [0,1]},
        symbol="Stroke" if y_stroke is not None else None,
        opacity=0.65,
        title="PCA Projection (K=2, clusters aligned to Stroke)"
    )
    fig_scatter.update_layout(height=420, margin=dict(t=50, b=10, l=10, r=10), legend_title_text="Cluster")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # --- KPI row ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Features used", f"{X_all.shape[1]}")
    m2.metric("Rows used", f"{X_all.shape[0]}")
    m3.metric("K (clusters)", "2")
    m4.metric("Silhouette score", f"{sil:.3f}" if np.isfinite(sil) else "—")

    # --- If Stroke is available: show confusion-style summary ---
    if y_stroke is not None:
        # Build confusion between aligned clusters (0/1) and true Stroke (0/1)
        cm = confusion_matrix(y_stroke.values, aligned_labels, labels=[0,1])
        cm_df = pd.DataFrame(
            cm,
            index=["True: No Stroke", "True: Stroke"],
            columns=[f"Pred: {legend_map[0]}", f"Pred: {legend_map[1]}"]
        )
        st.markdown("**Alignment to Stroke (counts)**")
        st.dataframe(cm_df, use_container_width=True)

    # --- Cluster profiles (means of original feature space) ---
    prof_df = (
        pd.concat([pd.DataFrame(X_all.reset_index(drop=True)),
                   pd.Series(aligned_labels, name="Cluster")], axis=1)
        .groupby("Cluster")
        .mean()
        .reset_index()
    )
    # Add stroke rate per aligned cluster if available
    if y_stroke is not None:
        sr = pd.DataFrame({"Cluster": aligned_labels, "Stroke": y_stroke.reset_index(drop=True)})
        stroke_rate = sr.groupby("Cluster")["Stroke"].mean().mul(100).rename("Stroke Rate (%)").reset_index()
        prof_df = prof_df.merge(stroke_rate, on="Cluster", how="left")
        prof_df["Cluster"] = prof_df["Cluster"].map(legend_map)

    st.markdown("**Cluster Profiles** (feature means; Stroke Rate if available). Scroll horizontally for wide tables.")
    st.dataframe(prof_df.round(2), use_container_width=True)