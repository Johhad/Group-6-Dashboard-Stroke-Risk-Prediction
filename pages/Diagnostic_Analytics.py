#Diagnostic page

from utils.ui_safety import begin_page
root = begin_page("🩺 Diagnostic Analytics")
with root:

    import streamlit as st
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import plotly.express as px
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    # Safety: avoid cross-page Matplotlib bleed
    import matplotlib.pyplot as plt
    plt.close('all')
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

                    # ---- FULL-WIDTH but compact height ----
                    plt.rcParams.update({
                        "font.size": 9,
                        "axes.titlesize": 11,
                        "axes.labelsize": 9,
                    })
                    fig_w = 8.5   # wide enough to fit full page
                    fig_h = min(4.5, 2.0 + 0.22 * (len(top_feats) + 1))  # adaptively short

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

                    # Full page width display, balanced height
                    st.pyplot(fig, clear_figure=True, use_container_width=True)
                    plt.close(fig)

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
        st.plotly_chart(fig_scatter, use_container_width=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Features used", f"{X_all.shape[1]}")
        m2.metric("Rows used", f"{X_all.shape[0]}")
        m3.metric("K (clusters)", "2")
        m4.metric("Silhouette score", f"{sil:.3f}" if np.isfinite(sil) else "—")