import streamlit as st
import pandas as pd
from pathlib import Path

st.title("🩺 Diagnostic Analytics")
st.caption("This page shows diagnostic analysis of the dataset that the project dashboard utilized and trained on")

ART = Path("artifacts/diagnostic")

# --- heatmap image ---
st.subheader("Correlation Heatmap")
st.markdown("This heatmap shows pairwise correlations among all encoded features in the dataset. "
            "It provides a quick overview of linear relationships between variables.")
hm_path = ART / "correlation_heatmap.png"
if hm_path.exists():
    st.image(str(hm_path), use_container_width=True)
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


