# preventive.py  (Streamlit page)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle, json, warnings
from pathlib import Path

st.set_page_config(page_title="Preventive Insights", page_icon="🛡️", layout="wide")
st.title("Preventive Insights 🛡️")
st.caption("Patient-specific explanation using SHAP (impact on predicted stroke probability).")

# -------------------------------------------------------------------
# 0) Require a prior submission from the Risk page
# -------------------------------------------------------------------
if "rp_input" not in st.session_state:
    st.warning("No patient input found. Please submit data on the **Risk Prediction** page first.")
    st.stop()

patient = st.session_state["rp_input"]

# -------------------------------------------------------------------
# 1) Load model + expected columns
# -------------------------------------------------------------------
@st.cache_resource
def load_model_and_meta():
    model_path = Path("assets/trained_model_final.pickle")
    if not model_path.exists():
        st.error(f"Trained model not found at: {model_path.resolve()}")
        return None, []

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    try:
        expected_cols = list(model.named_steps["prep"].transformers_[0][2])
    except Exception:
        expected_cols = []

    return model, expected_cols

model, EXPECTED_COLS = load_model_and_meta()
if model is None or not EXPECTED_COLS:
    st.stop()

# -------------------------------------------------------------------
# 2) Tiny background for SHAP (kept small for speed)
# -------------------------------------------------------------------
@st.cache_data
def load_background(n=60):
    df = pd.read_csv("./jupyter-notebooks/processed_data.csv")
    if len(df) > n:
        df = df.sample(n, random_state=42).reset_index(drop=True)
    return df

bg_raw = load_background()

# -------------------------------------------------------------------
# 3) Helpers (match Risk page encoding exactly)
# -------------------------------------------------------------------
def build_model_input(expected_cols,
                      age, sex, hypertension, heart_disease,
                      work_type, residence_type, glucose, bmi, smoking):
    row = {col: 0.0 for col in expected_cols}
    # numeric/binary
    if "Age" in row:           row["Age"] = float(age)
    if "Hypertension" in row:  row["Hypertension"] = float(hypertension)
    if "Heart Disease" in row: row["Heart Disease"] = float(heart_disease)
    if "Glucose" in row:       row["Glucose"] = float(glucose)
    if "BMI" in row:           row["BMI"] = float(bmi)
    # Sex (Female baseline)
    if "Sex_Male" in row:      row["Sex_Male"]  = 1.0 if sex == "Male" else 0.0
    if "Sex_Other" in row:     row["Sex_Other"] = 1.0 if sex == "Other" else 0.0
    # Work Type
    if "Work Type_Private" in row:         row["Work Type_Private"] = 1.0 if work_type == "Private" else 0.0
    if "Work Type_Self-employed" in row:   row["Work Type_Self-employed"] = 1.0 if work_type == "Self-employed" else 0.0
    if "Work Type_children" in row:        row["Work Type_children"] = 1.0 if work_type == "children" else 0.0
    if "Work Type_Never_worked" in row:    row["Work Type_Never_worked"] = 1.0 if work_type == "Never_worked" else 0.0
    if "Work Type_Govt_job" in row:        row["Work Type_Govt_job"] = 1.0 if work_type == "Govt_job" else 0.0
    # Residence
    if "Residence Type_Urban" in row:      row["Residence Type_Urban"] = 1.0 if residence_type == "Urban" else 0.0
    if "Residence Type_Rural" in row:      row["Residence Type_Rural"] = 1.0 if residence_type == "Rural" else 0.0
    # Smoking
    if "Smoking?_formerly smoked" in row:  row["Smoking?_formerly smoked"] = 1.0 if smoking == "formerly smoked" else 0.0
    if "Smoking?_never smoked" in row:     row["Smoking?_never smoked"]  = 1.0 if smoking == "never smoked"  else 0.0
    if "Smoking?_smokes" in row:           row["Smoking?_smokes"]        = 1.0 if smoking == "smokes"         else 0.0
    if "Smoking?_Unknown" in row:          row["Smoking?_Unknown"]       = 1.0 if smoking == "Unknown"        else 0.0
    return pd.DataFrame([row], columns=expected_cols)

def build_bg_matrix(expected_cols, df_raw):
    recs = []
    for _, r in df_raw.iterrows():
        age  = float(r.get("Age", 50))
        sex  = r.get("Sex", "Female") if pd.notna(r.get("Sex", "Female")) else "Female"
        hyp  = int(r.get("Hypertension", 0))
        hd   = int(r.get("Heart Disease", 0))
        work = r.get("Work Type", "Private")
        res  = r.get("Residence Type", "Urban")
        glu  = float(r.get("Glucose", 100))
        bmi  = float(r.get("BMI", 25))
        smk  = r.get("Smoking Status", "never smoked")
        recs.append(build_model_input(expected_cols, age, sex, hyp, hd, work, res, glu, bmi, smk))
    return pd.concat(recs, ignore_index=True)

# -------------------------------------------------------------------
# 4) Build patient row and background
# -------------------------------------------------------------------
sex_val = patient.get("Sex", "Female")
X_patient = build_model_input(
    EXPECTED_COLS,
    age=patient["Age"],
    sex=sex_val,
    hypertension=int(patient.get("Hypertension", 0)),
    heart_disease=int(patient.get("Heart Disease", 0)),
    work_type=patient["Work Type"],
    residence_type=patient["Residence Type"],
    glucose=patient["Glucose"],
    bmi=patient["BMI"],
    smoking=patient["Smoking Status"],
)
X_bg = build_bg_matrix(EXPECTED_COLS, bg_raw)

# -------------------------------------------------------------------
# 5) SHAP (local only) – class=1 probability
# -------------------------------------------------------------------
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

import shap
f = lambda X: model.predict_proba(pd.DataFrame(X, columns=EXPECTED_COLS))[:, 1]
bg = X_bg.sample(min(60, len(X_bg)), random_state=42).values  # tiny background for speed
explainer = shap.KernelExplainer(f, bg)

# nsamples controls speed/accuracy; keep modest
phi = explainer.shap_values(X_patient.values, nsamples=200)
shap_values = phi[1].ravel() if isinstance(phi, list) else np.array(phi).ravel()

# Table sorted by SHAP value (highest -> lowest)
local_df = pd.DataFrame({
    "feature": EXPECTED_COLS,
    "input_value": X_patient.iloc[0].values,
    "shap_value": shap_values
}).sort_values("shap_value", ascending=False)

# -------------------------------------------------------------------
# 6) Display (stacked vertically, full width)
# -------------------------------------------------------------------
st.subheader("Patient-specific SHAP explanation")

# A) Table (full width)
st.markdown("**Per-feature impact (sorted by SHAP value)**")
st.dataframe(
    local_df.style.format({"input_value": "{:.3f}", "shap_value": "{:.5f}"}),
    use_container_width=True,
    height=480
)

# B) Bar chart (full width)
top_k = st.slider("Show top N features in the chart", min_value=5, max_value=min(25, len(local_df)), value=12, step=1)
top = local_df.head(top_k)

fig_local = px.bar(
    top[::-1],  # reverse for horizontal order (largest at top)
    x="shap_value", y="feature",
    orientation="h",
    color=np.where(top["shap_value"] > 0, "Pushes ↑ risk", "Pushes ↓ risk"),
    color_discrete_map={"Pushes ↑ risk": "crimson", "Pushes ↓ risk": "steelblue"},
    labels={"shap_value": "SHAP value (impact on P(stroke))", "feature": ""},
    title="Per-feature impact for this patient"
)
fig_local.update_layout(
    showlegend=True,
    height=520,
    margin=dict(t=50, l=10, r=10, b=10)
)
st.plotly_chart(fig_local, use_container_width=True)

with st.expander("How to read this"):
    st.markdown(
        """
- We explain the model’s **predicted probability of stroke (class = 1)**.
- **Positive SHAP** values push the probability **up** (toward stroke); **negative** push it **down**.
- The table is sorted by SHAP value (largest positive at top).  
  Use the slider to control how many features appear in the chart below.
        """
    )