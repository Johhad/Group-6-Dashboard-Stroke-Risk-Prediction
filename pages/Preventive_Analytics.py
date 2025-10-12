# preventive.py  (Streamlit page)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle, json, warnings
from pathlib import Path

st.set_page_config(page_title="Preventive Insights", page_icon="🛡️", layout="wide")
st.title("Preventive Insights 🛡️")
st.caption("Feature importance and per-patient explanation using SHAP (class-1 probability).")

# -----------------------------
# 0) Load patient input from the Risk page
# -----------------------------
if "rp_input" not in st.session_state:
    st.warning("No patient input found. Please submit data on the **Risk Prediction** page first.")
    st.stop()

patient = st.session_state["rp_input"]

# -----------------------------
# 1) Load model + expected columns + (optional) threshold
# -----------------------------
@st.cache_resource
def load_model_and_meta():
    model_path = Path("assets/trained_model_final.pickle")
    thr_path   = Path("assets/decision_threshold.json")

    if not model_path.exists():
        st.error(f"Trained model not found at: {model_path.resolve()}")
        return None, 0.5, []

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    thr = 0.5
    if thr_path.exists():
        try:
            with open(thr_path) as f:
                thr = float(json.load(f)["threshold"])
        except Exception:
            pass

    # Expected feature columns seen by the ColumnTransformer
    try:
        expected_cols = list(model.named_steps["prep"].transformers_[0][2])
    except Exception:
        expected_cols = []

    return model, thr, expected_cols

model, DECISION_THR, EXPECTED_COLS = load_model_and_meta()
if model is None or not EXPECTED_COLS:
    st.stop()

# -----------------------------
# 2) Load a small background dataset for SHAP (for speed)
# -----------------------------
@st.cache_data
def load_background(n=120):
    df = pd.read_csv("./jupyter-notebooks/processed_data.csv")
    if len(df) > n:
        df = df.sample(n, random_state=42).reset_index(drop=True)
    return df

bg_raw = load_background()

# -----------------------------
# 3) Helper: map raw fields -> model columns (SAME logic as Risk page)
# -----------------------------
def build_model_input(expected_cols,
                      age, sex, hypertension, heart_disease,
                      work_type, residence_type, glucose, bmi, smoking):
    """
    Build a single-row DataFrame with columns exactly matching the model training columns.
    Unknown one-hot columns default to 0.0.
    """
    row = {col: 0.0 for col in expected_cols}

    # numeric/binary
    if "Age" in row:           row["Age"] = float(age)
    if "Hypertension" in row:  row["Hypertension"] = float(hypertension)
    if "Heart Disease" in row: row["Heart Disease"] = float(heart_disease)
    if "Glucose" in row:       row["Glucose"] = float(glucose)
    if "BMI" in row:           row["BMI"] = float(bmi)

    # Sex dummies (Female baseline)
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
    """Map many rows from the raw dataframe to the model columns; be defensive with missing raw fields."""
    records = []
    for _, r in df_raw.iterrows():
        age  = float(r.get("Age", np.nan)) if pd.notna(r.get("Age", np.nan)) else 50.0
        sex  = r.get("Sex", "Female") if pd.notna(r.get("Sex", "Female")) else "Female"
        hyp  = int(r.get("Hypertension", 0))
        hd   = int(r.get("Heart Disease", 0))
        work = r.get("Work Type", "Private")
        res  = r.get("Residence Type", "Urban")
        glu  = float(r.get("Glucose", 100.0))
        bmi  = float(r.get("BMI", 25.0))
        smk  = r.get("Smoking Status", "never smoked")
        row_df = build_model_input(expected_cols, age, sex, hyp, hd, work, res, glu, bmi, smk)
        records.append(row_df)
    return pd.concat(records, ignore_index=True)

# -----------------------------
# 4) Build X_patient and X_background
# -----------------------------
sex_val = patient.get("Sex", "Female")              # from Risk page
hyp_val = int(patient.get("Hypertension", 0))
hd_val  = int(patient.get("Heart Disease", 0))

X_patient = build_model_input(
    EXPECTED_COLS,
    age=patient["Age"],
    sex=sex_val,
    hypertension=hyp_val,
    heart_disease=hd_val,
    work_type=patient["Work Type"],
    residence_type=patient["Residence Type"],
    glucose=patient["Glucose"],
    bmi=patient["BMI"],
    smoking=patient["Smoking Status"],
)

X_bg = build_bg_matrix(EXPECTED_COLS, bg_raw)

# -----------------------------
# 5) SHAP explanation (probability of class=1)
# -----------------------------
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

import shap
# Use a model-agnostic explainer (works with any sklearn Pipeline)
f = lambda X: model.predict_proba(pd.DataFrame(X, columns=EXPECTED_COLS))[:, 1]

# Keep background small for speed (e.g., 60)
bg = X_bg.sample(min(60, len(X_bg)), random_state=42).values
explainer = shap.KernelExplainer(f, bg)

# Explain the single patient row
phi = explainer.shap_values(X_patient.values, nsamples=200)  # returns list for predict_proba; [class0, class1]
# pick class-1 contributions
shap_values = phi[1].ravel() if isinstance(phi, list) else np.array(phi).ravel()
base_value  = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

# Put into a tidy frame
local_df = pd.DataFrame({
    "feature": EXPECTED_COLS,
    "value":   X_patient.iloc[0].values,
    "shap":    shap_values,
    "abs_shap": np.abs(shap_values)
}).sort_values("abs_shap", ascending=False)

# -----------------------------
# 6) Display – local explanation (patient level)
# -----------------------------
st.subheader("Patient-specific explanation")
cols = st.columns([1,1])
with cols[0]:
    st.markdown("**Top contributors (|SHAP|):**")
    st.dataframe(local_df.head(12)[["feature", "value", "shap"]].style.format({"value":"{:.3f}", "shap":"{:.4f}"}), use_container_width=True)

with cols[1]:
    top = local_df.head(12)
    fig_local = px.bar(
        top[::-1],  # reverse for horizontal order
        x="shap", y="feature",
        orientation="h",
        color=np.where(top["shap"]>0, "Pushes ↑ risk", "Pushes ↓ risk"),
        color_discrete_map={"Pushes ↑ risk":"crimson","Pushes ↓ risk":"steelblue"},
        labels={"shap":"SHAP value (impact on P(stroke))", "feature":""},
        title="Per-feature impact (class=1 probability)"
    )
    fig_local.update_layout(showlegend=True, height=420, margin=dict(t=40, l=10, r=10, b=10))
    st.plotly_chart(fig_local, use_container_width=True)

# -----------------------------
# 7) Display – global importance (mean |SHAP| over background)
# -----------------------------
st.subheader("Global feature importance (on a small background sample)")
# Compute shap on a small slice for speed
bg_slice = X_bg.sample(min(80, len(X_bg)), random_state=123).values
phi_bg = explainer.shap_values(bg_slice, nsamples=200)
shap_bg = phi_bg[1] if isinstance(phi_bg, list) else np.array(phi_bg)
mean_abs = np.abs(shap_bg).mean(axis=0)
global_df = pd.DataFrame({"feature": EXPECTED_COLS, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)

fig_global = px.bar(
    global_df.head(15)[::-1],
    x="mean_abs_shap", y="feature", orientation="h",
    labels={"mean_abs_shap":"Mean |SHAP|", "feature":""},
    title="Most influential features overall"
)
fig_global.update_layout(height=520, margin=dict(t=40, l=10, r=10, b=10))
st.plotly_chart(fig_global, use_container_width=True)

# -----------------------------
# 8) Context & tips
# -----------------------------
with st.expander("How to read this"):
    st.markdown(
        """
- SHAP explains the **prediction probability for class = 1 (stroke)**.
- **Positive SHAP** → pushes probability **up** (toward stroke).  
  **Negative SHAP** → pushes probability **down**.
- *Patient-specific* chart shows which features mattered most **for this input**.  
  *Global* chart aggregates mean |SHAP| over a small sample to show overall influence.
- These explanations follow your training pipeline (same scaling and one-hot encoding).
        """
    )