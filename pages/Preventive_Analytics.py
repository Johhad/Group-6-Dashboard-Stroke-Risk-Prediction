# pages/preventive.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle, json, warnings
from pathlib import Path
import plotly.graph_objects as go

st.set_page_config(page_title="Preventive Insights", page_icon="🛡️", layout="wide")
st.title("Preventive Insights 🛡️")
st.caption("Patient-specific explanation using SHAP (impact on predicted stroke probability).")

# ------------------ require prior submission ------------------
if "rp_input" not in st.session_state:
    st.warning("No patient input found. Please submit data on the **Risk Prediction** page first.")
    st.stop()
pt = st.session_state["rp_input"]

# ------------------ load model + columns + threshold ------------------
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
    try:
        expected_cols = list(model.named_steps["prep"].transformers_[0][2])
    except Exception:
        expected_cols = []
    return model, thr, expected_cols

model, DECISION_THR, EXPECTED_COLS = load_model_and_meta()
if model is None or not EXPECTED_COLS:
    st.stop()

# ------------------ tiny background for SHAP ------------------
@st.cache_data
def load_background(n=60):
    df = pd.read_csv("./jupyter-notebooks/processed_data.csv")
    if len(df) > n:
        df = df.sample(n, random_state=42).reset_index(drop=True)
    return df
bg_raw = load_background()

# ------------------ helpers (match Risk page encoding) ------------------
def build_model_input(expected_cols,
                      age, sex, hypertension, heart_disease,
                      work_type, residence_type, glucose, bmi, smoking):
    row = {col: 0.0 for col in expected_cols}
    if "Age" in row:           row["Age"] = float(age)
    if "Hypertension" in row:  row["Hypertension"] = float(hypertension)
    if "Heart Disease" in row: row["Heart Disease"] = float(heart_disease)
    if "Glucose" in row:       row["Glucose"] = float(glucose)
    if "BMI" in row:           row["BMI"] = float(bmi)
    if "Sex_Male" in row:      row["Sex_Male"]  = 1.0 if sex == "Male" else 0.0
    if "Sex_Other" in row:     row["Sex_Other"] = 1.0 if sex == "Other" else 0.0
    if "Work Type_Private" in row:         row["Work Type_Private"] = 1.0 if work_type == "Private" else 0.0
    if "Work Type_Self-employed" in row:   row["Work Type_Self-employed"] = 1.0 if work_type == "Self-employed" else 0.0
    if "Work Type_children" in row:        row["Work Type_children"] = 1.0 if work_type == "children" else 0.0
    if "Work Type_Never_worked" in row:    row["Work Type_Never_worked"] = 1.0 if work_type == "Never_worked" else 0.0
    if "Work Type_Govt_job" in row:        row["Work Type_Govt_job"] = 1.0 if work_type == "Govt_job" else 0.0
    if "Residence Type_Urban" in row:      row["Residence Type_Urban"] = 1.0 if residence_type == "Urban" else 0.0
    if "Residence Type_Rural" in row:      row["Residence Type_Rural"] = 1.0 if residence_type == "Rural" else 0.0
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

# ------------------ build patient row + risk ------------------
sex_val = pt.get("Sex", "Female")
X_pt = build_model_input(
    EXPECTED_COLS,
    age=pt["Age"],
    sex=sex_val,
    hypertension=int(pt.get("Hypertension", 0)),
    heart_disease=int(pt.get("Heart Disease", 0)),
    work_type=pt["Work Type"],
    residence_type=pt["Residence Type"],
    glucose=pt["Glucose"],
    bmi=pt["BMI"],
    smoking=pt["Smoking Status"],
)
risk_prob = float(model.predict_proba(X_pt)[0, 1])
decision  = "Stroke risk (1)" if risk_prob >= DECISION_THR else "No stroke risk (0)"

# ------------------ TOP: patient summary metrics ------------------
st.subheader("Patient summary & risk score")
row1 = st.columns([1,1,1,1])
row1[0].metric("Risk probability", f"{risk_prob:.3f}")
row1[1].metric("Decision threshold", f"{DECISION_THR:.3f}")
row1[2].metric("Decision", decision)
row1[3].metric("Age", str(pt["Age"]))

row2 = st.columns([1,1,1,1,1])
row2[0].metric("Sex", sex_val)
row2[1].metric("Hypertension", "Yes" if int(pt.get("Hypertension",0))==1 else "No")
row2[2].metric("Heart Disease", "Yes" if int(pt.get("Heart Disease",0))==1 else "No")
row2[3].metric("Glucose", f"{float(pt['Glucose']):.1f}")
row2[4].metric("BMI", f"{float(pt['BMI']):.1f}")

row3 = st.columns([1,1,1])
row3[0].metric("Work Type", pt["Work Type"])
row3[1].metric("Residence", pt["Residence Type"])
row3[2].metric("Smoking", pt["Smoking Status"])

st.markdown("---")

# ------------------ Local SHAP (horizontal bars) ------------------
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
import shap

X_bg = build_bg_matrix(EXPECTED_COLS, bg_raw)
bg   = X_bg.sample(min(60, len(X_bg)), random_state=42).values

f = lambda X: model.predict_proba(pd.DataFrame(X, columns=EXPECTED_COLS))[:, 1]
explainer = shap.KernelExplainer(f, bg)
phi = explainer.shap_values(X_pt.values, nsamples=200)
shap_values = phi[1].ravel() if isinstance(phi, list) else np.array(phi).ravel()

# ---- Top 8 + "Other features" (no baseline/prediction bars) ----
TOP_K = 8
order = np.argsort(-np.abs(shap_values))
top_idx = order[:TOP_K]
other_val = shap_values[order[TOP_K:]].sum() if len(order) > TOP_K else 0.0

labels = [EXPECTED_COLS[i] for i in top_idx]
vals   = [shap_values[i] for i in top_idx]
if abs(other_val) > 0:
    labels.append("Other features")
    vals.append(other_val)

# Reverse for horizontal bars (top at top)
labels = labels[::-1]
vals   = vals[::-1]

# Colors by sign
colors = ["#d62728" if v > 0 else "#2ca02c" for v in vals]  # red = ↑risk, green = ↓risk

st.subheader("Top contributors (patient-specific)")

c1, c2 = st.columns([1,1])
with c1:
    fig = go.Figure()

    # Format SHAP labels with signs (+/-)
    text_labels = [f"{v:+.3f}" for v in vals]

    fig.add_bar(
        x=vals,
        y=labels,
        orientation="h",
        marker_color=colors,                # red = ↑ risk, green = ↓ risk
        text=text_labels,                   # show SHAP values
        textposition="inside",              # ← place labels within bars
        insidetextanchor="middle",
        textfont=dict(size=12, color="white", family="Arial"),
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:.3f}<extra></extra>",
    )

    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#666")

    fig.update_layout(
        height=420,
        margin=dict(t=30, l=10, r=10, b=10),
        xaxis_title="SHAP value (impact on P(stroke))",
        yaxis_title="",
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(zeroline=False, automargin=True),
    )

    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown(
        f"""
**How to read this**

- Bars show how each feature **pushes the probability** up (red) or down (green).
- Listed are the **top {TOP_K} features by absolute impact**, plus **“Other features”** (all remaining effects).
- Positive bar means higher predicted risk; negative bar means lower risk.
- Current patient’s predicted probability: **{risk_prob:.3f}**  
  Decision (threshold {DECISION_THR:.3f}): **{decision}**.
"""
    )

st.info("These SHAP values explain the model’s probability for this single patient only.")