# pages/preventive.py 
import streamlit as st

PAGE_ID = "preventive-page"
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
begin_page("Preventive Insights 🛡️")

import pandas as pd
import numpy as np
import pickle, json, warnings
from pathlib import Path
import plotly.graph_objects as go
import seaborn as sns
import re

#st.title("Preventive Insights 🛡️")
st.caption("Impact on predicted stroke probability based Patient data using SHAP .")

# ------------------ require prior submission ------------------
st.subheader("Summary of Patient Input Data")
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
                    age, sex, married, hypertension, heart_disease,
                    work_type, residence_type, glucose, bmi, smoking):
    row = {col: 0.0 for col in expected_cols}
    if "Age" in row:           row["Age"] = float(age)
    if "Hypertension" in row:  row["Hypertension"] = float(hypertension)
    if "Heart Disease" in row: row["Heart Disease"] = float(heart_disease)
    if "Married" in row:       row["Married"] = float(married)           # <— NEW (single binary column)
    if "Glucose" in row:       row["Glucose"] = float(glucose)
    if "BMI" in row:           row["BMI"] = float(bmi)

    if "Sex_Male" in row:      row["Sex_Male"]  = 1.0 if sex == "Male" else 0.0
    if "Sex_Other" in row:     row["Sex_Other"] = 1.0 if sex == "Other" else 0.0

    if "Work Type_Private" in row:         row["Work Type_Private"] = 1.0 if work_type == "Private" else 0.0
    if "Work Type_Self-employed" in row:   row["Work Type_Self-employed"] = 1.0 if work_type == "Self-employed" else 0.0
    if "Work Type_NA/underage" in row:     row["Work Type_children"] = 1.0 if work_type == "children" else 0.0
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
        married = int(r.get("Married", 0))  # <— NEW
        hyp  = int(r.get("Hypertension", 0))
        hd   = int(r.get("Heart Disease", 0))
        work = r.get("Work Type", "Private")
        res  = r.get("Residence Type", "Urban")
        glu  = float(r.get("Glucose", 100))
        bmi  = float(r.get("BMI", 25))
        smk  = r.get("Smoking Status", "never smoked")
        recs.append(build_model_input(
            expected_cols, age, sex, married, hyp, hd, work, res, glu, bmi, smk
        ))
    return pd.concat(recs, ignore_index=True)

# ------------------ build patient row (no risk display here) ------------------
sex_val = pt.get("Sex", "Female")
married_val = int(pt.get("Married", 0))  # from rp_input
X_pt = build_model_input(
    EXPECTED_COLS,
    age=pt["Age"],
    sex=sex_val,
    married=married_val,
    hypertension=int(pt.get("Hypertension", 0)),
    heart_disease=int(pt.get("Heart Disease", 0)),
    work_type=pt["Work Type"],
    residence_type=pt["Residence Type"],
    glucose=pt["Glucose"],
    bmi=pt["BMI"],
    smoking=pt["Smoking Status"],
)

# We still compute prob internally for SHAP baseline, but do NOT show it on top.
risk_prob = float(model.predict_proba(X_pt)[0, 1])

# ------------------ TOP: compact patient summary (no risk/threshold) ------------------
# High-contrast badges so it doesn’t look messy or spaced-out
st.markdown("""
<style>
.summary-card {
background: #ffffff;
border: 1px solid #dce2ea;
border-radius: 14px;
padding: 12px 12px 6px 12px;
margin: 4px 0 10px 0;
}
.summary-grid {
display: flex; flex-wrap: wrap; gap: 8px;
}
.badge {
display: inline-flex; align-items: center; gap: 6px;
padding: 6px 10px;
border-radius: 999px;
border: 1px solid #cbd5e1;
background: #f8fafc;
color: #0f172a; font-weight: 600; font-size: 0.92rem;
}
.badge .label {
opacity: 0.7; font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

def pill(label, value):
    return f"""<span class="badge"><span class="label">{label}</span> {value}</span>"""

summary_html = f"""
<div class="summary-card">
<div class="summary-grid">
    {pill("Age", pt['Age'])}
    {pill("Sex", sex_val)}
    {pill("Married", "Yes" if married_val==1 else "No")}
    {pill("Hypertension", "Yes" if int(pt.get("Hypertension",0))==1 else "No")}
    {pill("Heart disease", "Yes" if int(pt.get("Heart Disease",0))==1 else "No")}
    {pill("BMI", f"{float(pt['BMI']):.1f}")}
    {pill("Glucose", f"{float(pt['Glucose']):.1f}")}
    {pill("Residence", pt['Residence Type'])}
    {pill("Work", pt['Work Type'])}
    {pill("Smoking", pt['Smoking Status'])}
</div>
</div>
"""
st.markdown(summary_html, unsafe_allow_html=True)

# ------------------ SHAP (horizontal bars) ------------------
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
import shap

X_bg = build_bg_matrix(EXPECTED_COLS, bg_raw)
bg   = X_bg.sample(min(60, len(X_bg)), random_state=42).values

f = lambda X: model.predict_proba(pd.DataFrame(X, columns=EXPECTED_COLS))[:, 1]
explainer = shap.KernelExplainer(f, bg)
phi = explainer.shap_values(X_pt.values, nsamples=200)
shap_values = phi[1].ravel() if isinstance(phi, list) else np.array(phi).ravel()

# ---- Top K + "Other features" (no baseline/prediction bars) ----
TOP_K = 10
order = np.argsort(-np.abs(shap_values))
top_idx = order[:TOP_K]
other_val = shap_values[order[TOP_K:]].sum() if len(order) > TOP_K else 0.0

labels = [EXPECTED_COLS[i] for i in top_idx]
vals   = [shap_values[i] for i in top_idx]
if abs(other_val) > 0:
    labels.append("Other features")
    vals.append(other_val)

labels, vals = labels[::-1], vals[::-1]
colors = ["#d62728" if v > 0 else "#2ca02c" for v in vals]  # red ↑risk, green ↓risk

st.subheader("Top contributors (patient-specific)")
c1, c2 = st.columns([1,1])

with c1:
    fig = go.Figure()
    fig.add_bar(
        x=vals, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in vals],
        textposition="inside", insidetextanchor="middle",
        textfont=dict(size=12, color="white", family="Arial"),
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:.3f}<extra></extra>",
    )
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#666")
    fig.update_layout(
        height=600,
        margin=dict(t=40, l=80, r=40, b=40),
        xaxis_title="SHAP value (impact on P(stroke))",
        yaxis_title="",
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickfont=dict(size=13), range=[-0.06, 0.12]),
        yaxis=dict(tickfont=dict(size=13)),
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown(
        f"""
**How to read this**

This chart shows which patient factors most strongly influenced the stroke-risk prediction for this specific person.
Each bar represents how much a factor increased or decreased the predicted probability of stroke.
 - Red bars → Factors that increase the predicted risk of stroke for this patient.
 - Green bars → Factors that reduce the predicted risk of stroke for this patient.
 - The longer the bar, the stronger the effect on the prediction.

Example - A patient with Glucose level as 190 mg/dl and BMI as 28

Glucose (+0.070): Having high glucose than normal makes the model to predict this patient with increase probability of stroke.

BMI (–0.047): Not having too high BMI values of the patient reduces the predicted stroke risk slightly for him or her.
"""
    )

st.info("Note: These SHAP values explain the model’s probability for this patient only. It is important to note that these values are not general risk factors.")

st.markdown("<div style='height:100vh;background-color:white;'></div>", unsafe_allow_html=True)