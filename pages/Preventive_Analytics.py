# pages/preventive.py 
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json, pickle

# ===== Page frame / styling =====
PAGE_ID = "risk-page"
st.markdown(f"<div id='{PAGE_ID}'>", unsafe_allow_html=True)
st.markdown(f"""
<style>
#{PAGE_ID} div[role='radiogroup'] label {{
  background:#fff; border:2px solid #cbd5e1; border-radius:10px;
  padding:6px 16px; margin:4px; color:#1e293b; font-weight:600; transition:all .25s;
}}
#{PAGE_ID} div[role='radiogroup'] label:hover {{
  background:#e2e8f0; border-color:#94a3b8;
}}
#{PAGE_ID} div[role='radiogroup'] label:has(input:checked) {{
  background:#2563eb !important; color:#fff !important; border-color:#1e40af !important;
  box-shadow:0 0 4px rgba(37,99,235,.6);
}}
#{PAGE_ID} div[role='radiogroup'] {{
  display:flex; gap:6px; flex-wrap:wrap;
}}
</style>
""", unsafe_allow_html=True)

try:
    from utils.ui_safety import begin_page
    begin_page("Risk Prediction 🧑‍⚕️")
except Exception:
    st.title("Risk Prediction 🧑‍⚕️")

st.markdown(
    """
    <p style='font-size:16px; color:#333; margin-top:-5px;'>
       This page predicts stroke risk using the trained model based on the input features below.
    </p>
    """,
    unsafe_allow_html=True,
)

# ===== Data (for sensible defaults) =====
@st.cache_data
def load_data():
    try:
        return pd.read_csv("./jupyter-notebooks/processed_data.csv")
    except Exception:
        return pd.DataFrame()

df = load_data()

def _median(col, fallback):
    return float(df[col].median()) if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) else fallback

def _mode(col, fallback):
    if col in df.columns and df[col].dropna().size:
        return df[col].mode().iloc[0]
    return fallback

default_age   = int(round(_median('Age', 60)))
default_gluc  = round(_median('Glucose', 100.0), 1)
# BMI is the only body-size input now
default_bmi   = round(_median('BMI', 27.0), 1)
default_hyp   = _mode('Hypertension', 1)
default_hd    = _mode('Heart Disease', 0)
default_work  = _mode('Work Type', 'Private')
default_res   = _mode('Residence Type', 'Urban')
default_smoke = _mode('Smoking Status', 'smokes')
default_married = int(_mode('Married', 0))
married_default_index = 1 if default_married == 1 else 0

sex_opts       = ['Female', 'Male', 'Other']
yes_no_display = ['No', 'Yes']
work_type_opts = ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']
res_type_opts  = ['Urban', 'Rural']
smoke_opts     = ['formerly smoked', 'never smoked', 'smokes', 'Unknown']

def _yesno_from01(val01):  # helper for 0/1 defaults -> 'Yes'/'No'
    return 'Yes' if str(val01) == '1' else 'No'

# ===== Model loader (pipeline handles preprocessing) =====
@st.cache_resource(show_spinner=True)
def load_model_and_threshold():
    base = Path(__file__).resolve().parents[1]
    model_path = base / "assets" / "trained_model_final.pickle"
    thr_path   = base / "assets" / "decision_threshold.json"

    if not model_path.exists():
        assets_list = [p.name for p in (base / "assets").glob("*")] if (base / "assets").exists() else []
        raise FileNotFoundError(
            f"Model file not found at: {model_path}\nAvailable under /assets: {assets_list}"
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    decision_thr = 0.5
    if thr_path.exists():
        try:
            with open(thr_path, "r", encoding="utf-8") as f:
                decision_thr = float(json.load(f).get("threshold", decision_thr))
        except Exception:
            pass

    return model, decision_thr

try:
    model, DECISION_THR = load_model_and_threshold()
except Exception as e:
    st.error("Model failed to load. See details below.")
    st.exception(e)
    st.stop()

# ===== Form (BMI only; types are consistent) =====
with st.form("patient_form"):
    st.markdown("### Enter Patient Data")
    c1, c2 = st.columns(2)

    with c1:
        age = st.number_input("Age", min_value=1, max_value=120, value=int(default_age), step=1)
        hypertension_disp = st.radio(
            "Hypertension", yes_no_display, horizontal=True,
            index=yes_no_display.index(_yesno_from01(default_hyp))
        )
        heart_disease_disp = st.radio(
            "Heart Disease", yes_no_display, horizontal=True,
            index=yes_no_display.index(_yesno_from01(default_hd))
        )
        sex = st.selectbox("Gender", sex_opts, index=sex_opts.index('Female'))
        married_disp = st.radio("Ever Married?", yes_no_display, horizontal=True, index=married_default_index)

    with c2:
        work_type = st.selectbox(
            "Work Type", work_type_opts,
            index=work_type_opts.index(default_work) if default_work in work_type_opts else 0
        )
        residence_type = st.selectbox(
            "Residence Type", res_type_opts,
            index=res_type_opts.index(default_res) if default_res in res_type_opts else 0
        )

        glucose = st.number_input(
            "Glucose (mg/dl)",
            min_value=0.0, max_value=1000.0,
            value=float(default_gluc), step=0.1
        )

        # ✅ BMI: single float input (no height/weight on this page)
        bmi_value = st.number_input(
            "BMI",
            min_value=5.0, max_value=80.0,
            value=float(default_bmi), step=0.1
        )

        smoking = st.selectbox(
            "Smoking Status", smoke_opts,
            index=smoke_opts.index(default_smoke) if default_smoke in smoke_opts else 1
        )

    submitted = st.form_submit_button("Predict")

# ===== Build raw input row (pipeline will encode) =====
def build_model_input_raw(age, hypertension_disp, heart_disease_disp, sex,
                          married_disp, work_type, residence_type, glucose, bmi, smoking):
    return pd.DataFrame([{
        "Age": float(age),
        "Hypertension": 1.0 if hypertension_disp == "Yes" else 0.0,
        "Heart Disease": 1.0 if heart_disease_disp == "Yes" else 0.0,
        "Married": 1.0 if married_disp == "Yes" else 0.0,
        "Glucose": float(glucose),
        "BMI": float(bmi),
        # raw categoricals; your sklearn Pipeline should handle OHE
        "Sex": sex,
        "Work Type": work_type,
        "Residence Type": residence_type,
        "Smoking?": smoking,
    }])

# ===== Gauge helpers =====
def band_and_color(score: float, thr_pct: float = 50.0):
    if score < min(33.0, thr_pct * 0.66):
        return "Low", "#2ca02c"
    elif score < max(66.0, thr_pct):
        return "Moderate", "#ff7f0e"
    return "High", "#d62728"

def render_risk_gauge(score: float, title="Estimated Risk Score", decision_thr: float = 0.5):
    band, color = band_and_color(score, thr_pct=decision_thr * 100)
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:12px; margin: 6px 0 8px 0;">
            <h3 style="margin:0;">{title}</h3>
            <span style="
                display:inline-block; padding:6px 12px; border-radius:999px;
                background:{color}1A; color:{color}; font-weight:600; font-size:0.95rem;">
                {band} Risk • {score:.0f}/100
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'suffix': "/100", 'font': {'size': 46}},
        gauge={
            'shape': "angular",
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#888"},
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': "rgba(0,0,0,0)",
            'steps': [
                {'range': [0, 33], 'color': '#e8f5e9'},
                {'range': [33, 66], 'color': '#fff3e0'},
                {'range': [66, 100], 'color': '#ffebee'}
            ],
            'threshold': {'line': {'color': color, 'width': 4}, 'thickness': 0.75, 'value': score}
        },
        title={'text': ""}
    ))
    fig.update_layout(
        margin=dict(t=10, b=0, l=10, r=10),
        height=320,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14)
    )
    fig.add_annotation(
        text=f"Decision threshold ≈ {decision_thr:.3f}",
        x=0.5, y=0.90, showarrow=False,
        font=dict(size=14, color="#666")
    )
    st.plotly_chart(fig, use_container_width=True)

# ===== Predict on submit =====
if submitted:
    X_user = build_model_input_raw(
        age=age,
        hypertension_disp=hypertension_disp,
        heart_disease_disp=heart_disease_disp,
        sex=sex,
        married_disp=married_disp,
        work_type=work_type,
        residence_type=residence_type,
        glucose=glucose,
        bmi=bmi_value,
        smoking=smoking
    )

    try:
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X_user)[0][1])
        elif hasattr(model, "decision_function"):
            from scipy.special import expit
            prob = float(expit(model.decision_function(X_user))[0])
        else:
            pred_raw = model.predict(X_user)[0]
            prob = float(pred_raw) if isinstance(pred_raw, (int, float, np.floating)) else float(pred_raw == 1)
    except Exception as e:
        st.error("Prediction failed. See details below.")
        st.exception(e)
        st.stop()

    pred = int(prob >= DECISION_THR)

    render_risk_gauge(prob * 100.0, title="Model-Estimated Stroke Risk", decision_thr=DECISION_THR)
    st.markdown(f"**Predicted probability:** `{prob:.3f}`")
    st.markdown(f"**Decision (threshold = {DECISION_THR:.3f}):** {'**Stroke risk (1)**' if pred==1 else '**Low stroke risk (0)**'}")

    # NOTE: No height/weight in the session payload now — Preventive page does not need them.
    st.session_state['rp_input'] = {
        'Age': age,
        'Sex': sex,
        'Married': 1 if married_disp == 'Yes' else 0,
        'Hypertension': 1 if hypertension_disp == 'Yes' else 0,
        'Heart Disease': 1 if heart_disease_disp == 'Yes' else 0,
        'Work Type': work_type,
        'Residence Type': residence_type,
        'Glucose': float(glucose),
        'BMI': float(bmi_value),
        'Smoking Status': smoking,
        'pred_proba': prob,
        'predicted': pred
    }
else:
    st.info("Fill the form and click **Predict** to see the model-estimated risk.")

st.markdown("<div style='height:100vh;background-color:white;'></div>", unsafe_allow_html=True)