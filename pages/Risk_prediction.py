# pages/Risk_prediction.py
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json, pickle

# ==============================
# Page header / styling
# ==============================
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
#{PAGE_ID} div[role='radiogroup'] {{ display:flex; gap:6px; flex-wrap:wrap; }}
</style>
""", unsafe_allow_html=True)

try:
    from utils.ui_safety import begin_page
    begin_page("Risk Prediction 🧑‍⚕️")
except Exception:
    st.title("Risk Prediction 🧑‍⚕️")

st.caption("This page predicts stroke risk using the trained model. Inputs are mapped to the exact feature columns used in training.")

# ==============================
# Small helpers
# ==============================
def ni_float(label, min_v, max_v, value, step, **kwargs):
    return st.number_input(label, min_value=float(min_v), max_value=float(max_v),
                           value=float(value), step=float(step), **kwargs)

def ni_int(label, min_v, max_v, value, step, **kwargs):
    return st.number_input(label, min_value=int(min_v), max_value=int(max_v),
                           value=int(value), step=int(step), **kwargs)

def _positive_class_index(model) -> int:
    """Return the predict_proba column index that corresponds to the positive (1) class."""
    try:
        classes = getattr(model, "classes_", None)
        if classes is None and hasattr(model, "named_steps"):
            for _, step in model.named_steps.items():
                if hasattr(step, "classes_"):
                    classes = step.classes_
                    break
        if classes is not None:
            idx = int(np.where(np.asarray(classes) == 1)[0][0])
            return idx
    except Exception:
        pass
    return 1  # default assumption if unknown

# ==============================
# Load model + threshold
# ==============================
@st.cache_resource(show_spinner=True)
def load_model_and_threshold():
    base = Path(__file__).resolve().parents[1]
    model_path = base / "assets" / "trained_model_final.pickle"
    thr_path   = base / "assets" / "decision_threshold.json"

    if not model_path.exists():
        assets = [p.name for p in (base / "assets").glob("*")] if (base / "assets").exists() else []
        raise FileNotFoundError(f"Model not found at {model_path}. Assets folder has: {assets}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    thr = 0.5
    if thr_path.exists():
        try:
            thr = float(json.loads(thr_path.read_text()).get("threshold", thr))
        except Exception:
            pass

    return model, thr

try:
    model, DECISION_THR = load_model_and_threshold()
except Exception as e:
    st.error("Model failed to load.")
    st.exception(e)
    st.stop()

# ==============================
# Training schema (fixed)
# ==============================
TRAIN_COLS = [
    "Age", "Hypertension", "Heart Disease", "Married", "Glucose", "BMI",
    "Sex_Male", "Sex_Other",
    "Work Type_Never_worked", "Work Type_Private", "Work Type_Self-employed", "Work Type_children",
    "Residence Type_Urban",
    "Smoking?_formerly smoked", "Smoking?_never smoked", "Smoking?_smokes",
    # "Stroke" exists in dataset but is NOT a model input; omitted intentionally
]

# ==============================
# Defaults (nice UX)
# ==============================
@st.cache_data
def load_defaults():
    try:
        df = pd.read_csv("./jupyter-notebooks/processed_data.csv")
    except Exception:
        df = pd.DataFrame()

    def _median(col, fallback):
        return float(df[col].median()) if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) else fallback

    def _mode(col, fallback):
        if col in df.columns and df[col].dropna().size:
            return df[col].mode().iloc[0]
        return fallback

    defaults = dict(
        age = int(round(_median('Age', 60))),
        gluc = round(_median('Glucose', 100.0), 1),
        height_cm = 150,
        weight_kg = 60.0,
        hyp = int(_mode('Hypertension', 0)),
        hd  = int(_mode('Heart Disease', 0)),
        work = _mode('Work Type', 'Private'),
        res  = _mode('Residence Type', 'Urban'),
        smoke= _mode('Smoking Status', 'Unknown'),
        married = int(_mode('Married', 0)),
    )
    return defaults

D = load_defaults()
sex_opts       = ['Female', 'Male', 'Other']           # Female is baseline
work_type_opts = ['Govt_job', 'Private', 'Self-employed', 'children', 'Never_worked']  # Govt_job baseline
res_type_opts  = ['Rural', 'Urban']                    # Rural baseline
smoke_opts     = ['Unknown', 'formerly smoked', 'never smoked', 'smokes']  # Unknown baseline
yes_no_display = ['No', 'Yes']

# ==============================
# Input form
# ==============================
with st.form("patient_form"):
    st.subheader("Enter Patient Data")
    c1, c2 = st.columns(2)

    with c1:
        age = ni_int("Age", 1, 120, D['age'], 1)
        hypertension_disp = st.radio("Hypertension", yes_no_display, horizontal=True,
                                     index=1 if D['hyp']==1 else 0)
        heart_disease_disp = st.radio("Heart Disease", yes_no_display, horizontal=True,
                                      index=1 if D['hd']==1 else 0)
        sex = st.selectbox("Gender", sex_opts, index=sex_opts.index('Female'))
        married_disp = st.radio("Ever Married?", yes_no_display, horizontal=True,
                                index=1 if D['married']==1 else 0)

    with c2:
        work_type = st.selectbox("Work Type", work_type_opts,
                                 index=work_type_opts.index(D['work']) if D['work'] in work_type_opts else 0)
        residence_type = st.selectbox("Residence Type", res_type_opts,
                                      index=res_type_opts.index(D['res']) if D['res'] in res_type_opts else 0)
        glucose = ni_float("Glucose (mg/dl)", 0.0, 1000.0, D['gluc'], 0.1)

        height_cm = ni_int("Height (cm)", 50, 300, D['height_cm'], 1, key="risk_height")
        weight_kg = ni_float("Weight (kg)", 10.0, 300.0, D['weight_kg'], 1.0, key="risk_weight")
        height_m = height_cm / 100.0
        bmi_value = float(weight_kg) / (height_m**2) if height_m > 0 else 0.0

        smoking = st.selectbox("Smoking Status", smoke_opts,
                               index=smoke_opts.index(D['smoke']) if D['smoke'] in smoke_opts else 0)

    submitted = st.form_submit_button("Predict")

# ==============================
# Build EXACT encoded input row
# ==============================
def build_encoded_row(
    age, hyp_disp, hd_disp, married_disp, glucose, bmi,
    sex, work_type, residence_type, smoking
) -> pd.DataFrame:
    """
    Build a single-row DataFrame matching TRAIN_COLS exactly.
    Baselines (all-zero for that group):
      Sex: Female
      Work Type: Govt_job
      Residence: Rural
      Smoking: Unknown
    """
    row = {col: 0.0 for col in TRAIN_COLS}

    # numeric/binary
    row["Age"] = float(age)
    row["Hypertension"] = 1.0 if hyp_disp == "Yes" else 0.0
    row["Heart Disease"] = 1.0 if hd_disp == "Yes" else 0.0
    row["Married"] = 1.0 if married_disp == "Yes" else 0.0
    row["Glucose"] = float(glucose)
    row["BMI"] = float(bmi)

    # sex (Female baseline)
    if sex == "Male":
        row["Sex_Male"] = 1.0
    elif sex == "Other":
        row["Sex_Other"] = 1.0

    # work type (Govt_job baseline)
    if work_type == "Never_worked":
        row["Work Type_Never_worked"] = 1.0
    elif work_type == "Private":
        row["Work Type_Private"] = 1.0
    elif work_type == "Self-employed":
        row["Work Type_Self-employed"] = 1.0
    elif work_type == "children":
        row["Work Type_children"] = 1.0
    # else Govt_job -> all zeros for work-type dummies

    # residence (Rural baseline)
    if residence_type == "Urban":
        row["Residence Type_Urban"] = 1.0

    # smoking (Unknown baseline)
    if smoking == "formerly smoked":
        row["Smoking?_formerly smoked"] = 1.0
    elif smoking == "never smoked":
        row["Smoking?_never smoked"] = 1.0
    elif smoking == "smokes":
        row["Smoking?_smokes"] = 1.0

    return pd.DataFrame([row], columns=TRAIN_COLS)

# ==============================
# Threshold-aware gauge
# ==============================
def risk_badge(score_pct: float, thr_pct: float, margin: float = 5.0):
    if score_pct < thr_pct - margin:
        return "Below threshold", "#2ca02c"
    elif score_pct <= thr_pct + margin:
        return "Borderline", "#ff7f0e"
    else:
        return "Above threshold", "#d62728"

def render_risk_gauge(score_pct: float, title="Model-Estimated Stroke Risk", decision_thr: float = 0.5):
    thr_pct = float(decision_thr) * 100.0
    label, color = risk_badge(score_pct, thr_pct, margin=5.0)

    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:12px; margin: 6px 0 8px 0;">
            <h3 style="margin:0;">{title}</h3>
            <span style="
                display:inline-block; padding:6px 12px; border-radius:999px;
                background:{color}1A; color:{color}; font-weight:600; font-size:0.95rem;">
                {label} • {score_pct:.0f}/100
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    margin = 5.0
    lo = max(0.0, thr_pct - margin)
    hi = min(100.0, thr_pct + margin)
    steps = [
        {'range': [0, lo], 'color': '#e8f5e9'},
        {'range': [lo, hi], 'color': '#fff3e0'},
        {'range': [hi, 100], 'color': '#ffebee'}
    ]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score_pct,
        number={'suffix': "/100", 'font': {'size': 46}},
        gauge={
            'shape': "angular",
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#888"},
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': "rgba(0,0,0,0)",
            'steps': steps,
            'threshold': {
                'line': {'color': '#fb8c00', 'width': 4},
                'thickness': 0.75,
                'value': thr_pct
            }
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
        text=f"Decision threshold = {decision_thr:.3f}",
        x=0.5, y=0.90, showarrow=False,
        font=dict(size=14, color="#666")
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(
        """
        <div style="
            background-color:#f7fbff;
            border-left:4px solid #1f77b4;
            padding:12px 14px;
            border-radius:8px;
            margin-top:15px;
            margin-bottom:25px;">
            <b>How to read the prediction graph and values </b><br><br>
            • The gauge shows how likely the patient is to experience a stroke based on their personal health profile and risk factors.<br>
            • The color zones indicate <b>low</b>, <b>moderate</b>, or <b>high</b> risk levels — helping to quickly identify patients who may need closer monitoring.<br>
            • The orange line represents the <b>decision threshold</b> used by the model to separate lower-risk from higher-risk cases.<br>
            • This estimate is meant to <b>support clinical judgment</b> — it does not replace diagnosis, but helps prioritize prevention and follow-up actions.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================
# Predict
# ==============================
if submitted:
    # 1) build encoded row that matches training matrix
    try:
        X_user = build_encoded_row(
            age=age,
            hyp_disp=hypertension_disp,
            hd_disp=heart_disease_disp,
            married_disp=married_disp,
            glucose=glucose,
            bmi=bmi_value,
            sex=sex,
            work_type=work_type,
            residence_type=residence_type,
            smoking=smoking
        )
    except Exception as e:
        st.error("Failed to prepare features for the model.")
        st.exception(e)
        st.stop()

    # 2) predict probability for the positive class
    try:
        if hasattr(model, "predict_proba"):
            pos_idx = _positive_class_index(model)
            prob = float(model.predict_proba(X_user)[0][pos_idx])
        elif hasattr(model, "decision_function"):
            from scipy.special import expit
            prob = float(expit(model.decision_function(X_user))[0])
        else:
            pred_raw = model.predict(X_user)[0]
            prob = float(pred_raw) if isinstance(pred_raw, (int, float, np.floating)) else float(pred_raw == 1)
    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
        st.stop()

    pred = int(prob >= DECISION_THR)

    # 3) show results
    render_risk_gauge(prob * 100.0, title="Model-Estimated Stroke Risk", decision_thr=DECISION_THR)
    st.markdown(f"**Predicted probability:** `{prob:.3f}`")
    st.markdown(f"**Decision (threshold = {DECISION_THR:.3f}):** {'**Stroke risk (1)**' if pred==1 else '**Low stroke risk (0)**'}")

    # keep for other pages
    st.session_state['rp_input'] = {
        'Age': age,
        'Sex': sex,
        'Married': 1 if married_disp == 'Yes' else 0,
        'Hypertension': 1 if hypertension_disp == 'Yes' else 0,
        'Heart Disease': 1 if heart_disease_disp == 'Yes' else 0,
        'Work Type': work_type,
        'Residence Type': residence_type,
        'Glucose': float(glucose),
        'Height (cm)': float(height_cm),
        'Weight (kg)': float(weight_kg),
        'BMI': float(bmi_value),
        'Smoking Status': smoking,
        'pred_proba': prob,
        'predicted': pred
    }
else:
    st.info("Fill the form and click **Predict** to see the model-estimated risk.")

st.markdown("<div style='height:100vh;background-color:white;'></div>", unsafe_allow_html=True)









