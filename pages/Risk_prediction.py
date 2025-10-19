# pages/Risk_prediction.py
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json, pickle

# ---------- Page header / styling ----------
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

st.markdown(
    """
    <p style='font-size:16px; color:#333; margin-top:-5px;'>
       This page predicts stroke risk using the trained model based on the input features below.
    </p>
    """,
    unsafe_allow_html=True,
)

# ---------- Helpers to avoid mixed number_input types ----------
def ni_float(label, min_v, max_v, value, step, **kwargs):
    return st.number_input(label,
        min_value=float(min_v), max_value=float(max_v),
        value=float(value), step=float(step), **kwargs
    )

def ni_int(label, min_v, max_v, value, step, **kwargs):
    return st.number_input(label,
        min_value=int(min_v), max_value=int(max_v),
        value=int(value), step=int(step), **kwargs
    )

# ---------- Defaults from data ----------
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

default_age        = int(round(_median('Age', 60)))
default_gluc       = round(_median('Glucose', 100.0), 1)
default_height_cm  = 150
default_weight_kg  = 80.0
default_bmi_guess  = round(default_weight_kg / ((default_height_cm / 100) ** 2), 1)
default_hyp        = _mode('Hypertension', 1)
default_hd         = _mode('Heart Disease', 0)
default_work       = _mode('Work Type', 'Private')
default_res        = _mode('Residence Type', 'Urban')
default_smoke      = _mode('Smoking Status', 'smokes')
default_married    = int(_mode('Married', 0))
married_default_index = 1 if default_married == 1 else 0

sex_opts       = ['Female', 'Male', 'Other']
yes_no_display = ['No', 'Yes']
work_type_opts = ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']
res_type_opts  = ['Urban', 'Rural']
smoke_opts     = ['formerly smoked', 'never smoked', 'smokes', 'Unknown']

def _yesno_from01(val01):  # 0/1 -> Yes/No
    return 'Yes' if str(val01) == '1' else 'No'

# ---------- Load model + expected columns ----------
@st.cache_resource(show_spinner=True)
def load_model_thr_cols():
    base = Path(__file__).resolve().parents[1]
    model_path = base / "assets" / "trained_model_final.pickle"
    thr_path   = base / "assets" / "decision_threshold.json"

    if not model_path.exists():
        assets_list = [p.name for p in (base / "assets").glob("*")] if (base / "assets").exists() else []
        raise FileNotFoundError(f"Model file not found at: {model_path}\nAvailable: {assets_list}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # decision threshold (optional)
    decision_thr = 0.5
    if thr_path.exists():
        try:
            with open(thr_path, "r", encoding="utf-8") as f:
                decision_thr = float(json.load(f).get("threshold", decision_thr))
        except Exception:
            pass

    # try to read the expected training columns from the prep step
    expected_cols = None
    try:
        # typical: ColumnTransformer([("num", <scaler>, expected_cols)], remainder="drop")
        expected_cols = list(model.named_steps["prep"].transformers_[0][2])
    except Exception:
        pass

    return model, decision_thr, expected_cols

try:
    model, DECISION_THR, EXPECTED_COLS = load_model_thr_cols()
except Exception as e:
    st.error("Model failed to load.")
    st.exception(e)
    st.stop()

# ---------- Form ----------
with st.form("patient_form"):
    st.markdown("### Enter Patient Data")
    c1, c2 = st.columns(2)

    with c1:
        age = ni_int("Age", 1, 120, default_age, 1)
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
        glucose = ni_float("Glucose (mg/dl)", 0.0, 1000.0, default_gluc, 0.1)

        height_cm = ni_int("Height (cm)", 50, 300, default_height_cm, 1, key="risk_height")
        weight_kg = ni_float("Weight (kg)", 10.0, 300.0, default_weight_kg, 1.0, key="risk_weight")

        height_m = height_cm / 100.0
        bmi_value = float(weight_kg) / (height_m ** 2) if height_m > 0 else float(default_bmi_guess)

        smoking = st.selectbox(
            "Smoking Status", smoke_opts,
            index=smoke_opts.index(default_smoke) if default_smoke in smoke_opts else 1
        )

    submitted = st.form_submit_button("Predict")

# ---------- Build EXACT column frame the model expects ----------
def build_model_input_encoded(expected_cols,
    age, hypertension_disp, heart_disease_disp, sex,
    married_disp, work_type, residence_type, glucose, bmi, smoking
):
    """
    Build a single-row DataFrame with exactly `expected_cols` the model was trained on.
    Any missing columns are filled with 0; relevant dummies set based on current inputs.
    """
    if not expected_cols:
        # Fallback to raw; but your model expects encoded, so we keep the builder strict.
        raise RuntimeError("Model's expected column list is unavailable. Cannot encode features.")

    row = {col: 0.0 for col in expected_cols}

    # numerics / binaries
    if "Age" in row:           row["Age"] = float(age)
    if "Hypertension" in row:  row["Hypertension"] = 1.0 if hypertension_disp == "Yes" else 0.0
    if "Heart Disease" in row: row["Heart Disease"] = 1.0 if heart_disease_disp == "Yes" else 0.0
    if "Married" in row:       row["Married"] = 1.0 if married_disp == "Yes" else 0.0
    if "Glucose" in row:       row["Glucose"] = float(glucose)
    if "BMI" in row:           row["BMI"] = float(bmi)

    # Sex dummies (Female baseline)
    if "Sex_Male" in row:      row["Sex_Male"]  = 1.0 if sex == "Male"  else 0.0
    if "Sex_Other" in row:     row["Sex_Other"] = 1.0 if sex == "Other" else 0.0

    # Work Type dummies
    if "Work Type_Private" in row:       row["Work Type_Private"]       = 1.0 if work_type == "Private" else 0.0
    if "Work Type_Self-employed" in row: row["Work Type_Self-employed"] = 1.0 if work_type == "Self-employed" else 0.0
    if "Work Type_children" in row:      row["Work Type_children"]      = 1.0 if work_type == "children" else 0.0
    if "Work Type_Never_worked" in row:  row["Work Type_Never_worked"]  = 1.0 if work_type == "Never_worked" else 0.0
    if "Work Type_Govt_job" in row:      row["Work Type_Govt_job"]      = 1.0 if work_type == "Govt_job" else 0.0

    # Residence dummies
    if "Residence Type_Urban" in row:    row["Residence Type_Urban"] = 1.0 if residence_type == "Urban" else 0.0
    if "Residence Type_Rural" in row:    row["Residence Type_Rural"] = 1.0 if residence_type == "Rural" else 0.0

    # Smoking dummies (values match your dataset labels)
    if "Smoking?_formerly smoked" in row:  row["Smoking?_formerly smoked"] = 1.0 if smoking == "formerly smoked" else 0.0
    if "Smoking?_never smoked" in row:     row["Smoking?_never smoked"]    = 1.0 if smoking == "never smoked" else 0.0
    if "Smoking?_smokes" in row:           row["Smoking?_smokes"]          = 1.0 if smoking == "smokes" else 0.0
    if "Smoking?_Unknown" in row:          row["Smoking?_Unknown"]         = 1.0 if smoking == "Unknown" else 0.0

    # return with column order EXACTLY as expected
    return pd.DataFrame([row], columns=expected_cols)

# ---------- Gauge helpers ----------
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
        gauge={{
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
        }},
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

# ---------- Predict ----------
if submitted:
    try:
        X_user = build_model_input_encoded(
            EXPECTED_COLS,
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
    except Exception as e:
        st.error("Failed to prepare features for the model.")
        st.exception(e)
        st.stop()

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
