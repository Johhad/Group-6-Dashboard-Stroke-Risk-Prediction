# risk_prediction.py  (Streamlit page)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle, json
from pathlib import Path

st.set_page_config(page_title="Risk Prediction", page_icon="🧑‍⚕️", layout="wide")

st.title("Risk Prediction 🧑‍⚕️")
st.caption("This page predicts stroke risk using the trained SVM model based on the input features below.")

# -----------------------------
# Data loader (for sensible UI defaults)
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("./jupyter-notebooks/processed_data.csv")
    return df

df = load_data()

def _median(col, fallback):
    return float(df[col].median()) if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) else fallback

def _mode(col, fallback):
    if col in df.columns and df[col].dropna().size:
        return df[col].mode().iloc[0]
    return fallback

default_age   = int(round(_median('Age', 50)))
default_gluc  = round(_median('Glucose', 100.0), 1)
default_bmi   = round(_median('BMI', 25.0), 1)
default_hyp   = _mode('Hypertension', 1)      # 0/1 common
default_hd    = _mode('Heart Disease', 0)     # 0/1
default_work  = _mode('Work Type', 'Private')
default_res   = _mode('Residence Type', 'Urban')
default_smoke = _mode('Smoking Status', 'never smoked')

yes_no_display = ['No', 'Yes']
work_type_opts = ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']
res_type_opts  = ['Urban', 'Rural']
smoke_opts     = ['formerly smoked', 'never smoked', 'smokes', 'Unknown']

# NEW: Sex options (Female is baseline when both dummies are 0)
sex_opts = ['Female', 'Male', 'Other']
default_sex = 'Female'

def _yesno_from01(val01):
    return 'Yes' if str(val01) == '1' else 'No'

# -----------------------------
# Load trained model + threshold
# -----------------------------
@st.cache_resource
def load_model_and_meta():
    model_path = Path("assets/trained_model_final.pickle")
    thr_path   = Path("assets/decision_threshold.json")

    if not model_path.exists():
        st.error(f"Model file not found at: {model_path.resolve()}")
        return None, 0.5, []

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load threshold if exists, else default 0.5
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

    st.success(f"Model loaded from {model_path.name}")
    return model, thr, expected_cols

model, DECISION_THR, EXPECTED_COLS = load_model_and_meta()
if model is None:
    st.stop()

# -----------------------------
# Form
# -----------------------------
with st.form("patient_form"):
    st.markdown("### Enter Patient Data")

    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", min_value=1, max_value=120, value=default_age, step=1)
        hypertension_disp = st.radio(
            "Hypertension", yes_no_display, horizontal=True,
            index=yes_no_display.index(_yesno_from01(default_hyp))
        )
        heart_disease_disp = st.radio(
            "Heart Disease", yes_no_display, horizontal=True,
            index=yes_no_display.index(_yesno_from01(default_hd))
        )
        # NEW: Sex input
        sex = st.selectbox("Sex", sex_opts, index=sex_opts.index(default_sex))
    with c2:
        work_type = st.selectbox(
            "Work Type", work_type_opts,
            index=work_type_opts.index(default_work) if default_work in work_type_opts else 0
        )
        residence_type = st.selectbox(
            "Residence Type", res_type_opts,
            index=res_type_opts.index(default_res) if default_res in res_type_opts else 0
        )
        glucose = st.number_input("Glucose", min_value=0.0, value=default_gluc, step=0.1)
        bmi = st.number_input("BMI", min_value=0.0, value=default_bmi, step=0.1)
        smoking = st.selectbox(
            "Smoking?", smoke_opts,
            index=smoke_opts.index(default_smoke) if default_smoke in smoke_opts else 1
        )

    submitted = st.form_submit_button("Predict")

# -----------------------------
# Helper: map form -> model input
# -----------------------------
def build_model_input(expected_cols,
                      age, hypertension_disp, heart_disease_disp, sex,
                      work_type, residence_type, glucose, bmi, smoking):
    """
    Build a single-row DataFrame matching the model training columns.
    Unknown one-hot columns default to 0.0 (Female is baseline for Sex).
    """
    row = {col: 0.0 for col in expected_cols}

    # numeric/binary
    if "Age" in row:           row["Age"] = float(age)
    if "Hypertension" in row:  row["Hypertension"] = 1.0 if hypertension_disp == "Yes" else 0.0
    if "Heart Disease" in row: row["Heart Disease"] = 1.0 if heart_disease_disp == "Yes" else 0.0
    if "Glucose" in row:       row["Glucose"] = float(glucose)
    if "BMI" in row:           row["BMI"] = float(bmi)

    # Sex dummies (Female = both 0)
    if "Sex_Male" in row:      row["Sex_Male"]  = 1.0 if sex == "Male"  else 0.0
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

# -----------------------------
# Gauge helpers
# -----------------------------
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

# -----------------------------
# Predict on submit
# -----------------------------
if submitted:
    # 1) build model-ready row (now includes sex)
    X_user = build_model_input(
        EXPECTED_COLS,
        age=age,
        hypertension_disp=hypertension_disp,
        heart_disease_disp=heart_disease_disp,
        sex=sex,  # NEW
        work_type=work_type,
        residence_type=residence_type,
        glucose=glucose,
        bmi=bmi,
        smoking=smoking
    )

    # 2) predict probability and class (using saved threshold)
    try:
        prob = float(model.predict_proba(X_user)[0][1])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    pred = int(prob >= DECISION_THR)

    # 3) render gauge (prob * 100)
    render_risk_gauge(prob * 100.0, title="Model-Estimated Stroke Risk", decision_thr=DECISION_THR)

    # 4) show numeric outputs
    st.markdown(f"**Predicted probability:** `{prob:.3f}`")
    st.markdown(f"**Decision (threshold = {DECISION_THR:.3f}):** {'**Stroke risk (1)**' if pred==1 else '**Low stroke risk (0)**'}")

    # 5) stash inputs/outputs for other pages
    st.session_state['rp_input'] = {
        'Age': age,
        'Sex': sex,  # NEW
        'Hypertension': 1 if hypertension_disp == 'Yes' else 0,
        'Heart Disease': 1 if heart_disease_disp == 'Yes' else 0,
        'Work Type': work_type,
        'Residence Type': residence_type,
        'Glucose': float(glucose),
        'BMI': float(bmi),
        'Smoking Status': smoking,
        'pred_proba': prob,
        'predicted': pred
    }

    st.info("Prediction generated by the trained SVM (MinMax-scaled, class-weighted). "
            "The gauge reflects the predicted probability (0–100).")
else:
    st.info("Fill the form and click **Predict** to see the model-estimated risk.")