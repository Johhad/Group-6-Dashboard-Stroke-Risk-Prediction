import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.title("Risk Prediction 🧑‍⚕️")
st.caption("This page shows risk predication of stroke based on input features of the patient in the form")

# -----------------------------
# Data loader
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('./jupyter-notebooks/processed_data.csv')
    return df

df = load_data()

# -----------------------------
# Defaults from data
# -----------------------------
def _median(col, fallback):
    return float(df[col].median()) if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) else fallback

def _mode(col, fallback):
    if col in df.columns and df[col].dropna().size:
        return df[col].mode().iloc[0]
    return fallback

default_age   = int(round(_median('Age', 50)))
default_gluc  = round(_median('Glucose', 100.0), 1)
default_bmi   = round(_median('BMI', 25.0), 1)
default_hyp   = _mode('Hypertension', 0)      # 0/1 common
default_hd    = _mode('Heart Disease', 0)     # 0/1
default_mar   = _mode('Ever Married', 'No')   # 'Yes'/'No'
default_work  = _mode('Work Type', 'Private')
default_res   = _mode('Residence Type', 'Urban')
default_smoke = _mode('Smoking Status', 'never smoked')

yes_no_display = ['No', 'Yes']
work_type_opts = ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']
res_type_opts  = ['Urban', 'Rural']
smoke_opts     = ['formerly smoked', 'never smoked', 'smokes', 'Unknown']

def _yesno_from01(val01):
    return 'Yes' if str(val01) == '1' else 'No'

# -----------------------------
# Form
# -----------------------------
with st.form("my_form"):
    st.markdown("### Enter Patient Data")

    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", min_value=1, max_value=120, value=default_age, step=1)
        hypertension_disp = st.radio("Hypertension", yes_no_display, horizontal=True,
                                     index=yes_no_display.index(_yesno_from01(default_hyp)))
        heart_disease_disp = st.radio("Heart Disease", yes_no_display, horizontal=True,
                                      index=yes_no_display.index(_yesno_from01(default_hd)))
        married = st.radio("Married", yes_no_display, horizontal=True,
                           index=yes_no_display.index(default_mar if default_mar in yes_no_display else 'No'))
    with c2:
        work_type = st.selectbox("Work Type", work_type_opts,
                                 index=work_type_opts.index(default_work) if default_work in work_type_opts else 0)
        residence_type = st.selectbox("Residence Type", res_type_opts,
                                      index=res_type_opts.index(default_res) if default_res in res_type_opts else 0)
        glucose = st.number_input("Glucose", min_value=0.0, value=default_gluc, step=0.1)
        bmi = st.number_input("BMI", min_value=0.0, value=default_bmi, step=0.1)
        smoking = st.selectbox("Smoking?", smoke_opts,
                               index=smoke_opts.index(default_smoke) if default_smoke in smoke_opts else 1)

    submitted = st.form_submit_button("Submit")

# -----------------------------
# Heuristic risk score + viz
# -----------------------------
def compute_heuristic_score(a, hyp, hd, mar, work, res, glu, b, smoke):
    """
    Returns (score: 0-100, contributions: dict).
    Transparent placeholder logic you can replace with your model later.
    """
    contrib = {}

    # Base
    contrib['Base'] = 5.0

    # Age (0–25 pts): starts contributing >40 years
    contrib['Age'] = np.clip((a - 40) / 40, 0, 1) * 25

    # Hypertension (0 or +20)
    contrib['Hypertension'] = 20.0 if hyp == 'Yes' else 0.0

    # Heart disease (0 or +25)
    contrib['Heart Disease'] = 25.0 if hd == 'Yes' else 0.0

    # Glucose (0–15): starts > 7 mmol/L (adjust if your data uses mg/dL)
    glu_threshold = 7.0
    contrib['Glucose'] = np.clip((glu - glu_threshold) / glu_threshold, 0, 1) * 15

    # BMI (0–10): starts > 25
    contrib['BMI'] = np.clip((b - 25) / 10, 0, 1) * 10

    # Smoking (+10 current, +5 former)
    contrib['Smoking'] = 10.0 if smoke == 'smokes' else (5.0 if smoke == 'formerly smoked' else 0.0)

    # Married (protective -3 if Yes)
    contrib['Married'] = -3.0 if mar == 'Yes' else 0.0

    # Residence/work: small placeholder signals
    contrib['Residence Type'] = 1.0 if res == 'Urban' else 0.0
    contrib['Work Type'] = 1.0 if work in ['Private', 'Self-employed', 'Govt_job'] else 0.0

    score = float(np.clip(sum(contrib.values()), 0, 100))
    return score, contrib

# ---------- Gauge rendering (nice look) ----------
def band_and_color(score: float):
    if score < 33:
        return "Low", "#2ca02c"
    elif score < 66:
        return "Moderate", "#ff7f0e"
    return "High", "#d62728"

def render_risk_gauge(score: float, title="Estimated Risk Score"):
    band, color = band_and_color(score)

    # Title + badge
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

    # Semi-circular gauge
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
        text="Estimated Risk Score",
        x=0.5, y=0.9, showarrow=False,
        font=dict(size=14, color="#666")
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# On submit
# -----------------------------
if submitted:
    # compute score
    score, contributions = compute_heuristic_score(
        a=age,
        hyp=hypertension_disp,
        hd=heart_disease_disp,
        mar=married,
        work=work_type,
        res=residence_type,
        glu=glucose,
        b=bmi,
        smoke=smoking
    )

    # save inputs for Preventive page
    st.session_state['rp_input'] = {
        'Age': age,
        'Hypertension': 1 if hypertension_disp == 'Yes' else 0,
        'Heart Disease': 1 if heart_disease_disp == 'Yes' else 0,
        'Ever Married': married,
        'Work Type': work_type,
        'Residence Type': residence_type,
        'Glucose': float(glucose),
        'BMI': float(bmi),
        'Smoking Status': smoking
    }

    # render gauge
    render_risk_gauge(score)

       # disclaimer
    st.info(
        "This score is a temporary heuristic for UX only. It is not a diagnosis and should not be used for clinical decisions. "
        "When your ML model is ready, replace the scoring function so the gauge & contributions reflect the model’s output."
    )