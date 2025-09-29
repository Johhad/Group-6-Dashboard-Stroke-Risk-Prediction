# pages/Preventive.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.title("Preventive What-If 🛡️")
st.caption("Tweak modifiable factors and see how the estimated stroke risk changes. Use this to explore prevention strategies. (Placeholder score; replace with our final model later.)")

@st.cache_data
def load_data():
    return pd.read_csv('./jupyter-notebooks/processed_data.csv')

df = load_data()

# ---------- helpers ----------
def _median(col, fallback):
    return float(df[col].median()) if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) else fallback

def _mode(col, fallback):
    if col in df.columns and df[col].dropna().size:
        return df[col].mode().iloc[0]
    return fallback

yes_no = ['No', 'Yes']
work_type_opts = ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']
res_type_opts  = ['Urban', 'Rural']
smoke_opts     = ['formerly smoked', 'never smoked', 'smokes', 'Unknown']

def compute_heuristic_score(a, hyp, hd, work, res, glu, b, smoke):
    """
    Placeholder scoring (0-100). Swap with your trained model later.
    Returns (score, dict of contributions).
    """
    contrib = {}
    base = 5.0
    contrib['Base'] = base
    contrib['Age'] = np.clip((a - 40) / 40, 0, 1) * 25
    contrib['Hypertension'] = 20.0 if hyp == 'Yes' else 0.0
    contrib['Heart Disease'] = 25.0 if hd == 'Yes' else 0.0
    # assume mmol/L threshold; adjust if mg/dL in your dataset
    glu_threshold = 7.0
    contrib['Glucose'] = np.clip((glu - glu_threshold) / glu_threshold, 0, 1) * 15
    contrib['BMI'] = np.clip((b - 25) / 10, 0, 1) * 10
    contrib['Smoking'] = 10.0 if smoke == 'smokes' else (5.0 if smoke == 'formerly smoked' else 0.0)
    contrib['Residence Type'] = 1.0 if res == 'Urban' else 0.0
    contrib['Work Type'] = 1.0 if work in ['Private', 'Self-employed', 'Govt_job'] else 0.0
    score = float(np.clip(sum(contrib.values()), 0, 100))
    return score, contrib

# ---------- baseline (from Risk page if available) ----------
if 'rp_input' in st.session_state:
    b = st.session_state['rp_input']
    baseline = {
        'Age': int(b.get('Age', _median('Age', 50))),
        'Hypertension': 'Yes' if b.get('Hypertension', 0) == 1 else 'No',
        'Heart Disease': 'Yes' if b.get('Heart Disease', 0) == 1 else 'No',
        'Work Type': b.get('Work Type', 'Private'),
        'Residence Type': b.get('Residence Type', 'Urban'),
        'Glucose': float(b.get('Glucose', _median('Glucose', 100.0))),
        'BMI': float(b.get('BMI', _median('BMI', 25.0))),
        'Smoking Status': b.get('Smoking Status', 'never smoked'),
    }
else:
    baseline = {
        'Age': int(round(_median('Age', 50))),
        'Hypertension': 'Yes' if _mode('Hypertension', 0) == 1 else 'No',
        'Heart Disease': 'Yes' if _mode('Heart Disease', 0) == 1 else 'No',
        'Work Type': _mode('Work Type', 'Private'),
        'Residence Type': _mode('Residence Type', 'Urban'),
        'Glucose': round(_median('Glucose', 100.0), 1),
        'BMI': round(_median('BMI', 25.0), 1),
        'Smoking Status': _mode('Smoking Status', 'never smoked'),
    }

# Compute baseline score
baseline_score, baseline_contrib = compute_heuristic_score(
    a=baseline['Age'],
    hyp=baseline['Hypertension'],
    hd=baseline['Heart Disease'],
    work=baseline['Work Type'],
    res=baseline['Residence Type'],
    glu=baseline['Glucose'],
    b=baseline['BMI'],
    smoke=baseline['Smoking Status']
)

st.markdown("#### Baseline (from the patient data from Risk Prediciton page)")
bcols = st.columns(4)
bcols[0].metric("Age", baseline['Age'])
bcols[1].metric("Glucose", f"{baseline['Glucose']:.1f}")
bcols[2].metric("BMI", f"{baseline['BMI']:.1f}")
bcols[3].metric("Est. Risk", f"{baseline_score:.0f}/100")

st.divider()

# ---------- interactive what-if controls ----------
st.subheader("Adjust factors (what-if)")

c1, c2 = st.columns(2)
with c1:
    age = st.slider("Age", min_value=1, max_value=120, value=int(baseline['Age']), step=1)
    hypertension = st.radio("Hypertension", yes_no, horizontal=True, index=yes_no.index(baseline['Hypertension']))
    heart_disease = st.radio("Heart Disease", yes_no, horizontal=True, index=yes_no.index(baseline['Heart Disease']))
with c2:
    work_type = st.selectbox("Work Type", work_type_opts, index=work_type_opts.index(baseline['Work Type']) if baseline['Work Type'] in work_type_opts else 0)
    residence_type = st.selectbox("Residence Type", res_type_opts, index=res_type_opts.index(baseline['Residence Type']) if baseline['Residence Type'] in res_type_opts else 0)
    glucose = st.number_input("Glucose (mmol/L or mg/dL – match your data)", min_value=0.0, step=0.1, value=float(baseline['Glucose']))
    bmi = st.number_input("BMI", min_value=0.0, step=0.1, value=float(baseline['BMI']))
    smoking = st.selectbox("Smoking?", smoke_opts, index=smoke_opts.index(baseline['Smoking Status']) if baseline['Smoking Status'] in smoke_opts else 1)

# ---------- current score ----------
score, contrib = compute_heuristic_score(
    a=age, hyp=hypertension, hd=heart_disease,
    work=work_type, res=residence_type, glu=glucose, b=bmi, smoke=smoking
)

risk_band = "Low" if score < 33 else ("Moderate" if score < 66 else "High")
band_color = "#2ca02c" if risk_band == "Low" else ("#ff7f0e" if risk_band == "Moderate" else "#d62728")

st.markdown(
    f"<div style='display:inline-block; padding:8px 14px; border-radius:999px; "
    f"background:{band_color}1A; color:{band_color}; font-weight:600;'>"
    f"{risk_band} Risk • {score:.0f}/100</div>",
    unsafe_allow_html=True
)

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=score,
    number={'suffix': "/100"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': band_color},
        'steps': [
            {'range': [0, 33], 'color': '#e8f5e9'},
            {'range': [33, 66], 'color': '#fff3e0'},
            {'range': [66, 100], 'color': '#ffebee'}
        ],
        'threshold': {'line': {'color': band_color, 'width': 4}, 'thickness': 0.75, 'value': score}
    },
    title={'text': "Estimated Risk Score"}
))
fig_gauge.update_layout(height=260, margin=dict(t=30, b=10, l=10, r=10))
st.plotly_chart(fig_gauge, use_container_width=True)

# Contributions bar
st.markdown("#### What drives this score?")
contrib_df = (
    pd.DataFrame([{'Factor': k, 'Impact': v} for k, v in contrib.items()])
    .sort_values('Impact', ascending=True)
)
contrib_df['Direction'] = np.where(contrib_df['Impact'] >= 0, 'Increases risk', 'Decreases risk')
fig_imp = px.bar(
    contrib_df,
    x='Impact', y='Factor', orientation='h',
    color='Direction',
    title='Factor Contributions (heuristic)',
    labels={'Impact': 'Points', 'Factor': 'Feature'},
    color_discrete_map={'Increases risk': '#d62728', 'Decreases risk': '#2ca02c'}
)
fig_imp.update_layout(height=420, margin=dict(t=50, b=10, l=10, r=10))
st.plotly_chart(fig_imp, use_container_width=True)