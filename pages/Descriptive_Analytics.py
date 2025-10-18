#Descriptive page

import streamlit as st

PAGE_ID = "descriptive-page"
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

begin_page("Descriptive Analytics 📊") 

# Clearning up the unnecessaery data
if 'rp_input' in st.session_state:
    del st.session_state['rp_input']


import numpy as np
import plotly.express as px
import seaborn as sns
import re
import gc

#st.title("Descriptive Analytics 📊")
st.caption("This page shows key summary descriptive analysis of the dataset that the project dashboard utilized and trained on")

#🔗 Link: <https://plotly.com/python/scientific-charts/>

@st.cache_data
def load_data():
    import pandas as pd
    df = pd.read_csv('./jupyter-notebooks/processed_data.csv')
    return df

df = load_data()

# --- Key summary values ---
total_patients = len(df)
stroke_rate = (df['Stroke'] == 1).mean() * 100
mean_age = df['Age'].mean()
heart_disease_rate = (df['Heart Disease'] == 1).mean() * 100

# --- KPI big boxes at the top ---
col1, col2, col3, col4 = st.columns(4)
col1.markdown(f"<div style='text-align:center; background-color:#f0f2f6; padding:20px; border-radius:12px;'><h2>{total_patients}</h2><p>Total Patients</p></div>", unsafe_allow_html=True)
col2.markdown(f"<div style='text-align:center; background-color:#f0f2f6; padding:20px; border-radius:12px;'><h2>{stroke_rate:.1f}%</h2><p>Stroke Rate</p></div>", unsafe_allow_html=True)
col3.markdown(f"<div style='text-align:center; background-color:#f0f2f6; padding:20px; border-radius:12px;'><h2>{mean_age:.1f}</h2><p>Mean Age</p></div>", unsafe_allow_html=True)
col4.markdown(f"<div style='text-align:center; background-color:#f0f2f6; padding:20px; border-radius:12px;'><h2>{heart_disease_rate:.1f}%</h2><p>Heart Disease Rate</p></div>", unsafe_allow_html=True)



# -----------------------
# Demographics
# -----------------------
st.subheader("Demographics")
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# ---- Age distribution (with description + analysis) ----
with col1:
    with st.container():
        # Top description row
        st.markdown("**Age Distribution** — The histogram shows how patient ages are distributed across the dataset.")

        # Chart
        fig_age = px.histogram(df, x='Age', nbins=20, title='Age Distribution',
                            color_discrete_sequence=['#1f77b4'])
        fig_age.update_layout(height=400)
        st.plotly_chart(fig_age, use_container_width=True, key="desc_age_hist")
        gc.collect()

        # Bottom analysis row (auto)
        age_mean = df['Age'].mean()
        age_median = df['Age'].median()
        age_std = df['Age'].std()
        age_skew = df['Age'].skew()
        skew_txt = "right-skewed (more younger patients)" if age_skew > 0.3 else ("left-skewed (more older patients)" if age_skew < -0.3 else "roughly symmetric")
        st.markdown(
            f"**Quick read:** Mean age **{age_mean:.1f}**, median **{age_median:.1f}**, SD **{age_std:.1f}**; distribution is **{skew_txt}**."
        )

# ---- Gender distribution (with description + analysis) ----
with col2:
    with st.container():
        # Top description row
        st.markdown("**Gender Distribution** — Pie chart summarizes sex composition of the cohort.")

        # Chart
        gender_counts = df['Sex'].value_counts(dropna=False)
        fig_gender = px.pie(values=gender_counts.values, names=gender_counts.index,
                            title='Gender Distribution',
                            color_discrete_map={'Male': '#1f77b4', 'Female': '#ff7f0e'})
        fig_gender.update_layout(height=400)
        st.plotly_chart(fig_gender, use_container_width=True, key='gen_dis')
        gc.collect()

        # Bottom analysis row (auto)
        gender_pct = (gender_counts / gender_counts.sum() * 100).round(1).to_dict()
        male_pct = gender_pct.get('Male', 0.0)
        female_pct = gender_pct.get('Female', 0.0)
        st.markdown(
            f"**Quick read:** Female **{female_pct:.1f}%**, Male **{male_pct:.1f}%** of the sample."
        )

#---#----Age distribution by stroke status----
with col3:
    with st.container():
        # Top description row
        st.markdown("**Age Distribution by Stroke Status** — Overlaid bloxplot showing age distribution for patients with and without stroke.")
        # Chart
        fig_box = px.box(df, y='Age', x='Stroke', 
                title='Age Distribution: Stroke vs No Stroke',
                labels={'Age': 'Age (years)', 'Stroke': 'Had Stroke'},
                color='Stroke',
                color_discrete_map={0: '#3498db', 1: '#e74c3c'})
        fig_box.update_xaxes(ticktext=['No Stroke', 'Stroke'], tickvals=[0, 1])
        fig_box.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True,key='age_dis_str')
        gc.collect()
        # Bottom analysis row (auto)
        age_stroke_mean = df[df['Stroke'] == 1]['Age'].mean()
        age_nostroke_mean = df[df['Stroke'] == 0]['Age'].mean()
        st.markdown(
            f"**Quick read:** Mean age for **Stroke** patients: **{age_stroke_mean:.1f}** years; for **No Stroke**: **{age_nostroke_mean:.1f}** years."
        )        
# -----------------------
# Risk Factors
# -----------------------
st.subheader("Risk Factors")
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)
col5, col6 = st.columns(2)

# ---- Heart disease vs stroke (with description + analysis) ----
with col1:
    with st.container():
        # Top description row
        st.markdown("**Heart Disease and Stroke** — Counts of patients by **Heart Disease** status and **Stroke** outcome.")

        # Chart
        grp = df.groupby(['Heart Disease', 'Stroke']).size().reset_index(name='count')
        fig_heart = px.bar(
            grp,
            x='Heart Disease',
            y='count',
            color='Stroke',
            barmode='group',
            title='Heart Disease and Stroke',
            labels={'Heart Disease': 'Heart Disease', 'count': 'Count', 'Stroke': 'Stroke'}
        )
        fig_heart.update_layout(height=400, xaxis=dict(tickmode='array', tickvals=[0,1], ticktext=['No','Yes']))
        st.plotly_chart(fig_heart, use_container_width=True,key='ht_str')
        gc.collect()

        # Bottom analysis row (auto)
        # Stroke risk by heart disease status
        try:
            risk_hd = df[df['Heart Disease'] == 1]['Stroke'].mean() * 100
            risk_nohd = df[df['Heart Disease'] == 0]['Stroke'].mean() * 100
            rr = (risk_hd / max(risk_nohd, 1e-9)) if risk_nohd > 0 else np.nan
            rr_txt = f"{rr:.2f}× higher" if np.isfinite(rr) and rr >= 1 else ("similar" if np.isfinite(rr) else "—")
            st.markdown(
                f"**Quick read:** Stroke among **Heart Disease = Yes**: **{risk_hd:.1f}%**; **No**: **{risk_nohd:.1f}%** → **{rr_txt}** relative risk."
            )
        except Exception:
            st.markdown("**Quick read:** The graph shows the status of stroke in patients with no heart diseases and heart diseases. In terms of the graph, the proportion is high in patients with heart diseases.")
# -------Stroke rate by smoking status----
with col2:
    with st.container():
        # Top description row
        st.markdown("**Stroke Rate by Smoking Status** — Stroke rates for different smoking status categories.")

        # Chart
        grp_smoke = df.groupby('Smoking?')['Stroke'].mean().reset_index()
        grp_smoke['Stroke Rate (%)'] = grp_smoke['Stroke'] * 100
        fig_smoke = px.bar(
            grp_smoke,
            x='Smoking?',
            y='Stroke Rate (%)',
            title='Stroke Rate by Smoking Status',
            labels={'Smoking Status': 'Smoking Status', 'Stroke Rate (%)': 'Stroke Rate (%)'},
            color_discrete_sequence=["#43a6df"]
        )
        fig_smoke.update_layout(height=400)
        st.plotly_chart(fig_smoke, use_container_width=True,key='str_smk')
        gc.collect()

        # Bottom analysis row (auto)
        st.markdown(
            " **Quick read:** Current smokers have the highest stroke rate, followed by former smokers. Never smokers have the lowest stroke rate."
        )
#--- Percentage of patients with heart disease disaggregated by types of works and urban/rural residency
with col3:
    with st.container():
        # Top description row
        st.markdown("**Heart Disease by Work Type and Residence** — Percentage of patients with heart disease disaggregated by types of work and urban/rural residency.")

        # Chart
        grp_work = df.groupby(['Work Type', 'Residence Type'])['Heart Disease'].mean().reset_index()
        grp_work['Heart Disease Rate (%)'] = grp_work['Heart Disease'] * 100
        fig_work = px.bar(
            grp_work,
            x='Work Type',
            y='Heart Disease Rate (%)',
            color='Residence Type',
            barmode='group',
            title='Heart Disease by Work Type and Residence',
            labels={'Work Type': 'Work Type', 'Heart Disease Rate (%)': 'Heart Disease Rate (%)', 'Residence Type': 'Residence'},
            color_discrete_map={'Urban': "#2c5aa0", 'Rural': "#279fd6"}
        )
        fig_work.update_layout(height=400)
        st.plotly_chart(fig_work, use_container_width=True,key='hrt_wk_re')
        gc.collect()

        # Bottom analysis row (auto)
        st.markdown(
            " **Quick read:** The chart shows that the prevalence of heart disease varies by work type and residence, self employed showed higher rates."
        )

#-----Average glucose level of stroke and no-stroke patients
with col4:
    with st.container():
        # Top description row
        st.markdown("**Average Glucose Level by Stroke Status** — Average glucose levels for patients with and without stroke.")

        # Chart
        grp_glucose = df.groupby('Stroke')['Glucose'].mean().reset_index()
        fig_glucose = px.bar(
            grp_glucose,
            x='Stroke',
            y='Glucose',
            title='Average Glucose Level by Stroke Status',
            labels={'Stroke': 'Stroke Status', 'Glucose': 'Average Glucose Level (mg/dL)'},
            color_discrete_sequence=["#250feb"]
        )
        fig_glucose.update_xaxes(ticktext=['No Stroke', 'Stroke'], tickvals=[0, 1])
        fig_glucose.update_layout(height=400)
        st.plotly_chart(fig_glucose, use_container_width=True,key='glu_str')
        gc.collect()

        # Bottom analysis row (auto)
        glucose_stroke = grp_glucose[grp_glucose['Stroke'] == 1]['Glucose'].values[0]
        glucose_nostroke = grp_glucose[grp_glucose['Stroke'] == 0]['Glucose'].values[0]
        st.markdown(
            f"**Quick read:** Average glucose level for **Stroke** patients: **{glucose_stroke:.1f} mg/dL**; for **No Stroke**: **{glucose_nostroke:.1f} mg/dL**."
        )
#----Average BMI values vary between patients with hypertension and patients without hypertension
with col5:
    with st.container():
        # Top description row
        st.markdown("**Average BMI by Hypertension Status** — Average BMI values for patients with and without hypertension.")

        # Chart
        grp_bmi = df.groupby('Hypertension')['BMI'].mean().reset_index()
        fig_bmi = px.box(
            df,
            x='Hypertension',
            y='BMI',
            title='Average BMI by Hypertension Status',
            labels={'Hypertension': 'Hypertension Status', 'BMI': 'Body Mass Index (BMI)'},
            color_discrete_sequence=["#22aee6"]
        )
        fig_bmi.update_xaxes(ticktext=['No Hypertension', 'Hypertension'], tickvals=[0, 1])
        fig_bmi.update_layout(height=400)
        st.plotly_chart(fig_bmi, use_container_width=True,key='bmi_str')
        gc.collect()
        # Bottom analysis row (auto)
        bmi_hypertension = grp_bmi[grp_bmi['Hypertension'] == 1]['BMI'].values[0]
        bmi_nohypertension = grp_bmi[grp_bmi['Hypertension'] == 0]['BMI'].values[0]
        st.markdown(
            f"**Quick read:** Average BMI for **Hypertension** patients: **{bmi_hypertension:.1f}**; for **No Hypertension**: **{bmi_nohypertension:.1f}**."
        )

st.markdown("<div style='height:100vh;background-color:white;'></div>", unsafe_allow_html=True)