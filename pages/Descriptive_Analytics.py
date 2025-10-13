import streamlit as st
import plotly.express as px

st.title("Descriptive Analytics 📊")
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
        st.plotly_chart(fig_age, use_container_width=True)

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
        st.plotly_chart(fig_gender, use_container_width=True)

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
            st.markdown("**Age Distribution by Stroke Status** — Overlaid histograms showing age distribution for patients with and without stroke.")

            fig_box = px.box(df, y='Age', x='Stroke', 
                    title='Age Distribution: Stroke vs No Stroke',
                    labels={'Age': 'Age (years)', 'Stroke': 'Had Stroke'},
                    color='Stroke',
                    color_discrete_map={0: '#3498db', 1: '#e74c3c'})
            fig_box.update_xaxes(ticktext=['No Stroke', 'Stroke'], tickvals=[0, 1])
            fig_box.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

# -----------------------
# Risk Factors
# -----------------------
st.subheader("Risk Factors")
col1, col2 = st.columns(2)

# ---- Stroke distribution (with description + analysis) ----
with col1:
    with st.container():
        # Top description row
        st.markdown("**Percentages of Stroke and No-stroke** — Proportion of patients with recorded stroke outcome (1 = stroke, 0 = no stroke).")

        # Chart
        stroke_counts = df['Stroke'].value_counts().sort_index()  # expect 0,1
        names = ['No Stroke', 'Stroke'] if set(stroke_counts.index) <= {0, 1} else stroke_counts.index
        fig_stroke = px.pie(values=stroke_counts.values,
                            names=names,
                            title='Stroke Distribution',
                            color_discrete_sequence=['#2ca02c', '#d62728'])
        fig_stroke.update_layout(height=400)
        st.plotly_chart(fig_stroke, use_container_width=True)

        # Bottom analysis row (auto)
        stroke_pct = (df['Stroke'] == 1).mean() * 100
        st.markdown(f"**Quick read:** Stroke rate **{stroke_pct:.1f}%**; no-stroke **{100 - stroke_pct:.1f}%**. The dataset that the dashboard trained has imbalance dataset with only 4.8% of stroke patients. This has certain implications on machine learning model training stages.")

# ---- Heart disease vs stroke (with description + analysis) ----
with col2:
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
        st.plotly_chart(fig_heart, use_container_width=True)

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

