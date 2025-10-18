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

# Clean up unused state from other pages
if 'rp_input' in st.session_state:
    del st.session_state['rp_input']

import numpy as np
import pandas as pd
import plotly.express as px
import gc

st.caption("This page shows key summary descriptive analysis of the dataset that the project dashboard utilized and trained on")

@st.cache_data
def load_data():
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
# Utilities
# -----------------------
def quick_read_box(text: str, accent="#1f77b4", bg="#f7fbff"):
    st.markdown(
        f"""
        <div style="
            background:{bg};
            padding:10px 12px;
            border-left:4px solid {accent};
            border-radius:8px;
            margin:8px 0 18px;">
            <b>Quick read:</b> {text}
        </div>
        """,
        unsafe_allow_html=True,
    )

def _is_categorical(series, max_unique_for_numeric_as_cat=6):
    return series.dtype.name in ["object", "category", "bool"] or series.nunique(dropna=True) <= max_unique_for_numeric_as_cat

@st.cache_data(show_spinner=False)
def _get_col_types(_df: pd.DataFrame):
    num_cols = [c for c in _df.columns if pd.api.types.is_numeric_dtype(_df[c])]
    cat_cols = sorted({c for c in _df.columns if _is_categorical(_df[c])})
    return num_cols, cat_cols

# -----------------------
# Demographics
# -----------------------
st.subheader("Demographics")
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# ---- Age distribution (with description + analysis) ----
with col1:
    with st.container():
        st.markdown("**Age Distribution** — The histogram shows how patient ages are distributed across the dataset.")
        fig_age = px.histogram(df, x='Age', nbins=20, title='Age Distribution',
                               color_discrete_sequence=['#1f77b4'])
        fig_age.update_layout(height=400)
        st.plotly_chart(fig_age, use_container_width=True, key="desc_age_hist")
        gc.collect()

        age_mean = df['Age'].mean()
        age_median = df['Age'].median()
        age_std = df['Age'].std()
        age_skew = df['Age'].skew()
        skew_txt = "right-skewed (more younger patients)" if age_skew > 0.3 else ("left-skewed (more older patients)" if age_skew < -0.3 else "roughly symmetric")
        quick_read_box(
            f"Mean age <b>{age_mean:.1f}</b>, median <b>{age_median:.1f}</b>, SD <b>{age_std:.1f}</b>; "
            f"distribution is <b>{skew_txt}</b>."
        )

# ---- Gender distribution (with description + analysis) ----
with col2:
    with st.container():
        st.markdown("**Gender Distribution** — Pie chart summarizes sex composition of the cohort.")
        gender_counts = df['Sex'].value_counts(dropna=False)
        fig_gender = px.pie(values=gender_counts.values, names=gender_counts.index,
                            title='Gender Distribution',
                            color_discrete_map={'Male': '#1f77b4', 'Female': '#ff7f0e'})
        fig_gender.update_layout(height=400)
        st.plotly_chart(fig_gender, use_container_width=True, key='gen_dis')
        gc.collect()

        gender_pct = (gender_counts / gender_counts.sum() * 100).round(1).to_dict()
        male_pct = gender_pct.get('Male', 0.0)
        female_pct = gender_pct.get('Female', 0.0)
        quick_read_box(f"Female <b>{female_pct:.1f}%</b>, Male <b>{male_pct:.1f}%</b> of the sample.")

# ---- Age distribution by stroke status ----
with col3:
    with st.container():
        st.markdown("**Age Distribution by Stroke Status** — Boxplot showing age distribution for patients with and without stroke.")
        fig_box = px.box(
            df, y='Age', x='Stroke',
            title='Age Distribution: Stroke vs No Stroke',
            labels={'Age': 'Age (years)', 'Stroke': 'Had Stroke'},
            color='Stroke',
            color_discrete_map={0: '#3498db', 1: '#e74c3c'}
        )
        fig_box.update_xaxes(ticktext=['No Stroke', 'Stroke'], tickvals=[0, 1])
        fig_box.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True, key='age_dis_str')
        gc.collect()

        age_stroke_mean = df[df['Stroke'] == 1]['Age'].mean()
        age_nostroke_mean = df[df['Stroke'] == 0]['Age'].mean()
        quick_read_box(
            f"Mean age for <b>Stroke</b> patients: <b>{age_stroke_mean:.1f}</b> years; "
            f"for <b>No Stroke</b>: <b>{age_nostroke_mean:.1f}</b> years."
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
        st.markdown("**Heart Disease and Stroke** — Counts of patients by **Heart Disease** status and **Stroke** outcome.")
        grp = df.groupby(['Heart Disease', 'Stroke']).size().reset_index(name='count')
        fig_heart = px.bar(
            grp, x='Heart Disease', y='count', color='Stroke', barmode='group',
            title='Heart Disease and Stroke',
            labels={'Heart Disease': 'Heart Disease', 'count': 'Count', 'Stroke': 'Stroke'}
        )
        fig_heart.update_layout(height=400, xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['No', 'Yes']))
        st.plotly_chart(fig_heart, use_container_width=True, key='ht_str')
        gc.collect()

        try:
            risk_hd = df[df['Heart Disease'] == 1]['Stroke'].mean() * 100
            risk_nohd = df[df['Heart Disease'] == 0]['Stroke'].mean() * 100
            rr = (risk_hd / max(risk_nohd, 1e-9)) if risk_nohd > 0 else np.nan
            rr_txt = f"{rr:.2f}× higher" if np.isfinite(rr) and rr >= 1 else ("similar" if np.isfinite(rr) else "—")
            quick_read_box(
                f"Stroke among <b>Heart Disease = Yes</b>: <b>{risk_hd:.1f}%</b>; "
                f"<b>No</b>: <b>{risk_nohd:.1f}%</b> → <b>{rr_txt}</b> relative risk."
            )
        except Exception:
            quick_read_box("Patients with heart disease show a higher stroke proportion.")

# ---- Stroke rate by smoking status ----
with col2:
    with st.container():
        st.markdown("**Stroke Rate by Smoking Status** — Stroke rates for different smoking status categories.")
        grp_smoke = df.groupby('Smoking?')['Stroke'].mean().reset_index()
        grp_smoke['Stroke Rate (%)'] = grp_smoke['Stroke'] * 100
        fig_smoke = px.bar(
            grp_smoke, x='Smoking?', y='Stroke Rate (%)',
            title='Stroke Rate by Smoking Status',
            labels={'Smoking?': 'Smoking Status', 'Stroke Rate (%)': 'Stroke Rate (%)'},
            color_discrete_sequence=["#43a6df"]
        )
        fig_smoke.update_layout(height=400)
        st.plotly_chart(fig_smoke, use_container_width=True, key='str_smk')
        gc.collect()

        quick_read_box("Current smokers have the highest stroke rate, followed by former smokers; never smokers have the lowest rate.")

# ---- Heart disease by work type & residence ----
with col3:
    with st.container():
        st.markdown("**Heart Disease by Work Type and Residence** — Percentage of patients with heart disease disaggregated by types of work and urban/rural residency.")
        grp_work = df.groupby(['Work Type', 'Residence Type'])['Heart Disease'].mean().reset_index()
        grp_work['Heart Disease Rate (%)'] = grp_work['Heart Disease'] * 100
        fig_work = px.bar(
            grp_work, x='Work Type', y='Heart Disease Rate (%)',
            color='Residence Type', barmode='group',
            title='Heart Disease by Work Type and Residence',
            labels={'Work Type': 'Work Type', 'Heart Disease Rate (%)': 'Heart Disease Rate (%)', 'Residence Type': 'Residence'},
            color_discrete_map={'Urban': "#2c5aa0", 'Rural': "#279fd6"}
        )
        fig_work.update_layout(height=400)
        st.plotly_chart(fig_work, use_container_width=True, key='hrt_wk_re')
        gc.collect()

        quick_read_box("Prevalence varies by work type and residence; self-employed groups tend to show higher rates.")

# ---- Average glucose by stroke status ----
with col4:
    with st.container():
        st.markdown("**Average Glucose Level by Stroke Status** — Average glucose levels for patients with and without stroke.")
        grp_glucose = df.groupby('Stroke')['Glucose'].mean().reset_index()
        fig_glucose = px.bar(
            grp_glucose, x='Stroke', y='Glucose',
            title='Average Glucose Level by Stroke Status',
            labels={'Stroke': 'Stroke Status', 'Glucose': 'Average Glucose Level (mg/dL)'},
            color_discrete_sequence=["#250feb"]
        )
        fig_glucose.update_xaxes(ticktext=['No Stroke', 'Stroke'], tickvals=[0, 1])
        fig_glucose.update_layout(height=400)
        st.plotly_chart(fig_glucose, use_container_width=True, key='glu_str')
        gc.collect()

        glucose_stroke = grp_glucose[grp_glucose['Stroke'] == 1]['Glucose'].values[0]
        glucose_nostroke = grp_glucose[grp_glucose['Stroke'] == 0]['Glucose'].values[0]
        quick_read_box(
            f"Average glucose — <b>Stroke</b>: <b>{glucose_stroke:.1f} mg/dL</b>; "
            f"<b>No Stroke</b>: <b>{glucose_nostroke:.1f} mg/dL</b>."
        )

# ---- BMI distribution by hypertension status ----
with col5:
    with st.container():
        st.markdown("**BMI Distribution by Hypertension Status** — BMI spread for patients with and without hypertension.")
        grp_bmi = df.groupby('Hypertension')['BMI'].mean().reset_index()
        fig_bmi = px.box(
            df, x='Hypertension', y='BMI',
            title='BMI Distribution by Hypertension Status',
            labels={'Hypertension': 'Hypertension Status', 'BMI': 'Body Mass Index (BMI)'},
            color_discrete_sequence=["#22aee6"]
        )
        fig_bmi.update_xaxes(ticktext=['No Hypertension', 'Hypertension'], tickvals=[0, 1])
        fig_bmi.update_layout(height=400)
        st.plotly_chart(fig_bmi, use_container_width=True, key='bmi_str')
        gc.collect()

        bmi_hypertension = grp_bmi[grp_bmi['Hypertension'] == 1]['BMI'].values[0]
        bmi_nohypertension = grp_bmi[grp_bmi['Hypertension'] == 0]['BMI'].values[0]
        quick_read_box(
            f"Average BMI — <b>Hypertension</b>: <b>{bmi_hypertension:.1f}</b>; "
            f"<b>No Hypertension</b>: <b>{bmi_nohypertension:.1f}</b>."
        )

# =======================
# Interactive Descriptive Explorer
# =======================
st.subheader("Interactive Descriptives")

num_cols, cat_cols = _get_col_types(df)

# -----------------------
# Interactive: Categorical Explorer (counts and %)
# -----------------------
with st.container():
    st.markdown("**Categorical Explorer** — Compare counts or percentages across a categorical feature, optionally grouped by another category.")

    c1, c2, c3, c4 = st.columns([1.6, 1.2, 1.2, 1.2])
    with c1:
        default_x = "Heart Disease" if "Heart Disease" in cat_cols else (cat_cols[0] if cat_cols else None)
        cat_x = st.selectbox("Categorical feature (X)", options=cat_cols, index=cat_cols.index(default_x) if default_x in cat_cols else 0, key="ie_cat_x")
    with c2:
        group_options = ["(none)"] + cat_cols
        default_group = "Stroke" if "Stroke" in cat_cols else "(none)"
        cat_color = st.selectbox("Group by (optional)", options=group_options, index=group_options.index(default_group), key="ie_cat_color")
        cat_color = None if cat_color == "(none)" else cat_color
    with c3:
        stat = st.radio("Show", options=["Count", "Row % within X"], index=0, horizontal=True, key="ie_cat_stat")
    with c4:
        sort_by = st.radio("Sort bars by", options=["Label", "Value"], index=1, horizontal=True, key="ie_cat_sort")

    if cat_x:
        if cat_color:
            ct = pd.crosstab(df[cat_x], df[cat_color], dropna=False)
            if stat == "Row % within X":
                plot_df = (ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0) * 100).reset_index().melt(id_vars=cat_x, var_name=cat_color, value_name="value")
                y_label = "Row %"
                hover_tmpl = "<b>%{x}</b><br>" + f"{cat_color}: " + "%{customdata[0]}<br>Row %: %{y:.2f}%<extra></extra>"
            else:
                plot_df = ct.reset_index().melt(id_vars=cat_x, var_name=cat_color, value_name="value")
                y_label = "Count"
                hover_tmpl = "<b>%{x}</b><br>" + f"{cat_color}: " + "%{customdata[0]}<br>Count: %{y}<extra></extra>"
            if sort_by == "Value":
                order_x = plot_df.groupby(cat_x)["value"].sum().sort_values(ascending=False).index.tolist()
            else:
                order_x = sorted(plot_df[cat_x].astype(str).unique().tolist())

            fig_cat = px.bar(
                plot_df, x=cat_x, y="value", color=cat_color, barmode="group",
                labels={cat_x: cat_x, "value": y_label},
            )
            fig_cat.update_traces(customdata=plot_df[[cat_color]].to_numpy(), hovertemplate=hover_tmpl)
            fig_cat.update_layout(height=420, xaxis={'categoryorder': 'array', 'categoryarray': order_x})
        else:
            ct = df[cat_x].value_counts(dropna=False).rename_axis(cat_x).reset_index(name="Count")
            if stat == "Row % within X":
                ct["value"] = (ct["Count"] / ct["Count"].sum() * 100).round(2)
                y_label = "Row %"
                hover_tmpl = "<b>%{x}</b><br>Row %: %{y:.2f}%<extra></extra>"
            else:
                ct["value"] = ct["Count"]
                y_label = "Count"
                hover_tmpl = "<b>%{x}</b><br>Count: %{y}<extra></extra>"

            if sort_by == "Value":
                order_x = ct.sort_values("value", ascending=False)[cat_x].astype(str).tolist()
            else:
                order_x = sorted(ct[cat_x].astype(str).tolist())

            fig_cat = px.bar(ct, x=cat_x, y="value", labels={cat_x: cat_x, "value": y_label})
            fig_cat.update_traces(hovertemplate=hover_tmpl)
            fig_cat.update_layout(height=420, xaxis={'categoryorder': 'array', 'categoryarray': order_x})

        st.plotly_chart(fig_cat, use_container_width=True, key="desc_interactive_cat")

        try:
            if cat_color:
                top_combo = plot_df.sort_values("value", ascending=False).iloc[0]
                quick_read_box(
                    f"Top category: <b>{cat_x} = {top_combo[cat_x]}</b> "
                    f"(group <b>{cat_color} = {top_combo[cat_color]}</b>) with "
                    f"<b>{top_combo['value']:.2f}{'%' if stat=='Row % within X' else ''}</b>."
                )
            else:
                top_cat = ct.sort_values("value", ascending=False).iloc[0]
                quick_read_box(
                    f"Top category: <b>{cat_x} = {top_cat[cat_x]}</b> with "
                    f"<b>{top_cat['value']:.2f}{'%' if stat=='Row % within X' else ''}</b>."
                )
        except Exception:
            pass


# Spacer to avoid bottom bleed
st.markdown("<div style='height:100vh;background-color:white;'></div>", unsafe_allow_html=True)