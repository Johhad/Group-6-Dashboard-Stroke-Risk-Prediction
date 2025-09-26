import streamlit as st

@st.cache_data
def load_data():
    import pandas as pd
    df = pd.read_csv('./jupyter-notebooks/processed_data.csv')
    return df

df = load_data()
with st.form("my_form"):
    sex = st.text_input("Enter your gender")
    age = st.number_input("Enter your age", min_value=1)
    blood_pressure: st.number_input("Enter systolic blood pressure") # type: ignore
    cholesterol= st.number_input("Enter cholesterol in mmol/L")
    blood_sugar = st.number_input("Add fasting blood sugar")
        
    submitted = st.form_submit_button("Submit")