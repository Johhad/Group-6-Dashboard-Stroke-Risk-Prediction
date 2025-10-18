import streamlit as st

st.set_page_config(
    page_title="NeuroInsight Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar configuration
st.sidebar.image("./assets/logo_NeuroInsight.png",)

#--- Page setup---

about_page        = st.Page(page="pages/About.py",                 title="About",                icon=":material/info:", default=True)
patient_pred_page = st.Page(page="pages/Risk_prediction.py",       title="Risk Prediction",     icon="🧑‍⚕️")
preventive_page   = st.Page(page="pages/Preventive_Analytics.py",  title="Preventive Analytics",icon="🛡️")
desc_page         = st.Page(page="pages/Descriptive_Analytics.py", title="Descriptive Analytics",icon="📊")
diag_page         = st.Page(page="pages/Diagnostic_Analytics.py",  title="Diagnostic Analytics", icon="🩺")

# ---- Navigation ----
pg = st.navigation(
    {
        "Info":    [about_page],
        "Patient": [patient_pred_page, preventive_page],
        "Project": [desc_page, diag_page],
    }
)
# ---- Run ----
pg.run()

