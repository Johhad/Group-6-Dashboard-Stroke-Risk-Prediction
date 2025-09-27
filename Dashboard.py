import streamlit as st

st.set_page_config(
    page_title="PROHI Dashboard",
    page_icon="👋",
)
st.write("# Welcome to Stroke prediction Dashboard ")

# Sidebar configuration
st.sidebar.image("./assets/NeuroPredict.png",)
st.sidebar.success("Select a tab above.")

# Page configuration
st.set_page_config(
    page_title="NeuroPredict Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page configuration
st.set_page_config(
    page_title="NeuroPredict Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


#--- Page setup---

about_page = st.Page(page= "pages/About.py",title= "About", icon= ":material/info:",default=True,)
patient_data_page= st.Page(page="pages/Patient_Data.py",title="Patient Data",icon="🧑‍⚕️")
project_1_page = st.Page(page= "pages/Descriptive_Analytics.py", title ="Descriptive Analytics", icon = "📊")
project_2_page = st.Page(page= "pages/Diagnostic_Analytics.py", title= "Diagnostic Analytics", icon= "🩺")
project_3_page = st. Page(page= "pages/Preventive_Analytics.py", title= "Preventive Analytics", icon = "🛡️")
patient_data_page = st.Page(page= "pages/Risk_prediction.py", title= "Risk Prediction", icon = "🧑‍⚕️")

#--Navigation setup [sections]--
pg = st.navigation(
    {
        "Info": [about_page],
        "Patient": [patient_data_page],
        "Project": [project_1_page, project_2_page, project_3_page],
        
    }
)
#---Run navigation---
pg.run()

