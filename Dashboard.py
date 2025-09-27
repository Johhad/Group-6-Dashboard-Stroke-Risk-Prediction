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

# Custom CSS for styling
st.markdown("""""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #ffffff;
        text-align: center;
        padding: 1rem;
        background-color: #0e2a47;
        border: 1px solid #cccccc;
        border-radius: 8px;
        margin-bottom: 2rem;
        width : 100%;
    }
    .risk-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 2rem auto;
    }
    .high-risk {
        background-color: #ffebee;
        color: #c62828;
        border: 3px solid #c62828;
    }
    .medium-risk {
        background-color: #fff3e0;
        color: #ef6c00;
        border: 3px solid #ef6c00;
    }
    .low-risk {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 3px solid #2e7d32;
    }
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dddddd;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# # Page information


#--- Page setup---

about_page = st.Page(page= "pages/About.py",title= "About", icon= ":material/info:",default=True,)
patient_data_page= st.Page(page="pages/Patient_Data.py",title="Patient Data",icon="🧑‍⚕️")
project_1_page = st.Page(page= "pages/Descriptive_Analytics.py", title ="Descriptive Analytics", icon = "📊")
project_2_page = st.Page(page= "pages/Diagnostic_Analytics.py", title= "Diagnostic Analytics", icon = "🩺")
project_3_page = st.Page(page= "pages/Risk_prediction.py", title= "Rish Prediction Analytics", icon = "🤖")
project_4_page = st.Page(page= "pages/Preventive_Analytics.py", title= "What-If/Preventive Analytics", icon = "🛡️")

#--Navigation setup [sections]--
pg = st.navigation(
    {
        "Info": [about_page],
        "Patient": [patient_data_page],
        "Project": [project_1_page, project_2_page, project_3_page, project_4_page],
        
    }
)
#---Run navigation---
pg.run()

