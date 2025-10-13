import streamlit as st

st.set_page_config(
    page_title="PROHI Dashboard",
    page_icon="👋",
)

# Sidebar configuration
st.sidebar.image("./assets/logo_NeuroInsight.png",)

# Page configuration
st.set_page_config(
    page_title="NeuroInsight Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page configuration
st.set_page_config(
    page_title="NeuroInsight Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


#--- Page setup---

about_page = st.Page(page= "pages/About.py",title= "About", icon= ":material/info:",default=True,)
patient_data_page= st.Page(page="pages/Patient_Data.py",title="Patient Data",icon="🧑‍⚕️")
project_1_page = st.Page(page= "pages/Descriptive_Analytics.py", title ="Descriptive Analytics", icon = "📊")
project_2_page = st.Page(page= "pages/Diagnostic_Analytics.py", title= "Diagnostic Analytics", icon= "🩺")
patient_data_page1 = st.Page(page= "pages/Risk_prediction.py", title= "Risk Prediction", icon = "🧑‍⚕️")
patient_data_page2 = st. Page(page= "pages/Preventive_Analytics.py", title= "Preventive Analytics", icon = "🛡️")

#--Navigation setup [sections]--
pg = st.navigation(
    {
        "Info": [about_page],
        "Patient": [patient_data_page1, patient_data_page2],
        "Project": [project_1_page, project_2_page]
        
    }
)
#---Run navigation---
pg.run()

#---Color of sidebar
st.markdown(
    """
    <style>
    /* Sidebar background color */
    [data-testid="stSidebar"] {
        background-color: #0e2a47;
        color: #ffffff; /* set the text color to white */
    }
    /* Ensure all sidebar text elements are white */
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    /* Optional: if sidebar contains header text, make sure it's white */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6 {
        color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True
)
