import streamlit as st

st.set_page_config(
    page_title="PROHI Dashboard",
    page_icon="👋",
)

# Sidebar configuration
st.sidebar.image("./assets/NeuroPredict.png",)
st.sidebar.success("Select a tab above.")

# # Page information

st.write("# Welcome to Stroke prediction Dashboard ")

#--- Page setup---

about_page = st.Page(page= "pages/About.py",title= "About", icon= ":material/info:",default=True,)
project_1_page = st.Page(page= "pages/Descriptive_Analytics.py", title ="Descriptive Analytics", icon = "📊")
project_2_page = st.Page(page= "pages/Diagnostic_Analytics.py", title ="Diagnostic Analytics", icon = "🩺")
project_3_page = st.Page(page= "pages/Predictive_Analytics.py", title= "Predictive Analytics", icon = "📈")
project_4_page = st.Page(page= "pages/What-If/Preventive_Analytics.py", title= "What-If/Preventive Analytics", icon = "🛡️")

#--Navigation setup [sections]--
pg = st.navigation(
    {
        "Info": [about_page],
        "Project": [project_1_page, project_2_page],
    }
)
#---Run navigation---
pg.run()


# You can also add text right into the web as long comments (""")
"""
The final project aims to apply data science concepts and skills on a 
medical case study that you and your team select from a public data source.
The project assumes that you bring the technical Python skills from 
previous courses (*DSHI*: Data Science for Health Informatics), as well as 
the analytical skills to argue how and why specific techniques could
enhance the problem domain related to the selected dataset.
"""

# DATAFRAME MANAGEMENT
import numpy as np

dataframe = np.random.randn(10, 20)
st.dataframe(dataframe)

# Add a selectbox to the sidebar:
add_selectbox = st.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone')
)

# Add a slider to the sidebar:
add_slider = st.slider(
    'Select a range of values',
    0.0, 100.0, (25.0, 75.0)
)
