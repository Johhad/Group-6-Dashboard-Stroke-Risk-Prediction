import streamlit as st

from forms.contact import contact_form

#---define forms for contact---
@st.dialog("Contact Us")
def show_contact_form():
    contact_form()


#--Neuro predict Team---
col1, col2 = st.columns(2, gap="small", vertical_alignment="center")
with col1: 
    st.image("./assets/SU_Logo.png", width=400)
with col2:
    st. title("Neuro Predict Team", anchor =False)
    st.write(" We are a team of healthcare professionals currently studying a Masters in Health Informatics " \
    "dedicated to improve stroke prediction and patient outcomes. Our mission is to harness the power of data to provide actionable insights "
    "for healthcare professionals, ultimately enhancing patient care and reducing the incidence of stroke and the related complications.")
    if st.button('✉️Contact us'):
        show_contact_form()