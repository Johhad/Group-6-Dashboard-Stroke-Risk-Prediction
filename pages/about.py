import streamlit as st

from forms.contact import contact_form

#---define forms for contact---
@st.dialog("Contact Us")
def show_contact_form():
    contact_form()

st.markdown("""
<style>
.page-header-box {
    background-color: #0e2a47;   /* dark blue background */
    padding: 1.2rem;
    border-radius: 10px;          /* rounded corners */
    border: 1px solid #cccccc;   /* subtle border */
    margin-bottom: 1.5rem;
    text-align: center;          /* center the text */
}

.page-header-text {
    font-size: 2rem;             /* bigger font size */
    font-weight: bold;           /* bold text */
    color: #ffffff;                /* white text color */
    line-height: 1.2;            /* improve line height */
    margin: 0;                   /* remove default margins */
}
</style>
""", unsafe_allow_html=True)

# About Page

st.markdown(
    '<div class="page-header-box"><div class="page-header-text">About NeuroPredict</div></div>',
    unsafe_allow_html=True
)
    
col1, col2 = st.columns([2, 1])
    
with col1:
        st.image("./assets/SU_Logo.png", width=400)
        st.markdown("""
        ### Project Overview
        
        NeuroPredict is an advanced stroke risk prediction dashboard developed to assist 
        healthcare professionals and individuals in assessing and understanding stroke risk factors.
        
        ### Technology Stack
        - **Frontend:** Streamlit
        - **Backend:** Python
        - **ML Framework:** Scikit-learn
        - **Visualization:** Plotly
        - **Deployment:** Streamlit Cloud
        
        ### Model Information
        - **Algorithm:** Random Forest Classifier
        - **Features:** 10 risk factors
        - **Training Data:** 5,000 patient records
        - **Validation Accuracy:** ~92%
        - **Cross-validation:** 5-fold
        
        ### Key Features
        - Real-time risk prediction
        - Interactive data visualization
        - What-if scenario analysis
        - Prevention recommendations
        - Comprehensive risk factor analysis
        
        ### Data Privacy
        All patient data entered into this dashboard is processed locally and is not stored 
        or transmitted to any external servers. We prioritize patient privacy and data security.
        """)
    
with col2:
        st.markdown("""
        ### Team Members
        - Data Scientists
        - Healthcare Professionals
        - Software Engineers
        - UX/UI Designers
        
        ### Contact
        📧 Email: info@neuropredict.com
        🌐 Website: www.neuropredict.com
        📱 Support: +1-XXX-XXX-XXXX
        
        ### Version
        **Current Version:** 1.0.0
        **Last Updated:** 2024
        
        ### Disclaimer
        This tool is for educational and screening purposes only. It should not replace 
        professional medical advice, diagnosis, or treatment.
        """)

    if st.button('✉️Contact us'):
        show_contact_form()  
st.info("""
    ### Future Enhancements
    - Integration with Electronic Health Records (EHR)
    - Mobile application development
    - Multi-language support
    - Advanced deep learning models
    - Real-time monitoring capabilities
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    © 2024 NeuroPredict - Group 6 Dashboard | Stroke Risk Prediction System
</div>
""", unsafe_allow_html=True)
