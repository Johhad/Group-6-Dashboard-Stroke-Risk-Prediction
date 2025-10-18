#Dashboard.py
import streamlit as st
import time

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

# Force refresh whenever navigating to a new page because bleeds were happening and other options did not work
import streamlit as st
import time

# Identify which page is currently selected (based on Page.title)
try:
    current_page = pg._pages[pg._selected_page_index].title
except Exception:
    current_page = getattr(pg, "active_page", None) or "Unknown"

# Store the last visited page
last_page = st.session_state.get("last_page")

# If a different page is now selected → full refresh
if last_page is not None and last_page != current_page:
    st.session_state["last_page"] = current_page
    st.experimental_set_query_params(force_refresh=str(time.time()))
    st.rerun()
else:
    st.session_state["last_page"] = current_page
# =====================================================================

# ---- Run ----
pg.run()

