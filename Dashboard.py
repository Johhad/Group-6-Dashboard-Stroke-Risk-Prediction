#Dashboard.py
from pathlib import Path
import time
import streamlit as st

st.set_page_config(
    page_title="NeuroInsight Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar logo (make sure this file exists)
st.sidebar.image("./assets/logo_NeuroInsight.png")

# ---------- Helpers ----------
BASE = Path(__file__).parent  # folder containing Dashboard.py

def make_page(rel_path: str, **kwargs) -> st.Page:
    """Create a Streamlit Page from a path relative to this file, with a clear error if missing."""
    p = BASE / rel_path
    if not p.exists():
        # Raising here prevents silent fallback to legacy alphabetical nav.
        raise FileNotFoundError(f"Streamlit Page file not found: {p}")
    return st.Page(page=str(p), **kwargs)

# ---------- Define pages (match EXACT filenames in /pages) ----------
about_page        = make_page("pages/About.py",                 title="About",                 icon=":material/info:", default=True)
patient_pred_page = make_page("pages/Risk_prediction.py",       title="Risk Prediction",       icon="🧑‍⚕️")
preventive_page   = make_page("pages/Preventive_Analytics.py",  title="Preventive Analytics",  icon="🛡️")
desc_page         = make_page("pages/Descriptive_Analytics.py", title="Descriptive Analytics", icon="📊")
diag_page         = make_page("pages/Diagnostic_Analytics.py",  title="Diagnostic Analytics",  icon="🩺")

# ---------- Navigation (custom groups + order) ----------
nav = st.navigation(
    {
        "Info":    [about_page],
        "Patient": [patient_pred_page, preventive_page],
        "Project": [desc_page, diag_page],
    }
)

# Stores last page title and triggers a one-time rerun on change.
try:
    current_title = nav._pages[nav._selected_page_index].title
except Exception:
    current_title = getattr(nav, "active_page", None) or "Unknown"

last_title = st.session_state.get("last_page_title")
if last_title is not None and last_title != current_title:
    st.session_state["last_page_title"] = current_title
    st.experimental_set_query_params(_ts=str(time.time()))
    st.rerun()
else:
    st.session_state["last_page_title"] = current_title

# ---------- Run selected page ----------
nav.run()