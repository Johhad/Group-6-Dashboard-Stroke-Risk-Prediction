# utils/ui_safety.py
import matplotlib.pyplot as plt
import streamlit as st

def page_safety(title: str | None = None, layout: str = "wide"):
    """
    Safe per-page setup:
    - Closes any Matplotlib figures from other pages (prevents 'bleed').
    - Sets consistent layout for all pages.
    - Optionally prints a page title.
    """
    # 1️⃣ Close any leftover Matplotlib plots
    plt.close('all')

    # 2️⃣ Ensure layout consistency
    try:
        st.set_page_config(layout=layout)
    except Exception:
        # Ignore if already set by another page
        pass

    # 3️⃣ Optional title
    if title:
        st.title(title)