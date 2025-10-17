# utils/ui_safety.py

import streamlit as st
import matplotlib.pyplot as plt
import gc

def page_safety(title: str | None = None, layout: str = "wide"):
    """
    Global page-safety initializer.
    - Closes all active Matplotlib figures (prevents 'bleed').
    - Forces garbage collection to release memory between pages.
    - Sets consistent Streamlit layout.
    - Optionally sets page title.
    """
    try:
        plt.close('all')           # Close any lingering Matplotlib plots
        gc.collect()               # Clean any cached figure objects
    except Exception:
        pass

    try:
        st.set_page_config(layout=layout)
    except Exception:
        # Streamlit only allows set_page_config once per app
        pass

    if title:
        st.title(title)