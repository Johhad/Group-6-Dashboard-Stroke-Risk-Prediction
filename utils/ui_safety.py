# utils/ui_safety.py
import matplotlib
matplotlib.use("Agg")  # non-interactive backend prevents GUI bleed
import matplotlib.pyplot as plt
import streamlit as st
import gc

def begin_page(title: str | None = None):
    """
    Minimal per-page safety:
    - Switch to non-interactive MPL backend (Agg) (module-level)
    - Close all MPL figures
    - Force GC to drop figure handles
    - Optionally emit a title
    """
    try:
        plt.close('all')
        gc.collect()
    except Exception:
        pass

    if title:
        st.title(title)