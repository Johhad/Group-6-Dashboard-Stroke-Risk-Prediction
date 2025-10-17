# utils/ui_safety.py
import streamlit as st
import matplotlib.pyplot as plt
import gc

def begin_page(title: str | None = None, layout: str = "wide"):
    try:
        plt.close('all')
        gc.collect()
    except Exception:
        pass

    try:
        st.set_page_config(layout=layout)
    except Exception:
        pass

    prev_root_key = "_active_root_container"
    if prev_root_key in st.session_state and st.session_state[prev_root_key] is not None:
        try:
            st.session_state[prev_root_key].empty()
        except Exception:
            pass

    root = st.container()
    st.session_state[prev_root_key] = root

    if title:
        with root:
            st.title(title)

    return root