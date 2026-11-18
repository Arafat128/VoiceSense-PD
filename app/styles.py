"""Load app/styles.css into the page."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

_CSS_PATH = Path(__file__).with_name("styles.css")


def inject_styles() -> None:
    css = _CSS_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>\n{css}\n</style>", unsafe_allow_html=True)
