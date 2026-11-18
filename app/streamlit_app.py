"""VoiceSense website entry. Start here.

This file does not score a voice. It draws the frame and opens one screen:

  app/pages/live.py       microphone or audio upload  (default)
  app/pages/feature.py    spreadsheet lab
  app/pages/privacy.py    on-device notice and receipt
  app/pages/about.py      how the method is limited
  app/assistant_ui.py     project-only chat bubble

The engine map is the docstring in src/__init__.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.assistant_ui import render_assistant
from app.pages.about import render_about
from app.pages.feature import feature_page
from app.pages.live import live_page
from app.pages.privacy import render_privacy_banner, render_privacy_tab
from app.styles import inject_styles
from app.widgets import render_flash
from src.config import DISCLAIMER
from src.privacy import wipe_audio_keys

st.set_page_config(
    page_title="VoiceSense · PD Screening",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def render() -> None:
    inject_styles()

    wipe_audio_keys(st.session_state)
    if "audio_nonce" not in st.session_state:
        st.session_state["audio_nonce"] = 0

    st.markdown(
        """
        <div class="vs-hero">
          <div class="vs-kicker"><span></span> VoiceSense Research Platform</div>
          <h1 class="vs-title">Voice-based <em>Parkinson’s</em> screening</h1>
          <p class="vs-sub">
            Premium research interface for English &amp; Bangla voice analysis.
            Live microphone or feature-table input · explainable ML · screening support only.
          </p>
          <div class="vs-badges">
            <div class="vs-badge">On-device by default</div>
            <div class="vs-badge">Subject-aware models</div>
            <div class="vs-badge">Clinical acoustic features</div>
            <div class="vs-badge">Conservative PD threshold</div>
            <div class="vs-badge">Not a medical diagnosis</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(DISCLAIMER)
    render_flash()
    render_privacy_banner()

    pages = (
        "Live voice studio",
        "Feature lab",
        "Privacy",
        "About & method",
    )
    page = st.segmented_control(
        "Section",
        pages,
        default="Live voice studio",
        key="active_page",
        required=True,
        label_visibility="collapsed",
        width="stretch",
    )
    if page not in pages:
        page = "Live voice studio"

    if page == "Live voice studio":
        live_page()
    elif page == "Feature lab":
        feature_page()
    elif page == "Privacy":
        render_privacy_tab()
    elif page == "About & method":
        render_about()

    st.markdown(
        """
        <div class="vs-footer">
          VoiceSense · BAUET thesis research prototype · Explainable voice screening · Not for clinical diagnosis
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_assistant()


# app/pages/ holds imported screens. Calling navigation tells Streamlit not to
# turn those modules into a second sidebar menu.
st.navigation(
    [st.Page(render, title="VoiceSense", default=True)],
    position="hidden",
).run()
