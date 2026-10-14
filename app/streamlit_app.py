"""VoiceSense website entry. Start here.

This file does not score a voice. It draws the frame and opens one screen:

  app/live_page.py       microphone or audio upload  (default)
  app/feature_page.py    spreadsheet lab
  app/privacy_ui.py      privacy and optional enrollment
  the About block below  how the method is limited

The engine map is the docstring in src/__init__.py.
A readable walkthrough of every click is docs/VoiceSense-app-flows.pdf.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.feature_page import feature_page
from app.live_page import live_page
from app.notices import render_flash
from app.privacy_ui import (
    _session_enrollment,
    render_privacy_banner,
    render_privacy_enrollment_tab,
)
from app.styles import PREMIUM_CSS
from src.config import DISCLAIMER
from src.privacy import wipe_audio_keys

st.set_page_config(
    page_title="VoiceSense · PD Screening",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

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
_enroll = _session_enrollment()
if _enroll and _enroll.get("identity_fingerprint"):
    fp = str(_enroll["identity_fingerprint"])
    mock = " (mock)" if _enroll.get("mock") else ""
    st.caption(f"Study enrollment{mock} · fingerprint `{fp[:12]}…` · voice is not linked in this badge.")

PAGES = (
    "Live voice studio",
    "Feature lab",
    "Privacy & enrollment",
    "About & method",
)
page = st.segmented_control(
    "Section",
    PAGES,
    default="Live voice studio",
    key="active_page",
    required=True,
    label_visibility="collapsed",
    width="stretch",
)
if page not in PAGES:
    page = "Live voice studio"

if page == "Live voice studio":
    live_page()
elif page == "Feature lab":
    feature_page()
elif page == "Privacy & enrollment":
    render_privacy_enrollment_tab()
elif page == "About & method":
    st.markdown(
        """
        <div class="vs-card">
          <h3>Method highlights</h3>
          <p class="vs-muted">
            English UCI &amp; Bengali BenSParX feature models · Live acoustic model trained with the
            same Praat/librosa extractor used at inference · subject/speaker-aware validation.
            A live call is shown only when the clip is at least 4 seconds, clean enough to trust
            pitch and loudness, and the random-forest and SVM checks agree. Three separate vowels
            are read together. One home-microphone clip is not treated as a final answer.
          </p>
          <h3>About the IDM popup</h3>
          <p class="vs-muted">
            Internet Download Manager sometimes intercepts browser media streams (Error 0x80080005).
            That is <b>not</b> a model crash. Leave audio preview off, or exclude localhost from IDM.
            Analysis uses in-memory WAV conversion and does not require a file download.
          </p>
          <h3>Privacy</h3>
          <p class="vs-muted">
            Live screening is designed to run on your machine. Successful analysis
            discards the recording. Optional Primus enrollment proves a Web2 account
            or age condition and stores only a fingerprint under
            <code>~/.voicesense/</code> — never the WAV.
          </p>
          <h3>Where the code lives</h3>
          <p class="vs-muted">
            Screens: <code>app/live_page.py</code>, <code>app/feature_page.py</code>, <code>app/privacy_ui.py</code>.
            Engine map: <code>src/__init__.py</code>.
            Click-by-click guide: <code>docs/VoiceSense-app-flows.pdf</code>.
          </p>
          <h3>Disclaimer</h3>
          <p class="vs-muted">Research screening support only. Not a clinical diagnostic device.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="vs-footer">
      VoiceSense · BAUET thesis research prototype · Explainable voice screening · Not for clinical diagnosis
    </div>
    """,
    unsafe_allow_html=True,
)
