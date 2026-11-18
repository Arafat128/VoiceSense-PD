"""About and method screen. It does not score a voice."""
from __future__ import annotations

import streamlit as st


def render_about() -> None:
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
            discards the recording. The receipt download does not include the WAV.
            Identity proofs are not part of this repository.
          </p>
          <h3>Where the code lives</h3>
          <p class="vs-muted">
            Screens: <code>app/pages/live.py</code>, <code>app/pages/feature.py</code>,
            <code>app/pages/privacy.py</code>.
            Scoring: <code>src/predict.py</code>.
            Model loading: <code>src/models/registry.py</code>.
          </p>
          <h3>Disclaimer</h3>
          <p class="vs-muted">Research screening support only. Not a clinical diagnostic device.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
