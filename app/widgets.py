"""Shared cards used by the live studio and the feature lab.

Open this file when you want to change how a result looks, not how it is calculated.
The calculation lives under src/.
"""
from __future__ import annotations

import html
import io
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

# Project folder: VoiceSense-PD/
ROOT = Path(__file__).resolve().parents[1]


def _live_audio_available() -> bool:
    if (ROOT / "models" / "live_audio_default.joblib").exists():
        return True
    return any((ROOT / "models").glob("live_audio_italian__*.joblib"))


def _to_wav_bytes(raw: bytes) -> bytes:
    """
    Normalize any browser/upload audio to WAV bytes for analysis.
    Avoids download-manager interception of odd media containers on analyze.
    """
    try:
        import librosa
        import soundfile as sf
        import numpy as np
        import tempfile

        # Detect container roughly
        suffix = ".wav"
        if raw[:4] == b"OggS":
            suffix = ".ogg"
        elif raw[:4] == b"\x1aE\xdf\xa3":
            suffix = ".webm"
        elif len(raw) > 12 and raw[4:8] == b"ftyp":
            suffix = ".mp4"
        elif raw[:3] == b"ID3" or raw[:2] == b"\xff\xfb":
            suffix = ".mp3"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            path = tmp.name
        try:
            y, sr = librosa.load(path, sr=22050, mono=True)
        finally:
            Path(path).unlink(missing_ok=True)

        # Peak normalize lightly for stable feature extraction
        peak = float(np.max(np.abs(y)) + 1e-9)
        y = (0.95 * y / peak).astype("float32")
        buf = io.BytesIO()
        sf.write(buf, y, sr, format="WAV")
        return buf.getvalue()
    except Exception:
        # Fall back to original bytes if conversion fails
        return raw


def _render_result_banner(primary: Dict[str, Any]) -> None:
    code = primary.get("decision_code") or (
        "PD" if primary.get("pred_label") == 1 else "HC" if primary.get("pred_label") == 0 else "UNCERTAIN"
    )
    css = "pd" if code == "PD" else "hc" if code == "HC" else "unc"
    st.markdown(
        f"""
        <div class="vs-result {css}">
          <div class="label">Primary screening decision</div>
          <div class="value">{primary.get('prediction', code)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_explanation(exp: dict, title: str = "Clinical-style explanation") -> None:
    st.markdown(f'<div class="vs-section">{title}</div>', unsafe_allow_html=True)
    with st.container():
        for line in exp.get("summary_lines", []):
            st.markdown(f"- {line}")

        left, right = st.columns(2)
        with left:
            st.markdown("#### Signals toward PD")
            items = exp.get("why_pd") or []
            if not items:
                st.caption("No strong PD-favoring cues in the top features.")
            for item in items[:8]:
                with st.expander(f"{item['feature']} · score {item.get('local_score', 0):.3f}"):
                    st.write(item.get("comparison", ""))
                    st.caption(item.get("meaning", ""))
        with right:
            st.markdown("#### Signals toward healthy")
            items = exp.get("why_not_pd") or []
            if not items:
                st.caption("No strong healthy-favoring cues in the top features.")
            for item in items[:8]:
                with st.expander(f"{item['feature']} · score {item.get('local_score', 0):.3f}"):
                    st.write(item.get("comparison", ""))
                    st.caption(item.get("meaning", ""))

        top = exp.get("top_influential_features") or []
        if top:
            st.markdown("#### Top influential features")
            st.dataframe(pd.DataFrame(top), use_container_width=True, hide_index=True)


def _esc(value: Any) -> str:
    return html.escape(str(value or ""))


def _render_everyday(plain: Dict[str, Any]) -> None:
    """Non-technical reading. Technical metrics stay underneath, unchanged."""
    heard = "".join(f"<li>{_esc(line)}</li>" for line in (plain.get("heard") or []))
    steps = "".join(f"<li>{_esc(line)}</li>" for line in (plain.get("next_steps") or []))
    series = plain.get("series") or {}
    series_html = ""
    if series:
        series_html = (
            f"<h4>Your set of recordings</h4>"
            f"<p class='vs-muted'><b>{_esc(series.get('headline'))}</b> {_esc(series.get('detail'))}</p>"
        )
    st.markdown(
        f"""
        <div class="vs-card vs-plain">
          <h4>In everyday language</h4>
          <h3>{_esc(plain.get("headline"))}</h3>
          <p class="vs-muted">{_esc(plain.get("meaning"))}</p>
          <p class="vs-muted">{_esc(plain.get("chance"))}</p>
          <h4>What the voice showed</h4>
          <ul>{heard}</ul>
          <h4>How much to trust this</h4>
          <div class="vs-trust">{_esc(plain.get("trust_label"))}</div>
          <p class="vs-muted">{_esc(plain.get("trust_detail"))}</p>
          {series_html}
          <h4>What you can do</h4>
          <ul>{steps}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
