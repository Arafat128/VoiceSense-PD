"""Shared cards and notices used by the live studio and the feature lab.

Open this file when you want to change how a result looks, not how it is calculated.
The calculation lives under src/.
"""
from __future__ import annotations

import html
import io
import tempfile
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

Kind = Literal["success", "error", "warning", "info"]
FLASH_KEY = "vs_flash"


def set_flash(kind: Kind, text: str) -> None:
    st.session_state[FLASH_KEY] = {"kind": kind, "text": text}


def render_flash() -> None:
    flash = st.session_state.pop(FLASH_KEY, None)
    if not flash:
        return
    kind = flash.get("kind") or "info"
    text = str(flash.get("text") or "")
    if not text:
        return
    if kind == "success":
        st.success(text)
        st.toast(text, icon="✅")
    elif kind == "error":
        st.error(text)
        st.toast(text, icon="❌")
    elif kind == "warning":
        st.warning(text)
        st.toast(text, icon="⚠️")
    else:
        st.info(text)
        st.toast(text, icon="ℹ️")


def live_audio_available() -> bool:
    if (ROOT / "models" / "live_audio_default.joblib").exists():
        return True
    return any((ROOT / "models").glob("live_audio_italian__*.joblib"))


def to_wav_bytes(raw: bytes) -> bytes:
    """Normalize browser or upload audio to WAV bytes for analysis.

    A failed decode raises RuntimeError instead of handing the original
    container to the feature extractor.
    """
    if not raw:
        raise RuntimeError(
            "No audio bytes were captured. Record or upload a clip, then analyze again."
        )
    try:
        import librosa
        import numpy as np
        import soundfile as sf

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

        if y.size == 0:
            raise RuntimeError("the file decoded to silence")

        peak = float(np.max(np.abs(y)) + 1e-9)
        y = (0.95 * y / peak).astype("float32")
        buf = io.BytesIO()
        sf.write(buf, y, sr, format="WAV")
        return buf.getvalue()
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            "Could not convert this recording to WAV. "
            "Record a 5–8 second vowel, or upload a WAV, MP3, OGG, WEBM, M4A, or FLAC file. "
            f"Decoder said: {exc}"
        ) from exc


def render_result_banner(primary: dict[str, Any]) -> None:
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


def render_explanation(exp: dict, title: str = "Clinical-style explanation") -> None:
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
            st.dataframe(pd.DataFrame(top), width="stretch", hide_index=True)


def _esc(value: Any) -> str:
    return html.escape(str(value or ""))


def render_everyday(plain: dict[str, Any]) -> None:
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


def validation_caption(m: dict) -> str:
    """One line describing how the live model was validated."""
    if "person_auc" not in m:
        return (
            f"Train-time subject-aware: acc={m.get('accuracy', 0):.3f} · "
            f"F1={m.get('f1', 0):.3f} · AUC={m.get('roc_auc')}"
        )
    p_lo, p_hi = m.get("person_auc_ci95") or (0, 0)
    c_lo, c_hi = m.get("clip_auc_ci95") or (0, 0)
    return (
        f"Validation ({m.get('protocol', 'speaker-grouped CV')}): "
        f"per-person AUC {m['person_auc']:.2f} (95% CI {p_lo:.2f}–{p_hi:.2f}) · "
        f"per-clip AUC {m.get('clip_auc', 0):.2f} (95% CI {c_lo:.2f}–{c_hi:.2f}). "
        "Same-corpus estimate; other microphones and languages score lower."
    )
