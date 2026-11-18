"""On-device privacy banner and a receipt that contains no recording."""
from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

from src.privacy import is_local_runtime, make_screening_receipt, receipt_json


def render_privacy_banner() -> bool:
    local, reason = is_local_runtime()
    if local:
        st.markdown(
            f"""
            <div class="vs-privacy ok">
              <strong>On-device processing.</strong>
              Voice is scored in this Python process on your machine
              (<code>{reason}</code>). After a successful analysis the recording is
              dropped from memory.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="vs-privacy warn">
              <strong>Hosted runtime ({reason}).</strong>
              The browser must send audio to this Streamlit server to score it.
              That is <em>not</em> on-device privacy. Run locally
              (<code>python -m streamlit run app/streamlit_app.py</code>)
              if the recording must never leave your computer.
            </div>
            """,
            unsafe_allow_html=True,
        )
    return local


def render_export_receipt(result: Dict[str, Any], source_kind: Optional[str]) -> None:
    if not result or result.get("status") != "ok":
        return
    local, _ = is_local_runtime()
    st.markdown('<div class="vs-section">Export (no audio)</div>', unsafe_allow_html=True)
    include_features = st.checkbox(
        "Include top acoustic feature preview in the JSON receipt",
        value=False,
        key="receipt_features",
        help="Still no WAV. Uncheck unless you want to share feature numbers.",
    )
    receipt = make_screening_receipt(
        result,
        source_kind=source_kind,
        include_features=include_features,
        local_runtime=local,
    )
    payload = receipt_json(receipt)
    st.download_button(
        "Download screening receipt (JSON, no recording)",
        data=payload.encode("utf-8"),
        file_name="voicesense-screening-receipt.json",
        mime="application/json",
        key="dl_receipt",
        use_container_width=True,
    )
    with st.expander("Receipt preview"):
        st.code(payload, language="json")


def render_privacy_tab() -> None:
    local, reason = is_local_runtime()
    where = "this computer" if local else f"the host ({reason})"
    st.markdown(
        f"""
        <div class="vs-card">
          <h3>What happens to the recording</h3>
          <p class="vs-muted">
            Scoring runs in the Python process on {where}.
            After a successful analysis the microphone and upload widgets are cleared.
            A downloaded receipt has the decision and, if you ask for it, a few feature numbers.
            It does not contain the WAV or the upload file name.
          </p>
          <p class="vs-muted">
            Identity proofs are not part of this repository.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
