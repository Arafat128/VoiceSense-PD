"""Streamlit UI for on-device privacy + optional Primus enrollment."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional

import streamlit as st

try:
    from app.notices import set_flash
except ImportError:
    from notices import set_flash  # type: ignore

from src.enrollment import (
    clear_enrollment,
    configured_for_live_primus,
    load_enrollment,
    local_recipient,
    merge_enrollment,
    mock_uniqueness_record,
    parse_attestation,
    primus_settings,
    public_enrollment_view,
    save_enrollment,
)
from src.privacy import is_local_runtime, make_screening_receipt, receipt_json


def _session_enrollment() -> Optional[Dict[str, Any]]:
    if "enrollment" in st.session_state:
        return st.session_state["enrollment"]
    stored = load_enrollment()
    if stored:
        st.session_state["enrollment"] = stored
    return stored


def _commit_enrollment(record: Dict[str, Any]) -> Dict[str, Any]:
    merged = merge_enrollment(_session_enrollment(), record)
    merged["enrolled"] = bool(merged.get("identity_fingerprint"))
    st.session_state["enrollment"] = save_enrollment(merged)
    return st.session_state["enrollment"]


def render_privacy_banner() -> bool:
    local, reason = is_local_runtime()
    if local:
        st.markdown(
            f"""
            <div class="vs-privacy ok">
              <strong>On-device processing.</strong>
              Voice is scored in this Python process on your machine
              (<code>{reason}</code>). After a successful analysis the recording is
              dropped from memory. Nothing is uploaded to Primus.
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
        enrollment=public_enrollment_view(_session_enrollment()),
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


def _apply_widget_result(value: Any, purpose: str, template_id: str) -> None:
    if not value:
        return
    marker = hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    seen_key = f"last_primus_payload_{purpose}"
    if st.session_state.get(seen_key) == marker:
        return
    st.session_state[seen_key] = marker
    if value.get("ok") is False and value.get("error"):
        set_flash("error", f"Primus proof failed: {value['error']}")
        st.rerun()
        return
    if not value.get("ok"):
        return
    try:
        record = parse_attestation(
            value.get("attestation"),
            expected_template_id=template_id or value.get("template_id"),
            source="primus-zktls",
            sdk_verified=bool(value.get("sdk_verified")),
            purpose=purpose,
        )
        saved = _commit_enrollment(record)
        st.session_state[f"primus_widget_nonce_{purpose}"] = (
            int(st.session_state.get(f"primus_widget_nonce_{purpose}", 0)) + 1
        )
        if saved.get("plaintext_stripped"):
            set_flash(
                "warning",
                "Enrollment saved, but identity fields were plaintext and got hashed. "
                "Set SHA256 on the Primus template.",
            )
        else:
            set_flash(
                "success",
                "Enrollment succeeded. Fingerprint stored only — no handle, no voice.",
            )
        st.rerun()
    except Exception as exc:
        set_flash("error", f"Enrollment failed: {exc}")
        st.rerun()


def render_privacy_enrollment_tab() -> None:
    try:
        from app.primus_enroll import available as enroll_widget_available
        from app.primus_enroll import primus_enroll, sdk_bundle_present
    except ImportError:
        from primus_enroll import available as enroll_widget_available  # type: ignore
        from primus_enroll import primus_enroll, sdk_bundle_present  # type: ignore

    local, reason = is_local_runtime()
    settings = primus_settings()
    enrolled = _session_enrollment()
    public = public_enrollment_view(enrolled)

    st.markdown('<div class="vs-section">How your data is handled</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="vs-card">
              <h3>Voice stays in screening</h3>
              <p class="vs-muted">
                Microphone and upload buffers live only in this Streamlit session.
                A successful Analyze discards the recording immediately.
                Receipts and enrollment files never include WAV bytes or filenames.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="vs-card">
              <h3>Runtime</h3>
              <p class="vs-muted">
                Mode: <b>{"local / on-device" if local else "hosted"}</b> ({reason}).<br/>
                Primus is optional and is used only to prove an account or age
                condition — it is not on the audio path.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="vs-section">Enrollment status</div>', unsafe_allow_html=True)
    if public and public.get("enrolled"):
        fp = public.get("identity_fingerprint") or ""
        age = public.get("age_ok")
        age_txt = "not proven" if age is None else ("yes" if age else "no")
        mock_txt = " · mock (not a real zkTLS proof)" if public.get("mock") else ""
        st.markdown(
            f"""
            <div class="vs-privacy ok">
              <strong>Enrolled{mock_txt}.</strong>
              Fingerprint <code>{fp[:16]}…{fp[-8:]}</code><br/>
              Age/eligibility: {age_txt} · source {public.get("source")} ·
              {public.get("verified_at") or ""}
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Clear enrollment from this device", type="secondary", key="clr_enroll"):
            clear_enrollment()
            st.session_state.pop("enrollment", None)
            set_flash("success", "Enrollment cleared from this device.")
            st.rerun()
    else:
        st.info("No enrollment on this device. Screening works without it.")

    st.markdown('<div class="vs-section">Optional Primus uniqueness proof</div>', unsafe_allow_html=True)
    st.markdown(
        """
        Prove you control a Web2 account (GitHub, X, …) via Primus zkTLS.
        VoiceSense stores a **SHA-256 fingerprint**, not your handle.
        Create a template in the [Developer Hub](https://dev.primuslabs.xyz/).
        If the extension popup does not open inside the widget, complete the
        proof in a normal tab and paste the attestation JSON below.
        """
    )

    if not configured_for_live_primus():
        st.warning(
            "Set `PRIMUS_APP_ID` and `PRIMUS_UNIQUENESS_TEMPLATE_ID` in `.env` "
            "to enable the live Primus button. You can still paste a verified "
            "attestation JSON, or use mock mode for UI testing."
        )
        st.code(
            "PRIMUS_APP_ID=...\n"
            "PRIMUS_APP_SECRET=...   # local test-mode signing only\n"
            "PRIMUS_UNIQUENESS_TEMPLATE_ID=...\n"
            "PRIMUS_IDENTITY_FIELD=screen_name",
            language="bash",
        )

    if not sdk_bundle_present():
        st.caption(
            "SDK bundle not built. From the repo: `cd enrollment && npm install && npm run build`."
        )

    recipient = local_recipient()
    st.caption(
        f"Primus recipient id for this device (random, not your wallet): `{recipient[:10]}…`"
    )

    if enroll_widget_available() and configured_for_live_primus():
        unique_val = primus_enroll(
            app_id=settings["app_id"],
            template_id=settings["uniqueness_template_id"],
            recipient=recipient,
            app_secret=settings["app_secret"],
            att_field=settings["identity_field"],
            att_op="SHA256",
            button_label="Prove uniqueness with Primus",
            purpose="uniqueness",
            key=f"primus_unique_{int(st.session_state.get('primus_widget_nonce_uniqueness', 0))}",
        )
        _apply_widget_result(unique_val, "uniqueness", settings["uniqueness_template_id"])

        if settings["age_template_id"]:
            st.markdown("##### Optional age / eligibility")
            age_val = primus_enroll(
                app_id=settings["app_id"],
                template_id=settings["age_template_id"],
                recipient=recipient,
                app_secret=settings["app_secret"],
                att_field=settings["age_field"],
                att_op=settings["age_op"],
                att_value=settings["age_value"],
                button_label=f"Prove {settings['age_field']} {settings['age_op']} {settings['age_value']}",
                purpose="age",
                key=f"primus_age_{int(st.session_state.get('primus_widget_nonce_age', 0))}",
            )
            _apply_widget_result(age_val, "age", settings["age_template_id"])
        else:
            st.caption(
                "Optional: set `PRIMUS_AGE_TEMPLATE_ID` for a boolean age proof "
                "(comparison stays in zkTLS; VoiceSense only stores true/false)."
            )

    st.markdown("##### Paste a Primus attestation JSON")
    pasted = st.text_area(
        "Paste attestation",
        height=140,
        key="paste_att",
        placeholder='{"recipient":"0x…","data":"{…}","signatures":["0x…"]}',
    )
    purpose_paste = st.radio(
        "This proof is for",
        options=["uniqueness", "age"],
        horizontal=True,
        key="paste_purpose",
    )
    if st.button("Import attestation", key="import_att"):
        if not pasted.strip():
            set_flash("error", "Import failed: paste attestation JSON first.")
            st.rerun()
        else:
            try:
                raw = json.loads(pasted)
                tmpl = (
                    settings["age_template_id"]
                    if purpose_paste == "age"
                    else settings["uniqueness_template_id"]
                )
                record = parse_attestation(
                    raw,
                    expected_template_id=tmpl or None,
                    source="paste",
                    sdk_verified=False,
                    purpose=purpose_paste,
                )
                _commit_enrollment(record)
                set_flash("success", "Import succeeded. Raw identity fields were not stored.")
                st.rerun()
            except Exception as exc:
                set_flash("error", f"Import failed: {exc}")
                st.rerun()

    if settings["mock"]:
        st.markdown("##### Developer mock")
        st.caption("PRIMUS_MOCK=1 — does **not** contact Primus. For UI testing only.")
        if st.button("Simulate uniqueness enrollment", key="mock_enroll"):
            _commit_enrollment(mock_uniqueness_record())
            set_flash("success", "Mock enrollment succeeded (not a real zkTLS proof).")
            st.rerun()

    st.markdown(
        """
        <div class="vs-card">
          <h3>What is stored where</h3>
          <p class="vs-muted">
            Enrollment receipt: <code>~/.voicesense/enrollment.json</code>
            (fingerprint + timestamps only).<br/>
            Screening audio: not written under that folder, not written in the repo.<br/>
            Never combine a public PD label with this fingerprint on a chain.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
