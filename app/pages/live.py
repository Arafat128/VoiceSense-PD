"""Live voice studio — the screen people see first.

Read this file from top to bottom. It is the click path:

1. The person records a vowel or uploads an audio file. Nothing is scored yet.
2. Analyze voice calls src/predict.py, which listens and applies the safety rules.
3. The recording is then dropped from the page.
4. An everyday-language card is drawn first. Technical numbers stay underneath.
5. Up to three recordings are kept for this visit only, in the browser session.

Shared drawing helpers live in app/widgets.py.
Set VOICESENSE_DEBUG=1 to show the raw traceback after a failed analysis.
"""
from __future__ import annotations

import os
import traceback

import streamlit as st

from app.pages.privacy import render_export_receipt
from app.widgets import (
    decision_policy_text,
    friendly_error,
    live_audio_available,
    render_everyday,
    render_explanation,
    render_flash,
    render_result_banner,
    set_flash,
    to_wav_bytes,
    validation_caption,
)
from src.config import DISCLAIMER
from src.privacy import bump_audio_nonce


def _debug_enabled() -> bool:
    return os.getenv("VOICESENSE_DEBUG") == "1"

@st.fragment
def live_page() -> None:
    render_flash()
    st.markdown('<div class="vs-section">Capture</div>', unsafe_allow_html=True)

    if not live_audio_available():
        st.error("Live audio model missing. Run: `python -m training.train_audio`")
    else:
        c1, c2 = st.columns([1.1, 1])
        with c1:
            st.markdown(
                """
                <div class="vs-card">
                  <h3>Recording protocol</h3>
                  <p class="vs-muted">
                    1. Pick the prompt language you prefer<br/>
                    2. Quiet room · mic 15–20 cm away<br/>
                    3. Sustain <b>“aaa” / “আ—”</b> for 5–8 seconds<br/>
                    4. Repeat that vowel <b>three times</b>, with a pause between them<br/>
                    5. Click Analyze — no diagnosis is made
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class="vs-card">
                  <h3>Decision policy</h3>
                  <p class="vs-muted">
                    {decision_policy_text()}
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        lang = st.radio(
            "Prompt language (changes the on-screen prompt only, not the model)",
            options=["english", "bangla"],
            format_func=lambda x: "English" if x == "english" else "Bangla (বাংলা)",
            horizontal=True,
            key="lang_mode",
        )

        # IMPORTANT: optional preview only — auto audio widgets often trigger IDM popups
        preview = st.checkbox(
            "Preview captured audio in browser (optional — may open download managers like IDM)",
            value=False,
            key="preview_audio",
        )

        rec_col, up_col = st.columns(2)
        audio_raw: bytes | None = None
        source_kind = None
        audio_nonce = int(st.session_state.get("audio_nonce", 0))

        with rec_col:
            st.markdown("##### Microphone")
            if hasattr(st, "audio_input"):
                mic = st.audio_input(
                    "Record a sustained vowel",
                    key=f"live_mic_{audio_nonce}",
                )
                if mic is not None:
                    audio_raw = mic.getvalue()
                    source_kind = "microphone"
                    st.success("Recording is in memory only — not written to disk.")
                    if preview:
                        st.audio(audio_raw)
            else:
                st.warning("Upgrade Streamlit for microphone support, or use file upload.")

        with up_col:
            st.markdown("##### Upload file")
            up = st.file_uploader(
                "WAV / MP3 / OGG / WEBM / M4A / FLAC",
                type=["wav", "mp3", "ogg", "webm", "m4a", "flac"],
                key=f"voice_up_{audio_nonce}",
            )
            if up is not None:
                audio_raw = up.getvalue()
                source_kind = "upload"
                st.success(
                    f"Loaded {len(audio_raw)//1024} KB in memory. "
                    "The filename is not kept in session state."
                )
                if preview:
                    st.audio(audio_raw)

        act_l, act_r = st.columns(2)
        with act_l:
            analyze = st.button(
                "Analyze voice",
                type="primary",
                width="stretch",
                disabled=audio_raw is None,
                key="analyze_live",
            )
        with act_r:
            discard = st.button(
                "Discard recording",
                type="secondary",
                width="stretch",
                disabled=audio_raw is None,
                key="discard_live",
            )
        if discard:
            bump_audio_nonce(st.session_state)
            st.session_state["audio_discarded_after_analysis"] = False
            set_flash("success", "Recording discarded. Nothing was scored or saved.")
            st.rerun(scope="fragment")

        if analyze and audio_raw is not None:
            with st.spinner("Normalizing audio · extracting clinical features · scoring models…"):
                try:
                    from src.predict import predict_from_audio

                    # Convert to WAV in-memory to stabilize decode path (no file download)
                    wav_bytes = to_wav_bytes(audio_raw)
                    result = predict_from_audio(
                        wav_bytes,
                        language_mode=lang,
                        require_quality_ok=True,
                    )
                    st.session_state["last_live_result"] = result
                    st.session_state["last_live_source"] = source_kind
                    st.session_state.pop("last_live_error", None)
                    st.session_state.pop("last_live_trace", None)
                    if result.get("status") == "ok":
                        primary = result.get("primary") or {}
                        decision = primary.get("prediction") or primary.get("decision_code") or "done"
                        take = {
                            "probability_pd": float(primary.get("probability_pd") or 0.0),
                            "decision_code": primary.get("decision_code"),
                            "prediction": primary.get("prediction"),
                            "recording_ready": not (primary.get("recording_limits") or []),
                        }
                        takes = list(st.session_state.get("live_takes") or [])
                        takes.append(take)
                        st.session_state["live_takes"] = takes[-3:]
                        set_flash(
                            "success",
                            f"Analysis succeeded. Recording discarded from memory. Decision: {decision}",
                        )
                        bump_audio_nonce(st.session_state)
                        st.session_state["audio_discarded_after_analysis"] = True
                        st.rerun(scope="fragment")
                    elif result.get("status") == "unreadable_audio":
                        set_flash("error", result.get("message") or "This audio could not be analysed.")
                        st.rerun(scope="fragment")
                    elif result.get("status") == "rejected_quality":
                        reasons = "; ".join(str(m) for m in (result.get("messages") or [])[:3])
                        set_flash(
                            "error",
                            "Recording quality too low to analyse"
                            + (f" ({reasons})" if reasons else "")
                            + ". Re-record a steady 5–8 s “aaa” in a quiet room, 15–20 cm from the mic.",
                        )
                        st.rerun(scope="fragment")
                    else:
                        set_flash("error", f"Analysis failed: {result.get('status', 'unknown error')}")
                        st.rerun(scope="fragment")
                except Exception as e:  # noqa: BLE001
                    set_flash("error", friendly_error(e))
                    st.session_state["last_live_error"] = friendly_error(e)
                    st.session_state["last_live_trace"] = traceback.format_exc()
                    st.rerun(scope="fragment")

        if st.session_state.get("audio_discarded_after_analysis"):
            st.caption("Recording discarded from memory after analysis.")

        if st.session_state.get("last_live_error"):
            st.error(st.session_state["last_live_error"])
            if _debug_enabled():
                with st.expander("Technical traceback"):
                    st.code(st.session_state.get("last_live_trace") or "")
            if st.button("Clear error", key="clr_live_err"):
                st.session_state.pop("last_live_error", None)
                st.session_state.pop("last_live_trace", None)
                st.rerun(scope="fragment")

        result = st.session_state.get("last_live_result")
        if result:
            st.markdown('<div class="vs-section">Results</div>', unsafe_allow_html=True)
            st.caption(f"Source: {st.session_state.get('last_live_source', 'n/a')} · audio not retained")

            if result.get("status") == "rejected_quality":
                from src.plain_language import plain_quality_rejection

                st.error(plain_quality_rejection(result))
                for m in result.get("messages", []):
                    st.write(f"- {m}")
                with st.expander("Technical quality numbers"):
                    st.json(result.get("quality", {}))
                st.info("Tip: longer steady vowel, quieter room, closer mic.")
            else:
                from src.plain_language import build_plain_reading

                primary = result["primary"]
                takes = list(st.session_state.get("live_takes") or [])
                plain = build_plain_reading(result, takes)
                render_result_banner(primary)
                render_everyday(plain)
                st.caption(f"Recording set · {len(takes)} of 3 kept")
                st.caption(
                    "How much to trust this: on recordings from a setup the model never trained on, "
                    "its accuracy is close to chance (see About & method). Use it to explore, "
                    "not to decide anything about your health."
                )
                if takes and st.button("Start a new set of 3 recordings", key="reset_takes"):
                    st.session_state["live_takes"] = []
                    st.rerun(scope="fragment")

                st.markdown('<div class="vs-section">Technical details</div>', unsafe_allow_html=True)
                thr = primary.get("thresholds") or {}
                if thr:
                    st.caption(
                        f"Policy · a firm PD call needs P≥{thr['pd_high']:.0%}, "
                        f"a firm healthy call needs P≤{thr['hc_high']:.0%} (cut-offs from validation data), "
                        "and both also need a clean clip (at least 4s, low noise, mostly voiced) "
                        "plus agreement from the second model. Otherwise the result stays Uncertain."
                    )

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("P(PD)", f"{primary['probability_pd']:.1%}")
                m2.metric("P(Healthy)", f"{primary['probability_hc']:.1%}")
                m3.metric("Adjusted confidence", f"{primary.get('adjusted_confidence', 0):.1%}")
                q = result.get("quality", {})
                m4.metric("Voiced fraction", f"{q.get('voiced_fraction', 0):.0%}")

                st.markdown(
                    f"**Quality** · duration {q.get('duration_sec', 0):.2f}s · "
                    f"SNR proxy {q.get('snr_db_proxy', 0):.1f} dB · OK={q.get('ok')}"
                )
                for msg in q.get("messages", []):
                    st.caption(msg)

                if primary.get("pred_label") == 1:
                    st.warning(
                        "Possible PD pattern on this clip only. Not a diagnosis. "
                        "Re-record a quiet 5–8s vowel and compare. See a clinician for real symptoms."
                    )
                elif primary.get("recording_limits"):
                    st.info(
                        "The score was held back because this clip is too short, noisy, or incomplete "
                        "to trust. A longer vowel in a quieter room is the useful next step."
                    )
                elif (primary.get("agreement") or {}).get("peer_model") and not (primary.get("agreement") or {}).get("agree"):
                    st.info(
                        "The two voice checks did not agree, so the result stays uncertain "
                        "even if one of the scores looks high or low."
                    )
                elif primary.get("pred_label") == -1:
                    st.info(
                        "Uncertain is common on home mics and is safer than a false PD alarm. "
                        "Try one cleaner re-record if you want a clearer score."
                    )
                else:
                    st.success("No clear PD acoustic pattern on this sample.")

                agreement = primary.get("agreement") or {}
                if agreement.get("peer_model"):
                    st.caption(
                        f"Second check · `{agreement.get('peer_model')}` · "
                        f"P(PD)={float(agreement.get('peer_probability_pd') or 0):.1%} · "
                        f"{agreement.get('peer_decision_code')} · "
                        f"{'agrees' if agreement.get('agree') else 'does not agree'}"
                    )
                elif agreement.get("single_model"):
                    st.caption("Second model file was not found, so only the recording-quality safeguard was applied.")
                model_call = primary.get("model_call") or {}
                if model_call.get("decision_code") and model_call.get("decision_code") != primary.get("decision_code"):
                    st.caption(
                        f"Safeguard · model alone would have said `{model_call.get('decision_code')}` "
                        f"at {float(model_call.get('probability_pd') or 0):.1%}, "
                        "and that call was held back."
                    )

                st.markdown(
                    f"**Primary model:** {primary.get('display_name')} · `{primary.get('model_name')}` · "
                    f"completeness {primary.get('completeness', 0):.0%}"
                )
                if primary.get("metrics_at_train_time"):
                    st.caption(validation_caption(primary["metrics_at_train_time"]))

                render_explanation(primary.get("explanation") or {}, "Why this decision")

                warnings = list(result.get("warnings") or [])
                other_warnings = warnings

                for note in other_warnings:
                    st.caption(f"⚠ {note}")

                st.warning(DISCLAIMER)
                receipt_result = dict(result)
                receipt_result["plain"] = plain
                render_export_receipt(receipt_result, st.session_state.get("last_live_source"))
