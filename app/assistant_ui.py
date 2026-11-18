"""Floating project assistant. Chat history stays in the Streamlit session."""
from __future__ import annotations

import time

import streamlit as st

from src.assistant.policy import NOT_CONFIGURED, Quota
from src.assistant.service import submit_message

GREETING = (
    "Ask me how VoiceSense screening works. I only know this project. "
    "I cannot see your recording or your scores, and I cannot diagnose Parkinson's disease."
)
SUGGESTIONS = (
    "Is this a medical diagnosis?",
    "What data was the live model trained on?",
    "What happens to my recording?",
)


def _history() -> list[dict[str, str]]:
    return list(st.session_state.setdefault("vs_assistant_history", []))


def _quota() -> Quota:
    quota = st.session_state.get("vs_assistant_quota")
    if not isinstance(quota, Quota):
        quota = Quota()
        st.session_state["vs_assistant_quota"] = quota
    return quota


def _ask(text: str) -> None:
    result = submit_message(_history(), _quota(), text, now=time.time())
    st.session_state["vs_assistant_history"] = result.history
    st.session_state["vs_assistant_quota"] = result.quota
    st.session_state["vs_assistant_notice"] = result.notice


def render_assistant() -> None:
    open_panel = bool(st.session_state.get("vs_assistant_open"))
    if open_panel:
        with st.container(key="vs_assistant_panel", border=True):
            title, close = st.columns([4, 1])
            title.markdown("**VoiceSense assistant**")
            if close.button("Close", key="vs_assistant_close"):
                st.session_state["vs_assistant_open"] = False
                st.rerun()
            st.caption(GREETING)
            if not _history():
                for index, question in enumerate(SUGGESTIONS):
                    if st.button(question, key=f"vs_assistant_suggest_{index}", width="stretch"):
                        _ask(question)
                        st.rerun()
            for message in _history():
                role = "user" if message.get("role") == "user" else "assistant"
                with st.chat_message(role):
                    st.markdown(message.get("content") or "")
            notice = st.session_state.get("vs_assistant_notice")
            if notice:
                if notice == NOT_CONFIGURED:
                    st.info(notice)
                else:
                    st.warning(notice)
            typed = st.chat_input(
                "Ask about VoiceSense",
                key="vs_assistant_input",
                max_chars=500,
            )
            if typed:
                _ask(typed)
                st.rerun()

    if st.button("💬", key="vs_assistant_bubble", help="Ask about VoiceSense"):
        st.session_state["vs_assistant_open"] = not open_panel
        st.rerun()
