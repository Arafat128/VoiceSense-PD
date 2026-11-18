"""About and method screen. It does not score a voice.

Numbers are read from reports/ at render time, so this page always matches
the shipped model instead of repeating figures by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.config import REPORTS_DIR


def _load(name: str) -> dict:
    path = Path(REPORTS_DIR) / name
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _ci(block: dict) -> str:
    lo, hi = block.get("ci95") or (float("nan"), float("nan"))
    return f"{block.get('value', float('nan')):.2f} ({lo:.2f}–{hi:.2f})"


def render_about() -> None:
    live = _load("live_audio_metrics.json")
    default = live.get("app_default_model", "random_forest")
    model = (live.get("models") or {}).get(default, {})
    cv = model.get("cv") or {}
    lodo = model.get("lodo") or {}
    datasets = live.get("datasets") or {}

    st.markdown("### What the live model is")
    if datasets:
        rows = [
            f"- **{key}**: {d['pd_speakers']} Parkinson's and {d['hc_speakers']} healthy speakers, {d['clips']} clips"
            for key, d in datasets.items()
        ]
        st.markdown(
            "Trained on sustained vowels from three public corpora (all CC BY 4.0):\n\n" + "\n".join(rows)
        )
    st.markdown(
        "It uses 34 classical voice measures (pitch, jitter, shimmer, harmonics-to-noise, loudness range). "
        "Young healthy controls are left out because age alone separates them from Parkinson's speakers. "
        f"The primary model is `{default}`; the other model must agree before a firm call."
    )

    st.markdown("### How well it works")
    if cv:
        st.markdown(
            "**Same corpora, new speakers** (speaker-grouped 5-fold CV repeated 5 times, "
            "95% CI from a speaker bootstrap):\n\n"
            "| level | AUC | balanced accuracy | sensitivity | specificity |\n|---|---|---|---|---|\n"
            + "\n".join(
                f"| per {lvl} | {_ci(cv[lvl]['auc'])} | {_ci(cv[lvl]['balanced_accuracy'])} | "
                f"{_ci(cv[lvl]['sensitivity'])} | {_ci(cv[lvl]['specificity'])} |"
                for lvl in ("clip", "person")
                if lvl in cv
            )
        )
    if lodo:
        st.markdown(
            "**A corpus the model never saw** (train on two corpora, test on the third). "
            "This is the honest estimate for a new microphone:\n\n"
            "| held-out corpus | per-person AUC |\n|---|---|\n"
            + "\n".join(f"| {k} | {_ci(v['person']['auc'])} |" for k, v in lodo.items() if isinstance(v, dict))
        )
        st.warning(
            "Across corpora the AUC is close to 0.5 (chance). Recording setup changes these voice "
            "measures more than Parkinson's does, so a single home recording cannot screen anyone. "
            "Treat results as a research demonstration."
        )

    st.markdown("### English and Bangla")
    st.markdown(
        "The live model listens to a sustained vowel, so it does not depend on words. "
        "The language choice only changes the on-screen prompt. "
        "The UCI English and BenSParX Bengali models were trained on published feature tables from "
        "other recording setups; mapping a microphone clip onto them is not valid, so they are **not** "
        "used in live analysis. They remain in Feature lab as research comparisons on their own data."
    )

    st.markdown("### Privacy")
    st.markdown(
        "Local Streamlit scores audio on your machine and discards the recording after analysis. "
        "The receipt has no audio and no file name."
    )
    st.markdown("### Troubleshooting")
    st.markdown(
        "Internet Download Manager can intercept browser audio previews (Error 0x80080005). "
        "That is not a model crash: leave audio preview off or exclude localhost in IDM."
    )
    st.markdown("### Disclaimer")
    st.markdown("Research screening support only. Not a clinical diagnostic device.")
