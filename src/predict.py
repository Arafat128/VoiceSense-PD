"""Score a feature-table row or a live recording.

Model files are read only through src.models.registry, which caches them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import DISCLAIMER
from .explain import explain_prediction
from .features.audio import add_engineered, extract_features, vectorize
from .models.registry import load_live_audio_model, load_model


def _row_from_mapping(mapping: dict[str, Any], feature_names: list[str]) -> pd.DataFrame:
    missing = [name for name in feature_names if name not in mapping]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} required features. First missing: {missing[:5]}"
        )
    row = {name: float(mapping[name]) for name in feature_names}
    return pd.DataFrame([row], columns=feature_names)


def predict_from_features(
    dataset_key: str,
    features: dict[str, Any] | pd.Series | pd.DataFrame,
    model_name: str = "random_forest",
    top_k: int = 10,
) -> dict[str, Any]:
    pack = load_model(dataset_key, model_name)
    feature_names = pack["feature_names"]
    pipe = pack["pipeline"]

    if isinstance(features, pd.DataFrame):
        if len(features) != 1:
            raise ValueError("Provide exactly one sample row in the DataFrame")
        frame = features[feature_names]
        series = frame.iloc[0]
    elif isinstance(features, pd.Series):
        frame = _row_from_mapping(features.to_dict(), feature_names)
        series = frame.iloc[0]
    else:
        frame = _row_from_mapping(dict(features), feature_names)
        series = frame.iloc[0]

    proba = float(pipe.predict_proba(frame)[0, 1])
    pred = table_label(pack, proba)
    explanation = explain_prediction(pack, series, proba, pred, top_k=top_k)

    return {
        "dataset_key": dataset_key,
        "model_name": model_name,
        "language": pack.get("language"),
        "display_name": pack.get("display_name"),
        "prediction": explanation["decision"],
        "pred_label": pred,
        "probability_pd": proba,
        "probability_hc": 1.0 - proba,
        "explanation": explanation,
        "metrics_at_train_time": pack.get("metrics"),
        "thresholds": pack.get("decision_thresholds"),
        "disclaimer": pack.get("disclaimer"),
    }


def table_label(pack: dict[str, Any], proba: float) -> int:
    """1 = PD side, 0 = healthy side, -1 = between the model's data-derived cut-offs.

    Older table artifacts without saved cut-offs keep their 50% yes/no rule.
    """
    thr = pack.get("decision_thresholds")
    if not thr:
        return int(proba >= 0.5)
    if proba >= float(thr["pd_high"]):
        return 1
    if proba <= float(thr["hc_high"]):
        return 0
    return -1


def _add_engineered(feat_dict: dict[str, float]) -> dict[str, float]:
    """Same engineered clinical features as training (shared definition)."""
    return add_engineered(feat_dict)


# Only for artifacts trained before thresholds were derived from data.
LEGACY_THRESHOLDS = {"pd_high": 0.72, "hc_high": 0.42}


def thresholds_for(pack: dict[str, Any]) -> dict[str, float]:
    """Likely / Unlikely cut-offs saved in the model metadata at training time."""
    saved = pack.get("decision_thresholds") or {}
    if "pd_high" in saved and "hc_high" in saved:
        return {"pd_high": float(saved["pd_high"]), "hc_high": float(saved["hc_high"])}
    return dict(LEGACY_THRESHOLDS)


def _decide_from_proba(proba: float, thresholds: dict[str, float]) -> dict[str, Any]:
    """
    Conservative decision for consumer mics:
    - high p => PD
    - low p => Healthy
    - middle => Uncertain (do NOT scare users with weak PD flags)
    """
    thr_pd = float(thresholds["pd_high"])
    thr_hc = float(thresholds["hc_high"])
    if proba >= thr_pd:
        return {
            "pred_label": 1,
            "decision_code": "PD",
            "prediction": "Possible PD pattern (screening only)",
            "band": "high_pd",
        }
    if proba <= thr_hc:
        return {
            "pred_label": 0,
            "decision_code": "HC",
            "prediction": "No clear PD pattern (likely non-PD on this sample)",
            "band": "high_hc",
        }
    return {
        "pred_label": -1,
        "decision_code": "UNCERTAIN",
        "prediction": "Uncertain — not enough evidence to flag PD",
        "band": "uncertain",
    }


# A firm call needs a steady vowel. Jitter, shimmer, and HNR move with room
# noise and phone compression, so clips that only barely pass the hard reject
# gate are not allowed to say "Parkinson's" or "healthy".
MIN_CALL_DURATION_SEC = 4.0
MIN_CALL_SNR_DB = 10.0
MIN_CALL_VOICED = 0.35
MIN_CALL_COMPLETENESS = 0.90
_PEER_MODEL = {"random_forest": "svm_rbf", "svm_rbf": "random_forest"}


def _recording_limits(quality: dict[str, Any], completeness: float) -> list:
    """Reasons a clip is too weak for a firm PD or healthy call."""
    reasons = []
    duration = float(quality.get("duration_sec") or 0.0)
    snr = float(quality.get("snr_db_proxy") or 0.0)
    voiced = float(quality.get("voiced_fraction") or 0.0)
    if duration < MIN_CALL_DURATION_SEC:
        reasons.append(
            f"it is only {duration:.1f}s long (a steady vowel of at least 4 seconds is needed)"
        )
    if snr < MIN_CALL_SNR_DB:
        reasons.append("background noise is high enough to distort pitch and loudness measures")
    if voiced < MIN_CALL_VOICED:
        reasons.append("too little of the clip is a clear voiced sound")
    if float(completeness or 0.0) < MIN_CALL_COMPLETENESS:
        reasons.append("some voice measures could not be calculated and had to be filled in")
    return reasons


def _downgrade_to_uncertain(primary: dict[str, Any], reasons: list) -> None:
    original = primary.get("prediction")
    primary["downgraded_from"] = primary.get("decision_code")
    primary["pred_label"] = -1
    primary["decision_code"] = "UNCERTAIN"
    primary["prediction"] = "Uncertain — not enough evidence to flag PD"
    primary["decision_band"] = "uncertain"
    note = "A firm call was withheld because: " + "; ".join(reasons) + "."
    explanation = dict(primary.get("explanation") or {})
    kept = [
        line
        for line in (explanation.get("summary_lines") or [])
        if not str(line).startswith("Screening decision:")
    ]
    explanation["decision"] = primary["prediction"]
    explanation["summary_lines"] = [
        f"Screening decision: {primary['prediction']}",
        note,
        (
            f"Underlying model probability of PD: {primary['probability_pd']:.1%}. "
            f"Without the safeguard the call would have been: {original}."
        ),
    ] + kept
    primary["explanation"] = explanation


def _apply_accuracy_policy(
    primary: dict[str, Any],
    peer: dict[str, Any] | None,
    quality: dict[str, Any],
) -> dict[str, Any]:
    """Hold back a firm call unless the recording is usable and both models agree.

    Home microphones inflate jitter, shimmer, and noise measures. Requiring
    agreement, and refusing short or noisy clips, cuts false Parkinson's flags
    without inventing a new trained model.
    """
    limits = _recording_limits(quality, float(primary.get("completeness") or 0.0))
    raw_code = primary.get("decision_code")
    peer_code = peer.get("decision_code") if peer else None
    disagree = peer is not None and peer_code != raw_code
    primary["model_call"] = {
        "decision_code": raw_code,
        "prediction": primary.get("prediction"),
        "probability_pd": primary.get("probability_pd"),
    }
    primary["recording_limits"] = limits
    primary["agreement"] = {
        "peer_model": peer.get("model_name") if peer else None,
        "peer_probability_pd": peer.get("probability_pd") if peer else None,
        "peer_decision_code": peer_code,
        "agree": bool(peer is not None and not disagree),
        "single_model": peer is None,
    }
    reasons = []
    if raw_code in ("PD", "HC"):
        if disagree:
            reasons.append("the second voice check does not agree with the first")
        reasons.extend(limits)
    primary["policy_notes"] = reasons
    if reasons:
        _downgrade_to_uncertain(primary, reasons)
    elif limits:
        explanation = dict(primary.get("explanation") or {})
        lines = list(explanation.get("summary_lines") or [])
        extra = "This clip is also a weak recording: " + "; ".join(limits) + "."
        if extra not in lines:
            lines.append(extra)
        explanation["summary_lines"] = lines
        primary["explanation"] = explanation
    return primary


def _predict_with_pack(
    pack: dict[str, Any],
    feat_dict: dict[str, float],
    explain: bool = True,
) -> dict[str, Any]:
    feat_dict = _add_engineered(feat_dict)
    names = pack["feature_names"]
    medians = pack.get("train_medians") or {}
    vec, _imputed, completeness = vectorize(feat_dict, names, medians)
    frame = pd.DataFrame([vec], columns=names)
    pipe = pack["pipeline"]
    proba = float(pipe.predict_proba(frame)[0, 1])
    thresholds = thresholds_for(pack)
    decision = _decide_from_proba(proba, thresholds)
    explanation = None
    if explain:
        # For explanations, map uncertain toward the nearer class
        exp_label = 1 if proba >= 0.5 else 0
        series = frame.iloc[0]
        explanation = explain_prediction(pack, series, proba, exp_label, top_k=12)
        # Override summary decision line for uncertain
        if decision["pred_label"] == -1:
            explanation["decision"] = decision["prediction"]
            explanation["summary_lines"] = [
                f"Screening decision: {decision['prediction']}",
                f"Model probability of PD: {proba:.1%}",
                (
                    f"PD is only flagged when probability ≥ {thresholds['pd_high']:.0%}; "
                    f"healthy when ≤ {thresholds['hc_high']:.0%} "
                    "(cut-offs chosen from validation data)."
                ),
                "Your score is in the middle band — this often happens with short clips, new microphones, or normal voice variation.",
                "This is NOT a diagnosis. If you have concerns, consult a clinician.",
            ] + explanation.get("summary_lines", [])[3:]
        else:
            explanation["decision"] = decision["prediction"]
            explanation["summary_lines"][0] = f"Screening decision: {decision['prediction']}"

    return {
        "pred_label": decision["pred_label"],
        "decision_code": decision["decision_code"],
        "probability_pd": proba,
        "probability_hc": 1.0 - proba,
        "prediction": decision["prediction"],
        "decision_band": decision["band"],
        "thresholds": thresholds,
        "completeness": completeness,
        "imputed_features": _imputed,
        "explanation": explanation,
        "model_name": pack.get("model_name"),
        "display_name": pack.get("display_name"),
        "language": pack.get("language"),
        "metrics_at_train_time": pack.get("metrics"),
        "false_positive_note": pack.get("false_positive_note"),
    }


def predict_from_audio(
    source: str | Path | bytes | bytearray,
    language_mode: str = "english",
    live_model_name: str | None = None,
    require_quality_ok: bool = True,
) -> dict[str, Any]:
    """
    language_mode: 'english' | 'bangla' | 'auto'. Recorded for the receipt only.
    The decision always comes from the live vowel model (same extractor as training).
    """
    try:
        extraction = extract_features(source)
    except Exception as exc:  # noqa: BLE001 - a bad file should explain itself, not crash
        return {
            "status": "unreadable_audio",
            "message": (
                "This audio could not be analysed. Record a steady 5–8 second vowel, "
                "or upload a WAV, MP3, OGG, WEBM, M4A, or FLAC file."
            ),
            "detail": f"{type(exc).__name__}: {exc}"[:300],
            "disclaimer": DISCLAIMER,
        }
    quality = extraction.quality.as_dict()

    if require_quality_ok and not extraction.quality.ok:
        return {
            "status": "rejected_quality",
            "quality": quality,
            "message": "Recording quality is not sufficient for reliable screening.",
            "messages": extraction.quality.messages,
            "disclaimer": DISCLAIMER,
        }

    live_pack = load_live_audio_model(live_model_name)
    primary = _predict_with_pack(live_pack, extraction.common)
    peer = None
    peer_name = _PEER_MODEL.get(str(live_pack.get("model_name") or ""))
    if peer_name:
        try:
            peer = _predict_with_pack(
                load_live_audio_model(peer_name),
                extraction.common,
                explain=False,
            )
        except Exception as exc:  # noqa: BLE001
            peer = None
            primary["peer_error"] = str(exc)
    primary = _apply_accuracy_policy(primary, peer, quality)

    # Confidence blend: down-weight if quality borderline or many imputations
    q_factor = 1.0
    if extraction.quality.snr_db_proxy < 8:
        q_factor *= 0.9
    if extraction.quality.voiced_fraction < 0.25:
        q_factor *= 0.9
    if primary["completeness"] < 0.85:
        q_factor *= 0.85
    # For uncertain band, adjusted confidence reflects distance from decision edges
    if primary.get("pred_label") == -1:
        adjusted_conf = float(min(primary["probability_pd"], 1.0 - primary["probability_pd"]) * 2 * q_factor)
    elif primary.get("pred_label") == 1:
        adjusted_conf = float(primary["probability_pd"] * q_factor)
    else:
        adjusted_conf = float(primary["probability_hc"] * q_factor)

    warnings = [
        (
            "Live laptop/phone mics differ from clinic recordings. "
            "A single clip can be wrong — do not treat a PD flag as a diagnosis."
        ),
    ]
    if primary.get("policy_notes"):
        warnings.append(
            "A firm healthy or Parkinson's call was withheld: "
            + "; ".join(primary["policy_notes"])
            + "."
        )
    if primary.get("pred_label") == 1:
        warnings.append(
            "PD pattern flagged only because probability was high. "
            "Re-record a quiet 5–8s sustained “aaa”/“আ—” and compare results."
        )
    # The UCI English and BenSParX Bengali models were trained on feature
    # tables from other recording setups. Mapping a microphone clip onto them
    # is not valid, so they are not scored here (Feature lab still uses them).
    mode_note = (
        "Prompt language only changes the on-screen vowel prompt. The vowel model is the "
        "same for every language; sustained-vowel measures do not depend on words."
    )

    return {
        "status": "ok",
        "language_mode": language_mode,
        "mode_note": mode_note,
        "quality": quality,
        "primary": {
            **primary,
            "adjusted_confidence": adjusted_conf,
            "role": "PRIMARY (same feature extractor as training — most trustworthy for live audio)",
        },
        "warnings": warnings,
        "extraction_preview": {
            "common_top": {
                item["feature"]: extraction.common.get(item["feature"])
                for item in (live_pack.get("feature_importances") or [])[:5]
                if isinstance(item, dict) and item.get("feature") in extraction.common
            },
        },
        "disclaimer": DISCLAIMER,
    }
