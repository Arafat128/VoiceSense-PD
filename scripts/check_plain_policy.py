"""Checks for the live-voice safeguards and the everyday-language reading."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plain_language import build_plain_reading, combine_takes
from src.predict_audio import _apply_accuracy_policy
from src.privacy import assert_receipt_has_no_audio, make_screening_receipt


def _primary(code: str, proba: float, completeness: float = 1.0) -> dict:
    prediction = {
        "PD": "Possible PD pattern (screening only)",
        "HC": "No clear PD pattern (likely non-PD on this sample)",
        "UNCERTAIN": "Uncertain — not enough evidence to flag PD",
    }[code]
    label = {"PD": 1, "HC": 0, "UNCERTAIN": -1}[code]
    return {
        "pred_label": label,
        "decision_code": code,
        "probability_pd": proba,
        "probability_hc": 1.0 - proba,
        "prediction": prediction,
        "decision_band": code,
        "completeness": completeness,
        "explanation": {
            "decision": prediction,
            "summary_lines": [f"Screening decision: {prediction}", "Model probability of PD: x"],
            "why_pd": [
                {"feature": "praat_local_jitter", "supports_pd": True},
                {"feature": "praat_hnr", "supports_pd": True},
            ],
            "why_not_pd": [
                {"feature": "praat_local_shimmer", "supports_pd": False},
            ],
        },
    }


def _peer(code: str, proba: float) -> dict:
    return {"decision_code": code, "probability_pd": proba, "model_name": "svm_rbf"}


def _quality(duration: float = 6.0, snr: float = 18.0, voiced: float = 0.8) -> dict:
    return {"duration_sec": duration, "snr_db_proxy": snr, "voiced_fraction": voiced, "ok": True}


def test_short_clip_cannot_flag_pd() -> None:
    primary = _apply_accuracy_policy(
        _primary("PD", 0.91),
        _peer("PD", 0.88),
        _quality(duration=1.4, snr=18, voiced=0.8),
    )
    assert primary["decision_code"] == "UNCERTAIN"
    assert primary["downgraded_from"] == "PD"
    assert primary["model_call"]["probability_pd"] == 0.91


def test_disagreement_cannot_flag_pd() -> None:
    primary = _apply_accuracy_policy(
        _primary("PD", 0.80),
        _peer("HC", 0.20),
        _quality(),
    )
    assert primary["decision_code"] == "UNCERTAIN"
    assert primary["agreement"]["agree"] is False


def test_clean_agreement_keeps_pd() -> None:
    primary = _apply_accuracy_policy(
        _primary("PD", 0.86),
        _peer("PD", 0.79),
        _quality(),
    )
    assert primary["decision_code"] == "PD"
    assert primary["agreement"]["agree"] is True
    assert not primary["policy_notes"]


def test_two_clean_takes_needed() -> None:
    one = combine_takes(
        [{"probability_pd": 0.9, "decision_code": "PD", "recording_ready": True}]
    )
    assert one is not None and one["conclusive"] is False
    two = combine_takes(
        [
            {"probability_pd": 0.9, "decision_code": "PD", "recording_ready": True},
            {"probability_pd": 0.8, "decision_code": "PD", "recording_ready": True},
        ]
    )
    assert two is not None and two["decision_code"] == "PD" and two["conclusive"] is True
    mixed = combine_takes(
        [
            {"probability_pd": 0.9, "decision_code": "PD", "recording_ready": True},
            {"probability_pd": 0.2, "decision_code": "HC", "recording_ready": True},
        ]
    )
    assert mixed is not None and mixed["decision_code"] == "UNCERTAIN"


def test_plain_language_hides_feature_codes() -> None:
    primary = _apply_accuracy_policy(_primary("PD", 0.86), _peer("PD", 0.8), _quality())
    plain = build_plain_reading(
        {"primary": primary, "quality": _quality()},
        [{"probability_pd": 0.86, "decision_code": "PD", "recording_ready": True}],
    )
    blob = " ".join(
        [
            plain["headline"],
            plain["meaning"],
            plain["chance"],
            plain["trust_detail"],
            *plain["heard"],
            *plain["next_steps"],
        ]
    ).lower()
    assert "praat_" not in blob
    assert "jitter" not in blob
    assert "shimmer" not in blob
    assert "hnr" not in blob
    assert "parkinson" in blob
    assert plain["heard"]


def test_receipt_keeps_plain_words_and_drops_audio() -> None:
    primary = _apply_accuracy_policy(_primary("HC", 0.2), _peer("HC", 0.25), _quality())
    plain = build_plain_reading({"primary": primary}, [])
    receipt = make_screening_receipt(
        {
            "status": "ok",
            "primary": primary,
            "plain": plain,
            "quality": _quality(),
            "secondary": {},
            "warnings": [],
        },
        source_kind="upload:secret-name.wav",
        include_features=False,
    )
    assert_receipt_has_no_audio(receipt)
    assert receipt["plain_language"]["headline"]
    assert "secret-name" not in str(receipt)


def main() -> None:
    test_short_clip_cannot_flag_pd()
    test_disagreement_cannot_flag_pd()
    test_clean_agreement_keeps_pd()
    test_two_clean_takes_needed()
    test_plain_language_hides_feature_codes()
    test_receipt_keeps_plain_words_and_drops_audio()
    print("plain-policy-ok")


if __name__ == "__main__":
    main()
