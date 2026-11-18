"""Sanity checks for screening receipts (no audio, no identity proofs)."""
from __future__ import annotations

import json

from src.privacy import (
    assert_receipt_has_no_audio,
    make_screening_receipt,
    receipt_json,
)


def test_receipt_omits_audio() -> None:
    result = {
        "status": "ok",
        "language_mode": "english",
        "quality": {"ok": True, "duration_sec": 6.2},
        "primary": {
            "prediction": "Uncertain — not enough evidence to flag PD",
            "decision_code": "UNCERTAIN",
            "pred_label": -1,
            "probability_pd": 0.51,
            "probability_hc": 0.49,
            "model_name": "random_forest",
            "display_name": "live",
            "explanation": {
                "decision": "Uncertain",
                "summary_lines": ["ok"],
                "why_pd": [{"feature": "praat_local_jitter", "comparison": "higher"}],
                "top_influential_features": [{"feature": "praat_local_jitter", "value": 0.01}],
            },
            "imputed_features": ["secret"],
        },
        "secondary": {},
        "warnings": [],
        "extraction_preview": {"common_top": {"praat_local_jitter": 0.01}},
    }
    receipt = make_screening_receipt(
        result,
        source_kind="upload:patient-name.wav",
        include_features=False,
        local_runtime=True,
    )
    assert_receipt_has_no_audio(receipt)
    blob = receipt_json(receipt)
    assert "patient-name.wav" not in blob
    assert receipt["source_kind"] == "upload"
    assert receipt["audio_retained"] is False
    assert receipt["privacy"]["raw_audio_included"] is False
    assert "imputed_features" not in json.dumps(receipt["primary"])
    assert "top_influential_features" not in blob
    assert "enrollment" not in receipt
