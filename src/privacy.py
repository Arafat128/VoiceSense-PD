"""On-device privacy helpers for VoiceSense.

Voice recordings and raw clinical feature tables are treated as sensitive.
This module never writes audio to disk. Screening receipts omit WAVs and
upload filenames.
"""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

from .config import DISCLAIMER

RECEIPT_SCHEMA = "voicesense.screening.v1"

AUDIO_SESSION_KEYS = (
    "last_audio_raw",
    "last_audio_source",
)


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def is_local_runtime() -> tuple[bool, str]:
    """
    Best-effort local vs hosted detection.

    Local Streamlit is the only mode that can honestly claim the recording
    never left the user's machine. Hosted Streamlit still uploads audio to
    the Python process on a remote server.
    """
    if os.getenv("VOICESENSE_FORCE_CLOUD") == "1":
        return False, "forced-cloud"
    if os.getenv("VOICESENSE_FORCE_LOCAL") == "1":
        return True, "forced-local"

    if os.getenv("STREAMLIT_RUNTIME_ENV", "").lower() == "cloud":
        return False, "streamlit-cloud"
    if os.getenv("IS_STREAMLIT_CLOUD"):
        return False, "streamlit-cloud"

    host = ""
    try:
        import streamlit as st

        if hasattr(st, "context") and getattr(st.context, "headers", None):
            host = (st.context.headers.get("host") or "").lower()
    except Exception:  # noqa: BLE001
        host = ""

    if any(token in host for token in ("streamlit.app", "share.streamlit.io", "streamlitusercontent")):
        return False, host or "hosted"

    return True, "local"


def wipe_audio_keys(session_state: Any) -> None:
    """Remove any leftover raw audio from Streamlit session_state."""
    for key in AUDIO_SESSION_KEYS:
        if key in session_state:
            del session_state[key]


def bump_audio_nonce(session_state: Any) -> int:
    """Change widget keys so Streamlit drops in-memory mic/upload buffers."""
    wipe_audio_keys(session_state)
    next_nonce = int(session_state.get("audio_nonce", 0)) + 1
    session_state["audio_nonce"] = next_nonce
    return next_nonce


def redact_source_kind(source_kind: str | None) -> str:
    if not source_kind:
        return "unknown"
    if source_kind.startswith("upload"):
        return "upload"
    if source_kind in ("microphone", "session"):
        return source_kind
    return "other"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items() if not isinstance(v, bytes)}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value if not isinstance(v, bytes)]
    try:
        return json.loads(json.dumps(value, default=str))
    except TypeError:
        return str(value)


def _plain_for_receipt(plain: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep only the everyday sentences. Never the raw feature table."""
    if not plain:
        return None
    series = plain.get("series") or {}
    return {
        "headline": plain.get("headline"),
        "meaning": plain.get("meaning"),
        "chance": plain.get("chance"),
        "heard": list(plain.get("heard") or []),
        "trust": plain.get("trust_label"),
        "trust_detail": plain.get("trust_detail"),
        "series_headline": series.get("headline"),
        "series_detail": series.get("detail"),
        "next_steps": list(plain.get("next_steps") or []),
    }


def make_screening_receipt(
    result: dict[str, Any],
    *,
    source_kind: str | None = None,
    include_features: bool = False,
    local_runtime: bool = True,
) -> dict[str, Any]:
    """Build an exportable screening record that cannot contain audio bytes."""
    primary = dict(result.get("primary") or {})
    explanation = dict(primary.get("explanation") or {})
    if not include_features:
        explanation.pop("top_influential_features", None)
        primary["explanation"] = {
            "decision": explanation.get("decision"),
            "summary_lines": explanation.get("summary_lines") or [],
            "why_pd": [
                {"feature": x.get("feature"), "comparison": x.get("comparison")}
                for x in (explanation.get("why_pd") or [])[:6]
            ],
            "why_not_pd": [
                {"feature": x.get("feature"), "comparison": x.get("comparison")}
                for x in (explanation.get("why_not_pd") or [])[:6]
            ],
        }
    else:
        primary["explanation"] = explanation

    for drop in ("imputed_features",):
        primary.pop(drop, None)

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "generated_at": utc_now_iso(),
        "runtime": "local" if local_runtime else "hosted",
        "audio_retained": False,
        "source_kind": redact_source_kind(source_kind),
        "disclaimer": DISCLAIMER,
        "status": result.get("status"),
        "language_mode": result.get("language_mode"),
        "quality": result.get("quality"),
        "primary": {
            "prediction": primary.get("prediction"),
            "decision_code": primary.get("decision_code"),
            "pred_label": primary.get("pred_label"),
            "probability_pd": primary.get("probability_pd"),
            "probability_hc": primary.get("probability_hc"),
            "adjusted_confidence": primary.get("adjusted_confidence"),
            "thresholds": primary.get("thresholds"),
            "completeness": primary.get("completeness"),
            "model_name": primary.get("model_name"),
            "display_name": primary.get("display_name"),
            "explanation": primary.get("explanation"),
        },
        "secondary": result.get("secondary") or {},
        "warnings": result.get("warnings") or [],
        "plain_language": _plain_for_receipt(result.get("plain")),
        "privacy": {
            "raw_audio_included": False,
            "upload_filename_included": False,
            "identity_plaintext_included": False,
            "note": (
                "This receipt is a user-controlled export. "
                "The recording itself is not stored in the receipt or on disk."
            ),
        },
    }
    if include_features:
        receipt["extraction_preview"] = result.get("extraction_preview") or {}
    return _json_safe(receipt)


def receipt_json(receipt: dict[str, Any]) -> str:
    return json.dumps(receipt, indent=2, sort_keys=False)


def assert_receipt_has_no_audio(receipt: dict[str, Any]) -> None:
    blob = json.dumps(receipt)
    if "RIFF" in blob or "last_audio_raw" in blob:
        raise ValueError("Receipt appears to contain audio material")
    dumped = json.dumps(receipt, default=lambda o: "NONJSON")
    if "NONJSON" in dumped:
        raise ValueError("Receipt contains non-JSON values")
