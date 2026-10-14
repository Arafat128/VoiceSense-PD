"""Optional Primus zkTLS enrollment — identity proofs only.

Never receives, stores, or hashes voice recordings. Attested Web2 fields are
reduced to fingerprints (SHA-256). Plaintext handles are stripped if a
developer forgets to set SHA256 conditions on the template.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .privacy import utc_now_iso

ENROLLMENT_SCHEMA = "voicesense.enrollment.v1"
ENROLLMENT_DIR = Path.home() / ".voicesense"
ENROLLMENT_PATH = ENROLLMENT_DIR / "enrollment.json"
RECIPIENT_PATH = ENROLLMENT_DIR / "recipient.txt"

HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        root = Path(__file__).resolve().parents[1]
        load_dotenv(root / ".env")
    except Exception:
        pass


def primus_settings() -> Dict[str, str]:
    _load_dotenv()
    return {
        "app_id": (os.getenv("PRIMUS_APP_ID") or "").strip(),
        "app_secret": (os.getenv("PRIMUS_APP_SECRET") or "").strip(),
        "uniqueness_template_id": (
            os.getenv("PRIMUS_UNIQUENESS_TEMPLATE_ID")
            or os.getenv("PRIMUS_TEMPLATE_ID")
            or ""
        ).strip(),
        "identity_field": (os.getenv("PRIMUS_IDENTITY_FIELD") or "screen_name").strip(),
        "age_template_id": (os.getenv("PRIMUS_AGE_TEMPLATE_ID") or "").strip(),
        "age_field": (os.getenv("PRIMUS_AGE_FIELD") or "age").strip(),
        "age_op": (os.getenv("PRIMUS_AGE_OP") or ">=").strip(),
        "age_value": (os.getenv("PRIMUS_AGE_VALUE") or "18").strip(),
        "mock": (os.getenv("PRIMUS_MOCK") or "").strip() in ("1", "true", "TRUE", "yes"),
    }


def configured_for_live_primus() -> bool:
    s = primus_settings()
    return bool(s["app_id"] and s["uniqueness_template_id"])


def local_recipient() -> str:
    """Random 0x-address used only as a Primus recipient id — not a user wallet."""
    ENROLLMENT_DIR.mkdir(parents=True, exist_ok=True)
    if RECIPIENT_PATH.exists():
        value = RECIPIENT_PATH.read_text(encoding="utf-8").strip()
        if value.startswith("0x") and len(value) == 42:
            return value
    value = "0x" + secrets.token_hex(20)
    RECIPIENT_PATH.write_text(value + "\n", encoding="utf-8")
    return value


def load_enrollment() -> Optional[Dict[str, Any]]:
    if not ENROLLMENT_PATH.exists():
        return None
    try:
        data = json.loads(ENROLLMENT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("schema") != ENROLLMENT_SCHEMA:
        return None
    return data


def save_enrollment(record: Dict[str, Any]) -> Dict[str, Any]:
    ENROLLMENT_DIR.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record["schema"] = ENROLLMENT_SCHEMA
    record.setdefault("saved_at", utc_now_iso())
    ENROLLMENT_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def clear_enrollment() -> None:
    if ENROLLMENT_PATH.exists():
        ENROLLMENT_PATH.unlink()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sanitize_attested_data(data: Any) -> Tuple[Dict[str, Any], bool]:
    """Keep hashes/booleans; replace any leftover plaintext with SHA-256."""
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return {"value": _sha256_text(data)}, True
    if not isinstance(data, dict):
        return {"value": _sha256_text(str(data))}, True

    sanitized: Dict[str, Any] = {}
    stripped = False
    for key, raw in data.items():
        if isinstance(raw, bool):
            sanitized[str(key)] = raw
            continue
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            sanitized[str(key)] = _sha256_text(str(raw))
            stripped = True
            continue
        text = str(raw).strip()
        lower = text.lower()
        if lower in ("true", "false"):
            sanitized[str(key)] = lower == "true"
        elif HEX64.match(text):
            sanitized[str(key)] = text.lower()
        else:
            sanitized[str(key)] = _sha256_text(text)
            stripped = True
    return sanitized, stripped


def fingerprint_from_sanitized(sanitized: Dict[str, Any], template_id: str = "") -> str:
    payload = json.dumps(
        {"template_id": template_id, "data": sanitized},
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256_text(payload)


def _extract_attestation_object(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, str):
        payload = json.loads(payload)
    if isinstance(payload, list) and payload:
        payload = payload[0]
    if not isinstance(payload, dict):
        raise ValueError("Attestation must be a JSON object")
    if "attestation" in payload and isinstance(payload["attestation"], dict):
        inner = dict(payload["attestation"])
        for k in ("attestor", "signature", "signatures", "taskId"):
            if k in payload and k not in inner:
                inner[k] = payload[k]
        return inner
    return payload


def parse_attestation(
    payload: Any,
    *,
    expected_template_id: Optional[str] = None,
    source: str = "paste",
    sdk_verified: bool = False,
    purpose: str = "uniqueness",
) -> Dict[str, Any]:
    att = _extract_attestation_object(payload)
    data = att.get("data")
    sanitized, stripped = sanitize_attested_data(data)

    template_id = (
        expected_template_id
        or att.get("templateId")
        or att.get("template_id")
        or ""
    )
    fingerprint = fingerprint_from_sanitized(sanitized, template_id)

    signatures = att.get("signatures") or []
    if att.get("signature") and not signatures:
        signatures = [att.get("signature")]

    if not sdk_verified and source == "paste":
        if not signatures and not att.get("attestors"):
            raise ValueError(
                "This JSON does not look like a Primus attestation "
                "(missing signatures/attestors)."
            )

    age_ok: Optional[bool] = None
    if purpose == "age":
        bools = [v for v in sanitized.values() if isinstance(v, bool)]
        if bools:
            age_ok = all(bools)
        elif any(str(v).lower() == "true" for v in sanitized.values()):
            age_ok = True
        elif sanitized:
            age_ok = False

    record: Dict[str, Any] = {
        "schema": ENROLLMENT_SCHEMA,
        "purpose": purpose,
        "identity_fingerprint": fingerprint,
        "age_ok": age_ok,
        "template_id": template_id,
        "sanitized_fields": sorted(sanitized.keys()),
        "plaintext_stripped": stripped,
        "sdk_verified": bool(sdk_verified),
        "source": source,
        "mock": False,
        "recipient": att.get("recipient") or "",
        "timestamp": att.get("timestamp") or "",
        "verified_at": utc_now_iso(),
        "no_raw_identity": True,
        "no_audio": True,
    }
    return record


def merge_enrollment(
    existing: Optional[Dict[str, Any]],
    incoming: Dict[str, Any],
) -> Dict[str, Any]:
    """Combine uniqueness + age proofs without keeping raw identity."""
    base = dict(existing or {})
    purpose = incoming.get("purpose") or "uniqueness"
    if purpose == "age":
        base["age_ok"] = incoming.get("age_ok")
        base["age_template_id"] = incoming.get("template_id")
        base["age_verified_at"] = incoming.get("verified_at")
        base["age_source"] = incoming.get("source")
        base["age_mock"] = bool(incoming.get("mock"))
    else:
        for key in (
            "identity_fingerprint",
            "template_id",
            "sanitized_fields",
            "plaintext_stripped",
            "sdk_verified",
            "source",
            "recipient",
            "timestamp",
            "mock",
        ):
            if key in incoming:
                base[key] = incoming[key]
        base["verified_at"] = incoming.get("verified_at")
        if incoming.get("age_ok") is not None and base.get("age_ok") is None:
            base["age_ok"] = incoming.get("age_ok")

    base["schema"] = ENROLLMENT_SCHEMA
    base["no_raw_identity"] = True
    base["no_audio"] = True
    base["enrolled"] = bool(base.get("identity_fingerprint"))
    return base


def mock_uniqueness_record() -> Dict[str, Any]:
    token = secrets.token_hex(32)
    return {
        "schema": ENROLLMENT_SCHEMA,
        "purpose": "uniqueness",
        "identity_fingerprint": _sha256_text("mock|" + token),
        "age_ok": None,
        "template_id": "mock",
        "sanitized_fields": ["mock"],
        "plaintext_stripped": False,
        "sdk_verified": False,
        "source": "mock",
        "mock": True,
        "recipient": local_recipient(),
        "timestamp": "",
        "verified_at": utc_now_iso(),
        "no_raw_identity": True,
        "no_audio": True,
        "enrolled": True,
    }


def public_enrollment_view(record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not record:
        return None
    return {
        "enrolled": bool(record.get("identity_fingerprint")),
        "identity_fingerprint": record.get("identity_fingerprint"),
        "age_ok": record.get("age_ok"),
        "source": record.get("source"),
        "mock": bool(record.get("mock")),
        "verified_at": record.get("verified_at"),
        "plaintext_stripped": bool(record.get("plaintext_stripped")),
        "sdk_verified": bool(record.get("sdk_verified")),
    }
