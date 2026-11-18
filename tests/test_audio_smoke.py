"""Synthetic tone goes through WAV conversion and a live prediction."""
from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from app.widgets import to_wav_bytes
from src.predict import predict_from_audio


def _tone_wav(seconds: float = 5.0, freq: float = 150.0, sr: int = 22050) -> bytes:
    samples = int(sr * seconds)
    time = np.linspace(0, seconds, samples, endpoint=False)
    tone = (0.2 * np.sin(2 * np.pi * freq * time)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, tone, sr, format="WAV")
    return buffer.getvalue()


def test_to_wav_bytes_rejects_garbage() -> None:
    with pytest.raises(RuntimeError, match="WAV"):
        to_wav_bytes(b"this is not audio")


def test_to_wav_bytes_keeps_a_tone() -> None:
    wav = to_wav_bytes(_tone_wav())
    assert wav[:4] == b"RIFF"


def test_synthetic_tone_reaches_a_prediction() -> None:
    result = predict_from_audio(
        _tone_wav(),
        language_mode="english",
        require_quality_ok=False,
    )
    assert result["status"] == "ok"
    primary = result["primary"]
    assert primary["decision_code"] in {"PD", "HC", "UNCERTAIN"}
    assert 0.0 <= float(primary["probability_pd"]) <= 1.0
    assert primary["explanation"]["summary_lines"]
