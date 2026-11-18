"""Bad audio produces a message a user can act on instead of a crash."""
from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from app.widgets import MAX_UPLOAD_MB, friendly_error, to_wav_bytes
from src.predict import predict_from_audio


def _wav(y: np.ndarray, sr: int = 22050) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y.astype("float32"), sr, format="WAV")
    return buf.getvalue()


def test_oversized_upload_is_refused_before_decoding() -> None:
    with pytest.raises(RuntimeError, match="under 25 MB"):
        to_wav_bytes(b"\0" * (MAX_UPLOAD_MB * 1024 * 1024 + 1))


def test_silent_recording_points_at_the_microphone() -> None:
    with pytest.raises(RuntimeError, match="microphone permission"):
        to_wav_bytes(_wav(np.zeros(22050 * 3)))


def test_long_upload_is_trimmed_to_one_minute() -> None:
    sr = 8000
    t = np.arange(sr * 90) / sr
    out = to_wav_bytes(_wav(0.3 * np.sin(2 * np.pi * 150 * t), sr))
    y, out_sr = sf.read(io.BytesIO(out))
    assert len(y) / out_sr == pytest.approx(60.0, abs=0.1)


def test_undecodable_bytes_return_unreadable_status() -> None:
    result = predict_from_audio(b"this is not audio at all")
    assert result["status"] == "unreadable_audio"
    assert "5–8 second vowel" in result["message"]


def test_friendly_error_hides_unexpected_internals() -> None:
    assert "VOICESENSE_DEBUG" in friendly_error(KeyError("pd_high"))
    assert "models/" in friendly_error(FileNotFoundError("x.joblib"))
    assert friendly_error(RuntimeError("Upload a shorter clip")) == "Upload a shorter clip"
