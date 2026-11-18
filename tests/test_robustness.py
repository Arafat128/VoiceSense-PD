"""A steady synthetic vowel keeps its score when the channel gets worse."""
from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from src.features.audio import TARGET_SR
from src.predict import predict_from_audio
from training.augment import DEGRADATIONS, to_wav_bytes


def _vowel(seconds: float = 6.0, f0: float = 130.0, seed: int = 0) -> np.ndarray:
    """Harmonic source with slight pitch jitter and formant-like weighting."""
    rng = np.random.default_rng(seed)
    n = int(seconds * TARGET_SR)
    inst_f0 = f0 * (1 + 0.004 * rng.standard_normal(n).cumsum() / np.sqrt(n))
    phase = 2 * np.pi * np.cumsum(inst_f0) / TARGET_SR
    y = sum((0.9 ** k) * np.sin(k * phase) for k in range(1, 15))
    y = y / np.max(np.abs(y)) * 0.5
    return y.astype(np.float32)


def _p(y: np.ndarray) -> float:
    out = predict_from_audio(to_wav_bytes(y), require_quality_ok=False)
    return float(out["primary"]["probability_pd"])


def test_mild_degradations_move_the_score_little() -> None:
    y = _vowel()
    base = _p(y)
    rng = np.random.default_rng(1)
    for name in ("noise_snr20", "gain_minus12db", "lowpass_3k"):
        degraded = DEGRADATIONS[name](y, TARGET_SR, rng)
        assert abs(_p(degraded) - base) < 0.25, name


def test_wav_roundtrip_is_16bit_pcm() -> None:
    wav = to_wav_bytes(_vowel(1.0))
    info = sf.info(io.BytesIO(wav))
    assert info.subtype == "PCM_16" and info.samplerate == TARGET_SR
