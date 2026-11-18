"""Audio augmentation for training and fixed degradations for robustness tests.

Training uses random chains (noise, gain, low-pass, phone band, reverb) so the
model sees the channel changes it will meet on home microphones and phones.
The robustness report uses a fixed, seeded list so runs are comparable.
"""
from __future__ import annotations

import io
from collections.abc import Callable

import numpy as np
import soundfile as sf

from src.features.audio import TARGET_SR


def _phone_band(y: np.ndarray, sr: int) -> np.ndarray:
    """300-3400 Hz band and an 8 kHz round trip, like a voicemail line."""
    import librosa
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, [300, 3400], btype="bandpass", fs=sr, output="sos")
    band = sosfiltfilt(sos, y).astype(np.float32)
    narrow = librosa.resample(band, orig_sr=sr, target_sr=8000)
    return librosa.resample(narrow, orig_sr=8000, target_sr=sr).astype(np.float32)


def _reverb(y: np.ndarray, sr: int, rt60: float, rng: np.random.Generator) -> np.ndarray:
    """Convolve with a synthetic exponentially decaying room response."""
    from scipy.signal import fftconvolve

    n = int(rt60 * sr)
    t = np.arange(n) / sr
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    ir[0] = 1.0
    ir /= np.sqrt(np.sum(ir**2))
    wet = fftconvolve(y, ir)[: len(y)]
    return (0.6 * y + 0.4 * wet).astype(np.float32)


def _add_noise(y: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    power = float(np.mean(y**2)) + 1e-12
    noise = rng.standard_normal(len(y)) * np.sqrt(power / (10 ** (snr_db / 10)))
    return (y + noise).astype(np.float32)


def _lowpass(y: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    sos = butter(6, cutoff, btype="lowpass", fs=sr, output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


def _gain(y: np.ndarray, db: float) -> np.ndarray:
    return np.clip(y * (10 ** (db / 20)), -1.0, 1.0).astype(np.float32)


# Fixed degradations used by the robustness report and tests.
DEGRADATIONS: dict[str, Callable[[np.ndarray, int, np.random.Generator], np.ndarray]] = {
    "noise_snr20": lambda y, sr, rng: _add_noise(y, 20.0, rng),
    "noise_snr10": lambda y, sr, rng: _add_noise(y, 10.0, rng),
    "gain_minus12db": lambda y, sr, rng: _gain(y, -12.0),
    "lowpass_3k": lambda y, sr, rng: _lowpass(y, sr, 3000.0),
    "phone_band_8k": lambda y, sr, rng: _phone_band(y, sr),
    "reverb_rt60_0.4": lambda y, sr, rng: _reverb(y, sr, 0.4, rng),
}


def random_chain(y: np.ndarray, sr: int, seed: int) -> np.ndarray:
    """One random training augmentation. Uses audiomentations where it fits."""
    from audiomentations import AddGaussianSNR, Compose, Gain, LowPassFilter

    rng = np.random.default_rng(seed)
    chain = Compose(
        [
            AddGaussianSNR(min_snr_db=10.0, max_snr_db=35.0, p=0.6),
            Gain(min_gain_db=-12.0, max_gain_db=6.0, p=0.5),
            LowPassFilter(min_cutoff_freq=3000.0, max_cutoff_freq=7500.0, p=0.3),
        ]
    )
    np.random.seed(seed % (2**32 - 1))
    out = chain(samples=y.astype(np.float32), sample_rate=sr)
    if rng.random() < 0.3:
        out = _phone_band(out, sr)
    if rng.random() < 0.3:
        out = _reverb(out, sr, float(rng.uniform(0.2, 0.6)), rng)
    peak = float(np.max(np.abs(out))) + 1e-9
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32)


def load_mono(path: str, sr: int = TARGET_SR) -> np.ndarray:
    import librosa

    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def to_wav_bytes(y: np.ndarray, sr: int = TARGET_SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()
