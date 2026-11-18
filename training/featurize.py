"""Extract and cache Praat clinical features for a clip manifest.

Rows are keyed by (clip_id, variant). variant 'clean' is the original file;
'aug0', 'aug1', ... are random training augmentations; any key of
``augment.DEGRADATIONS`` is a fixed degradation for the robustness report.
The cache lives under .cache/ and is never committed.
"""
from __future__ import annotations

import hashlib
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CACHE_DIR
from src.features.audio import TARGET_SR, add_engineered, extract_features

# Classical dysphonia measures only. MFCC and spectral shape track the
# recording channel more than the voice, so they are left out.
CLINICAL_FEATURES = [
    "praat_f0_mean", "praat_f0_min", "praat_f0_max", "praat_f0_std",
    "praat_local_jitter", "praat_local_abs_jitter", "praat_rap_jitter",
    "praat_ppq5_jitter", "praat_ddp_jitter",
    "praat_local_shimmer", "praat_local_db_shimmer", "praat_apq3", "praat_apq5",
    "praat_apq11", "praat_dda",
    "praat_hnr", "praat_nhr", "praat_ppe", "praat_rpde", "praat_dfa",
    "praat_spread1", "praat_spread2", "praat_d2",
    "praat_mean_intensity", "praat_min_intensity", "praat_max_intensity",
    "praat_mean_period", "praat_std_period", "praat_mean_autocorr_harmonicity",
    "voiced_fraction",
    "f0_cv", "f0_range", "intensity_range", "jitter_shimmer_product",
]
QUALITY_COLUMNS = ["q_ok", "q_duration_sec", "q_snr_db", "q_voiced_fraction"]


def _seed(clip_id: str, variant: str) -> int:
    return zlib.crc32(f"{clip_id}|{variant}".encode()) & 0x7FFFFFFF


def _features_for(path: str, clip_id: str, variant: str) -> dict:
    if variant == "clean":
        source: str | bytes = path
    else:
        from training.augment import DEGRADATIONS, load_mono, random_chain, to_wav_bytes

        y = load_mono(path)
        seed = _seed(clip_id, variant)
        if variant in DEGRADATIONS:
            y = DEGRADATIONS[variant](y, TARGET_SR, np.random.default_rng(seed))
        elif variant.startswith("aug"):
            y = random_chain(y, TARGET_SR, seed)
        else:
            raise ValueError(f"Unknown variant {variant}")
        source = to_wav_bytes(y)
    try:
        res = extract_features(source)
    except Exception as exc:  # noqa: BLE001 - one bad clip should not stop the run
        return {"clip_id": clip_id, "variant": variant, "error": str(exc)[:200]}
    feats = add_engineered(res.common)
    row = {"clip_id": clip_id, "variant": variant, "error": ""}
    row.update({name: feats.get(name, np.nan) for name in CLINICAL_FEATURES})
    row.update(
        {
            "q_ok": int(res.quality.ok),
            "q_duration_sec": res.quality.duration_sec,
            "q_snr_db": res.quality.snr_db_proxy,
            "q_voiced_fraction": res.quality.voiced_fraction,
        }
    )
    return row


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"praat_{name}.csv"


def build_features(
    manifest: pd.DataFrame,
    variants: list[str] | tuple[str, ...] = ("clean",),
    *,
    cache_name: str = "features",
    n_jobs: int = -1,
    verbose: bool = True,
) -> pd.DataFrame:
    """Return one row per (clip, variant) with CLINICAL_FEATURES, cached on disk."""
    from joblib import Parallel, delayed

    cache = _cache_path(cache_name)
    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["clip_id", "variant"])
    have = set(zip(done["clip_id"], done["variant"]))
    todo = [
        (row.path, row.clip_id, variant)
        for row in manifest.itertuples()
        for variant in variants
        if (row.clip_id, variant) not in have
    ]
    if todo:
        if verbose:
            print(f"Extracting {len(todo)} clip variants into {cache.name} ...")
        new = Parallel(n_jobs=n_jobs, batch_size=4, verbose=5 if verbose else 0)(
            delayed(_features_for)(p, c, v) for p, c, v in todo
        )
        done = pd.concat([done, pd.DataFrame(new)], ignore_index=True)
        done.to_csv(cache, index=False)
    wanted = manifest[["clip_id"]].merge(
        pd.DataFrame({"variant": list(variants)}), how="cross"
    )
    out = wanted.merge(done, on=["clip_id", "variant"], how="left")
    out = out.merge(manifest, on="clip_id", how="left")
    return out


def manifest_fingerprint(manifest: pd.DataFrame) -> str:
    joined = "|".join(sorted(manifest["clip_id"]))
    return hashlib.sha1(joined.encode()).hexdigest()[:12]
