"""Optional feature sets for the comparison report: openSMILE eGeMAPS and WavLM.

Neither is needed by the app. Install with requirements-research.txt.
Both read the same 16 kHz mono signal (trimmed, peak-normalized, first 12 s)
so the comparison is about the representation, not the preprocessing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CACHE_DIR

SR = 16000
MAX_SECONDS = 12.0
WAVLM_MODEL = "microsoft/wavlm-base"
WAVLM_LAYER = 6  # middle layers carry more voice-quality detail than the last one


def _load(path: str) -> np.ndarray:
    import librosa

    y, _ = librosa.load(path, sr=SR, mono=True)
    yt, _ = librosa.effects.trim(y, top_db=30)
    if yt.size >= int(0.3 * SR):
        y = yt
    y = y[: int(MAX_SECONDS * SR)]
    return (0.95 * y / (np.max(np.abs(y)) + 1e-9)).astype(np.float32)


def egemaps_table(manifest: pd.DataFrame, cache_name: str = "vowels") -> pd.DataFrame:
    import opensmile

    cache = CACHE_DIR / f"egemaps_{cache_name}.csv"
    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["clip_id"])
    todo = manifest[~manifest["clip_id"].isin(done["clip_id"])]
    if len(todo):
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        rows = []
        for i, r in enumerate(todo.itertuples(), 1):
            feats = smile.process_signal(_load(r.path), SR).iloc[0].to_dict()
            feats["clip_id"] = r.clip_id
            rows.append(feats)
            if i % 100 == 0:
                print(f"  eGeMAPS {i}/{len(todo)}")
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        done.to_csv(cache, index=False)
    return manifest[["clip_id"]].merge(done, on="clip_id", how="left")


def wavlm_table(manifest: pd.DataFrame, cache_name: str = "vowels") -> pd.DataFrame:
    import torch
    from transformers import AutoFeatureExtractor, WavLMModel

    cache = CACHE_DIR / f"wavlm_l{WAVLM_LAYER}_{cache_name}.csv"
    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["clip_id"])
    todo = manifest[~manifest["clip_id"].isin(done["clip_id"])]
    if len(todo):
        torch.set_num_threads(8)
        fe = AutoFeatureExtractor.from_pretrained(WAVLM_MODEL)
        model = WavLMModel.from_pretrained(WAVLM_MODEL).eval()
        rows = []
        with torch.inference_mode():
            for i, r in enumerate(todo.itertuples(), 1):
                inputs = fe(_load(r.path), sampling_rate=SR, return_tensors="pt")
                hidden = model(**inputs, output_hidden_states=True).hidden_states[WAVLM_LAYER][0]
                vec = hidden.mean(dim=0).numpy()
                row = {f"wavlm_{j}": float(v) for j, v in enumerate(vec)}
                row["clip_id"] = r.clip_id
                rows.append(row)
                if i % 100 == 0:
                    print(f"  WavLM {i}/{len(todo)}")
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        done.to_csv(cache, index=False)
    return manifest[["clip_id"]].merge(done, on="clip_id", how="left")


def feature_columns(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c != "clip_id"]


__all__ = ["egemaps_table", "wavlm_table", "feature_columns", "Path"]
