"""Load every runtime model and holdout table once per process.

``functools.lru_cache`` keeps the joblib objects alive across Streamlit
reruns (imported modules are not re-executed) and works in pytest with no
Streamlit runtime. The app must load models through this module only.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ..config import MODELS_DIR

# live_audio_default.joblib is a pointer. Older copies store an absolute path
# from the machine that trained them. The model filename is fixed, so the
# pointer's default_model field selects it and that path is ignored.


@functools.lru_cache(maxsize=32)
def _read_joblib(path_str: str) -> dict[str, Any]:
    return joblib.load(path_str)


def read_joblib(path: Path) -> dict[str, Any]:
    return _read_joblib(str(Path(path).resolve()))


def artifact_path(dataset_key: str, model_name: str = "random_forest") -> Path:
    return MODELS_DIR / f"{dataset_key}__{model_name}.joblib"


def load_model(dataset_key: str, model_name: str = "random_forest") -> dict[str, Any]:
    path = artifact_path(dataset_key, model_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}. Run training first: python -m training.train_tables"
        )
    return read_joblib(path)


def list_available_models() -> list[dict[str, str]]:
    out = []
    for path in sorted(MODELS_DIR.glob("*.joblib")):
        if "__" in path.stem:
            dataset_key, model_name = path.stem.split("__", 1)
            out.append(
                {
                    "dataset_key": dataset_key,
                    "model_name": model_name,
                    "path": str(path),
                }
            )
    return out


def list_joblib_paths() -> list[Path]:
    return sorted(MODELS_DIR.glob("*.joblib"))


@functools.lru_cache(maxsize=8)
def _read_holdout(dataset_key: str) -> pd.DataFrame:
    path = MODELS_DIR / f"{dataset_key}__holdout_samples.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Holdout samples not found: {path}. Train models first."
        )
    return pd.read_csv(path)


def load_holdout_samples(dataset_key: str) -> pd.DataFrame:
    return _read_holdout(dataset_key).copy()


def live_model_path(model_name: str | None = None) -> Path:
    """Fixed filename under models/. Does not follow stored absolute paths."""
    name = model_name
    if not name:
        pointer = MODELS_DIR / "live_audio_default.joblib"
        if pointer.exists():
            packed = read_joblib(pointer)
            name = str(packed.get("default_model") or "random_forest")
        else:
            name = "random_forest"
    path = MODELS_DIR / f"live_audio_italian__{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Live audio model not found: {path}. "
            "Expected live_audio_italian__random_forest.joblib. "
            "Run: python -m training.train_audio"
        )
    return path


def load_live_audio_model(model_name: str | None = None) -> dict[str, Any]:
    return read_joblib(live_model_path(model_name))
