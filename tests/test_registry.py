"""Every runtime artifact in models/ must load through the registry."""
from __future__ import annotations

from src.config import MODELS_DIR
from src.models.registry import (
    list_available_models,
    list_joblib_paths,
    load_holdout_samples,
    load_live_audio_model,
    load_model,
    read_joblib,
)


def test_seven_joblib_artifacts_load() -> None:
    paths = list_joblib_paths()
    assert len(paths) == 7
    assert {path.name for path in paths} == {
        "english_uci__random_forest.joblib",
        "english_uci__svm_rbf.joblib",
        "bengali_bensparx__random_forest.joblib",
        "bengali_bensparx__svm_rbf.joblib",
        "live_audio_vowel__random_forest.joblib",
        "live_audio_vowel__svm_rbf.joblib",
        "live_audio_default.joblib",
    }
    for path in paths:
        loaded = read_joblib(path)
        assert isinstance(loaded, dict)
        again = read_joblib(path)
        assert again is loaded


def test_named_models_and_holdouts() -> None:
    listed = list_available_models()
    assert len(listed) == 6
    for dataset_key, model_name in (
        ("english_uci", "random_forest"),
        ("english_uci", "svm_rbf"),
        ("bengali_bensparx", "random_forest"),
        ("bengali_bensparx", "svm_rbf"),
        ("live_audio_vowel", "random_forest"),
        ("live_audio_vowel", "svm_rbf"),
    ):
        pack = load_model(dataset_key, model_name)
        assert "pipeline" in pack
        assert pack["feature_names"]
        assert load_model(dataset_key, model_name) is pack

    for dataset_key in ("english_uci", "bengali_bensparx"):
        holdout = load_holdout_samples(dataset_key)
        assert len(holdout) > 0
        assert {"sample_id", "subject_id", "true_label"} <= set(holdout.columns)
        assert (MODELS_DIR / f"{dataset_key}__holdout_samples.csv").exists()


def test_live_pointer_uses_fixed_filename() -> None:
    pointer = read_joblib(MODELS_DIR / "live_audio_default.joblib")
    stored = str(pointer.get("artifact") or "")
    assert stored.startswith(f"{pointer['dataset_key']}__")
    live = load_live_audio_model()
    assert live["dataset_key"] == pointer["dataset_key"] == "live_audio_vowel"
    assert set(live["training_config"]["datasets"]) == {"italian", "uams", "argentina"}
    assert live["model_name"] == pointer.get("default_model")
    peer = load_live_audio_model("random_forest")
    assert peer["model_name"] == "random_forest"
    assert peer is load_live_audio_model("random_forest")
