"""Train the live sustained-vowel screening model.

The app records a sustained vowel, so the model is trained on sustained
vowels (Italian VA..VU), not on reading passages. Features come from the same
extractor the app uses (src.features.audio.extract_features) plus the shared
ratio features (add_engineered).

Labels: Parkinson's vs ELDERLY healthy controls. Young healthy controls are
excluded by default because age alone separates them from Parkinson's
speakers (--include-young-hc turns them back on; the Italian young controls
have no vowel recordings, so the flag only matters with --italian-task connected).

Run: python -m training.train_audio --data-dir <dataset root>
"""
from __future__ import annotations

import argparse
import json
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from training.common import (
    DISCLAIMER,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    SPLIT_METHOD,
    resolve_data_dir,
    stratified_group_holdout,
)
from training.featurize import CLINICAL_FEATURES, build_features
from training.vowel_data import describe, load_manifest

# Config flag: young healthy controls are an age confound, so they are off.
INCLUDE_YOUNG_HC = False

DECISION_THRESHOLDS = {"pd_high": 0.72, "hc_high": 0.42}


def _models() -> dict[str, Pipeline]:
    rf = RandomForestClassifier(
        n_estimators=500, max_features="sqrt", class_weight="balanced_subsample",
        random_state=RANDOM_STATE, n_jobs=-1, min_samples_leaf=3, max_depth=8,
    )
    svm = CalibratedClassifierCV(
        SVC(kernel="rbf", C=5.0, gamma="scale", class_weight="balanced", random_state=RANDOM_STATE),
        method="sigmoid", cv=3,
    )
    return {
        name: Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("clf", clf)])
        for name, clf in (("random_forest", rf), ("svm_rbf", svm))
    }


def train_live_audio_model(data_dir, *, include_young_hc: bool = INCLUDE_YOUNG_HC,
                           italian_task: str = "vowels") -> dict[str, Any]:
    manifest = load_manifest(data_dir, ("italian",), include_young_hc=include_young_hc,
                             italian_task=italian_task)
    table = build_features(manifest, ("clean",), cache_name="vowels" if italian_task == "vowels" else "italian_connected")
    table = table[table["error"].fillna("") == ""].reset_index(drop=True)
    features = list(CLINICAL_FEATURES)
    X = table[features]
    y = table["label"].to_numpy(int)
    groups = table["subject"].to_numpy()
    train_idx, test_idx = stratified_group_holdout(y, groups)
    results: dict[str, Any] = {
        "dataset": "Italian sustained vowels, PD vs elderly healthy",
        "config": {"include_young_hc": include_young_hc, "italian_task": italian_task},
        "datasets": describe(manifest),
        "n_features": len(features),
        "split": {"method": SPLIT_METHOD, "train_samples": len(train_idx), "test_samples": len(test_idx)},
        "models": {},
        "disclaimer": DISCLAIMER,
        "feature_names": features,
    }
    for name, pipe in _models().items():
        pipe.fit(X.iloc[train_idx], y[train_idx])
        prob = pipe.predict_proba(X.iloc[test_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)
        yt = y[test_idx]
        metrics = {
            "accuracy": float(accuracy_score(yt, pred)),
            "precision": float(precision_score(yt, pred, zero_division=0)),
            "recall": float(recall_score(yt, pred, zero_division=0)),
            "f1": float(f1_score(yt, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(yt, prob)),
        }
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_f1 = cross_val_score(pipe, X, y, groups=groups, cv=cv, scoring="f1")
        metrics["cv_f1_mean"], metrics["cv_f1_std"] = float(cv_f1.mean()), float(cv_f1.std())
        pipe.fit(X, y)
        _save(name, pipe, X, y, features, metrics, DECISION_THRESHOLDS, results["config"])
        results["models"][name] = {"metrics": metrics}
        print(name, json.dumps(metrics))
    _write_pointer_and_report(results, features)
    return results


def _artifact(name, pipe, X, y, features, metrics, thresholds, config, extra=None) -> dict:
    importances = []
    clf = pipe.named_steps["clf"]
    est = getattr(clf, "calibrated_classifiers_", None)
    forest = clf if hasattr(clf, "feature_importances_") else None
    if forest is None and est:
        inner = [c.estimator for c in est if hasattr(c.estimator, "feature_importances_")]
        if inner:
            imp = np.mean([c.feature_importances_ for c in inner], axis=0)
            importances = [{"feature": features[i], "importance": float(imp[i])} for i in np.argsort(imp)[::-1]]
    elif forest is not None:
        imp = forest.feature_importances_
        importances = [{"feature": features[i], "importance": float(imp[i])} for i in np.argsort(imp)[::-1]]
    art = {
        "pipeline": pipe,
        "feature_names": features,
        "dataset_key": "live_audio_italian",
        "model_name": name,
        "language": "Language-independent sustained vowel (trained on Italian vowels)",
        "display_name": "Live vowel model (Italian sustained vowels)",
        "positive_name": "Parkinson's Disease (PD)",
        "negative_name": "Healthy Control (HC)",
        "feature_importances": importances,
        "train_medians": X.median(numeric_only=True).to_dict(),
        "train_means": X.mean(numeric_only=True).to_dict(),
        "train_stds": X.std(ddof=0, numeric_only=True).to_dict(),
        "class_means": {
            "PD": X[y == 1].mean(numeric_only=True).to_dict(),
            "HC": X[y == 0].mean(numeric_only=True).to_dict(),
        },
        "metrics": metrics,
        "disclaimer": DISCLAIMER,
        "input_type": "audio_wav",
        "extraction": "src.features.audio.extract_features.common + add_engineered",
        "quality_gates": {"min_duration_sec": 1.0, "min_voiced_fraction": 0.15, "min_snr_db": 5.0},
        "decision_thresholds": thresholds,
        "feature_policy": "clinical_dysphonia_only_no_mfcc",
        "training_config": config,
        "false_positive_note": (
            "Live mics differ from research recordings. PD is reported only when "
            "probability is high; mid-range scores are labeled Uncertain."
        ),
    }
    if extra:
        art.update(extra)
    return art


def _save(name, pipe, X, y, features, metrics, thresholds, config, extra=None) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    art = _artifact(name, pipe, X, y, features, metrics, thresholds, config, extra)
    joblib.dump(art, MODELS_DIR / f"{art['dataset_key']}__{name}.joblib", compress=3)


def _write_pointer_and_report(results: dict, features: list[str], dataset_key: str = "live_audio_italian") -> None:
    default = "random_forest"
    pointer = {
        "default_model": default,
        "artifact": f"{dataset_key}__{default}.joblib",
        "feature_names": features,
        "summary_metrics": results["models"][default]["metrics"],
    }
    joblib.dump(pointer, MODELS_DIR / "live_audio_default.joblib")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results["app_default_model"] = default
    with open(REPORTS_DIR / "live_audio_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the live sustained-vowel model.")
    parser.add_argument("--data-dir", default=None, help="Dataset root. Overrides VOICESENSE_DATA_DIR.")
    parser.add_argument("--include-young-hc", action="store_true",
                        help="Add young healthy controls (age confound; off by default).")
    parser.add_argument("--italian-task", choices=("vowels", "connected"), default="vowels",
                        help="vowels = VA..VU (default). connected = old reading-task baseline.")
    args = parser.parse_args(argv)
    train_live_audio_model(resolve_data_dir(args.data_dir), include_young_hc=args.include_young_hc,
                           italian_task=args.italian_task)
    print(DISCLAIMER)


if __name__ == "__main__":
    main()
