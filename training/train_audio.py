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

from src.config import VOWEL_DATASETS
from training.common import DISCLAIMER, MODELS_DIR, REPORTS_DIR, resolve_data_dir
from training.evaluate import bootstrap_ci, derive_thresholds, fit_final, repeated_grouped_oof
from training.featurize import CLINICAL_FEATURES, build_features
from training.vowel_data import describe, load_manifest

# Config flag: young healthy controls are an age confound, so they are off.
INCLUDE_YOUNG_HC = False

# Old fixed cut-offs, kept only for --fixed-thresholds comparisons.
DECISION_THRESHOLDS = {"pd_high": 0.72, "hc_high": 0.42}

MODEL_NAMES = ("random_forest", "svm_rbf")

# All three public vowel corpora. Leave-one-dataset-out showed that adding
# corpora raises the score on a corpus the model never saw (reports/).
DEFAULT_DATASETS = ("italian", "uams", "argentina")
# Same age rule as dropping young controls: where ages are known (UAMS),
# healthy speakers under 50 are left out.
MIN_HC_AGE = 50.0
PROTOCOL = "speaker-grouped stratified 5-fold CV, repeated {repeats}x; 95% CI = 1000x speaker bootstrap"


def _flat_metrics(ci: dict, repeats_auc: list[float], protocol: str) -> dict:
    """Compact metric block stored in the artifact and shown in the app."""
    out = {"protocol": protocol}
    for level in ("clip", "person"):
        for key in ("auc", "balanced_accuracy", "sensitivity", "specificity"):
            out[f"{level}_{key}"] = ci[level][key]["value"]
            out[f"{level}_{key}_ci95"] = ci[level][key]["ci95"]
        out[f"{level}_n"] = ci[level]["n"]
    out["clip_auc_repeat_mean"] = float(np.mean(repeats_auc))
    out["clip_auc_repeat_sd"] = float(np.std(repeats_auc))
    return out


def load_table(data_dir, datasets, *, include_young_hc, min_hc_age, italian_task, augment):
    manifest = load_manifest(data_dir, tuple(datasets), include_young_hc=include_young_hc,
                             min_hc_age=min_hc_age, italian_task=italian_task)
    variants = ("clean",) + tuple(f"aug{i}" for i in range(augment))
    cache = "vowels" if italian_task == "vowels" else "italian_connected"
    table = build_features(manifest, variants, cache_name=cache)
    bad = table["error"].fillna("") != ""
    if bad.any():
        print(f"[warn] {int(bad.sum())} clip variants failed extraction and are skipped")
    return manifest, table[~bad].reset_index(drop=True), variants


def _display_name(datasets) -> str:
    if tuple(datasets) == ("italian",):
        return "Live vowel model (Italian sustained vowels)"
    names = {"italian": "Italian", "uams": "UAMS", "argentina": "Argentina"}
    return "Live vowel model (" + " + ".join(names.get(d, d) for d in datasets) + " vowels)"


def leave_one_dataset_out(table, features, name, variants, datasets, calibrate) -> dict:
    """Train on all corpora but one, test on the held-out corpus (true external test)."""
    out = {}
    for held in datasets:
        train = table[table["dataset"] != held]
        test = table[(table["dataset"] == held) & (table["variant"] == "clean")].copy()
        pipe = fit_final(name, train, features, train_variants=variants, calibrate=calibrate)
        test["prob"] = pipe.predict_proba(test[features])[:, 1]
        out[held] = bootstrap_ci(test, n_boot=500)
    out["mean_person_auc"] = round(float(np.mean([out[d]["person"]["auc"]["value"] for d in datasets])), 4)
    return out


def evaluate_config(table, variants, *, repeats, calibrate, derive, features=None) -> dict:
    features = features or list(CLINICAL_FEATURES)
    out = {}
    for name in MODEL_NAMES:
        print(f"{name}: {repeats}x speaker-grouped 5-fold CV")
        oof = repeated_grouped_oof(table, features, name, repeats=repeats, train_variants=variants,
                                   calibrate=calibrate)
        clean = oof.clips[oof.clips["variant"] == "clean"]
        ci = bootstrap_ci(clean)
        block = {
            "cv": ci,
            "repeat_clip_auc": [round(m["clip"]["auc"], 4) for m in oof.per_repeat],
            "repeat_person_auc": [round(m["person"]["auc"], 4) for m in oof.per_repeat],
        }
        if clean["dataset"].nunique() > 1:
            block["by_dataset"] = {k: bootstrap_ci(part) for k, part in clean.groupby("dataset")}
        thresholds = derive_thresholds(clean) if derive else dict(DECISION_THRESHOLDS)
        if not derive:
            thresholds["method"] = "fixed (not derived from data)"
        block["thresholds"] = thresholds
        out[name] = {"block": block, "oof": oof}
    return out


def train_live_audio_model(
    data_dir,
    *,
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    include_young_hc: bool = INCLUDE_YOUNG_HC,
    min_hc_age: float | None = MIN_HC_AGE,
    italian_task: str = "vowels",
    repeats: int = 5,
    augment: int = 2,
    calibrate: bool = True,
    derive: bool = True,
    save: bool = True,
    report_name: str = "live_audio_metrics.json",
) -> dict[str, Any]:
    manifest, table, variants = load_table(
        data_dir, datasets, include_young_hc=include_young_hc, min_hc_age=min_hc_age,
        italian_task=italian_task, augment=augment,
    )
    features = list(CLINICAL_FEATURES)
    config = {
        "datasets": list(datasets),
        "include_young_hc": include_young_hc,
        "min_hc_age": min_hc_age,
        "italian_task": italian_task,
        "augment_copies": augment,
        "grouped_calibration": calibrate,
        "data_thresholds": derive,
        "repeats": repeats,
        "speaker_weighting": "each speaker weighs 1 regardless of clip count",
    }
    protocol = PROTOCOL.format(repeats=repeats)
    results: dict[str, Any] = {
        "dataset": " + ".join(datasets) + (" sustained vowels" if italian_task == "vowels" else " reading tasks"),
        "config": config,
        "datasets": describe(manifest),
        "n_features": len(features),
        "feature_names": features,
        "validation_protocol": protocol,
        "models": {},
        "disclaimer": DISCLAIMER,
    }
    evals = evaluate_config(table, variants, repeats=repeats, calibrate=calibrate, derive=derive)
    clean = table[table["variant"] == "clean"]
    dataset_key = "live_audio_italian" if tuple(datasets) == ("italian",) else "live_audio_vowel"
    for name in MODEL_NAMES:
        block = evals[name]["block"]
        flat = _flat_metrics(block["cv"], block["repeat_clip_auc"], protocol)
        if len(datasets) > 1:
            block["lodo"] = leave_one_dataset_out(table, features, name, variants, datasets, calibrate)
            flat["lodo_person_auc"] = {d: block["lodo"][d]["person"]["auc"]["value"] for d in datasets}
            flat["lodo_person_auc_ci95"] = {d: block["lodo"][d]["person"]["auc"]["ci95"] for d in datasets}
            flat["lodo_mean_person_auc"] = block["lodo"]["mean_person_auc"]
            print(f"  {name}: leave-one-dataset-out person AUC {flat['lodo_person_auc']}")
        results["models"][name] = {"metrics": flat, **block}
        print(f"  {name}: person AUC {flat['person_auc']:.3f} {flat['person_auc_ci95']}  "
              f"clip AUC {flat['clip_auc']:.3f} {flat['clip_auc_ci95']}")
        if save:
            pipe = fit_final(name, table, features, train_variants=variants, calibrate=calibrate)
            X = clean[features]
            y = clean["label"].to_numpy(int)
            thresholds = {k: block["thresholds"][k] for k in ("pd_high", "hc_high")}
            _save(name, pipe, X, y, features, flat, thresholds, config,
                  extra={"threshold_method": block["thresholds"].get("method"),
                         "threshold_observed": block["thresholds"].get("observed"),
                         "dataset_key": dataset_key,
                         "display_name": _display_name(datasets),
                         "language": "Language-independent sustained vowel ("
                                     + ", ".join(VOWEL_DATASETS[d]["language"] for d in datasets) + " speakers)"})
    if len(datasets) > 1:
        default = max(MODEL_NAMES, key=lambda n: results["models"][n]["metrics"]["lodo_mean_person_auc"])
        results["default_model_rule"] = "highest mean leave-one-dataset-out person AUC"
    else:
        default = "random_forest"
    if save:
        _write_pointer_and_report(results, features, dataset_key=dataset_key, report_name=report_name,
                                  default=default)
    else:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(REPORTS_DIR / report_name, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
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


def _write_pointer_and_report(results: dict, features: list[str], dataset_key: str = "live_audio_italian",
                              report_name: str = "live_audio_metrics.json", default: str = "random_forest") -> None:
    for stale in MODELS_DIR.glob("live_audio_*__*.joblib"):
        if not stale.name.startswith(f"{dataset_key}__"):
            stale.unlink()
    pointer = {
        "default_model": default,
        "dataset_key": dataset_key,
        "artifact": f"{dataset_key}__{default}.joblib",
        "feature_names": features,
        "summary_metrics": results["models"][default]["metrics"],
    }
    joblib.dump(pointer, MODELS_DIR / "live_audio_default.joblib")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results["app_default_model"] = default
    with open(REPORTS_DIR / report_name, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the live sustained-vowel model.")
    parser.add_argument("--data-dir", default=None, help="Dataset root. Overrides VOICESENSE_DATA_DIR.")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS),
                        choices=list(VOWEL_DATASETS), help="Vowel corpora to train on.")
    parser.add_argument("--min-hc-age", type=float, default=MIN_HC_AGE,
                        help="Drop healthy speakers younger than this where ages are known (UAMS). 0 = keep all.")
    parser.add_argument("--include-young-hc", action="store_true",
                        help="Add young healthy controls (age confound; off by default).")
    parser.add_argument("--italian-task", choices=("vowels", "connected"), default="vowels",
                        help="vowels = VA..VU (default). connected = old reading-task baseline.")
    parser.add_argument("--repeats", type=int, default=5, help="Repeats of speaker-grouped 5-fold CV.")
    parser.add_argument("--augment", type=int, default=2,
                        help="Random augmented copies per training clip (noise, gain, low-pass, phone band, reverb).")
    parser.add_argument("--no-calibration", action="store_true",
                        help="Skip speaker-grouped sigmoid calibration (for comparison only).")
    parser.add_argument("--fixed-thresholds", action="store_true",
                        help="Keep the old 0.72/0.42 cut-offs instead of deriving them from data.")
    parser.add_argument("--report-only", metavar="NAME",
                        help="Write reports/NAME and do not replace the shipped models.")
    args = parser.parse_args(argv)
    train_live_audio_model(
        resolve_data_dir(args.data_dir),
        datasets=tuple(args.datasets),
        min_hc_age=args.min_hc_age or None,
        include_young_hc=args.include_young_hc,
        italian_task=args.italian_task,
        repeats=args.repeats,
        augment=args.augment,
        calibrate=not args.no_calibration,
        derive=not args.fixed_thresholds,
        save=not args.report_only,
        report_name=args.report_only or "live_audio_metrics.json",
    )
    print(DISCLAIMER)


if __name__ == "__main__":
    main()
