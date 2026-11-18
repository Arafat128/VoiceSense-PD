"""Retrain the Bangla (BenSParX) feature-table model with person-grouped validation.

Same machinery as the live vowel model (training/evaluate.py): repeated
StratifiedGroupKFold, 95% CIs from a person-level bootstrap, sigmoid
calibration on grouped folds, and Likely/Unlikely cut-offs derived from
out-of-fold probabilities.

Two person definitions are evaluated:

- ``bensparx_speaker``: the speaker ids published with BenSParX (PD1, HC12 ...).
- ``uci_subject``: the UCI pd_speech_features subject each row was copied from
  (see training/bensparx_provenance.py). This is the one that actually keeps a
  person out of their own test fold, so the shipped model, its cut-offs and its
  headline numbers use it.

Offline only:  python -m training.train_bangla --data-dir <dataset root>
"""
from __future__ import annotations

import argparse
import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from training.bensparx_provenance import build as build_provenance
from training.bensparx_provenance import load_mapping
from training.common import (
    DISCLAIMER,
    MODELS_DIR,
    REPORTS_DIR,
    resolve_data_dir,
    stratified_group_holdout,
)
from training.data_loading import load_dataset
from training.evaluate import (
    _binary_metrics,
    bootstrap_ci,
    derive_thresholds,
    fit_final,
    repeated_grouped_oof,
)

DATASET_KEY = "bengali_bensparx"
MODEL_NAMES = ("random_forest", "svm_rbf", "logreg")
SHIPPED = ("random_forest", "svm_rbf")
GROUPINGS = ("bensparx_speaker", "uci_subject")
HEADLINE_GROUPING = "uci_subject"
PROTOCOL = ("person-grouped stratified 5-fold CV, repeated {repeats}x, persons = {grouping}; "
            "95% CI = 1000x person bootstrap")
OLD_REPORT = {
    "protocol": "single 80/20 split grouped by BenSParX speaker id (random_state 42)",
    "random_forest": {"accuracy": 0.9379, "f1": 0.9371, "roc_auc": 0.9867, "cv_f1": "0.931 ± 0.025"},
    "svm_rbf": {"accuracy": 0.9322, "f1": 0.9302, "roc_auc": 0.9667, "cv_f1": "0.932 ± 0.020"},
}


def flat_metrics(ci: dict, protocol: str) -> dict:
    out: dict[str, Any] = {"protocol": protocol}
    for level in ("clip", "person"):
        for key in ("auc", "balanced_accuracy", "sensitivity", "specificity"):
            out[f"{level}_{key}"] = ci[level][key]["value"]
            out[f"{level}_{key}_ci95"] = ci[level][key]["ci95"]
        out[f"{level}_n"] = ci[level]["n"]
    return out


def importances_of(pipe, features: list[str]) -> list[dict]:
    clf = pipe.named_steps["clf"]
    inner = [c.estimator for c in getattr(clf, "calibrated_classifiers_", [])
             if hasattr(c.estimator, "feature_importances_")]
    if hasattr(clf, "feature_importances_"):
        inner = [clf]
    if not inner:
        return []
    imp = np.mean([c.feature_importances_ for c in inner], axis=0)
    return [{"feature": features[i], "importance": float(imp[i])} for i in np.argsort(imp)[::-1]]


def build_table(data_dir) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    data = load_dataset(DATASET_KEY, data_dir)
    try:
        mapping = build_provenance(data_dir, write=True)[0]
    except FileNotFoundError:
        mapping = load_mapping()
    if list(mapping["bensparx_id"]) != list(data.sample_ids):
        raise RuntimeError("BenSParX provenance mapping is out of date; rebuild it")
    table = data.X.copy()
    features = list(data.feature_names)
    table.insert(0, "clip_id", data.sample_ids)
    table.insert(1, "variant", "clean")
    table.insert(2, "dataset", "bensparx")
    table.insert(3, "label", data.y)
    table.insert(4, "bensparx_speaker", data.groups)
    table.insert(5, "uci_subject", mapping["uci_subject"].to_numpy())
    table.insert(6, "uci_row", mapping["uci_row"].to_numpy())
    return table, features, mapping


def leakage_of_old_split(table: pd.DataFrame) -> dict:
    y = table["label"].to_numpy(int)
    tr, te = stratified_group_holdout(y, table["bensparx_speaker"].to_numpy())
    subj = table["uci_subject"].to_numpy()
    rows = table["uci_row"].to_numpy()
    return {
        "test_rows": len(te),
        "test_rows_with_same_uci_person_in_train": round(float(np.isin(subj[te], subj[tr]).mean()), 4),
        "test_rows_with_same_uci_recording_in_train": round(float(np.isin(rows[te], rows[tr]).mean()), 4),
    }


def _evaluate(table, features, grouping, name, repeats):
    t = table.assign(subject=table[grouping])
    oof = repeated_grouped_oof(t, features, name, repeats=repeats, log=lambda *_: None)
    clean = oof.clips
    return grouping, name, {
        "cv": bootstrap_ci(clean),
        "repeat_person_auc": [round(m["person"]["auc"], 4) for m in oof.per_repeat],
        "repeat_clip_auc": [round(m["clip"]["auc"], 4) for m in oof.per_repeat],
        "thresholds": derive_thresholds(clean),
    }


def train_bangla(data_dir, *, repeats: int = 5, save: bool = True, n_jobs: int = -1) -> dict[str, Any]:
    table, features, _mapping = build_table(data_dir)
    jobs = [(g, n) for g in GROUPINGS for n in MODEL_NAMES]
    done = Parallel(n_jobs=n_jobs)(delayed(_evaluate)(table, features, g, n, repeats) for g, n in jobs)
    blocks: dict[str, dict] = {g: {} for g in GROUPINGS}
    for g, n, block in done:
        block["metrics"] = flat_metrics(block["cv"], PROTOCOL.format(repeats=repeats, grouping=g))
        blocks[g][n] = block
        m = block["metrics"]
        print(f"[{g}] {n}: row AUC {m['clip_auc']:.3f} {m['clip_auc_ci95']}  "
              f"person AUC {m['person_auc']:.3f} {m['person_auc_ci95']}")

    head = blocks[HEADLINE_GROUPING]
    default = max(SHIPPED, key=lambda n: head[n]["metrics"]["person_auc"])
    y = table["label"].to_numpy(int)
    tr, te = stratified_group_holdout(y, table[HEADLINE_GROUPING].to_numpy())
    train = table.iloc[tr].assign(subject=lambda d: d[HEADLINE_GROUPING])
    test = table.iloc[te]
    results: dict[str, Any] = {
        "dataset_key": DATASET_KEY,
        "display_name": "BenSParX (Bengali, public CSV)",
        "language": "Bengali (as published; see provenance)",
        "n_samples": len(table),
        "n_features": len(features),
        "n_bensparx_speakers": int(table["bensparx_speaker"].nunique()),
        "n_uci_subjects": int(table["uci_subject"].nunique()),
        "class_counts": {"PD": int(y.sum()), "HC": int((y == 0).sum())},
        "provenance": json.loads((REPORTS_DIR / "bensparx_provenance.json").read_text(encoding="utf-8")),
        "old_model": {**OLD_REPORT, "leakage": leakage_of_old_split(table)},
        "headline_grouping": HEADLINE_GROUPING,
        "validation": {g: {n: {k: v for k, v in b.items()} for n, b in bl.items()} for g, bl in blocks.items()},
        "app_default_model": default,
        "best_model": default,
        "default_model_rule": "highest person AUC with persons = UCI subject (RF or SVM)",
        "split": {
            "method": "80/20 holdout grouped by UCI subject; shipped models train on the 80%",
            "train_samples": len(tr), "test_samples": len(te),
            "train_subjects": int(train["uci_subject"].nunique()),
            "test_subjects": int(test["uci_subject"].nunique()),
        },
        "feature_names": features,
        "disclaimer": DISCLAIMER,
        "models": {},
    }
    for name in SHIPPED:
        block = head[name]
        pipe = fit_final(name, train, features)
        p = pipe.predict_proba(test[features])[:, 1]
        hold = _binary_metrics(test["label"].to_numpy(int), p)
        thr = {k: block["thresholds"][k] for k in ("pd_high", "hc_high")}
        metrics = {**block["metrics"],
                   "holdout_auc": round(hold["auc"], 4),
                   "holdout_balanced_accuracy": round(hold["balanced_accuracy"], 4),
                   "speaker_id_grouping_person_auc": blocks["bensparx_speaker"][name]["metrics"]["person_auc"],
                   "speaker_id_grouping_person_auc_ci95": blocks["bensparx_speaker"][name]["metrics"]["person_auc_ci95"]}
        results["models"][name] = {"metrics": metrics, "thresholds": block["thresholds"],
                                   "top_features": importances_of(pipe, features)[:15],
                                   "artifact": f"models/{DATASET_KEY}__{name}.joblib"}
        if save:
            X = train[features]
            yt = train["label"].to_numpy(int)
            imps = importances_of(pipe, features)
            if not imps and "random_forest" in results["models"]:
                imps = joblib.load(MODELS_DIR / f"{DATASET_KEY}__random_forest.joblib")["feature_importances"]
            joblib.dump({
                "pipeline": pipe,
                "feature_names": features,
                "dataset_key": DATASET_KEY,
                "model_name": name,
                "language": "Bengali (as published; rows match the UCI Turkish table)",
                "display_name": "BenSParX (Bengali, public CSV)",
                "positive_name": "Parkinson's Disease (PD)",
                "negative_name": "Healthy Control (HC)",
                "feature_importances": imps,
                "train_medians": X.median().to_dict(),
                "train_means": X.mean().to_dict(),
                "train_stds": X.std(ddof=0).to_dict(),
                "class_means": {"PD": X[yt == 1].mean().to_dict(), "HC": X[yt == 0].mean().to_dict()},
                "class_stds": {"PD": X[yt == 1].std(ddof=0).to_dict(), "HC": X[yt == 0].std(ddof=0).to_dict()},
                "metrics": metrics,
                "decision_thresholds": thr,
                "threshold_method": block["thresholds"]["method"],
                "input_type": "feature_table",
                "disclaimer": DISCLAIMER,
            }, MODELS_DIR / f"{DATASET_KEY}__{name}.joblib", compress=3)
        print(f"  shipped {name}: holdout AUC {hold['auc']:.3f}, cut-offs {thr}")
    if save:
        hold_df = test[features].copy()
        hold_df.insert(0, "sample_id", test["clip_id"].to_numpy())
        hold_df.insert(1, "subject_id", test["bensparx_speaker"].to_numpy())
        hold_df.insert(2, "true_label", test["label"].to_numpy(int))
        hold_df.to_csv(MODELS_DIR / f"{DATASET_KEY}__holdout_samples.csv", index=False)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / f"{DATASET_KEY}__metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    _update_summary(results)
    return results


def _update_summary(results: dict) -> None:
    path = REPORTS_DIR / "training_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    summary[DATASET_KEY] = {
        "display_name": results["display_name"],
        "best_model": results["best_model"],
        "app_default_model": results["app_default_model"],
        "split": results["split"],
        "models": {n: m["metrics"] for n, m in results["models"].items()},
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Retrain the BenSParX model with person-grouped CV.")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)
    train_bangla(resolve_data_dir(args.data_dir), repeats=args.repeats, save=not args.no_save)


if __name__ == "__main__":
    main()
