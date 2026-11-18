"""Score the shipped live model on corpora it never saw.

Default targets: UAMS (US English speakers, phone /a/, 8 kHz) and the
Argentine Spanish /a/ corpus. A target that the model was trained on is
refused, so every number here is a true external test.

Run: python -m training.eval_external --data-dir <dataset root> [--targets uams argentina]
     python -m training.eval_external --artifact old_model.joblib --tag before
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.registry import live_model_path
from src.predict import MIN_CALL_DURATION_SEC, MIN_CALL_SNR_DB, MIN_CALL_VOICED, thresholds_for
from training.common import DISCLAIMER, REPORTS_DIR, resolve_data_dir
from training.evaluate import bootstrap_ci
from training.featurize import build_features
from training.vowel_data import describe, load_manifest

# UAMS healthy controls are much younger than its Parkinson's group
# (mean 48 vs 67 years). The age-matched view keeps healthy speakers >= 50.
AGE_MATCH_MIN_HC = 50.0


def _score(pack: dict, table: pd.DataFrame) -> np.ndarray:
    names = pack["feature_names"]
    frame = table.reindex(columns=names)
    return pack["pipeline"].predict_proba(frame)[:, 1]


def _bands(table: pd.DataFrame, thr: dict) -> dict:
    out = {}
    for label, name in ((0, "healthy"), (1, "parkinsons")):
        p = table.loc[table["label"] == label, "prob"].to_numpy()
        if not len(p):
            continue
        out[name] = {
            "likely": round(float((p >= thr["pd_high"]).mean()), 4),
            "uncertain": round(float(((p > thr["hc_high"]) & (p < thr["pd_high"])).mean()), 4),
            "unlikely": round(float((p <= thr["hc_high"]).mean()), 4),
        }
    return out


def evaluate_target(pack: dict, data_dir: Path, target: str, min_hc_age: float | None = None) -> dict:
    manifest = load_manifest(data_dir, (target,), min_hc_age=min_hc_age)
    table = build_features(manifest, ("clean",), cache_name="vowels")
    table = table[table["error"].fillna("") == ""].reset_index(drop=True)
    table["prob"] = _score(pack, table)
    thr = thresholds_for(pack)
    callable_mask = (
        (table["q_duration_sec"] >= MIN_CALL_DURATION_SEC)
        & (table["q_snr_db"] >= MIN_CALL_SNR_DB)
        & (table["q_voiced_fraction"] >= MIN_CALL_VOICED)
    )
    out = {
        "dataset": describe(manifest)[target],
        "metrics": bootstrap_ci(table),
        "band_shares_at_saved_cutoffs": _bands(table, thr),
        "cutoffs": thr,
        "mean_prob": {
            "healthy": round(float(table.loc[table["label"] == 0, "prob"].mean()), 4),
            "parkinsons": round(float(table.loc[table["label"] == 1, "prob"].mean()), 4),
        },
        "app_would_make_a_call_share": round(float(callable_mask.mean()), 4),
    }
    if callable_mask.sum() >= 20 and table.loc[callable_mask, "label"].nunique() == 2:
        out["metrics_on_clips_the_app_accepts"] = bootstrap_ci(table[callable_mask])
    return out


def run(data_dir: Path, targets: list[str], artifact: Path | None = None, tag: str = "current") -> dict:
    results: dict = {"tag": tag, "disclaimer": DISCLAIMER, "models": {}}
    paths = [artifact] if artifact else [live_model_path("random_forest"), live_model_path("svm_rbf")]
    for path in paths:
        pack = joblib.load(path)
        trained_on = list((pack.get("training_config") or {}).get("datasets") or ["italian"])
        name = f"{pack.get('model_name')} ({Path(path).name})"
        results["models"][name] = {"trained_on": trained_on, "targets": {}}
        for target in targets:
            if target in trained_on:
                results["models"][name]["targets"][target] = {"skipped": "model was trained on this corpus"}
                continue
            block = evaluate_target(pack, data_dir, target)
            if target == "uams":
                block["age_matched_hc_ge_50"] = evaluate_target(pack, data_dir, target, AGE_MATCH_MIN_HC)
            results["models"][name]["targets"][target] = block
            m = block["metrics"]["person"]
            print(f"{name} -> {target}: AUC {m['auc']['value']:.3f} {m['auc']['ci95']}  "
                  f"bal.acc {m['balanced_accuracy']['value']:.3f}")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"external_validation_{tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="External test of the live model.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--targets", nargs="+", default=["uams", "argentina"])
    parser.add_argument("--artifact", type=Path, default=None, help="Score this joblib instead of the shipped models.")
    parser.add_argument("--tag", default="current")
    args = parser.parse_args(argv)
    run(resolve_data_dir(args.data_dir), args.targets, args.artifact, args.tag)


if __name__ == "__main__":
    main()
