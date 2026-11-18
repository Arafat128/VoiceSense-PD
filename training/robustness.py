"""Does a prediction stay put when the same clip is degraded?

For each held-out speaker (speaker-grouped CV, so the model never saw them),
score the clean clip and fixed degraded copies (noise, gain, low-pass, phone
band, reverb). Report the mean absolute change in P(PD), how often the
Likely / Uncertain / Unlikely band flips, and the AUC under each degradation.
Run once with training augmentation and once without to show what it buys.

Run: python -m training.robustness --data-dir <dataset root>
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from training.augment import DEGRADATIONS
from training.common import DISCLAIMER, REPORTS_DIR, resolve_data_dir
from training.evaluate import derive_thresholds, repeated_grouped_oof, summarize
from training.featurize import CLINICAL_FEATURES, build_features
from training.vowel_data import load_manifest


def _band(p: np.ndarray, thr: dict) -> np.ndarray:
    return np.where(p >= thr["pd_high"], 2, np.where(p <= thr["hc_high"], 0, 1))


def robustness_block(oof_clips: pd.DataFrame) -> dict:
    wide = oof_clips.pivot_table(index="clip_id", columns="variant", values="prob")
    labels = oof_clips.drop_duplicates("clip_id").set_index("clip_id")
    clean = oof_clips[oof_clips["variant"] == "clean"]
    thr = derive_thresholds(clean)
    out = {"clean": summarize(clean)["clip"], "cutoffs": {k: thr[k] for k in ("pd_high", "hc_high")}, "degraded": {}}
    base = wide["clean"].to_numpy()
    for name in DEGRADATIONS:
        if name not in wide:
            continue
        p = wide[name].to_numpy()
        ok = np.isfinite(p) & np.isfinite(base)
        part = oof_clips[oof_clips["variant"] == name]
        out["degraded"][name] = {
            "mean_abs_change": round(float(np.mean(np.abs(p[ok] - base[ok]))), 4),
            "band_flip_rate": round(float(np.mean(_band(p[ok], thr) != _band(base[ok], thr))), 4),
            "flip_to_likely_from_healthy_band": round(float(np.mean((_band(p[ok], thr) == 2) & (_band(base[ok], thr) == 0))), 4),
            "clip_auc": round(float(summarize(part)["clip"]["auc"]), 4),
            "n": int(ok.sum()),
        }
    deg = out["degraded"].values()
    out["summary"] = {
        "mean_abs_change": round(float(np.mean([d["mean_abs_change"] for d in deg])), 4),
        "band_flip_rate": round(float(np.mean([d["band_flip_rate"] for d in deg])), 4),
    }
    del labels
    return out


def run(data_dir, datasets=("italian",), min_hc_age=None, augment=2, repeats=2, model="svm_rbf") -> dict:
    manifest = load_manifest(data_dir, tuple(datasets), min_hc_age=min_hc_age)
    variants = ("clean",) + tuple(f"aug{i}" for i in range(augment)) + tuple(DEGRADATIONS)
    table = build_features(manifest, variants, cache_name="vowels")
    table = table[table["error"].fillna("") == ""].reset_index(drop=True)
    feats = list(CLINICAL_FEATURES)
    eval_variants = ("clean",) + tuple(DEGRADATIONS)
    results = {"datasets": list(datasets), "model": model, "repeats": repeats,
               "degradations": list(DEGRADATIONS), "disclaimer": DISCLAIMER, "runs": {}}
    for label, train_variants in (("without_augmentation", ("clean",)),
                                  ("with_augmentation", ("clean",) + tuple(f"aug{i}" for i in range(augment)))):
        print(f"{label}: training on {train_variants}")
        oof = repeated_grouped_oof(table, feats, model, repeats=repeats, train_variants=train_variants,
                                   eval_variants=eval_variants)
        block = robustness_block(oof.clips.rename(columns={"prob": "prob"}))
        results["runs"][label] = block
        print(f"  mean |dP| {block['summary']['mean_abs_change']:.3f}  band flips {block['summary']['band_flip_rate']:.1%}")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "robustness.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Robustness of predictions to degraded audio.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--datasets", nargs="+", default=["italian"])
    parser.add_argument("--min-hc-age", type=float, default=None)
    parser.add_argument("--augment", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--model", default="random_forest")
    args = parser.parse_args(argv)
    run(resolve_data_dir(args.data_dir), args.datasets, args.min_hc_age, args.augment, args.repeats, args.model)


if __name__ == "__main__":
    main()
