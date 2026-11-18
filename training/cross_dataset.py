"""Do BenSParX patterns transfer to other Parkinson's voice tables, and back?

Four public feature tables share a small set of Praat/MDVP perturbation
measures. Each model is trained on one table and scored on another, using
only the shared measures. Two input versions are compared:

- ``raw``: units harmonised (Replicated jitter is in percent), values as given.
- ``standardized``: log, then z-scored within each table. This removes unit
  and recording-level offsets and is the generous case for transfer.

Persons: BenSParX rows are grouped by the UCI subject they were copied from
(training/bensparx_provenance.py), so BenSParX <-> UCI pd_speech is the same
people measured once and is reported as such, not as a transfer test.

Offline only:  python -m training.cross_dataset --data-dir <dataset root>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from training.bensparx_provenance import load_mapping
from training.common import REPORTS_DIR, resolve_data_dir
from training.evaluate import bootstrap_ci, fit_final, repeated_grouped_oof

# Canonical name -> column in each table.
COLUMNS = {
    "bensparx": {
        "jitter_local": "locPctJitter", "jitter_abs": "locAbsJitter", "jitter_rap": "rapJitter",
        "jitter_ppq5": "ppq5Jitter", "shimmer_local": "locShimmer", "shimmer_db": "locDbShimmer",
        "shimmer_apq3": "apq3Shimmer", "shimmer_apq5": "apq5Shimmer", "shimmer_apq11": "apq11Shimmer",
        "nhr": "meanAutoCorrHarmonicity", "hnr": "NHR",  # BenSParX names are swapped; see provenance
    },
    "uci_pd_speech": {
        "jitter_local": "locPctJitter", "jitter_abs": "locAbsJitter", "jitter_rap": "rapJitter",
        "jitter_ppq5": "ppq5Jitter", "shimmer_local": "locShimmer", "shimmer_db": "locDbShimmer",
        "shimmer_apq3": "apq3Shimmer", "shimmer_apq5": "apq5Shimmer", "shimmer_apq11": "apq11Shimmer",
        "nhr": "meanNoiseToHarmHarmonicity", "hnr": "meanHarmToNoiseHarmonicity",
    },
    "uci_replicated": {
        "jitter_local": "Jitter_rel", "jitter_abs": "Jitter_abs", "jitter_rap": "Jitter_RAP",
        "jitter_ppq5": "Jitter_PPQ", "shimmer_local": "Shim_loc", "shimmer_db": "Shim_dB",
        "shimmer_apq3": "Shim_APQ3", "shimmer_apq5": "Shim_APQ5", "shimmer_apq11": "Shi_APQ11",
    },
    "uci_parkinsons": {
        "jitter_local": "MDVP:Jitter(%)", "jitter_abs": "MDVP:Jitter(Abs)", "jitter_rap": "MDVP:RAP",
        "jitter_ppq5": "MDVP:PPQ", "shimmer_local": "MDVP:Shimmer", "shimmer_db": "MDVP:Shimmer(dB)",
        "shimmer_apq3": "Shimmer:APQ3", "shimmer_apq5": "Shimmer:APQ5", "shimmer_apq11": "MDVP:APQ",
        "nhr": "NHR", "hnr": "HNR",
    },
}
DESCRIPTION = {
    "bensparx": "BenSParX public CSV (labelled Bengali conversational speech)",
    "uci_pd_speech": "UCI PD Classification, Sakar 2019 (Turkish sustained /a/)",
    "uci_replicated": "UCI Replicated Acoustic, Naranjo 2016 (Spanish sustained /a/)",
    "uci_parkinsons": "UCI Parkinsons, Little 2007 (English-speaking, sustained phonation)",
}
SAME_PEOPLE = {frozenset({"bensparx", "uci_pd_speech"})}
MODELS = ("logreg", "random_forest")


def load_all(data_dir: Path) -> dict[str, pd.DataFrame]:
    d = Path(data_dir)
    ben = pd.read_csv(d / "02_BenSParX" / "BenSParX-main" / "BenSParX.csv")
    mapping = load_mapping(d)
    ben["subject"] = mapping["uci_subject"].to_numpy()
    ben["label"] = ben["class"]
    sak = pd.read_csv(d / "03_UCI_PD_Classification" / "pd_speech_features.csv", header=1)
    sak["subject"] = "uci" + sak["id"].astype(str)
    sak["label"] = sak["class"]
    rep = pd.read_csv(d / "05_UCI_Replicated_Acoustic" / "ReplicatedAcousticFeatures-ParkinsonDatabase.csv")
    rep["subject"] = rep["ID"].astype(str)
    rep["label"] = rep["Status"]
    rep["Jitter_rel"] = rep["Jitter_rel"] / 100.0  # percent -> fraction, as in the others
    par = pd.read_csv(d / "01_UCI_Parkinsons" / "parkinsons.data")
    par["subject"] = par["name"].str.split("_").str[2]
    par["label"] = par["status"]
    raw = {"bensparx": ben, "uci_pd_speech": sak, "uci_replicated": rep, "uci_parkinsons": par}
    out = {}
    for key, df in raw.items():
        cols = COLUMNS[key]
        t = pd.DataFrame({canon: pd.to_numeric(df[src], errors="coerce") for canon, src in cols.items()})
        t["subject"] = df["subject"].astype(str).to_numpy()
        t["label"] = df["label"].astype(int).to_numpy()
        t["dataset"] = key
        t["variant"] = "clean"
        t["clip_id"] = [f"{key}_{i}" for i in range(len(t))]
        out[key] = t
    return out


def standardize(t: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    t = t.copy()
    x = np.log(np.clip(t[feats].astype(float), 1e-9, None))
    t[feats] = (x - x.mean()) / x.std(ddof=0).replace(0, 1)
    return t


def run(data_dir: Path, repeats: int = 3) -> dict:
    tables = load_all(data_dir)
    keys = list(tables)
    results: dict = {"datasets": {k: {"description": DESCRIPTION[k], "rows": len(t),
                                      "persons": int(t["subject"].nunique()),
                                      "pd_rows": int(t["label"].sum())} for k, t in tables.items()},
                     "feature_sets": {}, "pairs": [], "within": []}
    for version in ("raw", "standardized"):
        for train_key in keys:
            for test_key in keys:
                feats = sorted(set(COLUMNS[train_key]) & set(COLUMNS[test_key]))
                tr = tables[train_key]
                te = tables[test_key]
                if version == "standardized":
                    tr, te = standardize(tr, feats), standardize(te, feats)
                for model in MODELS:
                    if train_key == test_key:
                        oof = repeated_grouped_oof(tr, feats, model, repeats=repeats, log=lambda *_: None)
                        ci = bootstrap_ci(oof.clips, n_boot=500)
                        results["within"].append({"version": version, "dataset": train_key, "model": model,
                                                  "features": feats, "person_auc": ci["person"]["auc"],
                                                  "row_auc": ci["clip"]["auc"]})
                        continue
                    pipe = fit_final(model, tr, feats)
                    scored = te[["clip_id", "subject", "dataset", "label"]].copy()
                    scored["prob"] = pipe.predict_proba(te[feats])[:, 1]
                    ci = bootstrap_ci(scored, n_boot=500)
                    results["pairs"].append({
                        "version": version, "train": train_key, "test": test_key, "model": model,
                        "n_features": len(feats), "features": feats,
                        "same_people": frozenset({train_key, test_key}) in SAME_PEOPLE,
                        "person_auc": ci["person"]["auc"], "row_auc": ci["clip"]["auc"],
                        "person_balanced_accuracy_at_0.5": ci["person"]["balanced_accuracy"],
                    })
    results["verdict"] = verdict(results)
    return results


def verdict(res: dict) -> str:
    def mean_auc(rows):
        vals = [r["person_auc"]["value"] for r in rows]
        return float(np.mean(vals)) if vals else float("nan")
    std = [r for r in res["pairs"] if r["version"] == "standardized" and not r["same_people"]]
    from_ben = mean_auc([r for r in std if r["train"] == "bensparx"])
    to_ben = mean_auc([r for r in std if r["test"] == "bensparx"])
    others = mean_auc([r for r in std if "bensparx" not in (r["train"], r["test"])
                       and "uci_pd_speech" not in (r["train"], r["test"])])
    same = [r for r in res["pairs"] if r["same_people"] and r["version"] == "standardized"]
    return (
        f"With the shared perturbation measures standardized per table, a BenSParX model reaches mean "
        f"person AUC {from_ben:.2f} on the Spanish and English tables, and models trained there reach "
        f"{to_ben:.2f} on BenSParX (Replicated <-> Parkinsons: {others:.2f}). "
        f"BenSParX <-> UCI pd_speech scores {mean_auc(same):.2f}, but those are the same people's "
        "recordings, so it shows the copy, not transfer."
    )


def to_markdown(res: dict) -> str:
    lines = ["# Cross-dataset test: BenSParX vs other Parkinson's voice tables", "",
             ("Person-level AUC with 95% bootstrap CI over test persons. "
              "`standardized` = log + z-score within each table."), "",
             "| version | train | test | model | features | person AUC (95% CI) | note |",
             "|---|---|---|---|---|---|---|"]
    for r in res["pairs"]:
        a = r["person_auc"]
        note = "same people (copied rows)" if r["same_people"] else ""
        lines.append(f"| {r['version']} | {r['train']} | {r['test']} | {r['model']} | {r['n_features']} | "
                     f"{a['value']:.2f} ({a['ci95'][0]:.2f}–{a['ci95'][1]:.2f}) | {note} |")
    lines += ["", "Within-table reference (person-grouped CV on the same shared measures):", "",
              "| version | dataset | model | person AUC (95% CI) |", "|---|---|---|---|"]
    for r in res["within"]:
        a = r["person_auc"]
        lines.append(f"| {r['version']} | {r['dataset']} | {r['model']} | "
                     f"{a['value']:.2f} ({a['ci95'][0]:.2f}–{a['ci95'][1]:.2f}) |")
    lines += ["", f"**Verdict.** {res['verdict']}", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Cross-dataset transfer for the BenSParX model.")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args(argv)
    res = run(resolve_data_dir(args.data_dir), repeats=args.repeats)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "bensparx_cross_dataset.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (REPORTS_DIR / "bensparx_cross_dataset.md").write_text(to_markdown(res), encoding="utf-8")
    print(res["verdict"])


if __name__ == "__main__":
    main()
