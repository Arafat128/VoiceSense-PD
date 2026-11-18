"""Compare Praat clinical features, openSMILE eGeMAPS, and WavLM embeddings.

Protocol: leave-one-dataset-out across the vowel corpora (train on the others,
test on the held-out corpus), plus the Italian <-> UAMS pair in both
directions. Person-level AUC with 95% speaker-bootstrap CIs.

Verdict rule: the live model keeps Praat features unless another feature set
beats it on held-out corpora: a higher mean held-out AUC by at least 0.02 AND
a higher AUC on a majority of held-out corpora. Otherwise this is a report.

eGeMAPS and WavLM need requirements-research.txt (opensmile, torch, transformers).
Run: python -m training.compare_features --data-dir <dataset root>
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from training.common import DISCLAIMER, REPORTS_DIR, resolve_data_dir
from training.evaluate import bootstrap_ci, fit_final
from training.featurize import CLINICAL_FEATURES, build_features
from training.vowel_data import describe, load_manifest

DATASETS = ("italian", "uams", "argentina")
PAIRS = (("italian", "uams"), ("uams", "italian"))
CLASSIFIERS = {"praat": ("logreg", "random_forest", "svm_rbf"),
               "egemaps": ("logreg", "random_forest", "svm_rbf"),
               "wavlm": ("logreg", "svm_rbf")}
MIN_GAIN = 0.02


def _tables(data_dir, sets):
    manifest = load_manifest(data_dir, DATASETS)
    base = build_features(manifest, ("clean",), cache_name="vowels", verbose=False)
    base = base[base["error"].fillna("") == ""].reset_index(drop=True)
    out = {"praat": (base, list(CLINICAL_FEATURES))}
    if "egemaps" in sets:
        from training.embeddings import egemaps_table, feature_columns
        eg = egemaps_table(manifest)
        out["egemaps"] = (base.merge(eg, on="clip_id", how="left"), feature_columns(eg))
    if "wavlm" in sets:
        from training.embeddings import feature_columns, wavlm_table
        wl = wavlm_table(manifest)
        out["wavlm"] = (base.merge(wl, on="clip_id", how="left"), feature_columns(wl))
    return manifest, out


def _held_out(table, feats, model, train_sets, test_set):
    tr = table[table["dataset"].isin(train_sets)]
    te = table[table["dataset"] == test_set].copy()
    te["prob"] = fit_final(model, tr, feats).predict_proba(te[feats])[:, 1]
    ci = bootstrap_ci(te, n_boot=500)["person"]["auc"]
    return {"auc": ci["value"], "ci95": ci["ci95"]}


def run(data_dir, sets=("praat", "egemaps", "wavlm")) -> dict:
    manifest, tables = _tables(data_dir, sets)
    results = {"datasets": describe(manifest), "protocol": __doc__.split("\n\n")[1].strip(),
               "lodo": {}, "pairs": {}, "disclaimer": DISCLAIMER}
    for fset, (table, feats) in tables.items():
        for model in CLASSIFIERS[fset]:
            key = f"{fset}/{model}"
            lodo = {}
            for test in DATASETS:
                lodo[test] = _held_out(table, feats, model, [d for d in DATASETS if d != test], test)
            lodo["mean_auc"] = round(float(np.mean([lodo[d]["auc"] for d in DATASETS])), 4)
            results["lodo"][key] = lodo
            results["pairs"][key] = {f"{a}->{b}": _held_out(table, feats, model, [a], b) for a, b in PAIRS}
            print(key, {d: lodo[d]["auc"] for d in DATASETS}, "mean", lodo["mean_auc"], flush=True)
    results["verdict"] = verdict(results["lodo"])
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "feature_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    (REPORTS_DIR / "feature_comparison.md").write_text(to_markdown(results), encoding="utf-8")
    print(results["verdict"]["text"])
    return results


def verdict(lodo: dict) -> dict:
    best = {}
    for key, block in lodo.items():
        fset = key.split("/")[0]
        if fset not in best or block["mean_auc"] > best[fset][1]["mean_auc"]:
            best[fset] = (key, block)
    inc_key, inc = best["praat"]
    winner, text = inc_key, f"Keep Praat ({inc_key}, mean held-out AUC {inc['mean_auc']:.3f})."
    for fset, (key, block) in best.items():
        if fset == "praat":
            continue
        wins = sum(block[d]["auc"] > inc[d]["auc"] for d in DATASETS)
        clears_rule = block["mean_auc"] >= inc["mean_auc"] + MIN_GAIN and wins >= 2
        if clears_rule and (winner == inc_key or block["mean_auc"] > lodo[winner]["mean_auc"]):
            winner = key
            text = (f"Switch to {key}: mean held-out AUC {block['mean_auc']:.3f} vs "
                    f"{inc['mean_auc']:.3f} for {inc_key}, better on {wins}/3 held-out corpora.")
    if winner == inc_key:
        others = ", ".join(f"{k} {b['mean_auc']:.3f}" for f, (k, b) in best.items() if f != "praat")
        text += f" No alternative cleared the rule (best alternatives: {others})."
    return {"winner": winner, "best_per_feature_set": {f: k for f, (k, _) in best.items()}, "text": text}


def to_markdown(results: dict) -> str:
    lines = ["# Feature comparison: leave-one-dataset-out", "", results["protocol"], "",
             "Person-level AUC on the held-out corpus (95% CI). 0.50 = chance.", "",
             "| features / classifier | held-out Italian | held-out UAMS | held-out Argentina | mean |",
             "|---|---|---|---|---|"]
    for key, b in results["lodo"].items():
        cells = [f"{b[d]['auc']:.2f} ({b[d]['ci95'][0]:.2f}-{b[d]['ci95'][1]:.2f})" for d in DATASETS]
        lines.append(f"| {key} | " + " | ".join(cells) + f" | {b['mean_auc']:.3f} |")
    lines += ["", "| features / classifier | Italian -> UAMS | UAMS -> Italian |", "|---|---|---|"]
    for key, b in results["pairs"].items():
        cells = [f"{b[p]['auc']:.2f} ({b[p]['ci95'][0]:.2f}-{b[p]['ci95'][1]:.2f})" for p in ("italian->uams", "uams->italian")]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    lines += ["", f"**Verdict:** {results['verdict']['text']}", ""]
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--sets", nargs="+", default=["praat", "egemaps", "wavlm"])
    args = parser.parse_args(argv)
    run(resolve_data_dir(args.data_dir), tuple(args.sets))


if __name__ == "__main__":
    main()
