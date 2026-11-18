"""Speaker-grouped validation, grouped calibration, and data-driven thresholds.

Every estimate here keeps all clips of a speaker on one side of a split.
Metrics are reported per clip and per person (a person's score is the mean
probability over their clips), with 95% confidence intervals from a
speaker-level (cluster) bootstrap of the out-of-fold predictions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42

# Threshold targets for the three bands shown in the app.
# "Likely" needs at most 10% of healthy clips above it (specificity >= 90%).
# "Unlikely" needs at most 10% of Parkinson's clips below it (sensitivity >= 90%).
TARGET_SPECIFICITY = 0.90
TARGET_SENSITIVITY = 0.90


def base_estimator(name: str):
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=3,
            max_depth=8,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    if name == "svm_rbf":
        return SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=RANDOM_STATE)
    if name == "logreg":
        return LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000)
    raise KeyError(name)


def grouped_splits(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int, mask_eval=None):
    """StratifiedGroupKFold over speakers. Optional mask keeps eval rows clean-only."""
    n_groups = len(np.unique(groups))
    k = max(2, min(n_splits, n_groups // 2))
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    out = []
    for tr, te in cv.split(np.zeros(len(y)), y, groups):
        if mask_eval is not None:
            te = te[mask_eval[te]]
        if len(np.unique(y[te])) == 2 and len(np.unique(y[tr])) == 2:
            out.append((tr, te))
    return out


def make_model(name: str, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
               is_clean: np.ndarray | None = None, sample_weight: np.ndarray | None = None,
               calib_splits: int = 3, seed: int = RANDOM_STATE, calibrate: bool = True) -> Pipeline:
    """Fit imputer+scaler+classifier with sigmoid calibration on speaker-grouped folds.

    Calibration folds hold out whole speakers, and calibrate only on their
    clean clips, so augmented copies of a speaker never calibrate their own model.
    """
    clean = np.ones(len(y), bool) if is_clean is None else is_clean
    if calibrate:
        splits = grouped_splits(y, groups, calib_splits, seed, mask_eval=clean)
        calibrated = CalibratedClassifierCV(base_estimator(name), method="sigmoid", cv=splits)
    else:
        calibrated = base_estimator(name)
        if name == "svm_rbf":
            calibrated.set_params(probability=True)
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", calibrated),
        ]
    )
    fit_kw = {}
    if sample_weight is not None:
        fit_kw["clf__sample_weight"] = sample_weight
    pipe.fit(X, y, **fit_kw)
    return pipe


def speaker_weights(groups: np.ndarray) -> np.ndarray:
    """Each speaker counts once, however many clips they have."""
    s = pd.Series(groups)
    return (1.0 / s.map(s.value_counts())).to_numpy(dtype=float)


@dataclass
class OOFResult:
    clips: pd.DataFrame  # clip_id, subject, dataset, label, prob (mean over repeats), prob_r{i}
    per_repeat: list[dict]


def repeated_grouped_oof(
    table: pd.DataFrame,
    features: list[str],
    model_name: str,
    *,
    repeats: int = 5,
    n_splits: int = 5,
    train_variants: tuple[str, ...] = ("clean",),
    eval_variants: tuple[str, ...] = ("clean",),
    weight_speakers: bool = True,
    calibrate: bool = True,
    log=print,
) -> OOFResult:
    """Repeated speaker-grouped K-fold. Train on train_variants of training speakers.

    Out-of-fold probabilities are produced for every eval variant of held-out speakers.
    """
    clean = table[table["variant"] == "clean"].reset_index(drop=True)
    y_c = clean["label"].to_numpy(int)
    g_c = clean["subject"].to_numpy()
    out = table[table["variant"].isin(eval_variants)][
        ["clip_id", "variant", "subject", "dataset", "label"]
    ].copy().reset_index(drop=True)
    per_repeat = []
    for r in range(repeats):
        probs = np.full(len(out), np.nan)
        for tr, te in grouped_splits(y_c, g_c, n_splits, RANDOM_STATE + r):
            train_subj = set(g_c[tr])
            test_subj = set(g_c[te])
            tr_rows = table[table["subject"].isin(train_subj) & table["variant"].isin(train_variants)]
            X_tr = tr_rows[features]
            y_tr = tr_rows["label"].to_numpy(int)
            g_tr = tr_rows["subject"].to_numpy()
            w = speaker_weights(g_tr) if weight_speakers else None
            model = make_model(model_name, X_tr, y_tr, g_tr,
                               is_clean=(tr_rows["variant"] == "clean").to_numpy(),
                               sample_weight=w, seed=RANDOM_STATE + r, calibrate=calibrate)
            mask = out["subject"].isin(test_subj).to_numpy()
            te_rows = out[mask].merge(table, on=["clip_id", "variant"], how="left", suffixes=("", "_t"))
            probs[mask] = model.predict_proba(te_rows[features])[:, 1]
        out[f"prob_r{r}"] = probs
        clean_mask = (out["variant"] == "clean").to_numpy()
        m = summarize(out[clean_mask], f"prob_r{r}")
        per_repeat.append(m)
        log(f"  repeat {r + 1}/{repeats}: clip AUC {m['clip']['auc']:.3f}  person AUC {m['person']['auc']:.3f}")
    out["prob"] = out[[f"prob_r{r}" for r in range(repeats)]].mean(axis=1)
    return OOFResult(clips=out, per_repeat=per_repeat)


def _binary_metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "brier": float(brier_score_loss(y, p)),
        "n": len(y),
    }


def person_table(clips: pd.DataFrame, prob_col: str = "prob") -> pd.DataFrame:
    return clips.groupby("subject").agg(label=("label", "first"), prob=(prob_col, "mean"),
                                        dataset=("dataset", "first")).reset_index()


def summarize(clips: pd.DataFrame, prob_col: str = "prob") -> dict:
    c = clips.dropna(subset=[prob_col])
    per = person_table(c, prob_col)
    return {
        "clip": _binary_metrics(c["label"].to_numpy(int), c[prob_col].to_numpy()),
        "person": _binary_metrics(per["label"].to_numpy(int), per["prob"].to_numpy()),
    }


def bootstrap_ci(clips: pd.DataFrame, prob_col: str = "prob", n_boot: int = 1000,
                 seed: int = RANDOM_STATE) -> dict:
    """Point estimate plus 95% CI from resampling speakers with replacement."""
    c = clips.dropna(subset=[prob_col]).reset_index(drop=True)
    point = summarize(c, prob_col)
    rng = np.random.default_rng(seed)
    subjects = c["subject"].unique()
    by_subj = {s: idx for s, idx in c.groupby("subject").indices.items()}
    keys = ["auc", "balanced_accuracy", "sensitivity", "specificity"]
    samples = {lvl: {k: [] for k in keys} for lvl in ("clip", "person")}
    for _ in range(n_boot):
        pick = rng.choice(subjects, size=len(subjects), replace=True)
        idx = np.concatenate([by_subj[s] for s in pick])
        b = c.iloc[idx].copy()
        # Make duplicated speakers distinct so person-level means do not merge them.
        b["subject"] = np.repeat(np.arange(len(pick)), [len(by_subj[s]) for s in pick])
        if b["label"].nunique() < 2:
            continue
        m = summarize(b, prob_col)
        for lvl in ("clip", "person"):
            for k in keys:
                samples[lvl][k].append(m[lvl][k])
    out = {}
    for lvl in ("clip", "person"):
        out[lvl] = {}
        for k in keys:
            arr = np.asarray(samples[lvl][k], float)
            arr = arr[np.isfinite(arr)]
            lo, hi = (np.percentile(arr, [2.5, 97.5]) if len(arr) else (np.nan, np.nan))
            out[lvl][k] = {"value": round(float(point[lvl][k]), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}
        out[lvl]["n"] = point[lvl]["n"]
        out[lvl]["brier"] = round(float(point[lvl]["brier"]), 4)
    return out


def derive_thresholds(clips: pd.DataFrame, prob_col: str = "prob") -> dict:
    """Pick the Likely / Unlikely cut-offs from out-of-fold clip probabilities."""
    c = clips.dropna(subset=[prob_col])
    hc = np.sort(c.loc[c["label"] == 0, prob_col].to_numpy())
    pdp = np.sort(c.loc[c["label"] == 1, prob_col].to_numpy())
    pd_high = float(np.quantile(hc, TARGET_SPECIFICITY))  # <=10% of healthy clips at or above
    hc_high = float(np.quantile(pdp, 1 - TARGET_SENSITIVITY))  # <=10% of PD clips at or below
    note = "uncertain band between the two cut-offs"
    if hc_high >= pd_high:
        mid = (hc_high + pd_high) / 2
        hc_high, pd_high = mid - 0.02, mid + 0.02
        note = "targets overlap; a narrow band around their midpoint is used"
    pd_high = float(np.clip(pd_high, 0.5, 0.95))
    hc_high = float(np.clip(hc_high, 0.05, pd_high - 0.02))
    y = c["label"].to_numpy(int)
    p = c[prob_col].to_numpy()
    band = np.where(p >= pd_high, "likely", np.where(p <= hc_high, "unlikely", "uncertain"))

    def share(mask_label: int, name: str) -> float:
        sel = y == mask_label
        return round(float((band[sel] == name).mean()), 4) if sel.any() else float("nan")

    return {
        "pd_high": round(pd_high, 4),
        "hc_high": round(hc_high, 4),
        "method": (
            f"From speaker-grouped out-of-fold calibrated clip probabilities: pd_high is the "
            f"{TARGET_SPECIFICITY:.0%} quantile of healthy clips, hc_high is the "
            f"{1 - TARGET_SENSITIVITY:.0%} quantile of Parkinson's clips; {note}."
        ),
        "observed": {
            "healthy_clips": {"unlikely": share(0, "unlikely"), "uncertain": share(0, "uncertain"), "likely": share(0, "likely")},
            "pd_clips": {"unlikely": share(1, "unlikely"), "uncertain": share(1, "uncertain"), "likely": share(1, "likely")},
        },
    }


def fit_final(name: str, table: pd.DataFrame, features: list[str],
              train_variants: tuple[str, ...] = ("clean",), weight_speakers: bool = True,
              calibrate: bool = True) -> Pipeline:
    rows = table[table["variant"].isin(train_variants)]
    g = rows["subject"].to_numpy()
    return make_model(
        name, rows[features], rows["label"].to_numpy(int), g,
        is_clean=(rows["variant"] == "clean").to_numpy(),
        sample_weight=speaker_weights(g) if weight_speakers else None,
        calibrate=calibrate,
    )


def clone_pipeline(p: Pipeline) -> Pipeline:
    return clone(p)
