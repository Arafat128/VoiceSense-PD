"""Score every row of an uploaded feature table with one feature-table model."""
from __future__ import annotations

import io
from typing import Any

import pandas as pd

from .models.registry import load_model
from .predict import table_label

MAX_ROWS = 20_000
MAX_BYTES = 25 * 1024 * 1024
ID_COLUMNS = ("sample_id", "id", "name", "ID", "subject_id")
LABEL_COLUMNS = ("true_label", "class", "status", "Status", "label")
DECISION_TEXT = {1: "PD side", 0: "Healthy side", -1: "Uncertain"}


class BatchError(ValueError):
    """A problem with the uploaded table that the user can fix."""


def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    if not raw:
        raise BatchError("The file is empty.")
    if len(raw) > MAX_BYTES:
        raise BatchError(f"The file is {len(raw) / 1024 / 1024:.0f} MB; upload a CSV under 25 MB.")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise BatchError(f"This does not read as a CSV file ({type(exc).__name__}).") from exc
    if df.empty:
        raise BatchError("The CSV has a header but no rows.")
    return df


def validate(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Numeric feature frame in model order, or a BatchError naming what is wrong."""
    if len(df) > MAX_ROWS:
        raise BatchError(f"The CSV has {len(df):,} rows; the limit is {MAX_ROWS:,}.")
    cols = {str(c).strip(): c for c in df.columns}
    missing = [f for f in feature_names if f not in cols]
    if missing:
        near = [c for c in cols if c.lower() in {m.lower() for m in missing}]
        hint = f" Check capitalisation of: {', '.join(near[:5])}." if near else ""
        raise BatchError(
            f"Missing {len(missing)} of {len(feature_names)} required columns, e.g. "
            f"{', '.join(missing[:8])}.{hint} Use the column names of the chosen dataset."
        )
    X = df[[cols[f] for f in feature_names]].copy()
    X.columns = feature_names
    num = X.apply(pd.to_numeric, errors="coerce")
    bad = num.isna() & X.notna()
    if bad.any().any():
        col = bad.any().idxmax()
        rows = (bad[col][bad[col]].index[:5] + 1).tolist()
        raise BatchError(f"Column '{col}' has non-numeric values (data rows {rows}).")
    empty = num.isna().any(axis=1)
    if empty.any():
        rows = (num.index[empty][:5] + 1).tolist()
        raise BatchError(f"{int(empty.sum())} rows have empty feature cells (data rows {rows}).")
    return num


def score_table(dataset_key: str, df: pd.DataFrame, model_name: str = "random_forest") -> pd.DataFrame:
    pack = load_model(dataset_key, model_name)
    X = validate(df, pack["feature_names"])
    proba = pack["pipeline"].predict_proba(X)[:, 1]
    labels = [table_label(pack, float(p)) for p in proba]
    out = pd.DataFrame({"row": range(1, len(df) + 1)})
    for c in ID_COLUMNS:
        if c in df.columns:
            out[c] = df[c].to_numpy()
            break
    out["probability_pd"] = proba.round(4)
    out["decision"] = [DECISION_TEXT[v] for v in labels]
    for c in LABEL_COLUMNS:
        if c in df.columns and set(pd.to_numeric(df[c], errors="coerce").dropna().unique()) <= {0, 1}:
            out["true_label"] = pd.to_numeric(df[c], errors="coerce").map({1: "PD", 0: "HC"}).to_numpy()
            break
    out["model"] = f"{dataset_key}/{model_name}"
    return out


def summary(scored: pd.DataFrame) -> dict[str, Any]:
    counts = scored["decision"].value_counts().to_dict()
    out: dict[str, Any] = {"rows": len(scored), "counts": counts}
    if "true_label" in scored:
        called = scored[scored["decision"] != "Uncertain"]
        if len(called):
            hit = ((called["decision"] == "PD side") == (called["true_label"] == "PD")).mean()
            out["agreement_on_called_rows"] = float(hit)
            out["called_rows"] = len(called)
    return out
