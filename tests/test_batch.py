"""Feature lab batch CSV: validation messages and scoring every row."""
from __future__ import annotations

import pandas as pd
import pytest

from src.batch import BatchError, read_csv_bytes, score_table, summary, validate
from src.models.registry import load_holdout_samples, load_model


def _csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def test_scores_every_row_and_keeps_ids_and_labels() -> None:
    hold = load_holdout_samples("bengali_bensparx").head(25)
    scored = score_table("bengali_bensparx", read_csv_bytes(_csv(hold)), "random_forest")
    assert len(scored) == 25
    assert list(scored["sample_id"]) == list(hold["sample_id"])
    assert scored["probability_pd"].between(0, 1).all()
    assert set(scored["decision"]) <= {"PD side", "Healthy side", "Uncertain"}
    s = summary(scored)
    assert s["rows"] == 25 and sum(s["counts"].values()) == 25


def test_missing_columns_are_named() -> None:
    feats = load_model("bengali_bensparx")["feature_names"]
    df = load_holdout_samples("bengali_bensparx").drop(columns=feats[:3]).head(3)
    with pytest.raises(BatchError, match=f"Missing 3 of {len(feats)} required columns, e.g. {feats[0]}"):
        validate(df, feats)


def test_wrong_dataset_columns_are_refused() -> None:
    feats = load_model("bengali_bensparx")["feature_names"]
    english = load_holdout_samples("english_uci").head(3)
    with pytest.raises(BatchError, match="required columns"):
        validate(english, feats)


def test_non_numeric_cells_point_to_the_row() -> None:
    feats = load_model("bengali_bensparx")["feature_names"]
    df = load_holdout_samples("bengali_bensparx").head(4).copy()
    df[feats[1]] = df[feats[1]].astype(object)
    df.loc[2, feats[1]] = "n/a?"
    with pytest.raises(BatchError, match=rf"Column '{feats[1]}' has non-numeric values \(data rows \[3\]\)"):
        validate(df, feats)


def test_empty_and_header_only_files() -> None:
    with pytest.raises(BatchError, match="empty"):
        read_csv_bytes(b"")
    with pytest.raises(BatchError, match="no rows"):
        read_csv_bytes(b"a,b,c\n")
