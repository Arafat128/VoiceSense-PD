"""BenSParX model: person-grouped validation, data cut-offs, and the provenance finding."""
from __future__ import annotations

import json

import pandas as pd

from src.config import ROOT
from src.models.registry import load_holdout_samples, load_model
from src.predict import predict_from_features, table_label

REPORTS = ROOT / "reports"


def _report() -> dict:
    return json.loads((REPORTS / "bengali_bensparx__metrics.json").read_text(encoding="utf-8"))


def test_provenance_report_matches_every_row() -> None:
    prov = json.loads((REPORTS / "bensparx_provenance.json").read_text(encoding="utf-8"))
    assert prov["rows"] == 900
    assert prov["rows_with_exact_scaled_match"] == 900
    assert prov["label_agreement"] == 1.0
    assert prov["column_check"]["max_median_rel_error"] < 0.01


def test_old_split_leaked_people_and_is_reported() -> None:
    leak = _report()["old_model"]["leakage"]
    assert leak["test_rows_with_same_uci_person_in_train"] > 0.5


def test_shipped_models_use_person_grouped_metrics_and_cutoffs() -> None:
    rep = _report()
    assert rep["headline_grouping"] == "uci_subject"
    for name in ("random_forest", "svm_rbf"):
        pack = load_model("bengali_bensparx", name)
        m = pack["metrics"]
        assert "uci_subject" in m["protocol"]
        lo, hi = m["person_auc_ci95"]
        assert lo <= m["person_auc"] <= hi
        thr = pack["decision_thresholds"]
        assert 0.05 <= thr["hc_high"] < thr["pd_high"] <= 0.95


def test_holdout_people_never_appear_in_training() -> None:
    mapping = pd.read_csv(REPORTS / "bensparx_to_uci_pd_speech.csv")
    hold = set(load_holdout_samples("bengali_bensparx")["sample_id"])
    subj = mapping.set_index("bensparx_id")["uci_subject"]
    test_people = set(subj[subj.index.isin(hold)])
    train_people = set(subj[~subj.index.isin(hold)])
    assert test_people and not (test_people & train_people)


def test_table_label_bands() -> None:
    pack = {"decision_thresholds": {"pd_high": 0.8, "hc_high": 0.4}}
    assert table_label(pack, 0.85) == 1
    assert table_label(pack, 0.6) == -1
    assert table_label(pack, 0.3) == 0
    assert table_label({}, 0.6) == 1


def test_bangla_row_scores_with_cutoffs() -> None:
    pack = load_model("bengali_bensparx", "random_forest")
    row = load_holdout_samples("bengali_bensparx").iloc[0][pack["feature_names"]]
    out = predict_from_features("bengali_bensparx", row, "random_forest")
    assert out["pred_label"] in (-1, 0, 1)
    assert out["thresholds"] == pack["decision_thresholds"]
