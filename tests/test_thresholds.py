"""Decision cut-offs come from the model metadata, not from code."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.registry import load_live_audio_model
from src.predict import LEGACY_THRESHOLDS, _decide_from_proba, thresholds_for
from training.evaluate import derive_thresholds


def test_live_models_carry_data_derived_thresholds() -> None:
    for name in ("random_forest", "svm_rbf"):
        pack = load_live_audio_model(name)
        thr = thresholds_for(pack)
        assert thr != LEGACY_THRESHOLDS
        assert 0.0 < thr["hc_high"] < thr["pd_high"] < 1.0
        assert "quantile" in str(pack.get("threshold_method"))


def test_bands_follow_the_saved_cutoffs() -> None:
    thr = {"pd_high": 0.6, "hc_high": 0.3}
    assert _decide_from_proba(0.65, thr)["decision_code"] == "PD"
    assert _decide_from_proba(0.45, thr)["decision_code"] == "UNCERTAIN"
    assert _decide_from_proba(0.25, thr)["decision_code"] == "HC"


def test_derive_thresholds_hits_targets() -> None:
    rng = np.random.default_rng(1)
    hc = rng.beta(2, 5, 400)
    pdp = rng.beta(5, 2, 400)
    clips = pd.DataFrame({"label": [0] * 400 + [1] * 400, "prob": np.r_[hc, pdp]})
    thr = derive_thresholds(clips)
    assert abs((hc >= thr["pd_high"]).mean() - 0.10) < 0.02
    assert abs((pdp <= thr["hc_high"]).mean() - 0.10) < 0.02
    assert thr["hc_high"] < thr["pd_high"]
