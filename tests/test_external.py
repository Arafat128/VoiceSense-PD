"""External-test helpers count the three bands correctly."""
from __future__ import annotations

import pandas as pd

from training.eval_external import _bands


def test_band_shares() -> None:
    table = pd.DataFrame({"label": [0, 0, 0, 0, 1, 1], "prob": [0.1, 0.2, 0.5, 0.9, 0.95, 0.3]})
    out = _bands(table, {"pd_high": 0.8, "hc_high": 0.25})
    assert out["healthy"] == {"likely": 0.25, "uncertain": 0.25, "unlikely": 0.5}
    assert out["parkinsons"] == {"likely": 0.5, "uncertain": 0.5, "unlikely": 0.0}
