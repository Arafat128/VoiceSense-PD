"""Speaker-grouped validation never leaks a speaker and reports sane CIs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from training.evaluate import bootstrap_ci, grouped_splits, repeated_grouped_oof


def _toy_table(n_speakers: int = 24, clips: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_speakers):
        label = s % 2
        center = rng.normal(1.5 * label, 1.0, size=3)
        for c in range(clips):
            x = center + rng.normal(0, 0.5, size=3)
            rows.append({"clip_id": f"s{s}c{c}", "variant": "clean", "subject": f"s{s}",
                         "dataset": "toy", "label": label, "f0": x[0], "f1": x[1], "f2": x[2]})
    return pd.DataFrame(rows)


def test_grouped_splits_keep_speakers_apart() -> None:
    table = _toy_table()
    y = table["label"].to_numpy()
    g = table["subject"].to_numpy()
    for tr, te in grouped_splits(y, g, 5, seed=3):
        assert not set(g[tr]) & set(g[te])


def test_repeated_oof_and_bootstrap() -> None:
    table = _toy_table()
    oof = repeated_grouped_oof(table, ["f0", "f1", "f2"], "logreg", repeats=2, log=lambda *_: None)
    assert oof.clips["prob"].notna().all()
    ci = bootstrap_ci(oof.clips, n_boot=200)
    for level in ("clip", "person"):
        lo, hi = ci[level]["auc"]["ci95"]
        assert lo <= ci[level]["auc"]["value"] <= hi
    assert ci["person"]["n"] == 24
    assert ci["clip"]["n"] == 96
