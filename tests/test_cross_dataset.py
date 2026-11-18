"""Cross-dataset report: copied rows are flagged, not counted as transfer."""
from __future__ import annotations

import json

from src.config import ROOT
from training.cross_dataset import COLUMNS


def _res() -> dict:
    return json.loads((ROOT / "reports" / "bensparx_cross_dataset.json").read_text(encoding="utf-8"))


def test_same_people_pairs_are_only_bensparx_and_uci_pd_speech() -> None:
    for r in _res()["pairs"]:
        assert r["same_people"] == ({r["train"], r["test"]} == {"bensparx", "uci_pd_speech"})


def test_every_direction_is_reported_with_ci() -> None:
    pairs = {(r["train"], r["test"]) for r in _res()["pairs"]}
    others = ("uci_replicated", "uci_parkinsons", "uci_pd_speech")
    for o in others:
        assert ("bensparx", o) in pairs and (o, "bensparx") in pairs
    for r in _res()["pairs"]:
        lo, hi = r["person_auc"]["ci95"]
        assert lo <= r["person_auc"]["value"] <= hi


def test_shared_measures_are_perturbation_only() -> None:
    shared = set.intersection(*(set(c) for c in COLUMNS.values()))
    assert shared and not any("mfcc" in f for f in shared)
