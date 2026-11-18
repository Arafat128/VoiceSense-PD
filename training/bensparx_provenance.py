"""Trace each public BenSParX row back to the UCI Parkinson's speech table it matches.

The public ``BenSParX.csv`` (github.com/riadEDU/BenSParX) is described as 900
ten-second segments of Bengali phone-call speech from 120 people. Every one of
its 900 rows is, column for column, a row of the UCI "Parkinson's Disease
Classification" table (Sakar et al. 2019, Turkish sustained /a/,
``pd_speech_features.csv``) multiplied by one constant per row (1.027-1.092),
with the same PD/healthy label. Several columns are renamed on the way
(e.g. ``Delta_MFCC_k`` holds the mean MFCC k, ``meanPitch`` holds VFER_NSR_SEO),
and ``MFCC_k`` is exactly 0.12 x ``Delta_MFCC_k``.

This module recomputes that match from the two public CSVs so the claim can be
checked, and so validation can group rows by the person they really come from.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from training.common import REPORTS_DIR, resolve_data_dir

BENSPARX_PATH = ("02_BenSParX", "BenSParX-main", "BenSParX.csv")
UCI_PD_SPEECH_PATH = ("03_UCI_PD_Classification", "pd_speech_features.csv")
MAPPING_CSV = REPORTS_DIR / "bensparx_to_uci_pd_speech.csv"
SUMMARY_JSON = REPORTS_DIR / "bensparx_provenance.json"

# Columns that keep their name and meaning in both tables.
SAME_NAME = [
    "numPulses", "numPeriodsPulses", "meanPeriodPulses", "stdDevPeriodPulses",
    "locPctJitter", "locAbsJitter", "rapJitter", "ppq5Jitter", "ddpJitter",
    "locShimmer", "locDbShimmer", "apq3Shimmer", "apq5Shimmer", "apq11Shimmer", "ddaShimmer",
    "minIntensity", "maxIntensity", "meanIntensity", "f1", "f2", "f3", "f4",
    "mean_Log_energy", "std_Log_energy", "std_delta_log_energy", "std_delta_delta_log_energy",
]
_ORD = {0: "0th", 1: "1st", 2: "2nd", 3: "3rd"}
# BenSParX column -> UCI pd_speech_features column holding the same numbers (x row scale).
RENAMED = {
    "AutoCorrHarmonicity": "meanAutoCorrHarmonicity",
    "meanAutoCorrHarmonicity": "meanNoiseToHarmHarmonicity",
    "NHR": "meanHarmToNoiseHarmonicity",
    "maxPitch": "GQ_std_cycle_open",
    "HNR": "GNE_NSR_SEO",
    "meanPitch": "VFER_NSR_SEO",
    **{f"Delta_MFCC_{k}": f"mean_MFCC_{_ORD.get(k, f'{k}th')}_coef" for k in range(13)},
    **{f"DeltaDelta_MFCC_{k}": f"std_MFCC_{_ORD.get(k, f'{k}th')}_coef" for k in range(13)},
}
COLUMN_MAP = {**{c: c for c in SAME_NAME}, **RENAMED}
EXACT_SD = 0.01  # sd of log-ratios across the shared columns; real matches sit near 0.0005


def load_tables(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ben = pd.read_csv(Path(data_dir).joinpath(*BENSPARX_PATH))
    uci = pd.read_csv(Path(data_dir).joinpath(*UCI_PD_SPEECH_PATH), header=1)
    return ben, uci


def match_rows(ben: pd.DataFrame, uci: pd.DataFrame) -> pd.DataFrame:
    """For each BenSParX row, the UCI row with the most constant value ratio."""
    B = np.log(np.abs(ben[SAME_NAME].to_numpy(float)) + 1e-12)
    U = np.log(np.abs(uci[SAME_NAME].to_numpy(float)) + 1e-12)
    rows = []
    for i in range(len(ben)):
        d = B[i][None, :] - U
        sd = d.std(axis=1)
        order = np.argsort(sd)
        j, second = int(order[0]), float(sd[order[1]])
        rows.append({
            "bensparx_id": ben["id"].iat[i],
            "bensparx_speaker": str(ben["id"].iat[i]).rsplit("_", 1)[0],
            "bensparx_label": int(ben["class"].iat[i]),
            "uci_row": j,
            "uci_subject": f"uci{int(uci['id'].iat[j])}",
            "uci_label": int(uci["class"].iat[j]),
            "scale": round(float(np.exp(d[j].mean())), 5),
            "logratio_sd": round(float(sd[j]), 6),
            "next_best_sd": round(second, 6),
        })
    return pd.DataFrame(rows)


def renamed_column_check(ben: pd.DataFrame, uci: pd.DataFrame, m: pd.DataFrame) -> dict[str, float]:
    """Median relative error of BenSParX column vs scale x mapped UCI column."""
    src = uci.iloc[m["uci_row"].to_numpy()].reset_index(drop=True)
    out = {}
    for b_col, u_col in COLUMN_MAP.items():
        pred = src[u_col].to_numpy(float) * m["scale"].to_numpy(float)
        v = ben[b_col].to_numpy(float)
        out[b_col] = round(float(np.median(np.abs(v - pred) / (np.abs(v) + 1e-12))), 6)
    mfcc = {}
    for k in range(13):
        r = ben[f"MFCC_{k}"] / ben[f"Delta_MFCC_{k}"]
        mfcc[f"MFCC_{k}"] = round(float(r.median()), 4)
    return {"max_median_rel_error": max(out.values()), "per_column": out, "MFCC_over_Delta_MFCC": mfcc}


def summarize(m: pd.DataFrame, columns: dict) -> dict:
    exact = m["logratio_sd"] < EXACT_SD
    per_speaker = m[exact].groupby("bensparx_speaker")["uci_subject"].nunique()
    per_subject = m[exact].groupby("uci_subject")["bensparx_speaker"].nunique()
    return {
        "rows": len(m),
        "rows_with_exact_scaled_match": int(exact.sum()),
        "label_agreement": round(float((m["bensparx_label"] == m["uci_label"]).mean()), 4),
        "logratio_sd_max": float(m["logratio_sd"].max()),
        "next_best_sd_median": float(m["next_best_sd"].median()),
        "scale_values": {str(k): int(v) for k, v in m["scale"].round(3).value_counts().items()},
        "distinct_uci_recordings": int(m["uci_row"].nunique()),
        "uci_recordings_used_twice_or_more": int((m["uci_row"].value_counts() > 1).sum()),
        "distinct_uci_subjects": int(m["uci_subject"].nunique()),
        "uci_subjects_split_across_bensparx_speakers": int((per_subject > 1).sum()),
        "median_uci_subjects_per_bensparx_speaker": float(per_speaker.median()),
        "column_check": columns,
        "conclusion": (
            "Every BenSParX row equals a UCI pd_speech_features row (Sakar et al. 2019, Turkish "
            "sustained /a/) times a per-row constant, with the same label. BenSParX 'speakers' are "
            "mixtures of several UCI subjects, so grouping by BenSParX speaker does not stop the "
            "same person (often the same recording) appearing in train and test."
        ),
    }


def build(data_dir: Path, write: bool = True) -> tuple[pd.DataFrame, dict]:
    ben, uci = load_tables(data_dir)
    m = match_rows(ben, uci)
    s = summarize(m, renamed_column_check(ben, uci, m))
    if write:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        m.to_csv(MAPPING_CSV, index=False)
        SUMMARY_JSON.write_text(json.dumps(s, indent=2), encoding="utf-8")
    return m, s


def load_mapping(data_dir: Path | None = None) -> pd.DataFrame:
    """Saved mapping if present, else recompute from the two CSVs."""
    if MAPPING_CSV.exists():
        return pd.read_csv(MAPPING_CSV)
    if data_dir is None:
        raise FileNotFoundError(f"{MAPPING_CSV} missing; pass --data-dir to rebuild it")
    return build(data_dir)[0]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args(argv)
    _, s = build(resolve_data_dir(args.data_dir))
    print(json.dumps({k: v for k, v in s.items() if k != "column_check"}, indent=2))
    print("max median relative error over mapped columns:", s["column_check"]["max_median_rel_error"])


if __name__ == "__main__":
    main()
