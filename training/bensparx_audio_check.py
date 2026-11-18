"""Can a recording be turned into a BenSParX-style row the Bangla model can score?

Research check, not shipped in the app. The public BenSParX columns hold the
UCI pd_speech_features values (Sakar et al. 2019) times a per-row constant,
several under other names (training/bensparx_provenance.py). This module
rebuilds as much of that row as Praat and librosa allow:

- measured with Praat's voice report, the same tool Sakar et al. name:
  pulses, periods, jitter, shimmer, autocorrelation, NHR, HNR (18 columns);
- approximated: intensity, formants f1-f4, MFCC mean/std and log-energy
  statistics, whose exact settings (frame sizes, MFCC toolbox, scaling) are
  not published (50 columns);
- not reproducible: the three Tsanas-toolbox glottal measures stored under
  maxPitch / HNR / meanPitch (GQ_std_cycle_open, GNE_NSR_SEO, VFER_NSR_SEO),
  filled with the training median (3 columns);
- then every value is multiplied by 1.07, the most common BenSParX row scale.

It then scores labelled sustained-vowel recordings (Italian, UAMS, Argentina)
with the shipped BenSParX model and reports how often each rebuilt column
falls inside the BenSParX range, and the per-person AUC.

Offline only:  python -m training.bensparx_audio_check --data-dir <vowel corpora root>
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from training.common import REPORTS_DIR, resolve_data_dir
from training.evaluate import bootstrap_ci

ROW_SCALE = 1.07
MEASURED = [
    "numPulses", "numPeriodsPulses", "meanPeriodPulses", "stdDevPeriodPulses",
    "locPctJitter", "locAbsJitter", "rapJitter", "ppq5Jitter", "ddpJitter",
    "locShimmer", "locDbShimmer", "apq3Shimmer", "apq5Shimmer", "apq11Shimmer", "ddaShimmer",
    "AutoCorrHarmonicity", "meanAutoCorrHarmonicity", "NHR",
]
NOT_REPRODUCIBLE = ["maxPitch", "HNR", "meanPitch"]
_REPORT = {
    "numPulses": r"Number of pulses: (\S+)",
    "numPeriodsPulses": r"Number of periods: (\S+)",
    "meanPeriodPulses": r"Mean period: (\S+) seconds",
    "stdDevPeriodPulses": r"Standard deviation of period: (\S+) seconds",
    "locPctJitter": r"Jitter \(local\): (\S+)%",
    "locAbsJitter": r"Jitter \(local, absolute\): (\S+) seconds",
    "rapJitter": r"Jitter \(rap\): (\S+)%",
    "ppq5Jitter": r"Jitter \(ppq5\): (\S+)%",
    "ddpJitter": r"Jitter \(ddp\): (\S+)%",
    "locShimmer": r"Shimmer \(local\): (\S+)%",
    "locDbShimmer": r"Shimmer \(local, dB\): (\S+) dB",
    "apq3Shimmer": r"Shimmer \(apq3\): (\S+)%",
    "apq5Shimmer": r"Shimmer \(apq5\): (\S+)%",
    "apq11Shimmer": r"Shimmer \(apq11\): (\S+)%",
    "ddaShimmer": r"Shimmer \(dda\): (\S+)%",
    "AutoCorrHarmonicity": r"Mean autocorrelation: (\S+)",
    "meanAutoCorrHarmonicity": r"Mean noise-to-harmonics ratio: (\S+)",
    "NHR": r"Mean harmonics-to-noise ratio: (\S+) dB",
}
_PERCENT = {k for k, v in _REPORT.items() if v.endswith("%")}


def _num(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return float("nan")


def bensparx_row(path: str, medians: dict[str, float]) -> dict[str, float]:
    """Best-effort BenSParX-style feature row for one recording."""
    import librosa
    import parselmouth
    from parselmouth.praat import call

    snd = parselmouth.Sound(path)
    pitch = call(snd, "To Pitch (cc)", 0, 75, 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14, 600)
    pulses = call(snd, "To PointProcess (periodic, cc)", 75, 600)
    report = call([snd, pitch, pulses], "Voice report", 0, 0, 75, 600, 1.3, 1.6, 0.03, 0.45)
    row: dict[str, float] = {}
    for col, pat in _REPORT.items():
        m = re.search(pat, report)
        v = _num(m.group(1)) if m else float("nan")
        row[col] = v / 100.0 if col in _PERCENT else v

    inten = call(snd, "To Intensity", 75, 0, "yes")
    row["minIntensity"] = call(inten, "Get minimum", 0, 0, "Parabolic")
    row["maxIntensity"] = call(inten, "Get maximum", 0, 0, "Parabolic")
    row["meanIntensity"] = call(inten, "Get mean", 0, 0, "energy")
    formant = call(snd, "To Formant (burg)", 0, 5, 5500, 0.025, 50)
    for k in range(1, 5):
        row[f"f{k}"] = call(formant, "Get mean", k, 0, 0, "hertz")

    y, sr = librosa.load(path, sr=16000, mono=True)
    pcm = y * 32768.0
    frame, hop = int(0.025 * sr), int(0.010 * sr)
    frames = librosa.util.frame(np.pad(pcm, (0, frame)), frame_length=frame, hop_length=hop)
    log_e = np.log10(np.sum(frames.astype(float) ** 2, axis=0) + 1e-9)
    d1 = librosa.feature.delta(log_e[None, :], order=1)[0]
    d2 = librosa.feature.delta(log_e[None, :], order=2)[0]
    row["mean_Log_energy"] = float(np.mean(log_e))
    row["std_Log_energy"] = float(np.std(log_e))
    row["std_delta_log_energy"] = float(np.std(d1))
    row["std_delta_delta_log_energy"] = float(np.std(d2))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=frame, hop_length=hop, n_mels=26, htk=True) / 20.0
    for k in range(13):
        row[f"Delta_MFCC_{k}"] = float(np.mean(mfcc[k]))
        row[f"DeltaDelta_MFCC_{k}"] = float(np.std(mfcc[k]))
        row[f"MFCC_{k}"] = 0.12 * row[f"Delta_MFCC_{k}"]
    for col in NOT_REPRODUCIBLE:
        row[col] = float(medians[col]) / ROW_SCALE
    return {k: float(v) * ROW_SCALE for k, v in row.items()}


def run(data_dir, n_jobs: int = -1) -> dict:
    from src.models.registry import load_holdout_samples, load_model
    from training.vowel_data import load_manifest

    pack = load_model("bengali_bensparx", "random_forest")
    feats = pack["feature_names"]
    ben_csv = Path(data_dir) / "02_BenSParX" / "BenSParX-main" / "BenSParX.csv"
    # Reference range: the full BenSParX table if present, else the shipped hold-out rows.
    ben = pd.read_csv(ben_csv) if ben_csv.exists() else load_holdout_samples("bengali_bensparx")
    manifest = load_manifest(data_dir, ("italian", "uams", "argentina"), min_hc_age=50.0)
    rows = Parallel(n_jobs=n_jobs)(delayed(_safe_row)(p, pack["train_medians"]) for p in manifest["path"])
    X = pd.DataFrame(rows)
    ok = X[feats].notna().all(axis=1).to_numpy()
    table = manifest.loc[ok, ["clip_id", "subject", "dataset", "label"]].copy().reset_index(drop=True)
    X = X[ok].reset_index(drop=True)
    out: dict = {"clips_scored": int(ok.sum()), "clips_failed": int((~ok).sum()), "recipe": __doc__.split("\n\n")[1]}
    ref = ben[feats]
    lo, hi = ref.quantile(0.01), ref.quantile(0.99)
    inside = ((X[feats] >= lo) & (X[feats] <= hi)).mean()
    groups = {"measured": MEASURED, "not_reproducible": NOT_REPRODUCIBLE,
              "approximated": [f for f in feats if f not in MEASURED and f not in NOT_REPRODUCIBLE]}
    out["share_inside_bensparx_1_99_range"] = {
        g: {"mean": round(float(inside[c].mean()), 3), "per_column": inside[c].round(3).to_dict()}
        for g, c in groups.items()
    }
    out["median_ratio_rebuilt_to_bensparx"] = (X[feats].median() / ref.median()).round(3).to_dict()
    table["prob"] = pack["pipeline"].predict_proba(X[feats])[:, 1]
    out["auc_all"] = bootstrap_ci(table, n_boot=500)["person"]["auc"]
    out["auc_by_dataset"] = {d: bootstrap_ci(part, n_boot=500)["person"]["auc"] for d, part in table.groupby("dataset")}
    thr = pack["decision_thresholds"]
    band = np.where(table["prob"] >= thr["pd_high"], "pd", np.where(table["prob"] <= thr["hc_high"], "hc", "uncertain"))
    out["band_shares"] = {lab: pd.Series(band[table["label"] == v]).value_counts(normalize=True).round(3).to_dict()
                          for lab, v in (("pd_clips", 1), ("healthy_clips", 0))}
    out["mean_prob"] = round(float(table["prob"].mean()), 3)
    a = out["auc_all"]["value"]
    measured_in = out["share_inside_bensparx_1_99_range"]["measured"]["mean"]
    approx_in = out["share_inside_bensparx_1_99_range"]["approximated"]["mean"]
    out["ship_in_ui"] = bool(a >= 0.65 and approx_in >= 0.8)
    out["verdict"] = (
        f"Rebuilt rows: {measured_in:.0%} of Praat-measured values and {approx_in:.0%} of approximated values "
        f"fall inside the BenSParX 1-99% range; 3 columns cannot be reproduced. Scored by the BenSParX "
        f"model, labelled vowel recordings give person AUC {a:.2f} "
        f"({out['auc_all']['ci95'][0]:.2f}-{out['auc_all']['ci95'][1]:.2f}). "
        + ("Shipped as experimental." if out["ship_in_ui"] else
           "Not close enough to ship: the audio-to-BenSParX path stays out of the app.")
    )
    return out


def _safe_row(path, medians):
    try:
        return bensparx_row(str(path), medians)
    except Exception:  # noqa: BLE001 - a failed clip is counted, not fatal
        return {}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Audio -> BenSParX-style row feasibility check.")
    ap.add_argument("--data-dir", default=None, help="Root holding the vowel corpora.")
    args = ap.parse_args(argv)
    res = run(resolve_data_dir(args.data_dir))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "bensparx_audio_check.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(res["verdict"])


if __name__ == "__main__":
    main()
