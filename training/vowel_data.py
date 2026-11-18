"""Build a clip manifest for the sustained-vowel corpora.

Each row is one recording: path, dataset, speaker, label (1 PD, 0 healthy),
cohort, age, sex, vowel. Speaker IDs are prefixed with the dataset so groups
never collide across corpora.

Italian corpus notes (found while building this loader):
- Young healthy controls have no vowel recordings at all, only reading tasks.
  ``include_young_hc`` exists for the connected-speech baseline only.
- Three Parkinson's speakers were recorded twice and filed in two folders
  (Roberto R, Nicola S, Vito S). Their file codes match, so the folders are
  merged into one speaker. Splitting by folder would leak a person across
  train and test.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.config import VOWEL_DATASETS

ITALIAN_VOWEL = re.compile(r"^V([AEIOU])(\d)", re.IGNORECASE)
ITALIAN_CODE = re.compile(
    r"^(?:VA|VE|VI|VO|VU|B|PR|D|FB)\d{1,2}(.+?)(\d{2})([MF])\d{6,}", re.IGNORECASE
)
CONNECTED_PREFIXES = ("B1", "B2", "PR1", "PR2")

MANIFEST_COLUMNS = [
    "clip_id", "path", "dataset", "subject", "label", "cohort", "age", "sex", "vowel", "task",
]


def dataset_root(data_dir: Path, key: str) -> Path:
    return Path(data_dir).joinpath(*VOWEL_DATASETS[key]["relative_path"])


def _italian_cohort(path: Path) -> str | None:
    for part in path.parts:
        if "Parkinson" in part:
            return "PD"
        if "Elderly Healthy" in part:
            return "EHC"
        if "Young Healthy" in part:
            return "YHC"
    return None


def _italian_speaker_code(name: str) -> str | None:
    match = ITALIAN_CODE.match(name)
    if not match:
        return None
    letters = re.sub(r"[^a-z]", "", match.group(1).lower())
    return "".join(sorted(letters)) + match.group(2) + match.group(3).upper()


def load_italian(
    root: Path, *, task: str = "vowels", include_young_hc: bool = False
) -> pd.DataFrame:
    """task='vowels' keeps VA..VU. task='connected' keeps B1/B2/PR1/PR2 (old live model)."""
    if not root.exists():
        raise FileNotFoundError(f"Italian dataset not found: {root}")
    rows = []
    for wav in sorted(root.rglob("*.wav")):
        cohort = _italian_cohort(wav.relative_to(root))
        if cohort is None or (cohort == "YHC" and not include_young_hc):
            continue
        name = wav.name
        if task == "vowels":
            match = ITALIAN_VOWEL.match(name)
            if not match:
                continue
            vowel = match.group(1).lower()
        elif task == "connected":
            if not name.upper().startswith(CONNECTED_PREFIXES):
                continue
            vowel = ""
        else:
            raise ValueError(f"Unknown Italian task: {task}")
        folder = str(wav.parent.relative_to(root))
        rows.append(
            {
                "path": str(wav),
                "folder": folder,
                "cohort": cohort,
                "label": 1 if cohort == "PD" else 0,
                "code": _italian_speaker_code(name) if cohort == "PD" else None,
                "vowel": vowel,
                "task": task,
                "file": name,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    # Merge PD folders that hold the same speaker (identical file code).
    folder_to_subject = {f: f for f in df["folder"].unique()}
    pd_rows = df[df["code"].notna()]
    for _code, folders in pd_rows.groupby("code")["folder"]:
        uniq = sorted(set(folders))
        for folder in uniq[1:]:
            folder_to_subject[folder] = folder_to_subject[uniq[0]]
    df["subject"] = "italian::" + df["folder"].map(folder_to_subject)
    df["dataset"] = "italian"
    df["age"] = float("nan")
    df["sex"] = None
    df["clip_id"] = "italian::" + df["folder"] + "/" + df["file"]
    return df[MANIFEST_COLUMNS].reset_index(drop=True)


def load_uams(root: Path, *, min_hc_age: float | None = None) -> pd.DataFrame:
    demo_path = root / "Demographics_age_sex.xlsx"
    if not demo_path.exists():
        raise FileNotFoundError(f"UAMS demographics not found: {demo_path}")
    demo = pd.read_excel(demo_path)
    demo["Sample ID"] = demo["Sample ID"].astype(str).str.strip()
    meta = demo.set_index("Sample ID")
    rows = []
    for folder, label in (("PD_AH", 1), ("HC_AH", 0)):
        for wav in sorted((root / folder).glob("*.wav")):
            sid = wav.stem
            age = float(meta.loc[sid, "Age"]) if sid in meta.index else float("nan")
            sex = str(meta.loc[sid, "Sex"]) if sid in meta.index else None
            if label == 0 and min_hc_age is not None and not (age >= min_hc_age):
                continue
            rows.append(
                {
                    "clip_id": f"uams::{sid}",
                    "path": str(wav),
                    "dataset": "uams",
                    "subject": f"uams::{sid}",
                    "label": label,
                    "cohort": "PD" if label else "HC",
                    "age": age,
                    "sex": sex,
                    "vowel": "a",
                    "task": "vowels",
                }
            )
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def load_argentina(root: Path) -> pd.DataFrame:
    info_path = root / "data_information.csv"
    info = pd.read_csv(info_path, sep=";").dropna(how="all") if info_path.exists() else None
    ages = {}
    sexes = {}
    if info is not None:
        for _, row in info.iterrows():
            ages[str(int(row["ID"]))] = float(row["Age"])
            sexes[str(int(row["ID"]))] = str(row["Sex"])
    rows = []
    for folder, label in (("A_EP_Audios", 1), ("A_noEP_Audios", 0)):
        for wav in sorted((root / folder).glob("*.wav")):
            sid = re.sub(r"[^0-9]", "", wav.stem)
            rows.append(
                {
                    "clip_id": f"argentina::{wav.stem}",
                    "path": str(wav),
                    "dataset": "argentina",
                    "subject": f"argentina::{sid}",
                    "label": label,
                    "cohort": "PD" if label else "HC",
                    "age": ages.get(sid, float("nan")),
                    "sex": sexes.get(sid),
                    "vowel": "a",
                    "task": "vowels",
                }
            )
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def load_manifest(
    data_dir: Path,
    datasets: tuple[str, ...] = ("italian",),
    *,
    include_young_hc: bool = False,
    min_hc_age: float | None = None,
    italian_task: str = "vowels",
    vowels: str = "aeiou",
) -> pd.DataFrame:
    frames = []
    for key in datasets:
        root = dataset_root(data_dir, key)
        if key == "italian":
            df = load_italian(root, task=italian_task, include_young_hc=include_young_hc)
            if italian_task == "vowels":
                df = df[df["vowel"].isin(list(vowels.lower()))]
        elif key == "uams":
            df = load_uams(root, min_hc_age=min_hc_age)
        elif key == "argentina":
            df = load_argentina(root)
        else:
            raise KeyError(f"Unknown vowel dataset: {key}. Choose from {list(VOWEL_DATASETS)}")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=MANIFEST_COLUMNS)
    if out.empty:
        raise RuntimeError(f"No recordings found for {datasets} under {data_dir}")
    return out


def describe(manifest: pd.DataFrame) -> dict:
    out = {}
    for key, part in manifest.groupby("dataset"):
        subj = part.drop_duplicates("subject")
        out[key] = {
            "clips": int(len(part)),
            "speakers": int(len(subj)),
            "pd_speakers": int((subj["label"] == 1).sum()),
            "hc_speakers": int((subj["label"] == 0).sum()),
            "pd_clips": int((part["label"] == 1).sum()),
            "hc_clips": int((part["label"] == 0).sum()),
            "mean_age_pd": _mean(subj.loc[subj["label"] == 1, "age"]),
            "mean_age_hc": _mean(subj.loc[subj["label"] == 0, "age"]),
        }
    return out


def _mean(series: pd.Series) -> float | None:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return round(float(vals.mean()), 1) if len(vals) else None
