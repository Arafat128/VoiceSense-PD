# VoiceSense — Parkinson’s Voice Screening

Premium research app for **voice-based Parkinson’s disease (PD) screening** with:

- **Live microphone / audio upload** (English & Bangla UI modes)
- **On-device processing** (local Streamlit: recording discarded after analysis)
- **Feature-table lab** (UCI English + BenSParX Bengali)
- **Explainable outputs** (why PD / why not / uncertain)
- **Conservative decision policy** to reduce false PD flags on laptop mics

> **Not a medical diagnosis.** Screening support / research only.

The live model is trained **only on Italian recordings** (public Italian Parkinson’s Voice). English and Bangla on the live screen are interface modes. They do not mean the live model heard English or Bangla speech. The UCI and BenSParX models are separate feature-table checks.

## Quick start

```bash
git clone https://github.com/Arafat128/VoiceSense-PD.git
cd VoiceSense-PD
python -m pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

Open the local URL (usually `http://localhost:8501`).

Set `VOICESENSE_DEBUG=1` before that command to show the raw traceback after a failed live analysis. Without it, the page shows the error message only.

### Project assistant

The round chat bubble on every page answers only from a short project note. It does not see recordings or scores, and it is not a diagnosis.

```bash
cp .env.example .env
```

Put your xAI key in `XAI_API_KEY` inside `.env` (or in gitignored `.streamlit/secrets.toml`). Restart the app. The key stays on the server. If the key is missing, the rest of the app still runs and the chat says the assistant is not configured.

`VOICESENSE_ASSISTANT_MODEL` overrides the model. The default is `grok-4`.

### Tests

```bash
python -m pip install ruff pytest
python -m ruff check .
python -m pytest
```

CI on GitHub runs the same ruff and pytest commands on Python 3.13.

### Train models (if needed)

Datasets are not in this repo. Point `VOICESENSE_DATA_DIR` or `--data-dir` at the folder that contains `01_UCI_Parkinsons`, `02_BenSParX`, and `07_Italian_PD_Voice`.

```bash
python -m training.train_tables --data-dir /path/to/datasets
python -m training.train_audio --data-dir /path/to/datasets
```

`training/` is offline. The Streamlit app does not import it.

## Live model numbers

The default live model is the Italian random forest (`live_audio_italian__random_forest.joblib`).

Subject-aware **5-fold CV F1 is about 0.90 ± 0.08** (0.899 ± 0.081). That is the stability estimate.

The hold-out is small: **34 clips from 12 subjects**, accuracy 0.971, F1 0.963, ROC-AUC 0.993. A number that high on 34 clips is not the same claim as the cross-validation. Read the CV figure next to it. The SVM hold-out F1 on the same split is 0.833.

## Privacy (on-device screening)

Local Streamlit is the supported privacy mode:

- Audio is scored in the Python process on **your machine**.
- A successful **Analyze voice** run **discards** the recording from Streamlit widget/session memory.
- **Discard recording** drops a clip without scoring.
- Export is an optional JSON **receipt with no WAV** and no upload filename.
- Decode may use a short-lived tempfile that is deleted immediately (librosa/container support).

**Streamlit Community Cloud is not on-device.** The browser must upload audio to the host. The app shows a warning in that case.

Identity proofs are not part of this repository.

## IDM popup (Error 0x80080005)

If **Internet Download Manager** shows *Cannot transfer the download to IDM*:

1. That is **not** an app crash — IDM intercepts browser media.
2. In the app, leave **Preview captured audio** **off** (default).
3. Or exclude `localhost` from IDM browser integration.

Analysis converts audio **in memory** to WAV and does not need a file download.

## Decision policy (live voice)

| Band | Rule (default) | Meaning |
|------|------------------|---------|
| Healthy / non-PD | P(PD) ≤ ~42% | No clear PD pattern |
| Uncertain | middle | Not enough evidence to flag PD |
| Possible PD | P(PD) ≥ ~72% | High score only — still not diagnosis |

A firm call also needs a clip of at least 4 seconds, low noise, mostly voiced audio, and agreement from the second model.

## Where to start

1. `app/streamlit_app.py` — the website frame (it does not score audio)
2. `app/pages/live.py` — record, upload, analyze
3. `src/__init__.py` — map of the engine, one line per file

## Project layout

```
app/streamlit_app.py     # start the site; picks one screen
app/pages/               # live, feature, privacy, about
app/assistant_ui.py      # floating project chat
app/widgets.py           # shared result cards and flash notices
app/styles.css           # visual design
app/styles.py            # loads that CSS
src/predict.py           # score a row or a recording
src/models/registry.py   # load all 7 joblib models and both holdout tables
src/features/audio.py    # Praat / librosa features
src/explain.py           # technical "why"
src/plain_language.py    # everyday sentences
src/privacy.py           # receipt with no WAV
src/assistant/           # grounded Grok chat; knowledge.md is the only source
src/config.py            # constants and the feature glossary
training/                # offline only; not imported by the app
models/                  # trained files; do not edit by hand
reports/                 # metrics written by training
tests/                   # pytest
```

## Deploy notes

| Target | Status |
|--------|--------|
| **Local Streamlit** | Full app (recommended) |
| **Streamlit Community Cloud** | Possible if system audio libs available (`packages.txt` installs libsndfile) |
| **Vercel** | Not used — the ML stack is not serverless-friendly |

## Citation / data

Uses public corpora including UCI Parkinsons, BenSParX, and Italian Parkinson’s Voice (for live-audio training). Cite original dataset authors in academic work.

## License

Research / educational use. Provide proper dataset citations. Not for clinical deployment without regulatory validation.
