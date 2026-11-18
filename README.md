# VoiceSense — Parkinson’s Voice Screening

Premium research app for **voice-based Parkinson’s disease (PD) screening** with:

- **Live microphone / audio upload** (English & Bangla prompt modes)
- **On-device processing** (local Streamlit: recording discarded after analysis)
- **Feature-table lab** (UCI English + BenSParX Bengali)
- **Explainable outputs** (why PD / why not / uncertain)
- **Conservative decision policy** to reduce false PD flags on laptop mics

> **Not a medical diagnosis.** Screening support / research only.

Three model families are included. The **live** model, used when you record, is trained on sustained vowels from three public corpora (Italian Parkinson’s Voice, UAMS/Iyer, Argentina vowel /a/: 679 clips, 231 speakers). **English** is the separate UCI Parkinson’s feature-table model (195 recordings, 32 subjects) and **Bangla** is the separate BenSParX feature-table model (900 recordings, 120 subjects). Both live only in Feature lab: their columns cannot be measured validly from a microphone clip, so they are not used in live analysis. English and Bangla mode on the live screen change the on-screen prompt only.

> **Accuracy, honestly.** On new speakers from the same corpora the live model reaches person-level AUC 0.72 (95% CI 0.66–0.79). On a corpus it never saw, AUC is 0.46–0.56, close to chance. Treat a live result as a research demo.

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

Put either an OpenAI key (`OPENAI_API_KEY`) or an xAI key (`XAI_API_KEY`) in `.env`, or in gitignored `.streamlit/secrets.toml`. Restart the app. The key stays on the server. If neither key is set, the rest of the app still runs and the chat says the assistant is not configured.

`VOICESENSE_ASSISTANT_PROVIDER` is `openai`, `xai`, or `auto` (the default). Auto uses OpenAI when `OPENAI_API_KEY` is set, and otherwise xAI when `XAI_API_KEY` is set. `VOICESENSE_ASSISTANT_MODEL` overrides the model for the provider that was selected. The defaults are `gpt-4o-mini` for OpenAI and `grok-4` for xAI.

### Tests

```bash
python -m pip install ruff pytest
python -m ruff check .
python -m pytest
```

CI on GitHub runs the same ruff and pytest commands on Python 3.13.

### Train models (if needed)

Datasets are not in this repo. Point `VOICESENSE_DATA_DIR` or `--data-dir` at the folder that contains `01_UCI_Parkinsons`, `02_BenSParX`, `07_Italian_PD_Voice`, `09_UAMS_Iyer_figshare` (figshare 23849127, CC BY 4.0) and `10_Argentina_vowelA_figshare` (figshare 21453867, CC BY 4.0).

```bash
python -m training.train_tables --data-dir /path/to/datasets
python -m training.train_audio --data-dir /path/to/datasets      # 3-corpus vowel model
python -m training.eval_external --data-dir /path/to/datasets    # held-out corpus test
python -m training.robustness --data-dir /path/to/datasets       # noise / phone / reverb
```

The feature comparison (`training/compare_features.py`, eGeMAPS and WavLM) needs `requirements-research.txt`; WavLM stays optional and the app never loads it.

`training/` is offline. The Streamlit app does not import it.

## Models

Figures below are the subject-aware numbers in `reports/`. A small hold-out is not the same claim as cross-validation.

**Live acoustic model** (`live_audio_vowel__svm_rbf.joblib`, with a random-forest second check). Sustained vowels from Italian Parkinson’s Voice (vowels only, elderly controls; 47 speakers, 495 clips), UAMS/Iyer (/a/, controls aged 50+; 58 speakers) and Argentina (/a/; 126 speakers). 34 Praat measures (jitter, shimmer, HNR, pitch, loudness; no MFCCs). Trained with noise / gain / low-pass / phone-band / reverb augmentation and sigmoid calibration; the Possible-PD and No-clear-pattern cut-offs (69% / 43%) come from validation data. Validation: speaker-grouped 5-fold CV repeated 5 times, 95% CIs from a speaker bootstrap.

| | AUC (95% CI) | balanced accuracy |
|---|---|---|
| per person, same corpora | 0.72 (0.66–0.79) | 0.66 (0.60–0.72) |
| per clip, same corpora | 0.69 (0.62–0.76) | 0.63 |
| held-out corpus: Italian | 0.56 (0.38–0.73) | |
| held-out corpus: UAMS | 0.46 (0.29–0.64) | |
| held-out corpus: Argentina | 0.53 (0.43–0.62) | |

The previous Italian-only model reported 0.97 hold-out accuracy; it was trained on reading passages against young controls and scored 0.45 AUC on UAMS and 0.41 on Argentina, so those figures are retired. eGeMAPS and WavLM features did not beat Praat on held-out corpora (`reports/feature_comparison.md`).

**English UCI model.** 195 recordings, 32 subjects (147 Parkinson’s, 48 healthy), 22 published dysphonia features. Used in Feature lab only. Random forest hold-out (43 recordings, 7 subjects): accuracy 0.721, F1 0.806, ROC-AUC 0.754, CV F1 0.839 ± 0.067. SVM hold-out F1 0.896, CV F1 0.862 ± 0.047.

**Bengali BenSParX model.** 900 recordings, 120 subjects (450 Parkinson’s, 450 healthy), 71 features including MFCCs. Used in Feature lab only. Random forest hold-out (177 recordings, 24 subjects): accuracy 0.938, F1 0.937, ROC-AUC 0.987, CV F1 0.931 ± 0.025. SVM hold-out F1 0.930, CV F1 0.932 ± 0.020.

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

Uses public corpora: UCI Parkinsons, BenSParX, and, for live-audio training, Italian Parkinson’s Voice, UAMS/Iyer et al. 2023 (figshare 23849127) and the Argentine vowel /a/ set (figshare 21453867). Cite original dataset authors in academic work.

## License

Research / educational use. Provide proper dataset citations. Not for clinical deployment without regulatory validation.
