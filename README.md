# VoiceSense — Parkinson’s Voice Screening

Premium research app for **voice-based Parkinson’s disease (PD) screening** with:

- **Live microphone / audio upload** (English & Bangla UI modes)
- **On-device processing** (local Streamlit: recording discarded after analysis)
- **Feature-table lab** (UCI English + BenSParX Bengali)
- **Explainable outputs** (why PD / why not / uncertain)
- **Conservative decision policy** to reduce false PD flags on laptop mics

> **Not a medical diagnosis.** Screening support / research only.

## Quick start

```bash
git clone https://github.com/Arafat128/VoiceSense-PD.git
cd VoiceSense-PD
python -m pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

Open the local URL (usually `http://localhost:8501`).

### Train models (if needed)

```bash
python -m src.train          # UCI + BenSParX feature models
python -m src.train_audio    # live acoustic model (Italian WAVs required under datasets/)
```

Datasets are expected under `../datasets/` relative to this project (see thesis folder layout), or adjust paths in `src/config.py` / `src/train_audio.py`.

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

## Where to start

Open these in order if you are new to the code:

1. `app/streamlit_app.py` — the website frame (it does not score audio)
2. `app/live_page.py` — record, upload, analyze
3. `src/__init__.py` — map of the engine, one line per file

## Project layout

```
app/streamlit_app.py   # start the site; picks one screen
app/live_page.py       # microphone and audio upload
app/feature_page.py    # spreadsheet lab
app/privacy_ui.py      # on-device banner and receipt
app/widgets.py         # shared result cards
src/                   # scoring, training, privacy (see src/__init__.py)
models/                # trained files; do not edit by hand
reports/               # metrics written by training
scripts/               # small checks
```

## Deploy notes

| Target | Status |
|--------|--------|
| **Local Streamlit** | Full app (recommended) |
| **Streamlit Community Cloud** | Possible if system audio libs available |
| **Vercel** | Not used — the ML stack is not serverless-friendly |

## Citation / data

Uses public corpora including UCI Parkinsons, BenSParX, and Italian Parkinson’s Voice (for live-audio training). Cite original dataset authors in academic work.

## License

Research / educational use. Provide proper dataset citations. Not for clinical deployment without regulatory validation.
