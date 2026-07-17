# VoiceSense — Parkinson’s Voice Screening

Premium research app for **voice-based Parkinson’s disease (PD) screening** with:

- **Live microphone / audio upload** (English & Bangla UI modes)
- **Feature-table lab** (UCI English + BenSParX Bengali)
- **Explainable outputs** (why PD / why not / uncertain)
- **Conservative decision policy** to reduce false PD flags on laptop mics

> **Not a medical diagnosis.** Screening support / research only.

## Quick start

```bash
cd voicesense-pd-screening
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

## Project layout

```
app/streamlit_app.py   # premium UI
app/styles.py          # production CSS
src/                   # train / predict / features / explain
models/                # trained joblib artifacts
reports/               # metrics JSON
landing/               # static Vercel marketing page
```

## Deploy notes

| Target | Status |
|--------|--------|
| **Local Streamlit** | Full app (recommended) |
| **Streamlit Community Cloud** | Possible if system audio libs available |
| **Vercel** | Landing page only — ML stack is not serverless-friendly |

## Citation / data

Uses public corpora including UCI Parkinsons, BenSParX, and Italian Parkinson’s Voice (for live-audio training). Cite original dataset authors in academic work.

## License

Research / educational use. Provide proper dataset citations. Not for clinical deployment without regulatory validation.
