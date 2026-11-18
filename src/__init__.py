"""VoiceSense engine. If you are new, open the files in this order.

Shared
    config.py              constants, dataset names, and the word glossary
    models/registry.py     load every joblib model and holdout table once

A person records a vowel
    features/audio.py      turn audio into pitch, loudness, and noise numbers
    predict.py             score a clip or a spreadsheet row
    plain_language.py      everyday sentences for that score
    explain.py             technical "why", comparing the clip with training averages

Privacy
    privacy.py             drop audio from memory, and build a receipt that has no WAV

Project assistant (no tools, no recording access)
    assistant/knowledge.md the only facts the chat is allowed to use
    assistant/client.py    server-side Grok call; the API key stays on the server

Training is offline and is not imported by the app
    training/train_tables.py    English UCI feature model
    training/train_bangla.py    Bengali BenSParX feature model (person-grouped CV)
    training/train_audio.py     Italian vowel recordings, same features used at score time
    training/data_loading.py    read those two tables
    training/common.py          shared subject-balanced hold-out and dataset root

The website that calls this package is app/streamlit_app.py.
"""
