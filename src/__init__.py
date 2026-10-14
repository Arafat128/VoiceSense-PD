"""VoiceSense engine. If you are new, open the files in this order.

Shared
    config.py           folder paths, dataset locations, and the word glossary

A person records a vowel
    audio_features.py   turn audio into pitch, loudness, and noise numbers
    predict_audio.py    score it, then refuse a firm call when the clip is weak
    plain_language.py   everyday sentences for that score
    explain.py          technical "why", comparing the clip with training averages

A spreadsheet row, not a microphone
    data_loading.py     read the English UCI table and the Bengali BenSParX table
    predict.py          score one row
    train.py            train those two spreadsheet models

The microphone model is trained separately
    train_audio.py      Italian vowel recordings, same features used at score time

Privacy
    privacy.py          drop audio from memory, and build a receipt that has no WAV

The website that calls this package is app/streamlit_app.py.
The click-by-click guide is docs/VoiceSense-app-flows.pdf.
"""
