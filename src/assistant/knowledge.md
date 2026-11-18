# VoiceSense-PD knowledge

VoiceSense-PD is a research app for voice-based Parkinson's screening. It is a screening aid, not a diagnosis. A neurologist, not this app, decides whether someone has Parkinson's disease.

## What the app does

The site has four screens:

- Live voice studio: record a sustained vowel or upload audio. English and Bangla are interface modes only.
- Feature lab: score one row from a published feature table (UCI English or BenSParX Bengali). It does not listen to a microphone.
- Privacy: explains what happens to the recording.
- About: method limits, including the Internet Download Manager popup, which is not a model crash.

A firm live call needs all of the following:

- Probability of a Parkinson's-like pattern at least about 72% for a "possible PD" flag, or at most about 42% for "no clear pattern".
- A clip of at least 4 seconds, low noise, and mostly voiced audio.
- Agreement between the random-forest and SVM checks.

Otherwise the result stays Uncertain. Up to three vowels can be kept for the visit. One home-microphone clip is not a final answer.

## Models and honest metrics

The live model is trained only on Italian recordings from the public Italian Parkinson's Voice set. It was not trained on English or Bangla speech. The default live model is an Italian random forest.

- Subject-aware 5-fold cross-validation F1 is about 0.90 ± 0.08 (0.899 ± 0.081). That is the stability estimate.
- The hold-out is small: 34 clips from 12 subjects, accuracy about 0.97, F1 about 0.96, ROC-AUC about 0.99. A number that high on 34 clips is not the same claim as the cross-validation.
- The Italian SVM hold-out F1 on that same small split is about 0.83.

Feature-table models are separate. They score spreadsheet numbers, not a live microphone:

- UCI English, random forest: hold-out F1 about 0.81 on 43 samples; 5-fold F1 about 0.84.
- UCI English, SVM: hold-out F1 about 0.90 on 43 samples; 5-fold F1 about 0.86. The healthy group in that hold-out is small.
- BenSParX Bengali, random forest: hold-out F1 about 0.94 on 177 samples; 5-fold F1 about 0.93.

Splits keep every recording from one person on one side of the split. They are not a substitute for a clinical trial.

## Privacy

Local Streamlit scores audio in the Python process on the user's machine. After a successful analysis the recording is discarded from the page. A downloaded receipt has no WAV and no upload filename. Decode may use a short-lived temp file that is deleted immediately.

Streamlit Community Cloud is not on-device. The browser must send audio to the host, and the app warns about that.

This chat cannot see a recording, a score, a receipt, or who is speaking.

## Limits

This tool does not diagnose Parkinson's disease, rule it out, or recommend treatment. Changes in speech, walking, or handwriting should be discussed with a clinician, preferably a neurologist. A cold, fatigue, a dry throat, or a noisy room can look similar to a Parkinson's-like voice pattern.
