# VoiceSense-PD knowledge

VoiceSense-PD is a research app for voice-based Parkinson's screening. It is a screening aid, not a diagnosis. A neurologist, not this app, decides whether someone has Parkinson's disease.

The app has three model families. Recording a voice does not mean only the Italian model exists. English and Bangla are covered by separate feature-table models. Those models do not replace the live decision.

## What the app does

The site has four screens:

- Live voice studio: record a sustained vowel or upload audio. The primary decision always comes from the Italian-trained live model.
- Feature lab: score one row from a published feature table. Choose the UCI English model or the BenSParX Bengali model, then a classifier. This screen does not listen to a microphone.
- Privacy: explains what happens to the recording.
- About: method limits, including the Internet Download Manager popup, which is not a model crash.

A firm live call needs all of the following:

- Probability of a Parkinson's-like pattern at least about 72% for a "possible PD" flag, or at most about 42% for "no clear pattern".
- A clip of at least 4 seconds, low noise, and mostly voiced audio.
- Agreement between the live random-forest and live SVM checks.

Otherwise the result stays Uncertain. Up to three vowels can be kept for the visit. One home-microphone clip is not a final answer.

On every live analysis the app also tries two secondary scores: the UCI English random forest and the BenSParX Bengali random forest. They are supportive only. They do not override the primary live decision. The UCI secondary is skipped when fewer than about 70% of its features could be filled from the clip. The BenSParX secondary is skipped when fewer than about 25% could be filled, because that mapping is only partial.

## English mode and Bangla mode

The Language mode control does not swap or retrain the model that makes the primary call. English mode and Bangla mode both keep the Italian live model as the primary score.

What the mode changes is the note and which secondary score the screen highlights. English mode highlights the UCI English secondary when that score is available. Bangla mode highlights the BenSParX secondary when that score is available. Both secondaries are still attempted either way.

Choosing English does not mean the live model heard English speech. Choosing Bangla does not mean it heard Bangla speech.

## Live acoustic model (used when you record)

Dataset: public Italian Parkinson's Voice. The training report counts 186 recordings from 61 subjects: 77 Parkinson's and 109 healthy controls (older and young healthy controls).

Input: a microphone or uploaded audio clip, turned into 33 Praat voice features by the same extractor used at training time. The features are classical dysphonia measures: jitter, shimmer, harmonics-to-noise ratio (HNR), pitch, and loudness, plus a few ratios of those measures. Mel-frequency cepstral coefficients are not in this model. The measures are not tied to English, Bangla, or Italian words, so they are treated as language-independent. They are still sensitive to the microphone and the recording channel. A home mic is not a clinic mic.

The default primary checker is the Italian random forest. The Italian SVM is the second check that must agree before a firm call.

Random forest, subject-aware 5-fold CV F1: 0.899 ± 0.081 (about 0.90 ± 0.08). That is the stability estimate. The hold-out is small: 34 clips from 12 subjects (21 healthy, 13 Parkinson's), accuracy 0.971, F1 0.963, ROC-AUC 0.993. A number that high on 34 clips is not the same claim as the cross-validation.

SVM on the same split: 5-fold CV F1 0.915 ± 0.046. Hold-out accuracy 0.882, F1 0.833, ROC-AUC 0.971.

This live model was trained only on Italian recordings. It was not trained on English speech or Bangla speech.

## English model (UCI Parkinson's)

Dataset: UCI Parkinson's (English). The report counts 195 recordings from 32 subjects: 147 Parkinson's and 48 healthy controls. Each row has 22 published dysphonia features (jitter, shimmer, HNR, PPE, RPDE, DFA, and related measures). The input in Feature lab is that feature table, not a live waveform.

Hold-out: 43 recordings from 7 subjects (12 healthy and 31 Parkinson's rows). Subject-balanced 80/20 split.

- Random forest: hold-out accuracy 0.721, F1 0.806, ROC-AUC 0.754. 5-fold CV F1 0.839 ± 0.067.
- RBF SVM: hold-out accuracy 0.837, F1 0.896, ROC-AUC 0.739. 5-fold CV F1 0.862 ± 0.047.

Feature lab can run either classifier. The app's default there, and the live secondary score, is the random forest. The live secondary is a mapping from the clip onto those 22 columns. It is not the primary decision.

## Bangla / Bengali model (BenSParX)

Dataset: BenSParX (Bengali). The report counts 900 recordings from 120 subjects: 450 Parkinson's and 450 healthy controls. Each row has 71 features, including jitter, shimmer, pitch, loudness, harmonicity, and mel-frequency cepstral coefficients. The input in Feature lab is that feature table, not a live waveform.

Hold-out: 177 recordings from 24 subjects. Subject-balanced 80/20 split.

- Random forest: hold-out accuracy 0.938, F1 0.937, ROC-AUC 0.987. 5-fold CV F1 0.931 ± 0.025.
- RBF SVM: hold-out accuracy 0.932, F1 0.930, ROC-AUC 0.967. 5-fold CV F1 0.932 ± 0.020.

Feature lab can run either classifier. The app's default there, and the live secondary score, is the random forest. The live secondary uses only the columns the clip can fill, so it is partial and supportive, not the primary decision.

Splits for all three families keep every recording from one person on one side. They are not a clinical trial.

## FAQ

### Was it trained on English or Bangla speech?

The live acoustic model was not. It was trained only on Italian recordings. English speech is represented by the separate UCI Parkinson's feature-table model (195 recordings, 32 subjects). Bangla speech is represented by the separate BenSParX feature-table model (900 recordings, 120 subjects). Those two do not listen to the microphone. They score published feature rows in Feature lab, and they can add a secondary score after a live recording.

### Which model is used when I record?

The primary score is the Italian live random forest, with the Italian live SVM as the agreeing second check. The UCI English random forest and the BenSParX Bengali random forest may appear underneath as secondary scores. They do not replace the live decision.

### What does English or Bangla mode change?

It does not change the primary model. English mode highlights the UCI secondary score. Bangla mode highlights the BenSParX secondary score. Both secondaries are still attempted. The mode is not evidence that the live model was trained in that language.

### What do the UCI and BenSParX models score?

They score a table of acoustic numbers, in Feature lab or as a secondary live score. They do not score the waveform by themselves, and they are not a diagnosis.

## Privacy

Local Streamlit scores audio in the Python process on the user's machine. After a successful analysis the recording is discarded from the page. A downloaded receipt has no WAV and no upload filename. Decode may use a short-lived temp file that is deleted immediately.

Streamlit Community Cloud is not on-device. The browser must send audio to the host, and the app warns about that.

This chat cannot see a recording, a score, a receipt, or who is speaking.

## Limits

This tool does not diagnose Parkinson's disease, rule it out, or recommend treatment. Changes in speech, walking, or handwriting should be discussed with a clinician, preferably a neurologist. A cold, fatigue, a dry throat, or a noisy room can look similar to a Parkinson's-like voice pattern.
