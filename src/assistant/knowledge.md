# VoiceSense-PD knowledge

VoiceSense-PD is a research app for voice-based Parkinson's screening. It is a screening aid, not a diagnosis. A neurologist, not this app, decides whether someone has Parkinson's disease.

The app has three model families. The live vowel model scores recordings. The UCI English and BenSParX Bengali models only score published feature tables in Feature lab. They are research comparisons and are not used on microphone audio.

## What the app does

The site has four screens:

- Live voice studio: record a sustained vowel or upload audio. The decision always comes from the live vowel model.
- Feature lab: score one row from a published feature table. Choose the UCI English model or the BenSParX Bengali model, then a classifier. This screen does not listen to a microphone.
- Privacy: explains what happens to the recording.
- About: the live model's data, honest accuracy numbers, and the Internet Download Manager popup, which is not a model crash.

A firm live call needs all of the following:

- A calibrated probability of a Parkinson's-like pattern of at least 69% for a "possible PD" flag, or at most 43% for "no clear pattern". These cut-offs are not hand-picked. They were chosen from validation data so that about 10% of healthy clips land above the upper cut-off and about 10% of Parkinson's clips land below the lower one.
- A clip of at least 4 seconds, low noise, and mostly voiced audio.
- Agreement between the live SVM (primary) and the live random forest (second check).

Otherwise the result stays Uncertain. In validation about 57% of healthy clips and 64% of Parkinson's clips fell in the Uncertain band, so Uncertain is the most common answer. Up to three vowels can be kept for the visit. One home-microphone clip is not a final answer.

## Prompt language (English or Bangla)

The language control only changes the on-screen prompt. It does not swap, retrain, or add a model. A sustained vowel ("aaa" or "আ—") has no words, so the same vowel model is used for everyone.

The live analysis no longer shows UCI English or BenSParX Bengali "secondary" scores. Those models were trained on feature tables from other recording setups, and mapping a microphone clip onto their columns is not valid. Their scores were dropped from live analysis. They do not replace the live decision because they are not part of it at all.

## Live vowel model (used when you record)

Training data: sustained vowels from three public corpora, all CC BY 4.0, downloadable without an access request:

- Italian Parkinson's Voice and Speech (Dimauro et al.): vowels /a e i o u/, 25 Parkinson's and 22 elderly healthy speakers, 495 clips. Three Parkinson's speakers were filed in two folders each; they are counted once.
- UAMS voice samples (Iyer, Prior et al., figshare 23849127): US speakers saying /a/ over a phone line at 8 kHz, 40 Parkinson's and 18 healthy speakers aged 50 or older.
- Audios_vowel_A_PD (Giuliano et al., figshare 21453867): Argentine Spanish /a/, 55 Parkinson's and 71 healthy speakers.

Young healthy controls are left out, because age alone separates them from Parkinson's speakers. UAMS healthy speakers under 50 are left out for the same reason. The Argentine corpus does not publish healthy speakers' ages.

Input: 34 Praat voice measures from the same extractor used at training time: pitch and its variation, jitter, shimmer, harmonics-to-noise ratio (HNR), loudness range, and a few ratios. Mel-frequency cepstral coefficients are not used. Each training clip also gets two augmented copies (added noise, gain, low-pass, phone band, room reverb). Each speaker counts once in training.

How it was checked:

- Same corpora, new speakers (speaker-grouped 5-fold cross-validation repeated 5 times, 95% confidence intervals from a speaker bootstrap). SVM: per-person AUC 0.72 (0.66–0.79), per-person balanced accuracy 0.66 (0.60–0.72), sensitivity 0.70, specificity 0.62. Per clip: AUC 0.69 (0.62–0.76). Random forest: per-person AUC 0.70 (0.63–0.77), per clip 0.75 (0.69–0.81).
- A corpus the model never saw (train on two corpora, test on the third). SVM per-person AUC: Italian 0.56 (0.38–0.73), UAMS 0.46 (0.29–0.64), Argentina 0.53 (0.43–0.62). Random forest: 0.36, 0.48, 0.51. An AUC of 0.5 is chance. This is the honest estimate for a new microphone, and it is close to chance.

Earlier numbers, and why they were wrong to quote:

- The old live model said about 97% accuracy and ROC-AUC 0.99. It was trained on Italian reading passages, not the vowel the app records, and its healthy group included young controls. Re-checked per speaker it still scores 0.99 inside that corpus, but on the UAMS and Argentine corpora it scores 0.45 and 0.41 AUC.
- A model trained only on the Italian vowels (elderly controls only) scores 0.97 per person inside the Italian corpus, but 0.39 on UAMS and 0.42 on Argentina, below chance. In the Italian corpus the Parkinson's speakers have lower shimmer and higher HNR than its controls, the opposite of the other two corpora, so that model learned the recording setup, not Parkinson's.

Other feature sets were tested the same way. openSMILE eGeMAPS reached a mean held-out AUC of 0.51 and WavLM speech embeddings 0.43, versus 0.51 for the Praat measures, so the app keeps the Praat measures.

Robustness: on held-out speakers, adding noise, cutting volume, low-pass filtering, phone-band filtering, or reverb changes the probability by 0.045 on average with augmentation (0.056 without), and moves 18% of clips to a different band (19% without). Strong noise (10 dB SNR) moves 38% of clips, so a quiet room matters.

## English model (UCI Parkinson's)


Dataset: UCI Parkinson's (English). The report counts 195 recordings from 32 subjects: 147 Parkinson's and 48 healthy controls. Each row has 22 published dysphonia features (jitter, shimmer, HNR, PPE, RPDE, DFA, and related measures). The input in Feature lab is that feature table, not a live waveform.

Hold-out: 43 recordings from 7 subjects (12 healthy and 31 Parkinson's rows). Subject-balanced 80/20 split.

- Random forest: hold-out accuracy 0.721, F1 0.806, ROC-AUC 0.754. 5-fold CV F1 0.839 ± 0.067.
- RBF SVM: hold-out accuracy 0.837, F1 0.896, ROC-AUC 0.739. 5-fold CV F1 0.862 ± 0.047.

Feature lab can run either classifier; its default is the random forest. These numbers describe that feature table only. They are not valid for a microphone clip, and the model is not used in live analysis.

## Bangla / Bengali model (BenSParX)

Dataset: BenSParX (Bengali). The report counts 900 recordings from 120 subjects: 450 Parkinson's and 450 healthy controls. Each row has 71 features, including jitter, shimmer, pitch, loudness, harmonicity, and mel-frequency cepstral coefficients. The input in Feature lab is that feature table, not a live waveform.

Hold-out: 177 recordings from 24 subjects. Subject-balanced 80/20 split.

- Random forest: hold-out accuracy 0.938, F1 0.937, ROC-AUC 0.987. 5-fold CV F1 0.931 ± 0.025.
- RBF SVM: hold-out accuracy 0.932, F1 0.930, ROC-AUC 0.967. 5-fold CV F1 0.932 ± 0.020.

Feature lab can run either classifier; its default is the random forest. These numbers describe that feature table only. They are not valid for a microphone clip, and the model is not used in live analysis.

Splits for all three families keep every recording from one person on one side. They are not a clinical trial.

## FAQ

### Was it trained on English or Bangla speech?

The live vowel model heard Italian, US English, and Argentine Spanish speakers saying sustained vowels. It heard no Bangla speakers. Because a sustained vowel has no words, language matters less than the microphone and the room. English speech also appears in the separate UCI Parkinson's feature-table model (195 recordings, 32 subjects), and Bangla speech in the separate BenSParX feature-table model (900 recordings, 120 subjects). Those two do not listen to the microphone; they score published feature rows in Feature lab only.

### Which model is used when I record?

The live vowel SVM makes the call, and the live vowel random forest must agree before a firm call. No other model adds a score to a live recording.

### What does English or Bangla mode change?

Only the on-screen prompt. The language control does not swap the model, and it is not evidence that the model was trained in that language.

### What do the UCI and BenSParX models score?

They score a table of acoustic numbers in Feature lab. They are research comparisons on their own data. They are not valid on microphone audio and are not used in live analysis. They are not a diagnosis.

### How accurate is it?

Within the three training corpora, on speakers it had not seen, the per-person AUC is about 0.72. On a whole corpus it had not seen, it is about 0.46 to 0.56, close to chance. Treat a result as a research demonstration, not a screening result.

## Privacy

Local Streamlit scores audio in the Python process on the user's machine. After a successful analysis the recording is discarded from the page. A downloaded receipt has no WAV and no upload filename. Decode may use a short-lived temp file that is deleted immediately.

Streamlit Community Cloud is not on-device. The browser must send audio to the host, and the app warns about that.

This chat cannot see a recording, a score, a receipt, or who is speaking.

## Limits

This tool does not diagnose Parkinson's disease, rule it out, or recommend treatment. Changes in speech, walking, or handwriting should be discussed with a clinician, preferably a neurologist. A cold, fatigue, a dry throat, or a noisy room can look similar to a Parkinson's-like voice pattern.
