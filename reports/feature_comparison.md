# Feature comparison: leave-one-dataset-out

Protocol: leave-one-dataset-out across the vowel corpora (train on the others,
test on the held-out corpus), plus the Italian <-> UAMS pair in both
directions. Person-level AUC with 95% speaker-bootstrap CIs.

Person-level AUC on the held-out corpus (95% CI). 0.50 = chance.

| features / classifier | held-out Italian | held-out UAMS | held-out Argentina | mean |
|---|---|---|---|---|
| praat/logreg | 0.33 (0.18-0.50) | 0.52 (0.39-0.66) | 0.59 (0.49-0.68) | 0.481 |
| praat/random_forest | 0.26 (0.12-0.40) | 0.56 (0.43-0.68) | 0.47 (0.36-0.57) | 0.427 |
| praat/svm_rbf | 0.36 (0.19-0.53) | 0.55 (0.43-0.67) | 0.62 (0.52-0.71) | 0.510 |
| egemaps/logreg | 0.22 (0.09-0.38) | 0.47 (0.33-0.59) | 0.60 (0.50-0.70) | 0.428 |
| egemaps/random_forest | 0.38 (0.20-0.54) | 0.49 (0.36-0.62) | 0.62 (0.53-0.71) | 0.499 |
| egemaps/svm_rbf | 0.41 (0.21-0.61) | 0.45 (0.31-0.58) | 0.68 (0.58-0.78) | 0.513 |
| wavlm/logreg | 0.19 (0.07-0.34) | 0.46 (0.33-0.59) | 0.47 (0.37-0.58) | 0.375 |
| wavlm/svm_rbf | 0.22 (0.10-0.37) | 0.49 (0.36-0.62) | 0.58 (0.48-0.68) | 0.431 |

| features / classifier | Italian -> UAMS | UAMS -> Italian |
|---|---|---|
| praat/logreg | 0.46 (0.34-0.58) | 0.24 (0.11-0.38) |
| praat/random_forest | 0.39 (0.27-0.51) | 0.40 (0.25-0.56) |
| praat/svm_rbf | 0.38 (0.25-0.51) | 0.24 (0.11-0.38) |
| egemaps/logreg | 0.37 (0.24-0.50) | 0.13 (0.04-0.25) |
| egemaps/random_forest | 0.31 (0.19-0.43) | 0.07 (0.00-0.17) |
| egemaps/svm_rbf | 0.32 (0.20-0.44) | 0.13 (0.04-0.24) |
| wavlm/logreg | 0.31 (0.19-0.42) | 0.20 (0.09-0.34) |
| wavlm/svm_rbf | 0.22 (0.12-0.33) | 0.11 (0.03-0.23) |

**Verdict:** Keep Praat (praat/svm_rbf, mean held-out AUC 0.510). No alternative cleared the rule (best alternatives: egemaps/svm_rbf 0.513, wavlm/svm_rbf 0.431).
