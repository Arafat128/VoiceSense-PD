# Cross-dataset test: BenSParX vs other Parkinson's voice tables

Person-level AUC with 95% bootstrap CI over test persons. `standardized` = log + z-score within each table.

| version | train | test | model | features | person AUC (95% CI) | note |
|---|---|---|---|---|---|---|
| raw | bensparx | uci_pd_speech | logreg | 11 | 0.76 (0.69–0.82) | same people (copied rows) |
| raw | bensparx | uci_pd_speech | random_forest | 11 | 0.89 (0.84–0.94) | same people (copied rows) |
| raw | bensparx | uci_replicated | logreg | 9 | 0.61 (0.47–0.73) |  |
| raw | bensparx | uci_replicated | random_forest | 9 | 0.58 (0.45–0.70) |  |
| raw | bensparx | uci_parkinsons | logreg | 11 | 0.83 (0.65–0.97) |  |
| raw | bensparx | uci_parkinsons | random_forest | 11 | 0.58 (0.35–0.81) |  |
| raw | uci_pd_speech | bensparx | logreg | 11 | 0.75 (0.68–0.82) | same people (copied rows) |
| raw | uci_pd_speech | bensparx | random_forest | 11 | 0.91 (0.87–0.94) | same people (copied rows) |
| raw | uci_pd_speech | uci_replicated | logreg | 9 | 0.62 (0.48–0.73) |  |
| raw | uci_pd_speech | uci_replicated | random_forest | 9 | 0.58 (0.45–0.71) |  |
| raw | uci_pd_speech | uci_parkinsons | logreg | 11 | 0.83 (0.65–0.97) |  |
| raw | uci_pd_speech | uci_parkinsons | random_forest | 11 | 0.79 (0.61–0.95) |  |
| raw | uci_replicated | bensparx | logreg | 9 | 0.66 (0.58–0.73) |  |
| raw | uci_replicated | bensparx | random_forest | 9 | 0.53 (0.44–0.61) |  |
| raw | uci_replicated | uci_pd_speech | logreg | 9 | 0.65 (0.57–0.73) |  |
| raw | uci_replicated | uci_pd_speech | random_forest | 9 | 0.54 (0.46–0.63) |  |
| raw | uci_replicated | uci_parkinsons | logreg | 9 | 0.82 (0.62–0.96) |  |
| raw | uci_replicated | uci_parkinsons | random_forest | 9 | 0.58 (0.31–0.81) |  |
| raw | uci_parkinsons | bensparx | logreg | 11 | 0.68 (0.60–0.76) |  |
| raw | uci_parkinsons | bensparx | random_forest | 11 | 0.73 (0.65–0.80) |  |
| raw | uci_parkinsons | uci_pd_speech | logreg | 11 | 0.68 (0.60–0.75) |  |
| raw | uci_parkinsons | uci_pd_speech | random_forest | 11 | 0.73 (0.66–0.80) |  |
| raw | uci_parkinsons | uci_replicated | logreg | 9 | 0.62 (0.49–0.74) |  |
| raw | uci_parkinsons | uci_replicated | random_forest | 9 | 0.61 (0.47–0.73) |  |
| standardized | bensparx | uci_pd_speech | logreg | 11 | 0.78 (0.72–0.84) | same people (copied rows) |
| standardized | bensparx | uci_pd_speech | random_forest | 11 | 0.85 (0.80–0.90) | same people (copied rows) |
| standardized | bensparx | uci_replicated | logreg | 9 | 0.60 (0.46–0.72) |  |
| standardized | bensparx | uci_replicated | random_forest | 9 | 0.56 (0.43–0.68) |  |
| standardized | bensparx | uci_parkinsons | logreg | 11 | 0.86 (0.73–0.97) |  |
| standardized | bensparx | uci_parkinsons | random_forest | 11 | 0.86 (0.73–0.97) |  |
| standardized | uci_pd_speech | bensparx | logreg | 11 | 0.78 (0.71–0.84) | same people (copied rows) |
| standardized | uci_pd_speech | bensparx | random_forest | 11 | 0.88 (0.82–0.92) | same people (copied rows) |
| standardized | uci_pd_speech | uci_replicated | logreg | 9 | 0.61 (0.46–0.73) |  |
| standardized | uci_pd_speech | uci_replicated | random_forest | 9 | 0.56 (0.43–0.69) |  |
| standardized | uci_pd_speech | uci_parkinsons | logreg | 11 | 0.85 (0.70–0.97) |  |
| standardized | uci_pd_speech | uci_parkinsons | random_forest | 11 | 0.89 (0.74–0.99) |  |
| standardized | uci_replicated | bensparx | logreg | 9 | 0.68 (0.59–0.75) |  |
| standardized | uci_replicated | bensparx | random_forest | 9 | 0.56 (0.49–0.64) |  |
| standardized | uci_replicated | uci_pd_speech | logreg | 9 | 0.67 (0.59–0.74) |  |
| standardized | uci_replicated | uci_pd_speech | random_forest | 9 | 0.56 (0.49–0.63) |  |
| standardized | uci_replicated | uci_parkinsons | logreg | 9 | 0.75 (0.54–0.91) |  |
| standardized | uci_replicated | uci_parkinsons | random_forest | 9 | 0.64 (0.33–0.88) |  |
| standardized | uci_parkinsons | bensparx | logreg | 11 | 0.72 (0.64–0.79) |  |
| standardized | uci_parkinsons | bensparx | random_forest | 11 | 0.73 (0.66–0.80) |  |
| standardized | uci_parkinsons | uci_pd_speech | logreg | 11 | 0.72 (0.64–0.79) |  |
| standardized | uci_parkinsons | uci_pd_speech | random_forest | 11 | 0.71 (0.65–0.79) |  |
| standardized | uci_parkinsons | uci_replicated | logreg | 9 | 0.62 (0.48–0.74) |  |
| standardized | uci_parkinsons | uci_replicated | random_forest | 9 | 0.60 (0.46–0.72) |  |

Within-table reference (person-grouped CV on the same shared measures):

| version | dataset | model | person AUC (95% CI) |
|---|---|---|---|
| raw | bensparx | logreg | 0.74 (0.66–0.81) |
| raw | bensparx | random_forest | 0.75 (0.67–0.82) |
| raw | uci_pd_speech | logreg | 0.74 (0.67–0.80) |
| raw | uci_pd_speech | random_forest | 0.77 (0.70–0.84) |
| raw | uci_replicated | logreg | 0.61 (0.48–0.73) |
| raw | uci_replicated | random_forest | 0.56 (0.44–0.69) |
| raw | uci_parkinsons | logreg | 0.79 (0.60–0.95) |
| raw | uci_parkinsons | random_forest | 0.57 (0.32–0.85) |
| standardized | bensparx | logreg | 0.76 (0.69–0.83) |
| standardized | bensparx | random_forest | 0.75 (0.67–0.82) |
| standardized | uci_pd_speech | logreg | 0.77 (0.70–0.83) |
| standardized | uci_pd_speech | random_forest | 0.77 (0.70–0.84) |
| standardized | uci_replicated | logreg | 0.64 (0.50–0.76) |
| standardized | uci_replicated | random_forest | 0.56 (0.44–0.69) |
| standardized | uci_parkinsons | logreg | 0.79 (0.58–0.96) |
| standardized | uci_parkinsons | random_forest | 0.57 (0.32–0.85) |

**Verdict.** With the shared perturbation measures standardized per table, a BenSParX model reaches mean person AUC 0.72 on the Spanish and English tables, and models trained there reach 0.67 on BenSParX (Replicated <-> Parkinsons: 0.65). BenSParX <-> UCI pd_speech scores 0.82, but those are the same people's recordings, so it shows the copy, not transfer.
