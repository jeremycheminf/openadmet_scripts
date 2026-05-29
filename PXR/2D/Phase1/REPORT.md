# OpenADMET PXR Activation Challenge — Submission Report

Task: predict pEC50 for 513 blinded compounds against human PXR (NR1I2).
Training set: 4,083 curated compounds (pEC50 1.6–8.6, mean 4.64, std 0.96).
Metric: Relative Absolute Error (RAE), where 1.0 = predicting the training mean.

## Best submission

**Sub28** — rank **18** on the public leaderboard (as of 2026-05-12).

| Metric | Value |
|--------|------:|
| MAE | 0.447 |
| RAE | 0.561 |
| R² | 0.542 |
| Spearman ρ | 0.831 |
| Kendall τ | 0.640 |

## Approach

- **Base learners** (frozen encoders + foundation regressors / tree models):
  - TabICL on HTS-continuous CheMeleon embeddings 
  - TabICL on HTS-binary CheMeleon embeddings
  - MolE (50-epoch fine-tune, regression)
  - UniMol v1 (50-epoch fine-tune, regression)

## What didn't work

- XGBoost / Ridge / ElasticNet meta-stackers (NNLS wins on blind every time).

