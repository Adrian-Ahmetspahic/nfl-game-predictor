# NFL Game Result Predictor (2023–2025)

This notebook-based project predicts NFL game outcomes using team strength (Elo), recent form, and optional play‑by‑play aggregates. It trains models on 2023–2024, evaluates on 2024 and 2025 Weeks 1–2, and produces win probabilities for upcoming matchups.

## Data
- Final scores (included):
  - `Cleaned_NFL_2023_Scores.csv`
  - `nfl_scores_2024_clean.csv`
  - `nfl_scores_2025_clean.csv` (Weeks 1–2 completed; later weeks may be blank)
- Optional play-by-play (not required). If present, place at repo root:
  - `pbp-2023.csv`, `pbp-2024.csv`

## Features engineered
- Elo strength: pre‑game Elo with K=20 and HOME_FIELD=55.
- Recent form (no leakage): rolling 3/5‑game stats (points for/against, margin).
- Head‑to‑head priors: prior home‑win rate and prior average total points for the matchup.
- Optional offense/defense (from play‑by‑play if available): yards/play, total yards, TDs, EPA allowed; rolling‑5 and to‑date. (MORE DETAILS IN THE NOTEBOOK)

## Models
- Logistic Regression (probabilistic baseline, with scaling).
- Random Forest (non‑linear baseline).

Metrics reported (holdout or 2025 W1–2):
- Accuracy (vs “always pick home” baseline ~0.55–0.57)
- ROC AUC (ranking quality)
- Brier score (probability calibration; lower is better)
- Log loss (probability quality; lower is better)

Target yardsticks:
- ROC AUC > 0.65 good; > 0.70 strong.
- Brier ≤ 0.21 good; ≤ 0.20 strong.
- LogLoss ≤ 0.65 decent; ≤ 0.62 strong.

## How to run
1. Open `EDA_notebook.ipynb` and run all cells top‑to‑bottom.
   - The notebook uses relative paths. Ensure the CSVs are present in the repo root.
   - If play‑by‑play files are absent, advanced O/D features are skipped automatically.
2. Artifacts saved to repo root:
   - `nfl_homewin_model_all.joblib` (Logistic, 2023–2024)
   - `nfl_homewin_rf_all.joblib` (RandomForest, 2023–2024)
   - `nfl_model_meta_all.joblib` (feature names, Elo params)

## Measured results (from notebook outputs)

### 2024 holdout (train 2023 → test 2024)
- Logistic Regression:
  - Accuracy: 0.6842
  - ROC AUC: 0.7358
  - Brier: 0.2145
  - LogLoss: 0.6183
- Random Forest:
  - Accuracy: 0.5860
  - ROC AUC: 0.6148
  - Brier: 0.2392

Interpretation: Logistic clearly outperforms RF on 2024—better ranking and probability quality (higher AUC/accuracy; lower Brier/LogLoss).

### 2025 Weeks 1–2 (small sample)
- Logistic Regression:
  - Accuracy: 0.7111
  - ROC AUC: 0.7449
  - Brier: 0.1984
- Random Forest:
  - Accuracy: 0.6667
  - ROC AUC: 0.7469
  - Brier: 0.2001
  - LogLoss: 0.5869

Interpretation: Both models are similar on AUC; Logistic slightly edges RF on accuracy and Brier (probability calibration). Given 2024 results, Logistic remains the preferred default.

## Analysis summary
- Elo captures broad team strength; rolling form adds short‑term performance; H2H priors stabilize early‑season matchups.
- RandomForest can capture non‑linear interactions among these features but did not surpass Logistic on the larger 2024 holdout.
- Probability metrics (Brier/LogLoss) are emphasized since we want calibrated win probabilities, not just class labels.

## Reproducibility
- Notebook sets seeds and prints package versions.
- All file paths are relative to the repository root.

## Repo contents
- `EDA_notebook.ipynb`: end‑to‑end workflow (feature engineering, training, evaluation, 2025 predictions).
- `requirements.txt`: dependencies.
- `.gitignore`: ignore venv, checkpoints, and artifacts.
- CSV files: cleaned final scores for 2023–2025; optional PBP CSVs.
