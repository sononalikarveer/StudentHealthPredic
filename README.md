# Student Health Prediction

A machine learning pipeline for predicting student health conditions using ensemble methods. This project uses CatBoost, LightGBM, and XGBoost with stacking ensemble to achieve strong prediction performance.

## Project Structure

```
health-kaggle/
├── data/                    # Input datasets (download from Kaggle)
│   ├── train.csv            # Place here after download
│   ├── test.csv             # Place here after download
│   └── sample_submission.csv
├── src/                     # Source code
│   ├── config.py            # Project paths and constants
│   ├── data.py              # Data loading and preprocessing
│   ├── features.py          # Feature engineering
│   ├── models.py            # Model wrappers (LightGBM, CatBoost, XGBoost)
│   ├── validation.py        # Cross-validation utilities
│   ├── train_baseline.py    # Baseline model training
│   ├── train_features.py    # Feature-engineered training
│   ├── train_final.py       # Final model training pipeline
│   ├── tune_optuna.py       # Hyperparameter tuning with Optuna
│   ├── ensemble.py          # Stacking ensemble (Logistic Regression meta-model)
│   ├── ensemble_pseudo.py   # Ensemble with pseudo-labeling
│   ├── pseudo_label.py      # Pseudo-label generation
│   ├── adversarial_validation.py  # Train/test distribution analysis
│   ├── train_pseudo.py      # Training with pseudo labels
│   └── run_eda.py           # Exploratory data analysis
├── tests/                   # Unit tests
│   ├── test_data.py
│   └── test_validation.py
├── outputs/
│   └── submissions/         # Model predictions
│       └── submission_best.csv
└── requirements.txt         # Python dependencies
```

## Data

Download the competition data from Kaggle:

🔗 **[Playground Series S6E7 — Student Health Prediction](https://www.kaggle.com/competitions/playground-series-s6e7/data)**

Place the downloaded files (`train.csv`, `test.csv`, `sample_submission.csv`) into the `health-kaggle/data/` directory.

Alternatively, use the Kaggle CLI:
```bash
kaggle competitions download -c playground-series-s6e7 -p health-kaggle/data/
unzip health-kaggle/data/playground-series-s6e7.zip -d health-kaggle/data/
```

## Setup

```bash
pip install -r health-kaggle/requirements.txt
```

## Usage

### 1. Run EDA (Optional)
```bash
python -m src.run_eda
```

### 2. Train Baseline Models
```bash
python -m src.train_baseline
```

### 3. Hyperparameter Tuning
```bash
python -m src.tune_optuna
```

### 4. Train Final Models
```bash
python -m src.train_final
```

### 5. Generate Ensemble Submission
```bash
python -m src.ensemble
```

## Models Used

- **LightGBM** — Gradient boosting with leaf-wise tree growth
- **CatBoost** — Gradient boosting with native categorical feature support
- **XGBoost** — Gradient boosting with level-wise tree growth
- **Stacking Ensemble** — Logistic Regression meta-model over base model predictions

## Key Techniques

- Stratified K-Fold cross-validation (5 folds)
- Optuna-based hyperparameter optimization
- Feature engineering (interaction & aggregate features)
- Pseudo-labeling for semi-supervised learning
- Adversarial validation to detect train/test distribution shift
- Balanced class weighting

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- catboost, lightgbm, xgboost
- optuna, matplotlib, seaborn, pytest
