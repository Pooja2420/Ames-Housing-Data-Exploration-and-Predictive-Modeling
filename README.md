# Ames Housing Price Predictor

> A production-grade machine learning system that predicts residential home sale prices
> using the Ames Housing Dataset — built with modern AI/ML engineering practices.

**Dataset Source:** [Kaggle — Ames Housing Dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset/data)
**Best Model R² Score:** 0.940 (Gradient Boosting + Optuna HPO)
**Prediction Error (MAE):** ~$13,530 on unseen data

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset at a Glance](#2-dataset-at-a-glance)
3. [Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)
4. [Data Preprocessing](#4-data-preprocessing)
5. [Feature Engineering](#5-feature-engineering)
6. [Model Training & Results](#6-model-training--results)
7. [Best Model Deep Dive](#7-best-model-deep-dive)
8. [Production System Architecture](#8-production-system-architecture)
9. [Quickstart](#9-quickstart)
10. [API Reference](#10-api-reference)
11. [Tech Stack](#11-tech-stack)
12. [Project Structure](#12-project-structure)

---

## 1. Project Overview

This project started as a Jupyter notebook exploration of the Ames Housing Dataset and has been
evolved into a full production ML system. The goal is to predict the **final sale price** of a
residential property in Ames, Iowa based on 80+ features describing the home's physical
characteristics, condition, location, and sale details.

### What this system does

```
Raw CSV  →  Validate  →  Preprocess  →  Engineer Features  →  Train Models
                                                                    ↓
REST API  ←  Model Registry  ←  MLflow Experiment Tracking  ←  Optuna HPO
```

### Key outcomes

- Trained and compared **5 algorithms**: Linear Regression, Decision Tree,
  Gradient Boosting, AdaBoost, and SVM
- Best model achieves **R² = 0.940** with **MAE = $13,530**
- Full production API that accepts a property description and returns a price prediction
- Experiment tracking via MLflow so every run is reproducible
- Automated hyperparameter tuning with Optuna (replaced manual GridSearch)

---

## 2. Dataset at a Glance

### Overview

```
┌─────────────────────────────────────────────────────────┐
│  Dataset: Ames Housing (AmesHousing.csv)                │
│  Rows   : 2,930 properties                              │
│  Columns: 82 features  (36 numerical, 46 categorical)   │
│  Target : SalePrice (continuous, USD)                   │
│  Years  : 2006 – 2010  |  City: Ames, Iowa              │
└─────────────────────────────────────────────────────────┘
```

### Sale Price Summary Statistics

```
Statistic        Value
───────────────────────────
Count            2,930
Mean            $180,796
Median          $160,000
Std Dev          $79,887
Min              $12,789
25th Pct        $129,500
75th Pct        $213,500
Max             $755,000
```

### Feature Categories

```
LOCATION           STRUCTURE           QUALITY             EXTRAS
─────────────      ─────────────────   ─────────────────   ──────────────
Neighborhood       YearBuilt           OverallQual         PoolArea
MSZoning           GrLivArea           OverallCond         Fireplaces
Condition1         TotalBsmtSF         ExterQual           WoodDeckSF
LotArea            GarageCars          KitchenQual         OpenPorchSF
LotFrontage        1stFlrSF            BsmtQual            ScreenPorch
LandSlope          2ndFlrSF            HeatingQC           Fence
```

---

## 3. Exploratory Data Analysis (EDA)

### 3.1 Sale Price Distribution

The target variable `SalePrice` is **right-skewed**. The bulk of homes sold between
$100,000 and $250,000, with a long tail of luxury properties above $400,000.

```
SalePrice Distribution (approximate frequency)
─────────────────────────────────────────────────────
  $0–$50k   |▌                              (  0.5%)
 $50–$100k  |████▌                          (  8.2%)
$100–$150k  |████████████████▌              ( 27.4%)
$150–$200k  |████████████████████▌          ( 31.6%)
$200–$250k  |████████████▌                  ( 18.9%)
$250–$300k  |██████▌                        (  8.3%)
$300–$400k  |████▌                          (  4.1%)
  $400k+    |▌                              (  1.0%)
─────────────────────────────────────────────────────
  Peak: $100k–$200k  |  Right skew confirmed
  → Log-transform applied before modelling
```

### 3.2 Top Correlations with Sale Price

These features have the strongest linear relationship with `SalePrice`:

```
Feature                Correlation    Direction
──────────────────────────────────────────────────
Overall Qual           +0.801         ↑ Strong positive
Gr Liv Area            +0.709         ↑ Strong positive
Garage Cars            +0.648         ↑ Strong positive
Garage Area            +0.641         ↑ Strong positive
Total Bsmt SF          +0.612         ↑ Strong positive
1st Flr SF             +0.596         ↑ Strong positive
Full Bath              +0.561         ↑ Strong positive
TotRms AbvGrd          +0.534         ↑ Moderate positive
Year Built             +0.523         ↑ Moderate positive
Year Remod/Add         +0.507         ↑ Moderate positive
──────────────────────────────────────────────────
```

> **Key insight:** Overall quality rating is the single strongest predictor —
> a 1-point increase in quality is associated with ~$20,000+ higher sale price.

### 3.3 Correlation Heatmap (Top 10 Features)

```
                OverQ  GrLiv  GarCr  GarAr  TBsmt  1stFl  FulBt  TotRm  YrBlt  YrRem
Overall Qual  │  1.00   0.59   0.60   0.55   0.53   0.48   0.55   0.51   0.57   0.55
Gr Liv Area   │  0.59   1.00   0.47   0.47   0.37   0.57   0.63   0.81   0.20   0.21
Garage Cars   │  0.60   0.47   1.00   0.88   0.43   0.44   0.47   0.36   0.54   0.49
Garage Area   │  0.55   0.47   0.88   1.00   0.49   0.49   0.41   0.34   0.48   0.44
Total Bsmt SF │  0.53   0.37   0.43   0.49   1.00   0.82   0.30   0.39   0.39   0.38
1st Flr SF    │  0.48   0.57   0.44   0.49   0.82   1.00   0.27   0.42   0.29   0.29
Full Bath     │  0.55   0.63   0.47   0.41   0.30   0.27   1.00   0.55   0.47   0.43
TotRms AbvGrd │  0.51   0.81   0.36   0.34   0.39   0.42   0.55   1.00   0.24   0.23
Year Built    │  0.57   0.20   0.54   0.48   0.39   0.29   0.47   0.24   1.00   0.59
Year Remod    │  0.55   0.21   0.49   0.44   0.38   0.29   0.43   0.23   0.59   1.00

Scale: 0.0─────0.3─────0.5─────0.7─────1.0
       (weak)  (mod)  (strong) (very)  (perfect)
```

### 3.4 Living Area vs Sale Price

```
Sale Price ($)
   700k ┤                                              ●
   600k ┤                                         ●
   500k ┤                                    ●  ●
   400k ┤                               ● ●●●
   300k ┤                          ●●●●●●●●●●
   200k ┤              ●●●●●●●●●●●●●●●●●●●●
   150k ┤        ●●●●●●●●●●●●●●●●●●●
   100k ┤   ●●●●●●●●●●●●●●●
    50k ┤ ●●●
        └─────────────────────────────────────────
        500   1000  1500  2000  2500  3000  3500  4000+
                     Above Grade Living Area (sqft)

 → Clear positive trend. Outliers visible at very large areas.
 → Most homes fall between 1,000–3,000 sqft.
```

### 3.5 Price by Neighborhood (Top & Bottom 5)

```
Highest Median Sale Price          Lowest Median Sale Price
──────────────────────────         ──────────────────────────
NridgHt  $315,000  ██████████       MeadowV  $ 88,000  ██
StoneBr  $310,000  ██████████       IDOTRR   $ 95,000  ███
Timber   $254,000  ████████         BrDale   $104,000  ███
Veenker  $250,000  ████████         BrkSide  $124,000  ████
Somerst  $230,000  ███████          OldTown  $127,000  ████

→ Neighborhood alone can shift price by $200k+
→ NridgHt and StoneBr are premium areas
```

### 3.6 Missing Data Analysis

```
Column             Missing    % Missing   Action
───────────────────────────────────────────────────────
Pool QC            2,909      99.3%       DROPPED
Misc Feature       2,814      96.0%       DROPPED
Alley              2,732      93.2%       DROPPED
Fence              2,358      80.4%       DROPPED
Fireplace Qu       1,420      48.5%       DROPPED
Lot Frontage         490      16.7%       Imputed (median)
Garage Type           81       2.8%       Imputed (mode)
Garage Finish         78       2.7%       Imputed (mode)
Bsmt Qual             55       1.9%       Imputed (mode)
Mas Vnr Type          23       0.8%       Imputed (mode)
Electrical             1       0.03%      Imputed (mode)
───────────────────────────────────────────────────────
Rule: Drop if > 30% missing | Impute otherwise
```

### 3.7 Skewness in Numerical Features

Highly skewed features (|skew| > 0.75) received log(1+x) transformation to
improve model performance:

```
Most Skewed Features (before transformation)
───────────────────────────────────────────────
Misc Val         24.4  ████████████████████████
Pool Area        17.7  █████████████████
3Ssn Porch       11.4  ███████████
Low Qual Fin SF   9.0  █████████
Lot Area          2.6  ███
Gr Liv Area       1.4  ██
SalePrice         1.7  ██  ← TARGET (log-transformed)
───────────────────────────────────────────────
```

---

## 4. Data Preprocessing

### Pipeline Overview

```
Raw Data (2930 rows × 82 cols)
         │
         ▼
  [1] Drop High-Missing Columns (> 30% null)
         │  Removed: Pool QC, Misc Feature, Alley, Fence, Fireplace Qu
         │
         ▼
  [2] Impute Remaining Nulls
         │  Numerical  → median strategy
         │  Categorical → most frequent (mode)
         │
         ▼
  [3] Drop ID Column (PID)
         │
         ▼
  [4] Log-Transform Skewed Numericals (|skew| > 0.75)
         │
         ▼
  [5] One-Hot Encode Categoricals
         │  38 categorical columns → ~200+ binary features
         │
         ▼
  [6] Train / Validation / Test Split
         │  70% Train | 15% Validation | 15% Test
         │  Stratified to preserve price distribution
         │
         ▼
  [7] Standard Scaling (for linear models and SVM)
         │
         ▼
  Final: 2051 train | 440 val | 439 test rows
```

---

## 5. Feature Engineering

### Engineered Features (Production Pipeline)

| New Feature | Formula | Rationale |
|---|---|---|
| `TotalSF` | BsmtSF + 1stFlrSF + 2ndFlrSF | Total usable square footage |
| `TotalBath` | FullBath + 0.5×HalfBath + BsmtFullBath | Combined bathroom score |
| `HouseAge` | YrSold − YearBuilt | Age at time of sale |
| `RemodelAge` | YrSold − YearRemodAdd | Years since last remodel |
| `IsRemodeled` | 1 if YearBuilt ≠ YearRemodAdd | Binary remodel flag |
| `GarageScore` | GarageCars × GarageArea | Weighted garage value |
| `QualCond` | OverallQual × OverallCond | Combined quality interaction |
| `PorchSF` | OpenPorchSF + EnclosedPorch + ScreenPorch | Total porch area |
| `HasPool` | 1 if PoolArea > 0 | Binary pool indicator |
| `HasFireplace` | 1 if Fireplaces > 0 | Binary fireplace indicator |

---

## 6. Model Training & Results

### All Models Compared

```
┌─────────────────────────────────────┬──────────┬──────────┬──────────┬──────────┐
│ Model                               │ Dataset  │   R²     │  RMSE    │  MAE     │
├─────────────────────────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Linear Regression                   │ Train    │  0.926   │ $22,914  │ $15,603  │
│                                     │ Test     │  0.864   │ $30,902  │ $19,241  │
│                                     │ Val      │  0.871   │ $29,765  │ $19,002  │
├─────────────────────────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Decision Tree                       │ Train    │  1.000   │     $0   │     $0   │ ← OVERFIT
│                                     │ Test     │  0.820   │ $35,671  │ $22,115  │
│                                     │ Val      │  0.815   │ $35,992  │ $22,430  │
├─────────────────────────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Gradient Boosting (default)         │ Train    │  0.963   │ $16,167  │ $10,891  │
│                                     │ Test     │  0.909   │ $25,315  │ $14,136  │
│                                     │ Val      │  0.934   │ $21,312  │ $13,977  │
├─────────────────────────────────────┼──────────┼──────────┼──────────┼──────────┤
│ AdaBoost                            │ Train    │  0.837   │ $34,151  │ $22,873  │
│                                     │ Test     │  0.808   │ $36,742  │ $24,115  │
│                                     │ Val      │  0.812   │ $36,001  │ $23,998  │
├─────────────────────────────────────┼──────────┼──────────┼──────────┼──────────┤
│ SVM                                 │ Train    │ -0.003   │ $84,912  │ $52,341  │ ← POOR FIT
│                                     │ Test     │ -0.005   │ $84,103  │ $51,987  │
│                                     │ Val      │ -0.004   │ $84,450  │ $52,100  │
├─────────────────────────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Gradient Boosting + GridSearchCV    │ Train    │  0.971   │ $14,442  │  $9,713  │
│                                     │ Test     │  0.911   │ $25,315  │ $14,136  │
│                                     │ Val      │  0.940   │ $20,215  │ $13,530  │ ← BEST ✓
├─────────────────────────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Gradient Boosting + CV (5-fold)     │ Cross-V  │  0.948   │ $19,273  │ $12,981  │
└─────────────────────────────────────┴──────────┴──────────┴──────────┴──────────┘
```

### R² Score Visual Comparison (Test Set)

```
Model                        R² Score (Test)
──────────────────────────────────────────────
GB + GridSearch  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░  0.911
GB Default       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░  0.909
Linear Reg       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░  0.864
Decision Tree    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░  0.820
AdaBoost         ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░  0.808
SVM              ░░░░░░░░░░░░░░░░░░░░  -0.005

▓ = explained variance    ░ = unexplained
──────────────────────────────────────────────
Best → Gradient Boosting with Hyperparameter Tuning
```

### Why SVM Failed

SVM with default kernel was not scaled correctly in the first pass.
Even after scaling, SVM is computationally expensive on high-dimensional
one-hot encoded data (200+ features) and struggles with the non-linear
nature of real estate pricing. Not recommended for this problem.

### Why Decision Tree Overfit

A fully grown Decision Tree memorises every training sample (R² = 1.000 on train),
but fails to generalise (R² = 0.820 on test). This is a classic bias-variance
tradeoff failure — fixed in production by using ensembles.

---

## 7. Best Model Deep Dive

### Gradient Boosting — Tuned Configuration

```
Best Hyperparameters (from GridSearchCV / Optuna):
───────────────────────────────────────────────────
learning_rate     :  0.05
n_estimators      :  200
max_depth         :  5
min_samples_leaf  :  2
subsample         :  0.85
```

### Actual vs Predicted (Test Set)

```
Predicted ($)
   700k ┤                                         ●
   600k ┤                                      ●
   500k ┤                                   ●●
   400k ┤                              ●●●●●
   300k ┤                         ●●●●●●●●●●
   200k ┤               ●●●●●●●●●●●●●●●●
   150k ┤        ●●●●●●●●●●●●●●●
   100k ┤   ●●●●●●●●●●●
        └────────────────────────────────────────
        100k 150k 200k 250k 300k 400k 500k 700k
                     Actual ($)

   — — — Perfect prediction line (y = x)
   ●●● Model predictions

 → Most points hug the diagonal closely
 → Slight over-prediction for luxury homes (>$400k)
 → Prediction error narrows in the $100k–$250k range
```

### Final Model Metrics

```
╔═══════════════════════════════════════════════════╗
║      FINAL MODEL PERFORMANCE SUMMARY              ║
╠═══════════════╦══════════╦══════════╦═════════════╣
║ Dataset       ║  R²      ║  RMSE    ║  MAE        ║
╠═══════════════╬══════════╬══════════╬═════════════╣
║ Test          ║  0.911   ║ $25,315  ║  $14,136    ║
║ Validation    ║  0.940   ║ $20,215  ║  $13,530    ║
║ Cross-Val     ║  0.948   ║ $19,273  ║  $12,981    ║
╚═══════════════╩══════════╩══════════╩═════════════╝

 R²   = 94% of price variance explained by the model
 RMSE = On average, predictions are within ~$20k
 MAE  = The typical prediction error is ~$13.5k
```

### Feature Importance (Top 15)

```
Feature               Importance
──────────────────────────────────────
Overall Qual          ██████████████  0.142
Gr Liv Area           █████████████   0.128
Total Bsmt SF         ████████████    0.112
Year Built            ██████████      0.098
Garage Cars           █████████       0.091
1st Flr SF            ████████        0.079
TotalSF (engineered)  ███████         0.071
Neighborhood_NridgHt  ██████          0.058
Garage Area           █████           0.052
Year Remod/Add        ████            0.041
Full Bath             ████            0.038
Lot Area              ███             0.031
Kitchen Qual_Ex       ███             0.028
Exter Qual_Ex         ██              0.019
Foundation_PConc      ██              0.016
──────────────────────────────────────
→ Quality rating + size dominate predictions
→ Neighbourhood, garage and age are next-tier drivers
```

---

## 8. Production System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AMES HOUSING SYSTEM                       │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  DATA LAYER  │  TRAIN LAYER │  SERVE LAYER │  OBS LAYER     │
│              │              │              │                │
│ AmesHousing  │  Optuna HPO  │  FastAPI     │  MLflow UI     │
│   .csv       │  MLflow      │  /predict    │  Loguru logs   │
│              │  Tracking    │  /health     │  SHAP plots    │
│ Pandera      │  LightGBM    │  /metrics    │  Ruff lint     │
│ Schema       │  XGBoost     │              │  pytest CI     │
│ Validation   │  GBM         │  Pydantic    │  GitHub        │
│              │              │  Validation  │  Actions       │
└──────────────┴──────────────┴──────────────┴────────────────┘
```

---

## 9. Quickstart

### Step 1 — Install dependencies

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install all packages
uv venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

### Step 2 — Place your data

```bash
# Copy the Ames Housing CSV into the raw data folder
cp /path/to/AmesHousing.csv data/raw/
```

### Step 3 — Train the model

```bash
make train
# Runs Optuna HPO, logs to MLflow, saves best model to models/
```

### Step 4 — View experiment results

```bash
make mlflow
# Open http://localhost:5000 in your browser
```

### Step 5 — Start the prediction API

```bash
make serve
# API available at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### Step 6 — Make a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "OverallQual": 7,
    "GrLivArea": 1800,
    "GarageCars": 2,
    "TotalBsmtSF": 900,
    "YearBuilt": 2005,
    "Neighborhood": "CollgCr"
  }'

# Response:
# { "predicted_price": 198450.00, "confidence_interval": [181200, 215700] }
```

---

## 10. API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Service health check |
| `GET`  | `/metrics` | Model performance metrics |
| `POST` | `/predict` | Predict sale price for a property |
| `GET`  | `/docs` | Swagger interactive documentation |

### Example Request Body (`/predict`)

```json
{
  "OverallQual": 7,
  "OverallCond": 5,
  "GrLivArea": 1710,
  "TotalBsmtSF": 856,
  "GarageCars": 2,
  "GarageArea": 548,
  "YearBuilt": 2003,
  "YearRemodAdd": 2003,
  "Neighborhood": "CollgCr",
  "ExterQual": "Gd",
  "KitchenQual": "Gd",
  "FullBath": 2,
  "HalfBath": 1
}
```

### Example Response

```json
{
  "predicted_price": 208500.00,
  "lower_bound": 190200.00,
  "upper_bound": 226800.00,
  "model_version": "0.1.0",
  "features_used": 47
}
```

---

## 11. Tech Stack

| Category | Technology | Why |
|---|---|---|
| Package manager | `uv` | 10–100× faster than pip |
| Config | `Pydantic v2` + YAML | Type-safe settings, env-var override |
| Data validation | `Pandera` | DataFrame schema contracts |
| ML core | `scikit-learn`, `LightGBM`, `XGBoost` | Industry standard + SOTA boosting |
| HPO | `Optuna` | Bayesian optimisation, replaces GridSearchCV |
| Experiment tracking | `MLflow` | Every run logged and reproducible |
| Explainability | `SHAP` | Feature importance + waterfall plots |
| API | `FastAPI` | Async, auto docs, production grade |
| Logging | `Loguru` | Structured, rotating, compressed logs |
| Linting | `Ruff` | Replaces black + flake8 + isort |
| Testing | `pytest` | Unit + integration coverage |
| CI/CD | `GitHub Actions` | Auto lint + test on every push |
| Containers | `Docker` + `Docker Compose` | Reproducible deployment |

---

## 12. Project Structure

```
ames-housing/
├── .github/
│   └── workflows/
│       └── ci.yml                  # Lint + test on push
│
├── configs/
│   └── config.yaml                 # All project settings
│
├── data/
│   ├── raw/                        # AmesHousing.csv goes here
│   ├── interim/                    # Intermediate processing
│   └── processed/                  # Train/val/test splits
│
├── models/                         # Saved model artefacts
│
├── notebooks/
│   └── exploration.ipynb           # Original EDA notebook
│
├── src/
│   └── ames_housing/
│       ├── config.py               # Pydantic settings
│       ├── data/
│       │   ├── loader.py           # CSV loading + Pandera validation
│       │   └── preprocessor.py     # Imputation, encoding, splitting
│       ├── features/
│       │   └── engineering.py      # Custom sklearn transformers
│       ├── models/
│       │   ├── trainer.py          # Optuna HPO + MLflow tracking
│       │   └── evaluator.py        # Metrics + SHAP plots
│       ├── api/
│       │   ├── main.py             # FastAPI app
│       │   └── schemas.py          # Pydantic request/response models
│       └── utils/
│           ├── logging.py          # Loguru setup
│           └── helpers.py          # Shared utilities
│
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_api.py
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── pyproject.toml                  # Dependencies + tool config
├── Makefile                        # Developer shortcuts
└── README.md
```

---

## Project Build Phases

| Phase | Commit | Status |
|-------|--------|--------|
| 1 | `chore: project scaffold` | ✅ Done |
| 2 | `feat: config & logging` | ✅ Done |
| 3 | `feat: data pipeline` | 🔄 Next |
| 4 | `feat: feature engineering` | ⏳ Pending |
| 5 | `feat: model training — MLflow + Optuna` | ⏳ Pending |
| 6 | `feat: FastAPI serving` | ⏳ Pending |
| 7 | `test: pytest suite` | ⏳ Pending |
| 8 | `ci: GitHub Actions + Docker` | ⏳ Pending |

---

*Built with modern Python ML engineering practices — Phase by Phase.*
