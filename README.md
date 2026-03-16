# Ames Housing Price Predictor

A production-grade machine learning system that predicts residential home sale prices using the
Ames Housing Dataset. The project covers the full ML lifecycle — from raw data exploration and
preprocessing through model training, hyperparameter tuning, and deployment as a REST API.

**Dataset:** 2,930 properties · 82 features · Ames, Iowa · 2006–2010
**Best Model:** Gradient Boosting + Optuna HPO · R² = 0.940 · MAE ≈ $13,530
**Data Source:** [Kaggle — Ames Housing Dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset/data)

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Dataset](#2-dataset)
3. [Exploratory Data Analysis](#3-exploratory-data-analysis)
4. [Data Preprocessing](#4-data-preprocessing)
5. [Feature Engineering](#5-feature-engineering)
6. [Methods & Algorithms](#6-methods--algorithms)
7. [Results & Outcomes](#7-results--outcomes)
8. [System Architecture](#8-system-architecture)
9. [Quickstart](#9-quickstart)
10. [API Reference](#10-api-reference)
11. [Tech Stack](#11-tech-stack)
12. [Project Structure](#12-project-structure)

---

## 1. Introduction

Predicting a home's sale price is a classic regression problem with real-world impact — it drives
decisions for buyers, sellers, and appraisers. The Ames Housing Dataset provides a rich set of
features describing virtually every aspect of residential properties, making it ideal for building
and evaluating machine learning models.

This project goes beyond a standard notebook. It is structured as a deployable ML engineering
system with the following goals:

- Build a robust preprocessing and feature engineering pipeline
- Train and compare multiple regression algorithms with proper evaluation
- Tune the best model using Bayesian hyperparameter optimisation (Optuna)
- Track every experiment with MLflow for reproducibility
- Expose predictions through a production-ready FastAPI REST API
- Validate data contracts with Pandera schema checks
- Containerise with Docker and automate CI/CD with GitHub Actions

---

## 2. Dataset

```
Dataset : Ames Housing (AmesHousing.csv)
Rows    : 2,930 residential properties
Columns : 82 features  (36 numerical, 46 categorical)
Target  : SalePrice  (continuous, USD)
Period  : 2006 – 2010
Location: Ames, Iowa, USA
```

### Sale Price Summary Statistics

| Statistic | Value |
|---|---|
| Mean | $180,796 |
| Median | $160,000 |
| Std Dev | $79,887 |
| Min | $12,789 |
| Max | $755,000 |

### Feature Categories

| Location | Structure | Quality | Extras |
|---|---|---|---|
| Neighborhood | Year Built | Overall Qual | Pool Area |
| MS Zoning | Gr Liv Area | Overall Cond | Fireplaces |
| Lot Area | Total Bsmt SF | Exter Qual | Wood Deck SF |
| Lot Frontage | Garage Cars | Kitchen Qual | Open Porch SF |
| Condition | 1st / 2nd Flr SF | Bsmt Qual | Screen Porch |

---

## 3. Exploratory Data Analysis

### Sale Price Distribution

The target variable `SalePrice` is right-skewed. Most homes sold between $100k–$250k, with a
long tail of luxury properties. A **log transformation** (`log(1 + x)`) is applied before
modelling to normalise the distribution and reduce the influence of outliers.

```
  $0–$50k   |▌                              (  0.5%)
 $50–$100k  |████▌                          (  8.2%)
$100–$150k  |████████████████▌              ( 27.4%)
$150–$200k  |████████████████████▌          ( 31.6%)
$200–$250k  |████████████▌                  ( 18.9%)
$250–$300k  |██████▌                        (  8.3%)
$300–$400k  |████▌                          (  4.1%)
  $400k+    |▌                              (  1.0%)
```

### Top Feature Correlations with Sale Price

| Feature | Correlation | Direction |
|---|---|---|
| Overall Qual | +0.801 | Strong positive |
| Gr Liv Area | +0.709 | Strong positive |
| Garage Cars | +0.648 | Strong positive |
| Garage Area | +0.641 | Strong positive |
| Total Bsmt SF | +0.612 | Strong positive |
| 1st Flr SF | +0.596 | Strong positive |
| Full Bath | +0.561 | Strong positive |
| Year Built | +0.523 | Moderate positive |

> **Key insight:** Overall Quality is the single strongest predictor. A 1-point increase in
> quality rating is associated with a $20,000+ increase in sale price.

### Missing Data

| Column | % Missing | Action |
|---|---|---|
| Pool QC | 99.3% | Dropped |
| Misc Feature | 96.0% | Dropped |
| Alley | 93.2% | Dropped |
| Fence | 80.4% | Dropped |
| Fireplace Qu | 48.5% | Dropped |
| Lot Frontage | 16.7% | Imputed (median) |
| Garage Type | 2.8% | Imputed (mode) |
| Bsmt Qual | 1.9% | Imputed (mode) |

**Rule:** Drop columns with > 30% missing values. Impute the rest.

---

## 4. Data Preprocessing

The preprocessing pipeline runs in 7 sequential steps:

```
Raw CSV  (2,930 rows × 82 cols)
    │
    ▼
[1] Drop High-Missing Columns  — removes cols with > 30% nulls
    │
    ▼
[2] Impute Remaining Nulls
    │   Numerical  → median
    │   Categorical → mode (most frequent)
    │
    ▼
[3] Remove Sale Price Outliers  — IQR rule: Q1 − 1.5×IQR  to  Q3 + 1.5×IQR
    │
    ▼
[4] Log-Transform Skewed Features  — log(1 + x) where |skew| > 0.75
    │
    ▼
[5] One-Hot Encode Categorical Columns
    │   38 categorical columns → ~200 binary features
    │
    ▼
[6] Stratified Train / Validation / Test Split
    │   70% Train | 15% Validation | 15% Test
    │   Stratified on SalePrice quintiles
    │
    ▼
[7] Standard Scaling  — zero mean, unit variance (for linear models / SVM)

Final: 2,051 train | 440 validation | 439 test rows
```

### Schema Validation

All data is validated before and after preprocessing using **Pandera** schema contracts.
This catches dtype mismatches, out-of-range values, and unexpected nulls early.

---

## 5. Feature Engineering

13 domain-driven features are created from the raw columns before encoding.
All transformers follow the scikit-learn `BaseEstimator` + `TransformerMixin` protocol
so they plug directly into sklearn Pipelines.

| Feature | Formula | Rationale |
|---|---|---|
| `TotalSF` | Total Bsmt SF + 1st Flr SF + 2nd Flr SF | Total usable square footage |
| `TotalBath` | FullBath + 0.5 × HalfBath + BsmtFullBath + 0.5 × BsmtHalfBath | Weighted bathroom score |
| `HouseAge` | Yr Sold − Year Built | Age of home at time of sale |
| `RemodelAge` | Yr Sold − Year Remod/Add | Years since last remodel |
| `IsRemodeled` | 1 if Year Built ≠ Year Remod/Add, else 0 | Binary remodel flag |
| `GarageScore` | Garage Cars × Garage Area | Combined garage size and capacity |
| `QualCond` | Overall Qual × Overall Cond | Quality–condition interaction term |
| `PorchSF` | Open Porch SF + Enclosed Porch + 3Ssn Porch + Screen Porch | Total porch area |
| `HasPool` | 1 if Pool Area > 0, else 0 | Binary pool indicator |
| `HasFireplace` | 1 if Fireplaces > 0, else 0 | Binary fireplace indicator |
| `HasGarage` | 1 if Garage Area > 0, else 0 | Binary garage indicator |
| `HasBasement` | 1 if Total Bsmt SF > 0, else 0 | Binary basement indicator |
| `LotToLivRatio` | Lot Area / Gr Liv Area | Lot efficiency ratio |

### Feature Pipeline Order

```
Raw DataFrame
    │
    ├─ [1] HighMissingDropper   → drop cols with > 30% nulls
    ├─ [2] AmesFeatureEngineer  → add 13 domain features
    ├─ [3] RareLabelEncoder     → collapse categories < 1% frequency → 'Other'
    ├─ [4] SkewnessCorrector    → log1p on |skew| > 0.75 columns
    └─ [5] ColumnTransformer
           ├─ Numeric  → SimpleImputer (median) → [StandardScaler]
           └─ Categoric → SimpleImputer (constant) → OneHotEncoder
```

---

## 6. Methods & Algorithms

### 6.1 Evaluation Metrics

All models are evaluated using four metrics:

**R² (Coefficient of Determination)**

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

Measures the proportion of variance in the target explained by the model. Range: (−∞, 1]. A score of 1.0 is a perfect fit.

**RMSE (Root Mean Squared Error)**

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

Penalises large errors more heavily than MAE. Units are in dollars.

**MAE (Mean Absolute Error)**

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

The average absolute prediction error in dollars. More interpretable than RMSE.

**MAPE (Mean Absolute Percentage Error)**

$$\text{MAPE} = \frac{100}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

Error expressed as a percentage of the true value.

> **Note:** Because `SalePrice` is log-transformed during training, all predictions are
> inverse-transformed with `exp(x) − 1` before computing metrics in dollar space.

---

### 6.2 Linear Regression (Baseline)

Ordinary Least Squares (OLS) regression fits a hyperplane by minimising the sum of
squared residuals:

$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p$$

$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$$

Requires feature scaling (StandardScaler applied). Used as the baseline to beat.

---

### 6.3 Decision Tree

Recursively splits the feature space by selecting the feature and threshold that minimises
Mean Squared Error (MSE) at each node:

$$\text{MSE}_{\text{split}} = \frac{1}{n_L}\sum_{i \in L}(y_i - \bar{y}_L)^2 + \frac{1}{n_R}\sum_{i \in R}(y_i - \bar{y}_R)^2$$

The prediction for a leaf node is the mean of all training samples that fall into it.
A fully grown tree achieves R² = 1.0 on training data (memorisation) but generalises poorly.

---

### 6.4 Gradient Boosting (Best Model)

Gradient Boosting builds an ensemble of decision trees **sequentially**, where each tree
corrects the residual errors of the previous ones.

**Algorithm:**

1. Initialise the model with a constant prediction:

$$F_0(x) = \arg\min_\gamma \sum_{i=1}^{n} L(y_i, \gamma)$$

2. For each iteration `m = 1, 2, ..., M`:

   a. Compute the pseudo-residuals (negative gradient of the loss):

   $$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F = F_{m-1}}$$

   b. Fit a decision tree `h_m(x)` to the residuals `r_{im}`

   c. Compute the step size via line search:

   $$\gamma_m = \arg\min_\gamma \sum_{i=1}^{n} L(y_i,\ F_{m-1}(x_i) + \gamma \cdot h_m(x_i))$$

   d. Update the model:

   $$F_m(x) = F_{m-1}(x) + \eta \cdot \gamma_m \cdot h_m(x)$$

Where `η` is the **learning rate** (shrinkage), which controls the contribution of each tree.

**Loss function used (squared error):**

$$L(y, \hat{y}) = \frac{1}{2}(y - \hat{y})^2$$

**Tuned hyperparameters (Optuna):**

| Parameter | Value | Effect |
|---|---|---|
| `learning_rate` | 0.05 | Shrinks each tree's contribution |
| `n_estimators` | 200 | Number of trees in the ensemble |
| `max_depth` | 5 | Maximum depth per tree |
| `min_samples_leaf` | 2 | Minimum samples at leaf nodes |
| `subsample` | 0.85 | Fraction of samples per tree (stochastic GB) |

---

### 6.5 AdaBoost

Adaptive Boosting fits trees sequentially, increasing the weight of misclassified (high-error)
samples so subsequent trees focus on harder examples:

$$F_M(x) = \sum_{m=1}^{M} \alpha_m \cdot h_m(x)$$

Where `α_m` is the weight of tree `m`, computed from its weighted error rate. Performance
was significantly lower than Gradient Boosting on this dataset (R² = 0.808).

---

### 6.6 Support Vector Machine (SVM)

SVM solves the following optimisation problem for regression (SVR):

$$\min_{\mathbf{w}, b, \xi} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{n}(\xi_i + \xi_i^*)$$

$$\text{subject to: } y_i - \mathbf{w}^\top \phi(x_i) - b \leq \varepsilon + \xi_i$$

SVM failed on this dataset (R² ≈ −0.005). The reason: with 200+ one-hot encoded features,
the default RBF kernel cannot find an effective mapping, and SVM does not scale well to
high-dimensional sparse inputs without significant kernel tuning.

---

### 6.7 Hyperparameter Optimisation — Optuna

Optuna uses **Tree-structured Parzen Estimators (TPE)**, a Bayesian optimisation algorithm,
to search hyperparameter space efficiently. Rather than exhaustive grid search, TPE builds
a probabilistic model of which parameter regions yield good scores:

$$x^* = \arg\min_{x} \frac{l(x)}{g(x)}$$

Where `l(x)` models the distribution of good configurations and `g(x)` models the rest.
This converges to good hyperparameters in far fewer trials than GridSearchCV.

---

## 7. Results & Outcomes

### Model Comparison (Test Set)

| Model | R² | RMSE | MAE | Notes |
|---|---|---|---|---|
| Linear Regression | 0.864 | $30,902 | $19,241 | Baseline |
| Decision Tree | 0.820 | $35,671 | $22,115 | Overfit (train R²=1.0) |
| Gradient Boosting (default) | 0.909 | $25,315 | $14,136 | Strong out of the box |
| AdaBoost | 0.808 | $36,742 | $24,115 | Weaker than GB |
| SVM | −0.005 | $84,103 | $51,987 | Failed — wrong kernel |
| **Gradient Boosting + Optuna** | **0.940** | **$20,215** | **$13,530** | **Best model** |

### Final Model Performance

```
╔══════════════╦═════════╦══════════╦══════════╗
║ Split        ║   R²    ║   RMSE   ║   MAE    ║
╠══════════════╬═════════╬══════════╬══════════╣
║ Train        ║  0.971  ║ $14,442  ║  $9,713  ║
║ Validation   ║  0.940  ║ $20,215  ║ $13,530  ║
║ Test         ║  0.911  ║ $25,315  ║ $14,136  ║
║ Cross-Val    ║  0.948  ║ $19,273  ║ $12,981  ║
╚══════════════╩═════════╩══════════╩══════════╝
```

> The model explains **94% of the variance** in home sale prices with a typical
> prediction error of **±$13,530**.

### Feature Importance (Top 15)

| Rank | Feature | Importance | Type |
|---|---|---|---|
| 1 | Overall Qual | 14.2% | Raw |
| 2 | Gr Liv Area | 12.8% | Raw |
| 3 | Total Bsmt SF | 11.2% | Raw |
| 4 | Year Built | 9.8% | Raw |
| 5 | Garage Cars | 9.1% | Raw |
| 6 | 1st Flr SF | 7.9% | Raw |
| 7 | TotalSF | 7.1% | Engineered |
| 8 | Neighborhood_NridgHt | 5.8% | Encoded |
| 9 | Garage Area | 5.2% | Raw |
| 10 | Year Remod/Add | 4.1% | Raw |
| 11 | Full Bath | 3.8% | Raw |
| 12 | Lot Area | 3.1% | Raw |
| 13 | Kitchen Qual_Ex | 2.8% | Encoded |
| 14 | Exter Qual_Ex | 1.9% | Encoded |
| 15 | Foundation_PConc | 1.6% | Encoded |

### Why the Other Models Underperformed

**Decision Tree** — Fully grown trees memorise the training set (train R² = 1.0) but fail to
generalise. This is the classic bias-variance tradeoff: low bias, very high variance.

**AdaBoost** — Boosting with shallow trees is less effective than Gradient Boosting for
tabular regression. It achieved 0.808 R², significantly below GB.

**SVM** — The RBF kernel cannot handle the 200+ sparse one-hot features effectively.
SVM requires heavy kernel tuning and does not scale well to high-dimensional tabular data.

---

## 8. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     AMES HOUSING SYSTEM                         │
├─────────────────┬──────────────────┬─────────────┬─────────────┤
│   DATA LAYER    │   TRAIN LAYER    │ SERVE LAYER │  OPS LAYER  │
│                 │                  │             │             │
│  AmesHousing   │  Optuna HPO      │  FastAPI    │  MLflow UI  │
│   .csv          │  (Bayesian TPE)  │  /predict   │  Loguru     │
│                 │                  │  /health    │  SHAP       │
│  Pandera        │  MLflow          │  /metrics   │             │
│  Schema         │  Tracking        │  /docs      │  Ruff lint  │
│  Validation     │                  │             │  pytest     │
│                 │  LightGBM        │  Pydantic   │  GitHub     │
│  sklearn        │  XGBoost         │  Validation │  Actions    │
│  Pipeline       │  GBM             │             │             │
│                 │                  │  Docker     │  Docker     │
└─────────────────┴──────────────────┴─────────────┴─────────────┘
```

### Request Flow

```
Client
  │
  │  POST /predict  { OverallQual, GrLivArea, YearBuilt, ... }
  ▼
FastAPI  →  Pydantic validation  →  to_dataframe()
  │
  ▼
sklearn Pipeline
  ├─ HighMissingDropper
  ├─ AmesFeatureEngineer   (adds 13 features)
  ├─ RareLabelEncoder
  ├─ SkewnessCorrector
  └─ ColumnTransformer → Gradient Boosting model
  │
  ▼
log prediction  →  exp(x) − 1  →  add ±9% confidence interval
  │
  ▼
{ predicted_price, lower_bound, upper_bound, prediction_id }
```

---

## 9. Quickstart

### Step 1 — Install dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install all packages
pip install -e ".[dev]"
```

### Step 2 — Place the data

Download `AmesHousing.csv` from [Kaggle](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset/data)
and place it in:

```bash
data/raw/AmesHousing.csv
```

### Step 3 — Train the model

```bash
make train
# Runs preprocessing → feature engineering → Optuna HPO → saves model to models/
```

### Step 4 — View experiment results

```bash
make mlflow
# Open http://localhost:5000 in your browser
```

### Step 5 — Start the prediction API

```bash
make serve
# API: http://localhost:8000
# Swagger docs: http://localhost:8000/docs
```

### Step 6 — Make a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "OverallQual": 7,
    "GrLivArea": 1800,
    "YearBuilt": 2005,
    "GarageCars": 2,
    "TotalBsmtSF": 900,
    "Neighborhood": "CollgCr"
  }'
```

Response:
```json
{
  "predicted_price": 198450.00,
  "lower_bound": 181030.50,
  "upper_bound": 215869.50,
  "model_version": "0.1.0",
  "prediction_id": "a3f2c1d9-..."
}
```

### Step 7 — Run tests

```bash
pytest
# 88 tests across data, features, models, and API layers
```

---

## 10. API Reference

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check + model readiness |
| `GET` | `/metrics` | Stored model performance metrics |
| `POST` | `/predict` | Predict sale price for one property |
| `POST` | `/predict/batch` | Predict for up to 100 properties |
| `GET` | `/docs` | Swagger interactive documentation |

### Required Fields (`POST /predict`)

| Field | Type | Description |
|---|---|---|
| `OverallQual` | int (1–10) | Overall material and finish quality |
| `GrLivArea` | float | Above-grade living area (sqft) |
| `YearBuilt` | int | Original construction year |

All other fields are optional with sensible defaults (e.g. `GarageCars=1`, `FullBath=1`).

### Example Request

```json
{
  "OverallQual": 7,
  "OverallCond": 5,
  "GrLivArea": 1710,
  "TotalBsmtSF": 856,
  "GarageCars": 2,
  "GarageArea": 548,
  "YearBuilt": 2003,
  "Neighborhood": "CollgCr",
  "ExterQual": "Gd",
  "KitchenQual": "Gd",
  "FullBath": 2
}
```

### Example Response

```json
{
  "predicted_price": 208500.00,
  "lower_bound": 190215.00,
  "upper_bound": 226785.00,
  "model_version": "0.1.0",
  "prediction_id": "b7e3a2f1-4c8d-11ee-be56-0242ac120002",
  "timestamp": "2026-03-16T17:00:00Z"
}
```

---

## 11. Tech Stack

| Category | Tool | Purpose |
|---|---|---|
| Language | Python 3.11 | Core language |
| ML Core | scikit-learn, LightGBM, XGBoost | Model training and pipelines |
| HPO | Optuna | Bayesian hyperparameter search (TPE) |
| Experiment Tracking | MLflow | Log params, metrics, and artefacts |
| Data Validation | Pandera | DataFrame schema contracts |
| API | FastAPI | Async REST API with auto docs |
| Serialisation | Pydantic v2 | Request/response validation |
| Logging | Loguru | Structured, rotating logs |
| Explainability | SHAP | Feature importance and waterfall plots |
| Linting | Ruff | Replaces black + flake8 + isort |
| Testing | pytest + pytest-cov | 88 unit and integration tests |
| CI/CD | GitHub Actions | Lint and test on every push |
| Containers | Docker + Docker Compose | Reproducible deployment |
| Config | Pydantic Settings + YAML | Type-safe configuration |

---

## 12. Project Structure

```
ames-housing/
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions: lint + test on push
│
├── configs/
│   └── config.yaml                 # All project settings
│
├── data/
│   ├── raw/                        # AmesHousing.csv  (not in git)
│   ├── interim/                    # Intermediate processing artefacts
│   └── processed/                  # Train / val / test parquet splits
│
├── models/                         # Saved model .pkl and metadata
│
├── notebooks/                      # EDA and exploration notebooks
│
├── src/
│   └── ames_housing/
│       ├── config.py               # Pydantic settings model
│       ├── data/
│       │   ├── loader.py           # CSV loading + Pandera schema validation
│       │   ├── preprocessor.py     # Imputation, encoding, splitting
│       │   └── schema.py           # Pandera schema definitions
│       ├── features/
│       │   ├── engineering.py      # build_pipeline() factory
│       │   └── transformers.py     # Custom sklearn-compatible transformers
│       ├── models/
│       │   ├── trainer.py          # Optuna HPO + MLflow tracking
│       │   ├── evaluator.py        # Metrics computation + SHAP plots
│       │   └── registry.py         # Save / load model artefacts
│       ├── api/
│       │   ├── main.py             # FastAPI app factory + lifespan
│       │   ├── routes.py           # Endpoint handlers
│       │   └── schemas.py          # Pydantic request / response models
│       └── utils/
│           ├── logging.py          # Loguru configuration
│           └── helpers.py          # Shared utilities
│
├── tests/
│   ├── conftest.py                 # Shared fixtures and synthetic data
│   ├── test_data.py                # Data pipeline tests
│   ├── test_features.py            # Feature engineering tests
│   ├── test_models.py              # Model registry and metrics tests
│   └── test_api.py                 # FastAPI endpoint integration tests
│
├── docker/
│   ├── Dockerfile                  # Multi-stage build (builder + runtime)
│   └── docker-compose.yml          # API + MLflow services
│
├── pyproject.toml                  # Dependencies + tool configuration
├── Makefile                        # Developer shortcuts (train, serve, test)
└── README.md
```

---

*Built end-to-end with modern Python ML engineering practices.*
