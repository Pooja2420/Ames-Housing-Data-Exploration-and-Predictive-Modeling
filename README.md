# Ames Housing Price Predictor

A production-grade machine learning system for predicting residential home sale prices using the [Ames Housing Dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset/data).

## Architecture

```
ames-housing/
├── src/ames_housing/
│   ├── config.py          # Pydantic settings
│   ├── data/              # Data loading & validation
│   ├── features/          # Feature engineering pipelines
│   ├── models/            # Training, evaluation, registry
│   ├── api/               # FastAPI serving layer
│   └── utils/             # Logging, helpers
├── configs/               # YAML configuration files
├── data/
│   ├── raw/               # Place AmesHousing.csv here
│   ├── interim/           # Intermediate data
│   └── processed/         # Final datasets
├── tests/                 # pytest suite
├── notebooks/             # EDA notebooks
└── docker/                # Container definitions
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Package manager | `uv` |
| Config | `Pydantic v2` + YAML |
| Data validation | `Pandera` |
| ML | `scikit-learn`, `LightGBM`, `XGBoost` |
| HPO | `Optuna` |
| Experiment tracking | `MLflow` |
| Explainability | `SHAP` |
| API | `FastAPI` |
| Logging | `Loguru` |
| Linting | `Ruff` |
| Testing | `pytest` |
| CI/CD | `GitHub Actions` |
| Containers | `Docker` + `Docker Compose` |

## Quickstart

### 1. Install dependencies

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

### 2. Add your data

```bash
cp /path/to/AmesHousing.csv data/raw/
```

### 3. Train a model

```bash
make train
# or
ames-train
```

### 4. Launch the prediction API

```bash
make serve
# or
ames-serve
```

API docs available at: `http://localhost:8000/docs`

### 5. Track experiments

```bash
make mlflow
# Open http://localhost:5000
```

## Development

```bash
make install-dev   # install with dev extras
make lint          # run ruff
make test          # run pytest with coverage
make format        # auto-format with ruff
make all           # lint + test
```

## Docker

```bash
docker compose up --build
```

## Project Phases

- [x] Phase 1 — Project scaffold
- [ ] Phase 2 — Config & logging
- [ ] Phase 3 — Data pipeline
- [ ] Phase 4 — Feature engineering
- [ ] Phase 5 — Model training (MLflow + Optuna)
- [ ] Phase 6 — FastAPI serving
- [ ] Phase 7 — Test suite
- [ ] Phase 8 — CI/CD & Docker
