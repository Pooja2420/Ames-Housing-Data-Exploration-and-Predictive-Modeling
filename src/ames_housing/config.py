"""Central configuration using Pydantic v2 Settings + YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Repo root (two levels up from this file: src/ames_housing/config.py) ──────
ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_FILE = ROOT_DIR / "configs" / "config.yaml"


# ── Sub-models ──────────────────────────────────────────────────────────────

class ProjectConfig(BaseModel):
    name: str = "ames-housing"
    version: str = "0.1.0"
    description: str = "Ames Housing Price Prediction"


class PathsConfig(BaseModel):
    data_raw: Path = ROOT_DIR / "data" / "raw" / "AmesHousing.csv"
    data_interim: Path = ROOT_DIR / "data" / "interim"
    data_processed: Path = ROOT_DIR / "data" / "processed"
    models: Path = ROOT_DIR / "models"
    logs: Path = ROOT_DIR / "logs"
    mlruns: Path = ROOT_DIR / "mlruns"

    def ensure_dirs(self) -> None:
        for attr in ("data_interim", "data_processed", "models", "logs", "mlruns"):
            getattr(self, attr).mkdir(parents=True, exist_ok=True)


class DataConfig(BaseModel):
    target_column: str = "SalePrice"
    test_size: float = Field(0.15, ge=0.05, le=0.40)
    val_size: float = Field(0.15, ge=0.05, le=0.40)
    random_state: int = 42
    missing_threshold: float = Field(0.30, ge=0.0, le=1.0)


class FeaturesConfig(BaseModel):
    numerical_impute_strategy: str = "median"
    categorical_impute_strategy: str = "most_frequent"
    skewness_threshold: float = 0.75
    log_transform_target: bool = True


class TrainingConfig(BaseModel):
    experiment_name: str = "ames-housing-experiment"
    run_name: str = "gradient-boosting-optuna"
    n_trials: int = 50
    cv_folds: int = 5
    scoring: str = "neg_root_mean_squared_error"
    random_state: int = 42


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "info"
    model_path: Path = ROOT_DIR / "models" / "best_model.pkl"
    scaler_path: Path = ROOT_DIR / "models" / "scaler.pkl"
    feature_names_path: Path = ROOT_DIR / "models" / "feature_names.json"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    file: str = "logs/ames_housing.log"
    rotation: str = "10 MB"
    retention: str = "7 days"


# ── Root settings ─────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AMES_", env_nested_delimiter="__")

    project: ProjectConfig = ProjectConfig()
    paths: PathsConfig = PathsConfig()
    data: DataConfig = DataConfig()
    features: FeaturesConfig = FeaturesConfig()
    training: TrainingConfig = TrainingConfig()
    api: ApiConfig = ApiConfig()
    logging: LoggingConfig = LoggingConfig()

    @classmethod
    def from_yaml(cls, path: Path = CONFIG_FILE) -> "Settings":
        """Load settings from YAML, then allow env-var overrides."""
        raw: dict[str, Any] = {}
        if path.exists():
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
        return cls(**raw)


# ── Singleton ─────────────────────────────────────────────────────────────────
settings = Settings.from_yaml()
