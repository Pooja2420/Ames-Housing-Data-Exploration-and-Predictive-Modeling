"""Build the full sklearn feature-engineering pipeline for Ames Housing.

The pipeline is designed to work on the **raw** cleaned DataFrame (after
loading + schema validation) and produces a fully numeric matrix ready for
model training.

Pipeline steps
--------------
1. HighMissingDropper   – remove columns with > 30 % nulls
2. AmesFeatureEngineer  – add 13 domain interaction features
3. RareLabelEncoder     – collapse rare categories (< 1 %) into 'Other'
4. Simple imputation    – median for numerics, constant for categoricals
5. SkewnessCorrector    – log1p on high-skew numerical columns
6. OneHotEncoder        – encode all remaining categoricals
7. StandardScaler       – unit-variance scaling (used by linear models/SVM)

Usage
-----
    from ames_housing.features.engineering import build_pipeline

    pipeline = build_pipeline()
    X_train_t = pipeline.fit_transform(X_train)
    X_test_t  = pipeline.transform(X_test)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ames_housing.config import settings
from ames_housing.features.transformers import (
    AmesFeatureEngineer,
    HighMissingDropper,
    RareLabelEncoder,
    SkewnessCorrector,
)


def build_pipeline(scale: bool = True) -> Pipeline:
    """Return the end-to-end feature engineering pipeline.

    Parameters
    ----------
    scale : bool
        Whether to include ``StandardScaler`` as the final step.
        Set ``False`` for tree-based models that don't need scaling.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline — call ``.fit_transform(X_train)`` to fit.
    """
    cfg = settings.features

    steps: list[tuple[str, object]] = [
        # Step 1 – drop high-missing columns (learns threshold on train)
        ("drop_missing", HighMissingDropper(threshold=settings.data.missing_threshold)),

        # Step 2 – inject 13 domain features (stateless, no fit needed)
        ("feature_engineer", AmesFeatureEngineer()),

        # Step 3 – collapse rare categories before encoding
        ("rare_label", RareLabelEncoder(threshold=0.01)),

        # Step 4 – log1p skewed numeric columns (must run here, while X is still a
        #           DataFrame; ColumnTransformer converts to ndarray internally)
        ("skew_correct", SkewnessCorrector(threshold=cfg.skewness_threshold)),
    ]

    if scale:
        steps.append(("scaler", _build_column_transformer(cfg)))
    else:
        steps.append(("column_transform", _build_column_transformer(cfg, scale=False)))

    pipeline = Pipeline(steps=steps)
    logger.debug("Feature pipeline built | scale={}", scale)
    return pipeline


def _build_column_transformer(cfg, scale: bool = True) -> ColumnTransformer:
    """Inner ColumnTransformer: impute + encode + optionally scale."""

    # ── Numeric sub-pipeline ───────────────────────────────────────────────────
    # Note: SkewnessCorrector is applied in the outer pipeline (before this
    # ColumnTransformer) where X is still a DataFrame with named columns.
    num_steps = [
        ("impute_num", SimpleImputer(strategy=cfg.numerical_impute_strategy)),
    ]
    if scale:
        num_steps.append(("scale", StandardScaler()))

    numeric_pipeline = Pipeline(steps=num_steps)

    # ── Categorical sub-pipeline ───────────────────────────────────────────────
    cat_pipeline = Pipeline(steps=[
        ("impute_cat", SimpleImputer(
            strategy=cfg.categorical_impute_strategy,
            fill_value="missing",
        )),
        ("ohe", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32,
        )),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, _numeric_selector),
            ("cat", cat_pipeline,     _categorical_selector),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ── Selector callables (deferred — receive the DataFrame at fit time) ──────────

def _numeric_selector(X: pd.DataFrame) -> list[str]:
    return X.select_dtypes(include=[np.number]).columns.tolist()


def _categorical_selector(X: pd.DataFrame) -> list[str]:
    return X.select_dtypes(include=["object", "category"]).columns.tolist()


# ── Convenience helpers ────────────────────────────────────────────────────────

def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Extract output feature names from a fitted pipeline.

    Works with scikit-learn ≥ 1.0's ``get_feature_names_out`` API.
    """
    try:
        ct: ColumnTransformer = pipeline.named_steps.get(
            "scaler", pipeline.named_steps.get("column_transform")
        )
        return ct.get_feature_names_out().tolist()
    except Exception as exc:
        logger.warning("Could not extract feature names: {}", exc)
        return []


def summarise_features(pipeline: Pipeline) -> None:
    """Log a human-readable summary of the fitted pipeline's output features."""
    names = get_feature_names(pipeline)
    n_num = sum(1 for n in names if not n.startswith("cat__"))
    n_cat = len(names) - n_num
    logger.info(
        "Feature summary | total={} | numeric={} | one-hot encoded={}",
        len(names),
        n_num,
        n_cat,
    )
