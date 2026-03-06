"""General-purpose utility functions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.debug("Saved JSON → {}", path)


def load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def dataframe_hash(df: pd.DataFrame) -> str:
    """Deterministic hash of a DataFrame for cache invalidation."""
    return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()  # noqa: S324


def memory_usage_mb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / 1_024 / 1_024


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numerics to reduce DataFrame memory footprint."""
    for col in df.select_dtypes(include=[np.number]).columns:
        col_min, col_max = df[col].min(), df[col].max()
        if df[col].dtype in (np.float64, float):
            if col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
        elif df[col].dtype in (np.int64, int):
            if col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
    logger.debug("Memory after reduction: {:.2f} MB", memory_usage_mb(df))
    return df
