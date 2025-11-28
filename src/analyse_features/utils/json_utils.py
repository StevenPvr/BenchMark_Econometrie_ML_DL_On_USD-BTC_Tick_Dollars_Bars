"""JSON serialization utilities for feature analysis results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src.config_logging import get_logger

logger = get_logger(__name__)


def convert_to_serializable(obj: Any) -> Any:
    """Convert numpy/pandas types to JSON-serializable Python types.

    Args:
        obj: Object to convert.

    Returns:
        JSON-serializable object.
    """
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {convert_to_serializable(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(data: dict[str, Any], filepath: Path) -> None:
    """Save dictionary to JSON file.

    Args:
        data: Dictionary to save.
        filepath: Path to JSON file.
    """
    serializable_data = convert_to_serializable(data)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    logger.info("Saved JSON: %s", filepath)

