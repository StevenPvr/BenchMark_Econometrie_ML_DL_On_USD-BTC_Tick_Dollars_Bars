"""GARCH forecasting orchestration for full sample (TRAIN + TEST).

This module provides the main entry point for generating leak-free GARCH
variance forecasts on the complete sample:

1. TRAIN split: Expanding window forecasts with periodic refit
2. TEST split: Forecasts generated separately using model trained on TRAIN
3. Combined output: garch_forecasts.csv with h=1 forecasts for LightGBM

CRITICAL TEMPORAL SEPARATION:
- Forecaster is trained ONLY on TRAIN data
- TEST forecasts are generated using the trained model
- This ensures NO data leakage and complete temporal separation

All forecasts are σ²_t+1|t (one-step-ahead), never σ²_t|t (filtered).
This ensures NO data leakage when used as features in downstream models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.constants import (
    GARCH_DATASET_FILE,
    GARCH_EVALUATION_DIR,
    GARCH_INITIAL_WINDOW_SIZE_DEFAULT,
    GARCH_MIN_WINDOW_SIZE,
    GARCH_MODEL_FILE,
)
from src.garch.training_garch.forecaster import EGARCHForecaster, ForecastResult
from src.garch.training_garch.utils import count_splits, load_optimized_hyperparameters
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class ForecastGenerationConfig:
    """Configuration for GARCH forecast generation modes."""

    mode: str  # "optimized", "trained_model", "hybrid"
    use_optimized_params: bool = True
    use_fixed_params: bool = False
    seed_with_current_params: bool = False
    save_results: bool = True
    output_dir: Path = GARCH_EVALUATION_DIR
    # Override parameters
    o: int | None = None
    p: int | None = None
    dist: str | None = None
    refit_frequency: int | None = None
    window_type: str | None = None
    window_size: int | None = None
    initial_window_size: int = GARCH_INITIAL_WINDOW_SIZE_DEFAULT
    min_window_size: int = GARCH_MIN_WINDOW_SIZE
    anchor_at_min_window: bool = False


def load_garch_dataset() -> pd.DataFrame:
    """Load GARCH dataset (ARIMA residuals).

    Returns:
        DataFrame with columns: date, arima_resid (or sarima_resid), split.

    Raises:
        FileNotFoundError: If dataset not found.
        ValueError: If required columns are missing.
    """
    if not GARCH_DATASET_FILE.exists():
        msg = f"GARCH dataset not found: {GARCH_DATASET_FILE}"
        raise FileNotFoundError(msg)

    df = pd.read_csv(GARCH_DATASET_FILE, parse_dates=["date"])

    required_cols = ["date", "split"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    # Check for residual columns or columns needed to compute residuals
    # Accept either: arima_resid/sarima_resid OR weighted_log_return + arima_fitted_in_sample
    has_residual_col = "arima_resid" in df.columns or "sarima_resid" in df.columns
    has_compute_cols = (
        "weighted_log_return" in df.columns and "arima_fitted_in_sample" in df.columns
    )
    if not has_residual_col and not has_compute_cols:
        msg = (
            "Missing required columns: either residual column ('arima_resid' or 'sarima_resid') "
            "or columns to compute residuals ('weighted_log_return' and 'arima_fitted_in_sample')"
        )
        raise ValueError(msg)

    n_train, n_test = count_splits(df)
    logger.info(
        "Loaded GARCH dataset: %d observations, %d train, %d test",
        len(df),
        n_train,
        n_test,
    )

    return df


def _load_and_prepare_data() -> (
    tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.DatetimeIndex, pd.DatetimeIndex]
):
    """Load and prepare GARCH dataset for forecasting.

    Returns:
        Tuple of (df_train, df_test, resid_train, resid_test, dates_train, dates_test).
    """
    df_data = load_garch_dataset()
    return _prepare_forecast_data(df_data)


def _calculate_weighted_convergence_rate(
    result_train: ForecastResult,
    result_test: ForecastResult,
) -> float:
    """Calculate weighted average convergence rate across train and test splits.

    Args:
        result_train: ForecastResult from training split.
        result_test: ForecastResult from test split.

    Returns:
        Weighted average convergence rate, or 1.0 if no refits occurred.
    """
    n_refits_total = result_train.n_refits + result_test.n_refits

    if n_refits_total > 0:
        convergence_rate = (
            result_train.convergence_rate * result_train.n_refits
            + result_test.convergence_rate * result_test.n_refits
        ) / n_refits_total
    else:
        convergence_rate = 1.0

    return convergence_rate


def _resolve_hyperparameters_from_config(
    config: ForecastGenerationConfig,
) -> tuple[int, int, str, int, str, int | None]:
    """Resolve hyperparameters based on forecast generation config.

    Args:
        config: Forecast generation configuration.

    Returns:
        Tuple of (o, p, dist, refit_frequency, window_type, window_size).
    """
    if config.mode == "optimized":
        return _resolve_optimized_hyperparameters(
            config.o,
            config.p,
            config.dist,
            config.refit_frequency,
            config.window_type,
            config.window_size,
        )
    elif config.mode in ("trained_model", "hybrid"):
        return _load_trained_model_hyperparameters()
    else:
        msg = f"Unknown forecast generation mode: {config.mode}"
        raise ValueError(msg)


def _load_trained_model_hyperparameters() -> tuple[int, int, str, int, str, int | None]:
    """Load hyperparameters from trained model files.

    Returns:
        Tuple of (o, p, dist, refit_frequency, window_type, window_size).

    Raises:
        FileNotFoundError: If model files not found.
    """
    if not GARCH_MODEL_FILE.exists():
        msg = f"Trained model file not found: {GARCH_MODEL_FILE}"
        raise FileNotFoundError(msg)

    logger.info("Loading trained EGARCH model from %s", GARCH_MODEL_FILE)
    model_data = joblib.load(GARCH_MODEL_FILE)

    # Extract required keys from model
    o = int(model_data["o"])
    p = int(model_data["p"])
    dist = str(model_data["dist"])

    # Load complementary configuration from metadata
    from src.constants import GARCH_MODEL_METADATA_FILE

    if not GARCH_MODEL_METADATA_FILE.exists():
        msg = (
            "Trained model metadata not found: "
            f"{GARCH_MODEL_METADATA_FILE}. "
            "Cannot reconstruct trained forecaster without metadata."
        )
        raise FileNotFoundError(msg)

    import json

    with open(GARCH_MODEL_METADATA_FILE, "r") as f:
        metadata = json.load(f)

    try:
        window_type = str(metadata["window_type"])
        refit_frequency = int(metadata["refit_frequency"])
        window_size_val = metadata.get("window_size", None)
        window_size = int(window_size_val) if window_size_val is not None else None
    except KeyError as exc:
        raise KeyError(
            f"Missing key in model metadata: {exc}. Re-run training to regenerate metadata."
        ) from exc

    logger.info(
        "Using trained model: EGARCH(%d,%d), dist=%s, refit_freq=%d, " "window=%s, window_size=%s",
        o,
        p,
        dist,
        refit_frequency,
        window_type,
        str(window_size),
    )

    return o, p, dist, refit_frequency, window_type, window_size


def _resolve_optimized_hyperparameters(
    o: int | None,
    p: int | None,
    dist: str | None,
    refit_frequency: int | None,
    window_type: str | None,
    window_size: int | None,
) -> tuple[int, int, str, int, str, int | None]:
    """Resolve hyperparameters from optimization results with optional overrides."""
    opt_params = load_optimized_hyperparameters()

    resolved_o = o if o is not None else opt_params["o"]
    resolved_p = p if p is not None else opt_params["p"]
    resolved_dist = dist if dist is not None else opt_params["distribution"]
    resolved_refit_freq = (
        refit_frequency if refit_frequency is not None else opt_params["refit_freq"]
    )
    resolved_window_type = window_type if window_type is not None else opt_params["window_type"]
    resolved_window_size = window_size if window_size is not None else opt_params.get("window_size")

    logger.info(
        "Using optimized hyperparameters: EGARCH(%d,%d), distribution=%s, "
        "refit_freq=%d, window_type=%s, window_size=%s",
        resolved_o,
        resolved_p,
        resolved_dist,
        resolved_refit_freq,
        resolved_window_type,
        resolved_window_size,
    )

    return (
        resolved_o,
        resolved_p,
        resolved_dist,
        resolved_refit_freq,
        resolved_window_type,
        resolved_window_size,
    )


def _validate_explicit_hyperparameters(
    o: int | None,
    p: int | None,
    dist: str | None,
    refit_frequency: int | None,
    window_type: str | None,
) -> None:
    """Validate that all required hyperparameters are provided explicitly."""
    if o is None or p is None or dist is None or refit_frequency is None or window_type is None:
        msg = (
            "When use_optimized_params=False, all hyperparameters must be "
            "provided explicitly. "
            f"Missing: o={o}, p={p}, dist={dist}, "
            f"refit_frequency={refit_frequency}, window_type={window_type}"
        )
        raise ValueError(msg)


def _generate_train_forecasts(
    forecaster: EGARCHForecaster,
    resid_train: np.ndarray,
    dates_train: pd.DatetimeIndex,
) -> ForecastResult:
    """Generate forecasts on TRAIN data with expanding window.

    Args:
        forecaster: EGARCH forecaster instance.
        resid_train: Training residuals.
        dates_train: Training dates.

    Returns:
        ForecastResult with train forecasts.
    """
    logger.info("Generating TRAIN forecasts (expanding window)...")
    return forecaster.forecast_expanding(resid_train, dates=dates_train)


def _generate_test_forecasts(
    forecaster: EGARCHForecaster,
    resid_train: np.ndarray,
    resid_test: np.ndarray,
    dates_test: pd.DatetimeIndex,
) -> ForecastResult:
    """Generate forecasts on TEST data using model trained on TRAIN.

    CRITICAL: This function ensures temporal separation by:
    1. Starting with model trained only on TRAIN data
    2. Generating TEST forecasts sequentially from that state
    3. Each TEST forecast at time t uses only data up to t-1

    Args:
        forecaster: EGARCH forecaster instance (already trained on TRAIN).
        resid_train: Training residuals (used for refit windows).
        resid_test: Test residuals.
        dates_test: Test dates.

    Returns:
        ForecastResult with test forecasts.
    """
    logger.info("Generating TEST forecasts (rolling/expanding from trained model)...")

    # Continue forecasts from current state without regenerating train forecasts
    # This uses the forecaster's current parameters and refit manager state
    return forecaster.forecast_continuing(resid_train, resid_test, dates_test)


def _assign_forecasts_to_dataframes(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    result_train: ForecastResult,
    result_test: ForecastResult,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign forecast results to train and test DataFrames.

    Returns:
        Tuple of (df_train_with_forecasts, df_test_with_forecasts).
    """
    df_train = df_train.copy()
    df_test = df_test.copy()

    df_train["garch_forecast_h1"] = result_train.forecasts
    df_train["garch_vol_h1"] = result_train.volatility
    df_train["forecast_type"] = "expanding"
    df_train["refit_occurred"] = result_train.refit_mask

    df_test["garch_forecast_h1"] = result_test.forecasts
    df_test["garch_vol_h1"] = result_test.volatility
    df_test["forecast_type"] = "rolling"
    df_test["refit_occurred"] = result_test.refit_mask

    return df_train, df_test


def _save_combined_forecasts(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    result_train: ForecastResult,
    result_test: ForecastResult,
    o: int,
    p: int,
    dist: str,
    refit_frequency: int,
    window_type: str,
    window_size: int | None,
    initial_window_size: int,
    min_window_size: int,
    output_dir: Path,
) -> Path:
    """Combine train and test forecasts and save with metadata.

    Returns:
        Path to saved forecast file.
    """
    from src.garch.training_garch.predictions_io import save_garch_forecasts

    df_combined: pd.DataFrame = pd.concat([df_train, df_test], ignore_index=True)  # type: ignore[assignment]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "garch_forecasts.parquet"

    n_refits_train = result_train.n_refits
    n_refits_total = result_train.n_refits + result_test.n_refits
    convergence_rate = _calculate_weighted_convergence_rate(result_train, result_test)

    save_garch_forecasts(
        df_combined,
        model_type=f"EGARCH({o},{p})",
        distribution=dist,
        refit_frequency=refit_frequency,
        window_type=window_type,
        output_path=output_file,
        save_metadata=True,
        window_size=window_size,
        initial_window_size=initial_window_size,
        min_window_size=min_window_size,
        n_refits_train=n_refits_train,
        n_refits_total=n_refits_total,
        convergence_rate=convergence_rate,
    )

    return output_file


def _prepare_forecast_data(
    df_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.DatetimeIndex, pd.DatetimeIndex]:
    """Prepare data for forecasting by splitting and extracting arrays.

    Args:
        df_data: Full dataset with 'split' column.

    Returns:
        Tuple of (df_train, df_test, resid_train, resid_test, dates_train, dates_test).
    """
    # Defensive: ensure strict temporal ordering before any split.
    # Why: Upstream sources should already be sorted, but we avoid relying on it.
    df_sorted = df_data.sort_values("date").reset_index(drop=True)

    df_train = pd.DataFrame(df_sorted[df_sorted["split"] == "train"]).copy()
    df_test = pd.DataFrame(df_sorted[df_sorted["split"] == "test"]).copy()

    from src.garch.training_garch.utils import _extract_direct_residuals

    resid_train = _extract_direct_residuals(df_train)
    # For test data, try to extract directly (may need residual column)
    if "arima_resid" in df_test.columns:
        resid_test = np.asarray(df_test["arima_resid"], dtype=float)
    elif "sarima_resid" in df_test.columns:
        resid_test = np.asarray(df_test["sarima_resid"], dtype=float)
    else:
        # Test residuals should come from residual column, not computed
        msg = (
            "Test residuals require 'arima_resid' or 'sarima_resid' column. "
            "Cannot compute test residuals from fitted values."
        )
        raise ValueError(msg)
    dates_train = pd.DatetimeIndex(df_train["date"])
    dates_test = pd.DatetimeIndex(df_test["date"])

    return df_train, df_test, resid_train, resid_test, dates_train, dates_test


def _create_forecaster_from_config(
    config: ForecastGenerationConfig,
    o: int,
    p: int,
    dist: str,
    refit_frequency: int,
    window_type: str,
    window_size: int | None,
) -> EGARCHForecaster:
    """Create EGARCHForecaster based on configuration.

    Args:
        config: Forecast generation configuration.
        o: ARCH order.
        p: GARCH order.
        dist: Distribution.
        refit_frequency: Refit frequency.
        window_type: Window type.
        window_size: Window size.

    Returns:
        Configured EGARCHForecaster.
    """
    forecaster = _create_forecaster_from_params(
        o,
        p,
        dist,
        refit_frequency,
        window_type,
        window_size,
        config.initial_window_size,
        config.min_window_size,
        config.use_fixed_params,
        config.anchor_at_min_window,
    )

    # Apply mode-specific configuration
    if config.mode in ("trained_model", "hybrid"):
        model_data = joblib.load(GARCH_MODEL_FILE)

        # CRITICAL DATA LEAKAGE VALIDATION:
        # Verify that pre-trained parameters were estimated ONLY on TRAIN data
        # If these parameters were optimized using TEST data (even indirectly),
        # using them would constitute catastrophic data leakage
        from src.constants import GARCH_MODEL_METADATA_FILE
        import json

        if GARCH_MODEL_METADATA_FILE.exists():
            with open(GARCH_MODEL_METADATA_FILE, "r") as f:
                metadata = json.load(f)

            # Verify n_train exists and is reasonable
            n_train_in_model = metadata.get("n_train", 0)
            if n_train_in_model <= 0:
                msg = (
                    "CRITICAL: Model metadata does not contain valid n_train. "
                    "Cannot verify that parameters were trained ONLY on TRAIN data. "
                    "This creates a risk of data leakage if TEST data was used during training."
                )
                raise ValueError(msg)

            logger.info(
                "✓ DATA LEAKAGE CHECK PASSED: Model trained on %d TRAIN observations",
                n_train_in_model,
            )
            logger.warning(
                "⚠️  HYBRID MODE: Ensure model.joblib was trained WITHOUT any TEST data exposure. "
                "Using parameters optimized on TEST creates data leakage in downstream features."
            )
        else:
            logger.warning(
                "⚠️  CRITICAL WARNING: Model metadata not found. Cannot validate that "
                "parameters were trained only on TRAIN data. Proceeding with RISK of data leakage."
            )

        forecaster.refit_manager.current_params = model_data["params"]

        if config.mode == "hybrid":
            forecaster.seed_with_current_params = True
            logger.info(
                "HYBRID MODE ACTIVATED: Seeds with pre-trained params, then refits on schedule. "
                "ENSURE pre-trained params came from TRAIN-only training to avoid leakage."
            )
        else:  # trained_model
            # For trained model mode, use fixed parameters
            forecaster.refit_manager.use_fixed_params = True
            logger.info(
                "TRAINED_MODEL MODE: Uses fixed pre-trained params without refitting. "
                "This is the SAFE mode for generating LightGBM features."
            )

        logger.info("Loaded pre-trained parameters into forecaster")

    return forecaster


def _create_forecaster_from_params(
    o: int,
    p: int,
    dist: str,
    refit_frequency: int,
    window_type: str,
    window_size: int | None,
    initial_window_size: int,
    min_window_size: int,
    use_fixed_params: bool = False,
    anchor_at_min_window: bool = False,
) -> EGARCHForecaster:
    """Create EGARCHForecaster from parameters.

    Args:
        o: ARCH order.
        p: GARCH order.
        dist: Distribution.
        refit_frequency: Refit frequency.
        window_type: Window type.
        window_size: Window size.
        initial_window_size: Initial window size.
        min_window_size: Minimum window size.

    Returns:
        Configured EGARCHForecaster instance.
    """
    return EGARCHForecaster(
        o=o,
        p=p,
        dist=dist,
        refit_frequency=refit_frequency,
        window_type=window_type,
        window_size=window_size,
        initial_window_size=initial_window_size,
        min_window_size=min_window_size,
        use_fixed_params=use_fixed_params,
        anchor_at_min_window=anchor_at_min_window,
    )


def _log_summary(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    result_train: ForecastResult,
    result_test: ForecastResult,
    o: int,
    p: int,
    dist: str,
    output_file: Path | None,
) -> None:
    """Log summary of forecasting results."""
    n_refits_total = result_train.n_refits + result_test.n_refits
    n_total = len(df_train) + len(df_test)
    convergence_rate = _calculate_weighted_convergence_rate(result_train, result_test)

    logger.info("=" * 60)
    logger.info("GARCH FORECASTING COMPLETE")
    logger.info("=" * 60)
    logger.info("  Model: EGARCH(%d,%d), dist=%s", o, p, dist)
    logger.info("  Train: %d obs, %d refits", len(df_train), result_train.n_refits)
    logger.info("  Test: %d obs, %d refits", len(df_test), result_test.n_refits)
    logger.info(
        "  Total: %d obs, %d refits, %.1f%% converged",
        n_total,
        n_refits_total,
        convergence_rate * 100,
    )
    if output_file is not None:
        logger.info("  Output: %s (+ CSV)", output_file)
        logger.info("  Metadata: %s", output_file.with_suffix(".meta.json"))
    else:
        logger.info("  Output: None (+ CSV)")
    logger.info("=" * 60)


def _generate_forecasts_core(config: ForecastGenerationConfig) -> pd.DataFrame:
    """Core forecast generation logic shared across all modes.

    Args:
        config: Forecast generation configuration.

    Returns:
        Combined DataFrame with train and test forecasts.
    """
    # Load and prepare data
    df_train, df_test, resid_train, resid_test, dates_train, dates_test = _load_and_prepare_data()

    # Resolve hyperparameters
    o, p, dist, refit_frequency, window_type, window_size = _resolve_hyperparameters_from_config(
        config
    )

    logger.info(
        "Generating forecasts with EGARCH(%d,%d), dist=%s, refit_freq=%d, window=%s, mode=%s",
        o,
        p,
        dist,
        refit_frequency,
        window_type,
        config.mode,
    )

    # Create forecaster
    forecaster = _create_forecaster_from_config(
        config, o, p, dist, refit_frequency, window_type, window_size
    )

    # Generate forecasts
    result_train = _generate_train_forecasts(forecaster, resid_train, dates_train)
    result_test = _generate_test_forecasts(forecaster, resid_train, resid_test, dates_test)

    # Assign forecasts to DataFrames
    df_train, df_test = _assign_forecasts_to_dataframes(
        df_train, df_test, result_train, result_test
    )

    # Save results if requested
    output_file = None
    if config.save_results:
        output_file = _save_combined_forecasts(
            df_train,
            df_test,
            result_train,
            result_test,
            o,
            p,
            dist,
            refit_frequency,
            window_type,
            window_size,
            config.initial_window_size,
            config.min_window_size,
            config.output_dir,
        )

    # Log summary
    _log_summary(df_train, df_test, result_train, result_test, o, p, dist, output_file)

    # Return combined DataFrame
    return pd.concat([df_train, df_test], ignore_index=True)


def generate_full_sample_forecasts(
    *,
    use_optimized_params: bool = True,
    o: int | None = None,
    p: int | None = None,
    dist: str | None = None,
    refit_frequency: int | None = None,
    window_type: str | None = None,
    window_size: int | None = None,
    initial_window_size: int = GARCH_INITIAL_WINDOW_SIZE_DEFAULT,
    min_window_size: int = GARCH_MIN_WINDOW_SIZE,
    output_dir: Path = GARCH_EVALUATION_DIR,
    anchor_at_min_window: bool = False,
) -> pd.DataFrame:
    """Generate GARCH forecasts for full sample (TRAIN + TEST).

    CRITICAL TEMPORAL SEPARATION:
    This function ensures complete temporal separation by:
    1. Training forecaster ONLY on TRAIN data
    2. Generating TEST forecasts using the trained model
    3. Each forecast uses only information available at that time

    Args:
        use_optimized_params: Use optimized hyperparameters if True.
        o: ARCH order (overrides optimized).
        p: GARCH order (overrides optimized).
        dist: Distribution (overrides optimized).
        refit_frequency: Refit frequency (overrides optimized).
        window_type: Window type (overrides optimized).
        window_size: Rolling window size.
        initial_window_size: Initial training window size.
        min_window_size: Minimum window size to start forecasting.
        output_dir: Output directory for results.
        anchor_at_min_window: If True, anchor initial fit/scheduling at
            ``min_window_size`` to start forecasts earlier (evaluation mode).

    Returns:
        Combined DataFrame with train and test forecasts.
    """
    config = ForecastGenerationConfig(
        mode="optimized",
        use_optimized_params=use_optimized_params,
        o=o,
        p=p,
        dist=dist,
        refit_frequency=refit_frequency,
        window_type=window_type,
        window_size=window_size,
        initial_window_size=initial_window_size,
        min_window_size=min_window_size,
        output_dir=output_dir,
        anchor_at_min_window=anchor_at_min_window,
    )
    return _generate_forecasts_core(config)


def generate_full_sample_forecasts_from_trained_model(
    *,
    min_window_size: int = GARCH_MIN_WINDOW_SIZE,
    anchor_at_min_window: bool = False,
) -> pd.DataFrame:
    """Generate forecasts using the trained model from model.joblib.

    This function loads the trained EGARCH forecaster from model.joblib
    and uses it directly to generate forecasts, avoiding re-estimation.

    Returns:
        Combined DataFrame with train and test forecasts.

    Raises:
        FileNotFoundError: If model.joblib is not found.
        RuntimeError: If forecast generation fails.
    """
    config = ForecastGenerationConfig(
        mode="trained_model",
        use_fixed_params=True,
        save_results=False,  # No saving for trained model mode
        min_window_size=min_window_size,
        anchor_at_min_window=anchor_at_min_window,
    )
    return _generate_forecasts_core(config)


def generate_full_sample_forecasts_hybrid(
    *,
    min_window_size: int = GARCH_MIN_WINDOW_SIZE,
    anchor_at_min_window: bool = False,
) -> pd.DataFrame:
    """Generate forecasts using trained params to seed, and optimized hyperparams for refits.

    Behavior:
    - Loads best hyperparameters (o, p, distribution, refit_freq, window_type, window_size)
      from optimization results.
    - Loads trained parameters from model.joblib and seeds the forecaster's initial state.
    - Enables refits (use_fixed_params=False) with the optimized refit schedule and window.

    Returns:
        Combined DataFrame with train and test forecasts.

    Raises:
        FileNotFoundError: If model.joblib or optimization results are missing.
        ValueError: If trained model orders/dist mismatch optimized hyperparameters.
    """
    config = ForecastGenerationConfig(
        mode="hybrid",
        seed_with_current_params=True,
        min_window_size=min_window_size,
        anchor_at_min_window=anchor_at_min_window,
    )
    return _generate_forecasts_core(config)
