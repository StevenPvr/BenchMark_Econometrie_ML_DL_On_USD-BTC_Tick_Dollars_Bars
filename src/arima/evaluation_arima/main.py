"""CLI entry point for ARIMA evaluation.

Keeps `src.*` imports intact as requested.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Sequence

import pandas as pd # type: ignore[import-untyped]

# Add project root to Python path for direct execution.
# This must be done before importing src modules.
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


from src.arima.evaluation_arima.evaluation_arima import (  # type: ignore
    backtest_full_series,
    evaluate_model,
    ljung_box_on_residuals,
    plot_residuals_acf_with_ljungbox,
    run_all_normality_tests,
    save_evaluation_results,
    save_ljung_box_results,
)
from src.arima.training_arima.training_arima import load_trained_model  # type: ignore
from src.arima.evaluation_arima.save_data_for_garch import (  # type: ignore
    regenerate_garch_dataset_from_rolling_predictions,
    save_garch_dataset,
)
from src.arima.evaluation_arima.model_performance import (  # type: ignore
    plot_predictions_vs_actual,
)
from src.arima.evaluation_arima.utils import detect_value_column  # type: ignore

# No default constants imported - all parameters must be provided explicitly
from src.path import (  # type: ignore
    ARIMA_RESULTS_DIR,
    PREDICTIONS_VS_ACTUAL_ARIMA_PLOT,
    ROLLING_PREDICTIONS_ARIMA_FILE,
    WEIGHTED_LOG_RETURNS_SPLIT_FILE,
)
from src.utils import ensure_output_dir, get_logger, load_csv_file, save_json_pretty  # type: ignore

logger = get_logger(__name__)


def _load_split_df() -> pd.DataFrame:
    path = Path(WEIGHTED_LOG_RETURNS_SPLIT_FILE)
    df = load_csv_file(path)
    if "date" not in df.columns:
        raise ValueError("Split file must contain a 'date' column.")
    return df


def _get_train_test_split_date(df: pd.DataFrame) -> str:
    """Extract the train/test split date from the dataframe."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], format='mixed')
    df = df.sort_values("date")

    if "split" in df.columns:
        if set(df["split"].unique()) >= {"train", "test"}:
            # Find the transition from train to test
            split_changes = df["split"] != df["split"].shift(1)
            test_starts = df[(df["split"] == "test") & split_changes]
            if not test_starts.empty:
                split_date = test_starts.loc[:, "date"].iloc[0]
                return split_date.strftime("%Y-%m-%d")

    if "is_test" in df.columns:
        # Find where is_test becomes True
        test_starts = df[(df["is_test"]) & (df["is_test"] != df["is_test"].shift(1))]
        if not test_starts.empty:
            split_date = test_starts.loc[:, "date"].iloc[0]
            return split_date.strftime("%Y-%m-%d")

    raise ValueError("Could not determine train/test split date from dataframe.")


def _split_train_test(df: pd.DataFrame, value_col: str) -> tuple[pd.Series, pd.Series]:
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], format='mixed')
    df = df.sort_values("date")

    if "split" in df.columns:
        if set(df["split"].unique()) >= {"train", "test"}:
            train_df = df[df["split"] == "train"].set_index("date")
            test_df = df[df["split"] == "test"].set_index("date")
            train: pd.Series = train_df[value_col]  # type: ignore[assignment]
            test: pd.Series = test_df[value_col]  # type: ignore[assignment]
            return train, test

    if "is_test" in df.columns:
        mask = df["is_test"].astype(bool)
        train_df = df[~mask].set_index("date")
        test_df = df[mask].set_index("date")
        train_series: pd.Series = train_df[value_col]  # type: ignore[assignment]
        test_series: pd.Series = test_df[value_col]  # type: ignore[assignment]
        return train_series, test_series

    # No implicit fallback allowed: explicit split markers required
    raise ValueError(
        "No explicit split markers found in dataframe. "
        "Expected 'split' column with values {'train','test'} or a boolean 'is_test' column."
    )


def _load_trained_model_and_order() -> tuple[Any, tuple[int, int, int]]:
    """Load a trained ARIMA model and extract order.

    Raises explicit errors instead of silently falling back to defaults.
    """
    try:
        from src.arima.training_arima.training_arima import load_trained_model  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("Failed to import ARIMA training loader 'load_trained_model'.") from exc

    fitted_model, model_info = load_trained_model()

    order: tuple[int, int, int] | None = None

    # Prefer explicit metadata
    if isinstance(model_info, dict):
        maybe_order = model_info.get("order")
        if maybe_order is not None:
            order = tuple(maybe_order)  # type: ignore[arg-type]

    # Fallback extraction from fitted model attributes is not allowed; enforce explicit presence
    if order is None:
        raise ValueError(
            "Trained model metadata must include 'order'. "
            "Extraction from fitted model attributes is not allowed."
        )

    return fitted_model, order


def _run_residual_diagnostics(residuals: Sequence[float], lags: int) -> None:
    """Generate residual diagnostics (ACF plot + Ljung–Box results)."""
    try:
        plot_residuals_acf_with_ljungbox(residuals, lags=lags)
        lb_result = ljung_box_on_residuals(residuals, lags=lags)
        save_ljung_box_results(lb_result)
    except Exception as exc:  # pragma: no cover
        logger.warning("Diagnostics could not be generated: %s", exc)


def _save_normality_tests(residuals: Sequence[float]) -> None:
    """Run and persist residual normality tests."""
    logger.info("=" * 60)
    logger.info("RUNNING NORMALITY TESTS ON RESIDUALS")
    logger.info("=" * 60)
    try:
        results = run_all_normality_tests(residuals)
        logger.info(
            "Jarque-Bera: statistic=%.4f, p-value=%.4f",
            results["jarque_bera"]["statistic"],
            results["jarque_bera"]["p_value"],
        )
        logger.info(
            "Shapiro-Wilk: statistic=%.4f, p-value=%.4f",
            results["shapiro_wilk"]["statistic"],
            results["shapiro_wilk"]["p_value"],
        )
        logger.info(
            "Anderson-Darling: statistic=%.4f",
            results["anderson_darling"]["statistic"],
        )
        output = Path(ARIMA_RESULTS_DIR) / "evaluation" / "normality_tests.json"
        ensure_output_dir(output.parent)
        save_json_pretty(results, output)
        logger.info("Saved normality test results → %s", output)
    except Exception as exc:  # pragma: no cover
        logger.warning("Normality tests could not be generated: %s", exc)


def _build_garch_outputs(
    results: dict[str, Any],
    fitted_model: Any | None,
    backtest_residuals: pd.DataFrame | None,
) -> None:
    """Persist datasets required by the GARCH pipeline."""
    logger.info("=" * 60)
    logger.info("BUILDING GARCH DATASET")
    logger.info("=" * 60)
    try:
        save_garch_dataset(
            results,
            fitted_model=fitted_model,
            backtest_residuals=backtest_residuals,
        )
        return
    except Exception as exc:  # pragma: no cover
        logger.warning("GARCH dataset generation failed: %s", exc)
    try:
        logger.info("Attempting to regenerate GARCH dataset from rolling_predictions.csv…")
        regenerate_garch_dataset_from_rolling_predictions()
    except Exception as exc:  # pragma: no cover
        logger.warning("GARCH dataset regeneration also failed: %s", exc)


def _setup_evaluation() -> (
    tuple[pd.Series, pd.Series, tuple[int, int, int], dict[str, Any], str, int]
):
    """Setup evaluation by loading data and model parameters."""
    df = _load_split_df()
    value_col = detect_value_column(df)
    train, test = _split_train_test(df, value_col=value_col)

    fitted_model, order = _load_trained_model_and_order()

    # Load model_info to get trend and refit_every parameters
    _, model_info = load_trained_model()
    if not isinstance(model_info, dict) or "params" not in model_info:
        raise ValueError("Model info must contain params with trend and refit_every")

    params = model_info["params"]
    trend = params.get("trend")
    refit_every = params.get("refit_every")

    if trend is None:
        raise ValueError("Model parameters must include 'trend'")
    if refit_every is None:
        raise ValueError("Model parameters must include 'refit_every'")

    return train, test, order, model_info, trend, refit_every


def _run_evaluation(
    train: pd.Series,
    test: pd.Series,
    order: tuple[int, int, int],
    refit_every: int,
    model_info: dict[str, Any],
    trend: str,
) -> dict[str, Any]:
    """Run the ARIMA model evaluation."""
    results = evaluate_model(
        train_series=train,
        test_series=test,
        order=order,
        refit_every=refit_every,
        verbose=True,
        model_info=model_info,
        trend=trend,
    )
    save_evaluation_results(results)
    return results


def _generate_outputs(results: dict[str, Any]) -> None:
    """Generate evaluation outputs (plots and diagnostics)."""
    # Generate predictions vs actual plot
    try:
        split_df = _load_split_df()
        split_date = _get_train_test_split_date(split_df)
        plot_predictions_vs_actual(
            predictions_file=str(ROLLING_PREDICTIONS_ARIMA_FILE),
            output_file=str(PREDICTIONS_VS_ACTUAL_ARIMA_PLOT),
            train_test_split_date=split_date,
        )
        logger.info("Generated ARIMA predictions vs actual plot")
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not generate predictions vs actual plot: %s", exc)

    residuals = results.get("residuals")
    if residuals is None:
        raise KeyError("Evaluation results must include 'residuals'.")

    # Use standard Ljung-Box lags for residual diagnostics
    ljungbox_lags = 20
    _run_residual_diagnostics(residuals, ljungbox_lags)
    _save_normality_tests(residuals)


def _run_backtesting(
    train: pd.Series,
    test: pd.Series,
    order: tuple[int, int, int],
    refit_every_eval: int,
    trend: str,
    results: dict[str, Any],
) -> pd.DataFrame:
    """Run backtesting for GARCH data generation."""
    logger.info(f"Generating train-only residuals for GARCH with refit_every={refit_every_eval}")

    try:
        logger.info("Running full series backtest for GARCH artifacts (errors propagate to CLI).")
        backtest_history = backtest_full_series(
            train_series=train,
            test_series=test,
            order=order,
            refit_every=refit_every_eval,
            verbose=True,
            trend=trend,
            include_test=False,
        )
    except RuntimeError as exc:
        logger.error("Full series backtest failed: %s", exc)
        raise

    return backtest_history


def main() -> None:
    logger.info("=" * 60)
    logger.info("ARIMA evaluation starting…")

    train, test, order, model_info, trend, refit_every = _setup_evaluation()
    results = _run_evaluation(train, test, order, refit_every, model_info, trend)
    _generate_outputs(results)

    # Generate walk-forward residuals on TRAIN only to avoid double-predicting TEST
    # CRITICAL: backtest_history contains unbiased walk-forward residuals for GARCH
    refit_every_eval = results.get("refit_every")
    if refit_every_eval is None:
        raise ValueError("Evaluation results must include 'refit_every' parameter")

    backtest_history = _run_backtesting(train, test, order, refit_every_eval, trend, results)

    # NOTE: full_fitted_model has look-ahead bias (diagnostics only)
    # save_garch_dataset will use backtest_history instead to avoid bias
    full_fitted_model = results.get("_fitted_model")
    _build_garch_outputs(results, full_fitted_model, backtest_history)

    logger.info("=" * 60)
    logger.info("ARIMA evaluation complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
