"""CLI entry point for training_arima module."""

from __future__ import annotations

from pathlib import Path
import sys

# Add project root to Python path for direct execution
# This must be done before importing src modules
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.arima.training_arima.training_arima import save_trained_model, train_best_model
from src.data_preparation.data_preparation import load_train_test_data
from src.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Main CLI function for ARIMA training."""
    logger.info("=" * 60)
    logger.info("ARIMA TRAINING MODULE")
    logger.info("=" * 60)

    # Load train/test split and extract train series
    logger.info("Loading split data (train/test)…")
    train_series, _ = load_train_test_data()

    # Train fixed ARIMA(0,0,0) model (white noise for residual extraction)
    # NOTE: ARIMA(0,0,0) has NO forecasting capability - it only generates residuals for GARCH
    logger.info("Training fixed ARIMA(0,0,0) model (residual extraction for GARCH)…")
    fitted_model_aic, model_info_aic = train_best_model(train_series)
    save_trained_model(fitted_model_aic, model_info_aic)

    logger.info(f"Trained model: {model_info_aic['params']}")
    logger.info(f"Model AIC: {fitted_model_aic.aic:.2f}")

    logger.info("=" * 60)
    logger.info("ARIMA training complete!")
    logger.info("Next step: Run evaluation_arima to evaluate the trained model")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
