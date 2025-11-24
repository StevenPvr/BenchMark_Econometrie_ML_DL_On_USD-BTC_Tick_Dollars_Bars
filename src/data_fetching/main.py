"""CLI entry point for data_fetching module."""

from __future__ import annotations

from pathlib import Path
import sys

# Add project root to Python path for direct execution
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.data_fetching.data_fetching import (
    download_sp500_data,
    fetch_sp500_index_and_compute_log_returns,
    fetch_sp500_tickers,
)
from src.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Main CLI function to fetch S&P 500 data."""
    logger.info("=" * 80)
    logger.info("PIPELINE DE PRÉVISION S&P 500 - RENDEMENTS PONDÉRÉS PAR LIQUIDITÉ")
    logger.info("=" * 80)
    logger.info("ÉTAPE 1: RÉCUPÉRATION DES TICKERS S&P 500")
    fetch_sp500_tickers()

    logger.info("ÉTAPE 2: TÉLÉCHARGEMENT DES DONNÉES HISTORIQUES DES TICKERS")
    download_sp500_data()

    logger.info("ÉTAPE 3: RÉCUPÉRATION DE L'INDICE S&P 500 ET CALCUL DES LOG RETURNS")
    fetch_sp500_index_and_compute_log_returns()


if __name__ == "__main__":
    main()
