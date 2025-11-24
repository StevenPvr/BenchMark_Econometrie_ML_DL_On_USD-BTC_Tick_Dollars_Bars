"""CLI entry point for stationnarity_check module."""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure project root on path for direct execution
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.arima.stationnarity_check.stationnarity_check import (
    run_stationarity_pipeline,
    save_stationarity_report,
)
from src.constants import (
    STATIONARITY_DEFAULT_ALPHA,
    STATIONARITY_REPORT_FILE,
    WEIGHTED_LOG_RETURNS_FILE,
)
from src.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Run stationarity checks on the project's weighted returns and save a JSON report.

    Activates all available tests:
    - ADF test (Augmented Dickey-Fuller)
    - KPSS test (Kwiatkowski-Phillips-Schmidt-Shin)
    - Zivot-Andrews test (structural break detection)
    """
    try:
        report = run_stationarity_pipeline(
            data_file=str(WEIGHTED_LOG_RETURNS_FILE),
            column="close",
            alpha=STATIONARITY_DEFAULT_ALPHA,
            test_structural_break=True,  # Enable all tests including Zivot-Andrews
        )
        save_stationarity_report(report, STATIONARITY_REPORT_FILE)
    except Exception as exc:  # surface early during CLI use
        logger.error("Stationarity check failed: %s", exc, exc_info=True)
        raise


if __name__ == "__main__":
    main()
