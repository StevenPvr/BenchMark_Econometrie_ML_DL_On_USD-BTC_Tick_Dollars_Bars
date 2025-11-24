"""CLI for running GARCH numerical tests (pre-EGARCH, post-ARIMA).

Runs all numerical tests on ARIMA residuals:
- Ljung-Box test on residuals
- Ljung-Box test on squared residuals
- Engle ARCH-LM test
- McLeod-Li test

Saves results to JSON file.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure project root before any src imports
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import json

from src.constants import (
    GARCH_DATASET_FILE,
    GARCH_LJUNG_BOX_LAGS_DEFAULT,
    GARCH_LM_LAGS_DEFAULT,
    GARCH_NUMERICAL_TESTS_FILE,
)
from src.garch.garch_numerical_test.garch_numerical import run_all_tests
from src.garch.garch_numerical_test.utils import log_test_summary, prepare_output
from src.garch.structure_garch.utils import load_garch_dataset, prepare_residuals
from src.utils import ensure_output_dir, get_logger

logger = get_logger(__name__)


def main() -> None:
    """Run all numerical tests on ARIMA residuals and save results."""
    logger.info("=" * 60)
    logger.info("GARCH NUMERICAL TESTS (Pre-EGARCH, Post-ARIMA)")
    logger.info("=" * 60)

    # Load dataset and extract residuals
    df = load_garch_dataset(str(GARCH_DATASET_FILE))
    resid_test = prepare_residuals(df, use_test_only=True)

    logger.info("Running numerical tests on %d test residuals", resid_test.size)

    # Run all tests
    results = run_all_tests(
        resid_test,
        ljung_box_lags=GARCH_LJUNG_BOX_LAGS_DEFAULT,
        arch_lm_lags=GARCH_LM_LAGS_DEFAULT,
        alpha=0.05,
    )

    # Prepare and save output
    output = prepare_output(results, resid_test.size)
    ensure_output_dir(GARCH_NUMERICAL_TESTS_FILE)
    with GARCH_NUMERICAL_TESTS_FILE.open("w") as f:
        json.dump(output, f, indent=2)

    logger.info("Saved numerical test results: %s", GARCH_NUMERICAL_TESTS_FILE)
    log_test_summary(results)


if __name__ == "__main__":
    main()
