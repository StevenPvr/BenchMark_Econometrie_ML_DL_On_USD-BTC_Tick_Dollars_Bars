"""CLI for GARCH identification (ARCH/GARCH detection).

Implements methodology for detecting conditional heteroskedasticity:
1. Extract residuals εt from ARIMA model (mean model)
2. Test for ARCH effect using Lagrange Multiplier test (ARCH-LM)
3. Inspect autocorrelation of squared residuals

Results saved to results/garch/structure/
Plots saved to plots/garch/structure/
"""

from __future__ import annotations

from pathlib import Path
import sys

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import json

import numpy as np

from src.utils import setup_project_path

# Ensure project root
setup_project_path()

from src.constants import (
    GARCH_DATASET_FILE,
    GARCH_DIAGNOSTICS_FILE,
)
from src.garch.structure_garch.detection import detect_heteroskedasticity
from src.garch.structure_garch.utils import load_garch_dataset, prepare_residuals
from src.utils import ensure_output_dir, get_logger

logger = get_logger(__name__)


def main() -> None:
    """Run ARCH/GARCH identification diagnostics.

    Methodology:
    1. Extract residuals εt from ARIMA model (mean model)
    2. Test for ARCH effect using Lagrange Multiplier test (ARCH-LM)
    3. Inspect autocorrelation of squared residuals

    Results saved to results/garch/structure/
    Plots saved to plots/garch/structure/
    """
    logger.info("=" * 60)
    logger.info("GARCH IDENTIFICATION (ARCH-LM + ACF(e^2))")
    logger.info("=" * 60)

    df = load_garch_dataset(str(GARCH_DATASET_FILE))
    resid_test = prepare_residuals(df, use_test_only=True)
    resid_test = resid_test[np.isfinite(resid_test)]

    # Use standard values for ARCH/GARCH diagnostics
    lm_lags = 12
    significance_level = 0.05

    results = detect_heteroskedasticity(
        resid_test,
        lags=lm_lags,
        acf_lags=lm_lags,
        alpha=significance_level,
    )

    out = {
        "source": str(GARCH_DATASET_FILE),
        "diagnostics": results,
        "n_test": int(resid_test.size),
    }
    ensure_output_dir(GARCH_DIAGNOSTICS_FILE)
    with GARCH_DIAGNOSTICS_FILE.open("w") as f:
        json.dump(out, f, indent=2)
    logger.info("Saved identification diagnostics: %s", GARCH_DIAGNOSTICS_FILE)


if __name__ == "__main__":
    main()
