"""Main entry point for dollar bars visualisation."""

from __future__ import annotations

from pathlib import Path
import sys

# Add project root to Python path for direct execution.
# This must be done before importing src modules.
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.path import DATA_PLOTS_DIR, DOLLAR_BARS_PARQUET
from src.utils import get_logger

from src.data_visualisation.analysis import run_full_analysis

logger = get_logger(__name__)


def main() -> None:
    """Execute the full dollar bars analysis with plots."""
    run_full_analysis(
        parquet_path=DOLLAR_BARS_PARQUET,
        output_dir=DATA_PLOTS_DIR,
        show_plots=True,
        sample_fraction=0.2,
    )


if __name__ == "__main__":
    main()
