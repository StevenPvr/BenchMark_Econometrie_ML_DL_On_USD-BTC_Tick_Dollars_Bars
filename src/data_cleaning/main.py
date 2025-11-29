"""CLI entry point for the data_cleaning module."""

from __future__ import annotations

from pathlib import Path
import sys

# Add project root to Python path for direct execution.
# This must be done before importing src modules.
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config_logging import setup_logging
from src.data_cleaning.cleaning import clean_ticks_data
from src.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Run the standard data-cleaning sequence from the command line.

    The sequence is:

    1. Run :func:`clean_ticks_data` to apply integrity fixes and persist
       a cleaned dataset.

    Any handled error leads to a non-zero exit code to ease automation.
    """
    setup_logging()
    logger.info("Launching data_cleaning CLI")

    try:
        clean_ticks_data()
        logger.info("Data cleaning completed successfully")
    except (FileNotFoundError, KeyError, ValueError, OSError) as e:
        logger.error("Data cleaning failed: %s", e)
        sys.exit(1)
    except Exception as e:  # pragma: no cover - defensive catch-all
        logger.exception("Unexpected error during data cleaning: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
