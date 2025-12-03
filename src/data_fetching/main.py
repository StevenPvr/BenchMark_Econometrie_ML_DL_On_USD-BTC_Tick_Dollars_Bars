"""Automated BTC/USD tick data fetching with auto-increment."""

from __future__ import annotations

import json
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import FrameType

from src.config_logging import setup_logging
from src.constants import (
    END_DATE,
    EXCHANGE_ID,
    FETCHING_AUTO_INCREMENT_DAYS,
    START_DATE,
    SYMBOL,
)
from src.data_fetching.fetching import download_ticks_in_date_range
from src.path import DATASET_RAW_PARQUET, PROJECT_ROOT
from src.utils import get_logger

logger = get_logger(__name__)

# State file location
DATES_STATE_FILE: Path = PROJECT_ROOT / "data" / "fetch_dates_state.json"

# Daemon configuration
MAX_CONSECUTIVE_ERRORS: int = 5
DEFAULT_DELAY_SECONDS: int = 300
EXPONENTIAL_BACKOFF_BASE: int = 30
EXPONENTIAL_BACKOFF_MAX: int = 300


def _load_dates_state() -> tuple[str, str]:
    """Load current start/end dates from state file.

    Falls back to constants if state file doesn't exist or is corrupted.

    Returns:
        Tuple of (start_date, end_date) in YYYY-MM-DD format.
    """
    if DATES_STATE_FILE.exists():
        try:
            with DATES_STATE_FILE.open("r") as f:
                state = json.load(f)
                return state["start_date"], state["end_date"]
        except Exception as e:
            logger.warning("Could not load dates state file: %s", e)

    return START_DATE, END_DATE


def _save_dates_state(start_date: str, end_date: str) -> None:
    """Save current dates to state file.

    Args:
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
    """
    DATES_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "start_date": start_date,
        "end_date": end_date,
        "last_updated": datetime.now().isoformat(),
    }
    with DATES_STATE_FILE.open("w") as f:
        json.dump(state, f, indent=2)


def _increment_dates(current_end: str) -> tuple[str, str]:
    """Increment dates by auto-increment days.

    Args:
        current_end: Current end date in YYYY-MM-DD format.

    Returns:
        Tuple of (new_start, new_end) in YYYY-MM-DD format.
    """
    end_dt = datetime.strptime(current_end, "%Y-%m-%d")
    new_start = end_dt
    new_end = new_start + timedelta(days=FETCHING_AUTO_INCREMENT_DAYS)
    return new_start.strftime("%Y-%m-%d"), new_end.strftime("%Y-%m-%d")


def _validate_dependencies() -> None:
    """Validate that required dependencies are available.

    Raises:
        RuntimeError: If a required dependency is missing.
    """
    try:
        import ccxt  # noqa: F401

        logger.info("ccxt library available")
    except ImportError as e:
        logger.error("Missing required dependency: ccxt")
        logger.error("Install with: pip install ccxt")
        msg = "ccxt library not available"
        raise RuntimeError(msg) from e

    try:
        import pandas  # noqa: F401

        logger.info("pandas library available")
    except ImportError as e:
        logger.error("Missing required dependency: pandas")
        msg = "pandas library not available"
        raise RuntimeError(msg) from e

    try:
        import pyarrow  # noqa: F401

        logger.info("pyarrow library available")
    except ImportError as e:
        logger.error("Missing required dependency: pyarrow")
        logger.error("Install with: pip install pyarrow")
        msg = "pyarrow library not available"
        raise RuntimeError(msg) from e


def _log_header() -> None:
    """Display the automated fetching banner with configuration."""
    logger.info("")
    logger.info("=" * 70)
    logger.info(" AUTOMATED BTC/USD TICK DATA FETCHING")
    logger.info("=" * 70)
    logger.info("Exchange:     %s", EXCHANGE_ID)
    logger.info("Symbol:       %s", SYMBOL)
    logger.info("Date Range:   %s -> %s", START_DATE, END_DATE)
    logger.info("Output PQ:    %s", DATASET_RAW_PARQUET)
    logger.info("=" * 70)


def _log_footer(success: bool = True) -> None:
    """Display completion summary.

    Args:
        success: Whether the operation succeeded.
    """
    logger.info("")
    logger.info("=" * 70)
    if success:
        logger.info(" FETCHING COMPLETE")
        logger.info("BTC/USD tick data successfully downloaded")
    else:
        logger.info(" FETCHING FAILED")
    logger.info("=" * 70)


def main(auto_increment: bool = True) -> None:
    """Automated main function for BTC/USD tick data fetching with auto-increment.

    Args:
        auto_increment: Whether to increment dates for the next run.
    """
    setup_logging()

    try:
        current_start, current_end = _load_dates_state()
        _validate_dependencies()
        _log_run_header(current_start, current_end)

        logger.info("Starting automated data fetching...")
        download_ticks_in_date_range(start_date=current_start, end_date=current_end)
        logger.info("Data fetching completed successfully")

        if auto_increment:
            next_start, next_end = _increment_dates(current_end)
            _save_dates_state(next_start, next_end)
            logger.info(
                "Dates auto-incremented: %s -> %s (next run)", next_start, next_end
            )

        _log_footer(success=True)

    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user")
        _log_footer(success=False)
        sys.exit(130)
    except (FileNotFoundError, PermissionError) as e:
        logger.error("File system error: %s", e)
        logger.error("Check that output directories exist and are writable")
        _log_footer(success=False)
        sys.exit(1)
    except (ConnectionError, TimeoutError) as e:
        logger.error("Network error: %s", e)
        logger.error("Check your internet connection and exchange availability")
        _log_footer(success=False)
        sys.exit(1)
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        logger.error("Check your exchange ID, symbol, and date formats")
        _log_footer(success=False)
        sys.exit(1)
    except RuntimeError as e:
        logger.error("Runtime error: %s", e)
        _log_footer(success=False)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error during data fetching: %s", e)
        _log_footer(success=False)
        sys.exit(1)


def _log_run_header(current_start: str, current_end: str) -> None:
    """Log the header for a data fetching run.

    Args:
        current_start: Start date for this run.
        current_end: End date for this run.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info(" AUTOMATED BTC/USD TICK DATA FETCHING (AUTO-INCREMENT)")
    logger.info("=" * 70)
    logger.info("Exchange:     %s", EXCHANGE_ID)
    logger.info("Symbol:       %s", SYMBOL)
    logger.info("Date Range:   %s -> %s", current_start, current_end)
    logger.info("Window Size:  %d days", FETCHING_AUTO_INCREMENT_DAYS)
    logger.info("Output:       %s", DATASET_RAW_PARQUET)
    logger.info("=" * 70)


def main_loop(
    max_iterations: int | None = None,
    delay_seconds: int = DEFAULT_DELAY_SECONDS,
) -> None:
    """Run data fetching in a continuous loop with auto-increment dates.

    This function runs indefinitely until interrupted, automatically incrementing
    dates and handling errors gracefully.

    Args:
        max_iterations: Maximum iterations (None = infinite).
        delay_seconds: Delay between iterations in seconds.
    """
    _setup_signal_handlers()

    iteration = 0
    consecutive_errors = 0

    _log_daemon_start(delay_seconds)

    while max_iterations is None or iteration < max_iterations:
        iteration += 1
        start_time = time.time()

        _log_iteration_start(iteration)

        try:
            main(auto_increment=True)
            execution_time = time.time() - start_time

            logger.info(
                "Iteration %d completed successfully in %.1fs", iteration, execution_time
            )
            consecutive_errors = 0

            if max_iterations is None or iteration < max_iterations:
                logger.info("Next iteration in %d seconds...", delay_seconds)
                time.sleep(delay_seconds)

        except KeyboardInterrupt:
            logger.info("Loop interrupted at iteration %d", iteration)
            break
        except Exception as e:
            execution_time = time.time() - start_time
            consecutive_errors += 1

            logger.error(
                "Iteration %d failed after %.1fs: %s", iteration, execution_time, e
            )
            logger.error(
                "Consecutive errors: %d/%d", consecutive_errors, MAX_CONSECUTIVE_ERRORS
            )

            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                logger.error(
                    "Too many consecutive errors (%d), stopping daemon",
                    consecutive_errors,
                )
                break

            retry_delay = _compute_retry_delay(consecutive_errors)
            logger.info("Retrying in %d seconds...", retry_delay)
            time.sleep(retry_delay)

    logger.info("Daemon stopped after %d iterations", iteration)


def _setup_signal_handlers() -> None:
    """Set up signal handlers for graceful shutdown."""

    def signal_handler(signum: int, frame: FrameType | None) -> None:
        logger.info("Received shutdown signal, stopping gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def _log_daemon_start(delay_seconds: int) -> None:
    """Log daemon startup information.

    Args:
        delay_seconds: Delay between iterations.
    """
    logger.info("Starting continuous data fetching daemon...")
    logger.info(
        "Delay between iterations: %d seconds (%d minutes)",
        delay_seconds,
        delay_seconds // 60,
    )
    logger.info("Use Ctrl+C or 'kill <pid>' to stop gracefully")
    logger.info("=" * 80)


def _log_iteration_start(iteration: int) -> None:
    """Log iteration start information.

    Args:
        iteration: Current iteration number.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("ITERATION %d - Starting...", iteration)
    logger.info("=" * 60)


def _compute_retry_delay(consecutive_errors: int) -> int:
    """Compute retry delay with exponential backoff.

    Args:
        consecutive_errors: Number of consecutive errors.

    Returns:
        Retry delay in seconds.
    """
    return min(
        EXPONENTIAL_BACKOFF_MAX,
        EXPONENTIAL_BACKOFF_BASE * (2**consecutive_errors),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Automated BTC/USD tick data fetching daemon"
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as daemon (infinite loop with error recovery)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run in auto-increment loop (legacy, use --daemon)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        help="Maximum iterations for loop/daemon mode",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=DEFAULT_DELAY_SECONDS,
        help=f"Delay between iterations in seconds (default: {DEFAULT_DELAY_SECONDS})",
    )
    parser.add_argument(
        "--no-increment",
        action="store_true",
        help="Disable auto-increment (test mode)",
    )

    args = parser.parse_args()

    if args.daemon or args.loop:
        main_loop(max_iterations=args.max_iterations, delay_seconds=args.delay)
    else:
        main(auto_increment=not args.no_increment)


__all__ = [
    "main",
    "main_loop",
]
