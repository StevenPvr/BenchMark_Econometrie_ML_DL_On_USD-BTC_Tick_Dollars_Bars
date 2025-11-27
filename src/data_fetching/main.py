"""Automated BTC/USD tick data fetching with auto-increment."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
import json
import sys

# Add project root to Python path for direct execution
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config_logging import setup_logging
from src.constants import END_DATE, EXCHANGE_ID, START_DATE, SYMBOL
from src.data_fetching.fetching import download_ticks_in_date_range
from src.path import DATASET_RAW_CSV, DATASET_RAW_PARQUET
from src.utils import get_logger

logger = get_logger(__name__)

# Auto-increment configuration
AUTO_INCREMENT_DAYS: int = 5  # 5-day windows
DATES_STATE_FILE = _project_root / "data" / "fetch_dates_state.json"


def _load_dates_state() -> tuple[str, str]:
    """Load current start/end dates from state file, fallback to constants."""
    if DATES_STATE_FILE.exists():
        try:
            with open(DATES_STATE_FILE, 'r') as f:
                state = json.load(f)
                return state['start_date'], state['end_date']
        except Exception as e:
            logger.warning("Could not load dates state file: %s", e)

    # Fallback to constants
    return START_DATE, END_DATE


def _save_dates_state(start_date: str, end_date: str) -> None:
    """Save current dates to state file."""
    DATES_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state = {
        'start_date': start_date,
        'end_date': end_date,
        'last_updated': datetime.now().isoformat()
    }
    with open(DATES_STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def _increment_dates(current_end: str) -> tuple[str, str]:
    """Increment dates by AUTO_INCREMENT_DAYS, handling month ends."""
    end_dt = datetime.strptime(current_end, "%Y-%m-%d")

    # New start = current end
    new_start = end_dt

    # New end = start + AUTO_INCREMENT_DAYS
    new_end = new_start + timedelta(days=AUTO_INCREMENT_DAYS)

    return new_start.strftime("%Y-%m-%d"), new_end.strftime("%Y-%m-%d")


def _validate_dependencies() -> None:
    """Validate that required dependencies are available."""
    try:
        import ccxt  # type: ignore[import-untyped]
        logger.info("✓ ccxt library available")
    except ImportError as e:
        logger.error("Missing required dependency: ccxt")
        logger.error("Install with: pip install ccxt")
        raise RuntimeError("ccxt library not available") from e

    try:
        import pandas as pd  # type: ignore[import-untyped]
        logger.info("✓ pandas library available")
    except ImportError as e:
        logger.error("Missing required dependency: pandas")
        raise RuntimeError("pandas library not available") from e

    try:
        import pyarrow  # type: ignore[import-untyped]
        logger.info("✓ pyarrow library available")
    except ImportError as e:
        logger.error("Missing required dependency: pyarrow")
        logger.error("Install with: pip install pyarrow")
        raise RuntimeError("pyarrow library not available") from e


def _log_header() -> None:
    """Display the automated fetching banner with configuration."""
    logger.info("\n")
    logger.info("=" * 70)
    logger.info(" AUTOMATED BTC/USD TICK DATA FETCHING")
    logger.info("=" * 70)
    logger.info(f"Exchange:     {EXCHANGE_ID}")
    logger.info(f"Symbol:       {SYMBOL}")
    logger.info(f"Date Range:   {START_DATE} → {END_DATE}")
    logger.info(f"Output CSV:   {DATASET_RAW_CSV}")
    logger.info(f"Output PQ:    {DATASET_RAW_PARQUET}")
    logger.info("=" * 70)


def _log_footer(success: bool = True) -> None:
    """Display completion summary."""
    logger.info("\n" + "=" * 70)
    if success:
        logger.info(" FETCHING COMPLETE")
        logger.info("✓ BTC/USD tick data successfully downloaded")
    else:
        logger.info(" FETCHING FAILED")
    logger.info("=" * 70)


def main(auto_increment: bool = True) -> None:
    """Automated main function for BTC/USD tick data fetching with auto-increment."""
    setup_logging()

    try:
        # Load current dates (from state file or constants)
        current_start, current_end = _load_dates_state()

        # Validate dependencies before proceeding
        _validate_dependencies()

        # Display header with current configuration
        logger.info("\n")
        logger.info("=" * 70)
        logger.info(" AUTOMATED BTC/USD TICK DATA FETCHING (AUTO-INCREMENT)")
        logger.info("=" * 70)
        logger.info(f"Exchange:     {EXCHANGE_ID}")
        logger.info(f"Symbol:       {SYMBOL}")
        logger.info(f"Date Range:   {current_start} → {current_end}")
        logger.info(f"Window Size:  {AUTO_INCREMENT_DAYS} days")
        logger.info(f"Output:       {DATASET_RAW_PARQUET}")
        logger.info("=" * 70)

        # Execute the fetching pipeline with current dates
        logger.info("Starting automated data fetching...")
        download_ticks_in_date_range(start_date=current_start, end_date=current_end)
        logger.info("✓ Data fetching completed successfully")

        # Auto-increment dates for next run
        if auto_increment:
            next_start, next_end = _increment_dates(current_end)
            _save_dates_state(next_start, next_end)
            logger.info("✓ Dates auto-incremented: %s → %s (next run)", next_start, next_end)

        # Display success footer
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


def main_loop(max_iterations: int | None = None, delay_seconds: int = 300) -> None:
    """Run data fetching in a continuous loop with auto-increment dates.

    This function runs indefinitely until interrupted, automatically incrementing
    dates and handling errors gracefully.

    Args:
        max_iterations: Maximum iterations (None = infinite)
        delay_seconds: Delay between iterations (default: 5 minutes = 300s)
    """
    import time
    import signal
    import sys

    # Handle graceful shutdown
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal, stopping gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    iteration = 0
    consecutive_errors = 0
    max_consecutive_errors = 5

    logger.info("🚀 Starting continuous data fetching daemon...")
    logger.info(f"⏰ Delay between iterations: {delay_seconds} seconds ({delay_seconds//60} minutes)")
    logger.info("🛑 Use Ctrl+C or 'kill <pid>' to stop gracefully")
    logger.info("="*80)

    while max_iterations is None or iteration < max_iterations:
        iteration += 1
        start_time = time.time()

        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 ITERATION {iteration} - Starting...")
        logger.info(f"{'='*60}")

        try:
            main(auto_increment=True)
            execution_time = time.time() - start_time

            logger.info(f"✅ Iteration {iteration} completed successfully in {execution_time:.1f}s")
            consecutive_errors = 0  # Reset error counter

            if max_iterations is None or iteration < max_iterations:
                logger.info(f"⏳ Next iteration in {delay_seconds} seconds...")
                time.sleep(delay_seconds)

        except KeyboardInterrupt:
            logger.info(f"🛑 Loop interrupted at iteration {iteration}")
            break
        except Exception as e:
            execution_time = time.time() - start_time
            consecutive_errors += 1

            logger.error(f"❌ Iteration {iteration} failed after {execution_time:.1f}s: {e}")
            logger.error(f"🔄 Consecutive errors: {consecutive_errors}/{max_consecutive_errors}")

            if consecutive_errors >= max_consecutive_errors:
                logger.error(f"💀 Too many consecutive errors ({consecutive_errors}), stopping daemon")
                break

            # Exponential backoff for retries
            retry_delay = min(300, 30 * (2 ** consecutive_errors))  # Max 5 minutes
            logger.info(f"⏳ Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

    logger.info(f"🏁 Daemon stopped after {iteration} iterations")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Automated BTC/USD tick data fetching daemon")
    parser.add_argument("--daemon", action="store_true",
                       help="Run as daemon (infinite loop with error recovery)")
    parser.add_argument("--loop", action="store_true",
                       help="Run in auto-increment loop (legacy, use --daemon)")
    parser.add_argument("--max-iterations", type=int,
                       help="Maximum iterations for loop/daemon mode")
    parser.add_argument("--delay", type=int, default=300,
                       help="Delay between iterations in seconds (default: 300 = 5 min)")
    parser.add_argument("--no-increment", action="store_true",
                       help="Disable auto-increment (test mode)")

    args = parser.parse_args()

    if args.daemon or args.loop:
        # Run as daemon (infinite loop unless max_iterations specified)
        main_loop(max_iterations=args.max_iterations, delay_seconds=args.delay)
    else:
        # Single execution
        main(auto_increment=not args.no_increment)
