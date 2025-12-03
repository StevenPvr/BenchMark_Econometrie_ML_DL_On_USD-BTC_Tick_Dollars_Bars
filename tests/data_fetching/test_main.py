"""Tests for data_fetching.main module."""

from __future__ import annotations

import json
import sys
from importlib import import_module
from unittest.mock import MagicMock, mock_open, patch

import pytest

# Import the module properly (not via __init__.py which exports the function)
main_module = import_module("src.data_fetching.main")

# Now import functions from the module
_compute_retry_delay = main_module._compute_retry_delay
_increment_dates = main_module._increment_dates
_load_dates_state = main_module._load_dates_state
_log_footer = main_module._log_footer
_log_header = main_module._log_header
_save_dates_state = main_module._save_dates_state
_validate_dependencies = main_module._validate_dependencies
main = main_module.main
main_loop = main_module.main_loop


# --- Fixtures & Mocks ---


@pytest.fixture
def mock_logger() -> MagicMock:
    """Mock the logger."""
    with patch.object(main_module, "logger") as mock:
        yield mock


# --- State File Tests ---


def test_load_dates_state_exists() -> None:
    """Test loading dates state from existing file."""
    state_content = json.dumps({"start_date": "2023-01-01", "end_date": "2023-01-05"})
    mock_path = MagicMock()
    mock_path.exists.return_value = True
    mock_path.open = mock_open(read_data=state_content)

    with patch.object(main_module, "DATES_STATE_FILE", mock_path):
        start, end = _load_dates_state()
        assert start == "2023-01-01"
        assert end == "2023-01-05"


def test_load_dates_state_missing(mock_logger: MagicMock) -> None:
    """Test loading dates state when file is missing."""
    mock_path = MagicMock()
    mock_path.exists.return_value = False

    with patch.object(main_module, "DATES_STATE_FILE", mock_path):
        with patch.object(main_module, "START_DATE", "2020-01-01"):
            with patch.object(main_module, "END_DATE", "2020-02-01"):
                start, end = _load_dates_state()
                assert start == "2020-01-01"
                assert end == "2020-02-01"


def test_load_dates_state_corrupt(mock_logger: MagicMock) -> None:
    """Test loading dates state with corrupted file."""
    mock_path = MagicMock()
    mock_path.exists.return_value = True
    mock_path.open = mock_open(read_data="{invalid json")

    with patch.object(main_module, "DATES_STATE_FILE", mock_path):
        with patch.object(main_module, "START_DATE", "2020-01-01"):
            with patch.object(main_module, "END_DATE", "2020-02-01"):
                start, end = _load_dates_state()
                assert start == "2020-01-01"
                assert end == "2020-02-01"
                mock_logger.warning.assert_called_once()


def test_save_dates_state() -> None:
    """Test saving dates state."""
    mock_path = MagicMock()
    mock_path.parent.mkdir = MagicMock()
    mock_path.open = mock_open()

    with patch.object(main_module, "DATES_STATE_FILE", mock_path):
        _save_dates_state("2023-01-01", "2023-01-05")

        mock_path.parent.mkdir.assert_called_with(parents=True, exist_ok=True)
        mock_path.open.assert_called_with("w")

        handle = mock_path.open()
        written_content = "".join(call.args[0] for call in handle.write.call_args_list)
        data = json.loads(written_content)
        assert data["start_date"] == "2023-01-01"
        assert data["end_date"] == "2023-01-05"


# --- Date Increment Tests ---


def test_increment_dates() -> None:
    """Test date increment functionality."""
    current_end = "2023-01-10"
    with patch.object(main_module, "FETCHING_AUTO_INCREMENT_DAYS", 5):
        new_start, new_end = _increment_dates(current_end)
        assert new_start == "2023-01-10"
        assert new_end == "2023-01-15"


def test_increment_dates_month_boundary() -> None:
    """Test date increment across month boundary."""
    current_end = "2023-01-30"
    with patch.object(main_module, "FETCHING_AUTO_INCREMENT_DAYS", 5):
        new_start, new_end = _increment_dates(current_end)
        assert new_start == "2023-01-30"
        assert new_end == "2023-02-04"


# --- Dependency Validation Tests ---


def test_validate_dependencies_success() -> None:
    """Test successful dependency validation."""
    with patch.dict(
        sys.modules, {"ccxt": MagicMock(), "pandas": MagicMock(), "pyarrow": MagicMock()}
    ):
        _validate_dependencies()


def test_validate_dependencies_missing_ccxt() -> None:
    """Test dependency validation with missing ccxt."""
    original_import = __import__

    def mock_import(name: str, *args, **kwargs):
        if name == "ccxt":
            raise ImportError("No module named ccxt")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        with pytest.raises(RuntimeError, match="ccxt library not available"):
            _validate_dependencies()


def test_validate_dependencies_failures(mock_logger: MagicMock) -> None:
    """Test dependency validation failures."""
    original_import = __import__

    def mock_import_ccxt(name: str, *args, **kwargs):
        if name == "ccxt":
            raise ImportError("No module named ccxt")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import_ccxt):
        with pytest.raises(RuntimeError, match="ccxt library not available"):
            _validate_dependencies()
        assert mock_logger.error.call_count >= 1

    def mock_import_pd(name: str, *args, **kwargs):
        if name == "pandas":
            raise ImportError("No module named pandas")
        if name == "ccxt":
            return MagicMock()
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import_pd):
        with pytest.raises(RuntimeError, match="pandas library not available"):
            _validate_dependencies()

    def mock_import_pa(name: str, *args, **kwargs):
        if name == "pyarrow":
            raise ImportError("No module named pyarrow")
        if name in ["ccxt", "pandas"]:
            return MagicMock()
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import_pa):
        with pytest.raises(RuntimeError, match="pyarrow library not available"):
            _validate_dependencies()


# --- Logging Tests ---


def test_log_header(mock_logger: MagicMock) -> None:
    """Test header logging."""
    _log_header()
    assert mock_logger.info.call_count > 0


def test_log_footer_success(mock_logger: MagicMock) -> None:
    """Test footer logging on success."""
    _log_footer(success=True)
    assert mock_logger.info.call_count > 0


def test_log_footer_failure(mock_logger: MagicMock) -> None:
    """Test footer logging on failure."""
    _log_footer(success=False)
    assert mock_logger.info.call_count > 0


# --- Main Function Tests ---


@patch.object(main_module, "download_ticks_in_date_range")
@patch.object(main_module, "_load_dates_state")
@patch.object(main_module, "_save_dates_state")
@patch.object(main_module, "_validate_dependencies")
def test_main_success(
    mock_validate: MagicMock,
    mock_save: MagicMock,
    mock_load: MagicMock,
    mock_download: MagicMock,
) -> None:
    """Test successful main execution."""
    mock_load.return_value = ("2023-01-01", "2023-01-05")
    with patch.object(main_module, "_increment_dates") as mock_incr:
        mock_incr.return_value = ("2023-01-05", "2023-01-10")
        main(auto_increment=True)
        mock_validate.assert_called_once()
        mock_download.assert_called_with(start_date="2023-01-01", end_date="2023-01-05")
        mock_save.assert_called_with("2023-01-05", "2023-01-10")


@patch.object(main_module, "download_ticks_in_date_range")
@patch.object(main_module, "_load_dates_state")
@patch.object(main_module, "_validate_dependencies")
def test_main_no_increment(
    mock_validate: MagicMock, mock_load: MagicMock, mock_download: MagicMock
) -> None:
    """Test main without auto-increment."""
    mock_load.return_value = ("2023-01-01", "2023-01-05")
    with patch.object(main_module, "_save_dates_state") as mock_save:
        main(auto_increment=False)
        mock_download.assert_called()
        mock_save.assert_not_called()


@patch.object(main_module, "download_ticks_in_date_range")
@patch.object(main_module, "_load_dates_state")
def test_main_interrupt(
    mock_load: MagicMock, mock_download: MagicMock, mock_logger: MagicMock
) -> None:
    """Test main with keyboard interrupt."""
    mock_load.return_value = ("2023-01-01", "2023-01-05")
    mock_download.side_effect = KeyboardInterrupt()
    with patch.object(main_module, "_validate_dependencies"):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 130
        mock_logger.warning.assert_called_with("Operation interrupted by user")


@patch.object(main_module, "download_ticks_in_date_range")
@patch.object(main_module, "_load_dates_state")
def test_main_file_error(
    mock_load: MagicMock, mock_download: MagicMock, mock_logger: MagicMock
) -> None:
    """Test main with file system error."""
    mock_load.return_value = ("2023-01-01", "2023-01-05")
    mock_download.side_effect = FileNotFoundError("Disk missing")
    with patch.object(main_module, "_validate_dependencies"):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        mock_logger.error.assert_any_call(
            "File system error: %s", mock_download.side_effect
        )


@patch.object(main_module, "download_ticks_in_date_range")
@patch.object(main_module, "_load_dates_state")
def test_main_network_error(
    mock_load: MagicMock, mock_download: MagicMock, mock_logger: MagicMock
) -> None:
    """Test main with network error."""
    mock_load.return_value = ("2023-01-01", "2023-01-05")
    mock_download.side_effect = ConnectionError("No internet")
    with patch.object(main_module, "_validate_dependencies"):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        mock_logger.error.assert_any_call("Network error: %s", mock_download.side_effect)


@patch.object(main_module, "download_ticks_in_date_range")
@patch.object(main_module, "_load_dates_state")
def test_main_config_error(
    mock_load: MagicMock, mock_download: MagicMock, mock_logger: MagicMock
) -> None:
    """Test main with configuration error."""
    mock_load.return_value = ("2023-01-01", "2023-01-05")
    mock_download.side_effect = ValueError("Bad date")
    with patch.object(main_module, "_validate_dependencies"):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        mock_logger.error.assert_any_call(
            "Configuration error: %s", mock_download.side_effect
        )


@patch.object(main_module, "download_ticks_in_date_range")
@patch.object(main_module, "_load_dates_state")
def test_main_runtime_error(
    mock_load: MagicMock, mock_download: MagicMock, mock_logger: MagicMock
) -> None:
    """Test main with runtime error."""
    mock_load.return_value = ("2023-01-01", "2023-01-05")
    mock_download.side_effect = RuntimeError("Crash")
    with patch.object(main_module, "_validate_dependencies"):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        mock_logger.error.assert_any_call("Runtime error: %s", mock_download.side_effect)


@patch.object(main_module, "download_ticks_in_date_range")
@patch.object(main_module, "_load_dates_state")
def test_main_unknown_error(
    mock_load: MagicMock, mock_download: MagicMock, mock_logger: MagicMock
) -> None:
    """Test main with unknown error."""
    mock_load.return_value = ("2023-01-01", "2023-01-05")
    mock_download.side_effect = Exception("Unknown")
    with patch.object(main_module, "_validate_dependencies"):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        mock_logger.exception.assert_called()


# --- Main Loop Tests ---


@patch.object(main_module, "main")
@patch("time.sleep")
def test_main_loop_success(
    mock_sleep: MagicMock, mock_main: MagicMock, mock_logger: MagicMock
) -> None:
    """Test successful main loop execution."""
    main_loop(max_iterations=2, delay_seconds=1)

    assert mock_main.call_count == 2
    assert mock_logger.info.call_count > 0
    mock_sleep.assert_called_with(1)


@patch.object(main_module, "main")
@patch("time.sleep")
def test_main_loop_interrupt(
    mock_sleep: MagicMock, mock_main: MagicMock, mock_logger: MagicMock
) -> None:
    """Test main loop with keyboard interrupt."""
    mock_main.side_effect = KeyboardInterrupt()
    main_loop(max_iterations=5, delay_seconds=1)

    assert mock_main.call_count == 1
    mock_logger.info.assert_any_call("Loop interrupted at iteration %d", 1)


@patch.object(main_module, "main")
@patch("time.sleep")
def test_main_loop_error_retry(
    mock_sleep: MagicMock, mock_main: MagicMock, mock_logger: MagicMock
) -> None:
    """Test main loop with error retry."""
    mock_main.side_effect = [
        Exception("Fail1"),
        Exception("Fail2"),
        Exception("Fail3"),
        None,
    ]

    main_loop(max_iterations=4, delay_seconds=1)

    assert mock_main.call_count == 4
    assert mock_logger.error.call_count >= 3
    assert mock_logger.info.call_count > 0


@patch.object(main_module, "main")
@patch("time.sleep")
def test_main_loop_max_consecutive_errors(
    mock_sleep: MagicMock, mock_main: MagicMock, mock_logger: MagicMock
) -> None:
    """Test main loop stops after max consecutive errors."""
    mock_main.side_effect = Exception("Fail")

    main_loop(max_iterations=10, delay_seconds=1)

    assert mock_main.call_count == 5
    mock_logger.error.assert_any_call(
        "Too many consecutive errors (%d), stopping daemon", 5
    )


# --- Retry Delay Tests ---


def test_compute_retry_delay() -> None:
    """Test exponential backoff retry delay computation."""
    assert _compute_retry_delay(1) == 60  # 30 * 2^1 = 60
    assert _compute_retry_delay(2) == 120  # 30 * 2^2 = 120
    assert _compute_retry_delay(3) == 240  # 30 * 2^3 = 240
    assert _compute_retry_delay(4) == 300  # 30 * 2^4 = 480 -> capped at 300
    assert _compute_retry_delay(10) == 300  # Capped at max
