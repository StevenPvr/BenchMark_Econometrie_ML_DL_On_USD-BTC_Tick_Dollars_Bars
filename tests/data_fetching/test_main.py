
import pytest
import sys
import json
import time
from unittest.mock import MagicMock, patch, mock_open
from datetime import datetime

from src.data_fetching import main as main_module
from src.data_fetching.main import (
    _load_dates_state,
    _save_dates_state,
    _increment_dates,
    _validate_dependencies,
    _log_header,
    _log_footer,
    main,
    main_loop,
)

# --- Fixtures & Mocks ---

@pytest.fixture
def mock_logger():
    with patch('src.data_fetching.main.logger') as mock:
        yield mock

# --- Tests ---

def test_load_dates_state_exists():
    state_content = json.dumps({'start_date': '2023-01-01', 'end_date': '2023-01-05'})
    with patch('src.data_fetching.main.DATES_STATE_FILE') as mock_path:
        mock_path.exists.return_value = True
        with patch('builtins.open', mock_open(read_data=state_content)):
            start, end = _load_dates_state()
            assert start == '2023-01-01'
            assert end == '2023-01-05'

def test_load_dates_state_missing(mock_logger):
    with patch('src.data_fetching.main.DATES_STATE_FILE') as mock_path:
        mock_path.exists.return_value = False
        with patch('src.data_fetching.main.START_DATE', '2020-01-01'), \
             patch('src.data_fetching.main.END_DATE', '2020-02-01'):
            start, end = _load_dates_state()
            assert start == '2020-01-01'
            assert end == '2020-02-01'
    # Ensure no warning was logged for missing file (as it just checks exists())
    # actually logic says: if exists, try read. if fail read, log warning. if not exists, fallback.
    # so here it just falls back.

def test_load_dates_state_corrupt(mock_logger):
    with patch('src.data_fetching.main.DATES_STATE_FILE') as mock_path:
        mock_path.exists.return_value = True
        with patch('builtins.open', mock_open(read_data="{invalid json")):
            with patch('src.data_fetching.main.START_DATE', '2020-01-01'), \
                 patch('src.data_fetching.main.END_DATE', '2020-02-01'):
                start, end = _load_dates_state()
                assert start == '2020-01-01'
                assert end == '2020-02-01'
                mock_logger.warning.assert_called_once()

def test_save_dates_state():
    with patch('src.data_fetching.main.DATES_STATE_FILE') as mock_path:
        mock_path.parent.mkdir = MagicMock()
        with patch('builtins.open', mock_open()) as mock_file:
            _save_dates_state('2023-01-01', '2023-01-05')

            mock_path.parent.mkdir.assert_called_with(parents=True, exist_ok=True)
            mock_file.assert_called_with(mock_path, 'w')

            handle = mock_file()
            written_content = "".join(call.args[0] for call in handle.write.call_args_list)
            data = json.loads(written_content)
            assert data['start_date'] == '2023-01-01'
            assert data['end_date'] == '2023-01-05'

def test_increment_dates():
    current_end = "2023-01-10"
    with patch('src.data_fetching.main.AUTO_INCREMENT_DAYS', 5):
        new_start, new_end = _increment_dates(current_end)
        assert new_start == "2023-01-10"
        assert new_end == "2023-01-15"

def test_validate_dependencies_success():
    with patch.dict(sys.modules, {'ccxt': MagicMock(), 'pandas': MagicMock(), 'pyarrow': MagicMock()}):
        _validate_dependencies()

def test_validate_dependencies_missing_ccxt():
    with patch.dict(sys.modules):
        sys.modules.pop('ccxt', None)
        # We need to ensure import raises ImportError.
        # mocking builtins.__import__ is one way, but simpler is using a side_effect on a fresh mock
        # However, since we are in a live env, simply popping might not be enough if it reloads.
        # Let's try to mock the module as None or raise ImportError
        with patch('builtins.__import__', side_effect=ImportError("No module named 'ccxt'")):
             with pytest.raises(RuntimeError, match="ccxt library not available"):
                _validate_dependencies()

# Since mocking __import__ affects everything, we must be careful.
# Better strategy for dependency testing:
def test_validate_dependencies_failures(mock_logger):
    # Mocking locally to the function
    original_import = __import__

    def mock_import(name, *args, **kwargs):
        if name == 'ccxt':
            raise ImportError("No module named ccxt")
        return original_import(name, *args, **kwargs)

    with patch('builtins.__import__', side_effect=mock_import):
        with pytest.raises(RuntimeError, match="ccxt library not available"):
             _validate_dependencies()
        assert mock_logger.error.call_count >= 1

    def mock_import_pd(name, *args, **kwargs):
        if name == 'pandas':
            raise ImportError("No module named pandas")
        if name == 'ccxt':
             return MagicMock()
        return original_import(name, *args, **kwargs)

    with patch('builtins.__import__', side_effect=mock_import_pd):
        with pytest.raises(RuntimeError, match="pandas library not available"):
             _validate_dependencies()

    def mock_import_pa(name, *args, **kwargs):
        if name == 'pyarrow':
            raise ImportError("No module named pyarrow")
        if name in ['ccxt', 'pandas']:
             return MagicMock()
        return original_import(name, *args, **kwargs)

    with patch('builtins.__import__', side_effect=mock_import_pa):
        with pytest.raises(RuntimeError, match="pyarrow library not available"):
             _validate_dependencies()

def test_log_header(mock_logger):
    _log_header()
    assert mock_logger.info.call_count > 0

def test_log_footer(mock_logger):
    _log_footer(success=True)
    assert mock_logger.info.call_count > 0
    _log_footer(success=False)
    assert mock_logger.info.call_count > 0

@patch('src.data_fetching.main.download_ticks_in_date_range')
@patch('src.data_fetching.main._load_dates_state')
@patch('src.data_fetching.main._save_dates_state')
@patch('src.data_fetching.main._validate_dependencies')
def test_main_success(mock_validate, mock_save, mock_load, mock_download):
    mock_load.return_value = ('2023-01-01', '2023-01-05')
    with patch('src.data_fetching.main._increment_dates') as mock_incr:
        mock_incr.return_value = ('2023-01-05', '2023-01-10')
        main(auto_increment=True)
        mock_validate.assert_called_once()
        mock_download.assert_called_with(start_date='2023-01-01', end_date='2023-01-05')
        mock_save.assert_called_with('2023-01-05', '2023-01-10')

@patch('src.data_fetching.main.download_ticks_in_date_range')
@patch('src.data_fetching.main._load_dates_state')
@patch('src.data_fetching.main._validate_dependencies')
def test_main_no_increment(mock_validate, mock_load, mock_download):
    mock_load.return_value = ('2023-01-01', '2023-01-05')
    with patch('src.data_fetching.main._save_dates_state') as mock_save:
        main(auto_increment=False)
        mock_download.assert_called()
        mock_save.assert_not_called()

@patch('src.data_fetching.main.download_ticks_in_date_range')
@patch('src.data_fetching.main._load_dates_state')
def test_main_interrupt(mock_load, mock_download, mock_logger):
    mock_load.return_value = ('2023-01-01', '2023-01-05')
    mock_download.side_effect = KeyboardInterrupt()
    with patch('src.data_fetching.main._validate_dependencies'):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 130
        mock_logger.warning.assert_called_with("Operation interrupted by user")

@patch('src.data_fetching.main.download_ticks_in_date_range')
@patch('src.data_fetching.main._load_dates_state')
def test_main_file_error(mock_load, mock_download, mock_logger):
    mock_load.return_value = ('2023-01-01', '2023-01-05')
    mock_download.side_effect = FileNotFoundError("Disk missing")
    with patch('src.data_fetching.main._validate_dependencies'):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        mock_logger.error.assert_any_call("File system error: %s", mock_download.side_effect)

@patch('src.data_fetching.main.download_ticks_in_date_range')
@patch('src.data_fetching.main._load_dates_state')
def test_main_network_error(mock_load, mock_download, mock_logger):
    mock_load.return_value = ('2023-01-01', '2023-01-05')
    mock_download.side_effect = ConnectionError("No internet")
    with patch('src.data_fetching.main._validate_dependencies'):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        mock_logger.error.assert_any_call("Network error: %s", mock_download.side_effect)

@patch('src.data_fetching.main.download_ticks_in_date_range')
@patch('src.data_fetching.main._load_dates_state')
def test_main_config_error(mock_load, mock_download, mock_logger):
    mock_load.return_value = ('2023-01-01', '2023-01-05')
    mock_download.side_effect = ValueError("Bad date")
    with patch('src.data_fetching.main._validate_dependencies'):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        mock_logger.error.assert_any_call("Configuration error: %s", mock_download.side_effect)

@patch('src.data_fetching.main.download_ticks_in_date_range')
@patch('src.data_fetching.main._load_dates_state')
def test_main_runtime_error(mock_load, mock_download, mock_logger):
    mock_load.return_value = ('2023-01-01', '2023-01-05')
    mock_download.side_effect = RuntimeError("Crash")
    with patch('src.data_fetching.main._validate_dependencies'):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        mock_logger.error.assert_any_call("Runtime error: %s", mock_download.side_effect)

@patch('src.data_fetching.main.download_ticks_in_date_range')
@patch('src.data_fetching.main._load_dates_state')
def test_main_unknown_error(mock_load, mock_download, mock_logger):
    mock_load.return_value = ('2023-01-01', '2023-01-05')
    mock_download.side_effect = Exception("Unknown")
    with patch('src.data_fetching.main._validate_dependencies'):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        mock_logger.exception.assert_called()

# --- Main Loop Tests ---

@patch('src.data_fetching.main.main')
@patch('time.sleep')
def test_main_loop_success(mock_sleep, mock_main, mock_logger):
    # Run loop 2 times
    main_loop(max_iterations=2, delay_seconds=1)

    assert mock_main.call_count == 2
    assert mock_logger.info.call_count > 0
    # Should sleep once between iterations
    mock_sleep.assert_called_with(1)

@patch('src.data_fetching.main.main')
@patch('time.sleep')
def test_main_loop_interrupt(mock_sleep, mock_main, mock_logger):
    mock_main.side_effect = KeyboardInterrupt()
    main_loop(max_iterations=5, delay_seconds=1)

    # Should stop at first interrupt
    assert mock_main.call_count == 1
    mock_logger.info.assert_any_call("🛑 Loop interrupted at iteration 1")

@patch('src.data_fetching.main.main')
@patch('time.sleep')
def test_main_loop_error_retry(mock_sleep, mock_main, mock_logger):
    # Fail 3 times then succeed
    mock_main.side_effect = [Exception("Fail1"), Exception("Fail2"), Exception("Fail3"), None]

    main_loop(max_iterations=4, delay_seconds=1)

    assert mock_main.call_count == 4
    # Check retries logic (backoff)
    # retry delays: 30, 60, 120
    # sleep calls: retry1, retry2, retry3, normal_delay(before 4th),
    # but wait, it succeeds on 4th, so it sleeps delay_seconds *after* 4th if max_iters wasn't hit
    # but here max_iters=4 so loop ends after 4th, so no final sleep.

    # We can check logs for "Retrying in"
    assert mock_logger.error.call_count >= 3
    assert mock_logger.info.call_count > 0

@patch('src.data_fetching.main.main')
@patch('time.sleep')
def test_main_loop_max_consecutive_errors(mock_sleep, mock_main, mock_logger):
    mock_main.side_effect = Exception("Fail")

    main_loop(max_iterations=10, delay_seconds=1)

    # Should stop after 5 consecutive errors (default max)
    assert mock_main.call_count == 5
    mock_logger.error.assert_any_call("💀 Too many consecutive errors (5), stopping daemon")
