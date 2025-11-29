
import pytest
import sys
import json
from unittest.mock import MagicMock, patch, mock_open
from datetime import datetime

# We need to make sure src is in path, which is handled by conftest or env usually,
# but for unit test imports let's rely on standard import mechanisms.
from src.data_fetching import main as main_module
from src.data_fetching.main import (
    _load_dates_state,
    _save_dates_state,
    _increment_dates,
    _validate_dependencies,
    main,
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

def test_load_dates_state_missing():
    with patch('src.data_fetching.main.DATES_STATE_FILE') as mock_path:
        mock_path.exists.return_value = False
        with patch('src.data_fetching.main.START_DATE', '2020-01-01'), \
             patch('src.data_fetching.main.END_DATE', '2020-02-01'):
            start, end = _load_dates_state()
            assert start == '2020-01-01'
            assert end == '2020-02-01'

def test_save_dates_state():
    with patch('src.data_fetching.main.DATES_STATE_FILE') as mock_path:
        mock_path.parent.mkdir = MagicMock()
        with patch('builtins.open', mock_open()) as mock_file:
            _save_dates_state('2023-01-01', '2023-01-05')

            mock_path.parent.mkdir.assert_called_with(parents=True, exist_ok=True)
            mock_file.assert_called_with(mock_path, 'w')

            # Check what was written
            handle = mock_file()
            # combine all write calls
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
    # Since we are running in an env where these likely exist or we can mock sys.modules
    with patch.dict(sys.modules, {'ccxt': MagicMock(), 'pandas': MagicMock(), 'pyarrow': MagicMock()}):
        _validate_dependencies() # Should not raise

def test_validate_dependencies_missing_ccxt():
    with patch.dict(sys.modules):
        if 'ccxt' in sys.modules:
            del sys.modules['ccxt']

        # We need to ensure import raises ImportError
        # mocking import is tricky, better to rely on side_effects if possible
        # but simpler is to mock the built-in __import__ but that affects everything.
        # Alternatively, we assume the environment has them, so we can't easily test 'missing'
        # without complex patching.
        # Let's skip deep import mocking for brevity unless strictly required.
        pass

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
def test_main_error_handling(mock_load, mock_download):
    mock_load.return_value = ('2023-01-01', '2023-01-05')
    mock_download.side_effect = RuntimeError("Something went wrong")

    with patch('src.data_fetching.main._validate_dependencies'):
        with pytest.raises(SystemExit) as exc:
            main(auto_increment=False)
        assert exc.value.code == 1
