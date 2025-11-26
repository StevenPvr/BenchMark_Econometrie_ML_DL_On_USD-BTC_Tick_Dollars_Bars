"""Tests for src/utils/io.py module."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd # type: ignore[import-untyped]
import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.io import (
    ensure_output_dir,
    get_parquet_path,
    load_and_validate_dataframe,
    load_csv_file,
    load_dataframe,
    load_json_data,
    load_parquet_file,
    read_dataset_file,
    save_json_pretty,
    save_parquet_and_csv,
    write_placeholder_file,
)


class TestGetParquetPath:
    """Test cases for get_parquet_path function."""

    def test_csv_to_parquet(self):
        """CSV path should convert to parquet."""
        result = get_parquet_path(Path("/data/file.csv"))
        assert result == Path("/data/file.parquet")

    def test_parquet_unchanged(self):
        """Parquet path should remain unchanged."""
        result = get_parquet_path(Path("/data/file.parquet"))
        assert result == Path("/data/file.parquet")

    def test_other_extension(self):
        """Other extensions should be replaced with .parquet."""
        result = get_parquet_path(Path("/data/file.txt"))
        assert result == Path("/data/file.parquet")


class TestLoadParquetFile:
    """Test cases for load_parquet_file function."""

    def test_loads_valid_parquet(self):
        """Should load valid parquet file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_path = Path(temp_dir) / "test.parquet"
            df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
            df.to_parquet(parquet_path)

            result = load_parquet_file(parquet_path)
            assert result is not None
            assert len(result) == 3


class TestLoadCsvFile:
    """Test cases for load_csv_file function."""

    def test_loads_valid_csv(self):
        """Should load valid CSV file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test.csv"
            df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
            df.to_csv(csv_path, index=False)

            result = load_csv_file(csv_path)
            assert len(result) == 3


class TestReadDatasetFile:
    """Test cases for read_dataset_file function."""

    def test_reads_csv(self):
        """Should read CSV file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test.csv"
            df = pd.DataFrame({"a": [1, 2]})
            df.to_csv(csv_path, index=False)

            result = read_dataset_file(csv_path)
            assert len(result) == 2

    def test_reads_parquet(self):
        """Should read parquet file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_path = Path(temp_dir) / "test.parquet"
            df = pd.DataFrame({"a": [1, 2]})
            df.to_parquet(parquet_path)

            result = read_dataset_file(parquet_path)
            assert len(result) == 2


class TestSaveParquetAndCsv:
    """Test cases for save_parquet_and_csv function."""

    def test_saves_parquet(self):
        """Should save parquet file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_path = Path(temp_dir) / "output.parquet"
            df = pd.DataFrame({"a": [1, 2, 3]})

            save_parquet_and_csv(df, parquet_path)

            assert parquet_path.exists()

    def test_saves_csv(self):
        """Should also save CSV file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_path = Path(temp_dir) / "output.parquet"
            csv_path = Path(temp_dir) / "output.csv"
            df = pd.DataFrame({"a": [1, 2, 3]})

            save_parquet_and_csv(df, parquet_path, csv_path)

            assert csv_path.exists()

    def test_creates_directories(self):
        """Should create parent directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_path = Path(temp_dir) / "nested" / "dir" / "output.parquet"
            df = pd.DataFrame({"a": [1]})

            save_parquet_and_csv(df, parquet_path)

            assert parquet_path.exists()

    def test_files_readable(self):
        """Saved files should be readable."""
        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_path = Path(temp_dir) / "output.parquet"
            csv_path = Path(temp_dir) / "output.csv"
            df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

            save_parquet_and_csv(df, parquet_path, csv_path)

            df_parquet = pd.read_parquet(parquet_path)
            df_csv = pd.read_csv(csv_path)

            assert len(df_parquet) == 3
            assert len(df_csv) == 3


class TestEnsureOutputDir:
    """Test cases for ensure_output_dir function."""

    def test_creates_directories(self):
        """Should create parent directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "a" / "b" / "c" / "file.txt"
            ensure_output_dir(path)

            assert path.parent.exists()

    def test_existing_directory_no_error(self):
        """Existing directory should not raise error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "file.txt"
            ensure_output_dir(path)
            ensure_output_dir(path)  # Second call should not raise


class TestWritePlaceholderFile:
    """Test cases for write_placeholder_file function."""

    def test_creates_file(self):
        """Should create placeholder file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "placeholder.txt"
            write_placeholder_file(path)

            assert path.exists()
            assert "Placeholder" in path.read_text()

    def test_creates_directories(self):
        """Should create parent directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "placeholder.txt"
            write_placeholder_file(path)

            assert path.exists()

    def test_custom_content(self):
        """Should write custom content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "custom.txt"
            write_placeholder_file(path, content="Custom content")

            assert path.read_text() == "Custom content"


class TestLoadDataframe:
    """Test cases for load_dataframe function."""

    def test_auto_detect_parquet(self):
        """Should auto-detect parquet format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test.parquet"
            df = pd.DataFrame({"a": [1, 2, 3]})
            df.to_parquet(path)

            result = load_dataframe(path)
            assert len(result) == 3

    def test_auto_detect_csv(self):
        """Should auto-detect CSV format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test.csv"
            df = pd.DataFrame({"a": [1, 2, 3]})
            df.to_csv(path, index=False)

            result = load_dataframe(path)
            assert len(result) == 3

    def test_unsupported_format_raises(self):
        """Should raise ValueError for unsupported format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test.txt"
            path.write_text("test")

            with pytest.raises(ValueError, match="Unsupported file format"):
                load_dataframe(path)


class TestLoadAndValidateDataframe:
    """Test cases for load_and_validate_dataframe function."""

    def test_loads_valid_parquet(self):
        """Should load valid parquet file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test.parquet"
            df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
            df.to_parquet(path)

            result = load_and_validate_dataframe(path)
            assert len(result) == 3

    def test_validates_required_columns(self):
        """Should validate required columns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test.parquet"
            df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
            df.to_parquet(path)

            result = load_and_validate_dataframe(path, required_columns=["a", "b"])
            assert "a" in result.columns
            assert "b" in result.columns

    def test_missing_column_raises(self):
        """Should raise ValueError for missing required column."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test.parquet"
            df = pd.DataFrame({"a": [1, 2]})
            df.to_parquet(path)

            with pytest.raises(ValueError, match="Missing required columns"):
                load_and_validate_dataframe(path, required_columns=["a", "b", "c"])


class TestLoadJsonData:
    """Test cases for load_json_data function."""

    def test_loads_valid_json(self):
        """Should load valid JSON file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test.json"
            data = {"key": "value", "number": 42}
            path.write_text(json.dumps(data))

            result = load_json_data(path)
            assert result["key"] == "value"
            assert result["number"] == 42

    def test_invalid_json_raises(self):
        """Invalid JSON should raise error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test.json"
            path.write_text("not valid json")

            with pytest.raises(json.JSONDecodeError):
                load_json_data(path)

    def test_file_not_found(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_json_data("/nonexistent/file.json")


class TestSaveJsonPretty:
    """Test cases for save_json_pretty function."""

    def test_saves_json(self):
        """Should save JSON file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test.json"
            data = {"key": "value"}

            save_json_pretty(data, path)

            assert path.exists()
            result = json.loads(path.read_text())
            assert result["key"] == "value"

    def test_saves_with_indent(self):
        """Should save with indentation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test.json"
            data = {"key": "value"}

            save_json_pretty(data, path)

            content = path.read_text()
            assert "  " in content  # Indented with 2 spaces

    def test_creates_directories(self):
        """Should create parent directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "dir" / "test.json"

            save_json_pretty({"key": "value"}, path)

            assert path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
