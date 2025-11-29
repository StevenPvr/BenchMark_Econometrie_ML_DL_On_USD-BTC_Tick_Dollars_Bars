"""Tests for src.utils.io."""

import pytest
import pandas as pd
import json
from pathlib import Path
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

class TestIO:
    def test_ensure_output_dir(self, tmp_path):
        """Test directory creation."""
        file_path = tmp_path / "subdir" / "file.txt"
        ensure_output_dir(file_path)
        assert (tmp_path / "subdir").exists()

    def test_get_parquet_path(self):
        """Test parquet path conversion."""
        path = Path("data/file.csv")
        result = get_parquet_path(path)
        assert result == Path("data/file.parquet")

    def test_load_and_validate_dataframe_csv(self, tmp_path):
        """Test loading and validating CSV."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "test.csv"
        df.to_csv(path, index=False)

        loaded = load_and_validate_dataframe(path, required_columns=["a", "b"])
        pd.testing.assert_frame_equal(df, loaded)

    def test_load_and_validate_dataframe_missing_col(self, tmp_path):
        """Test validation failure."""
        df = pd.DataFrame({"a": [1, 2]})
        path = tmp_path / "test.csv"
        df.to_csv(path, index=False)

        with pytest.raises(ValueError, match="Missing required columns"):
            load_and_validate_dataframe(path, required_columns=["a", "b"])

    def test_load_dataframe_parquet_dates(self, tmp_path):
        """Test loading parquet with date parsing."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "value": [1, 2]
        })
        path = tmp_path / "test.parquet"
        df.to_parquet(path)

        loaded = load_dataframe(path, date_columns=["date"])
        assert pd.api.types.is_datetime64_any_dtype(loaded["date"])
        pd.testing.assert_frame_equal(df, loaded)

    def test_load_json_data(self, tmp_path):
        """Test loading JSON."""
        data = {"key": "value"}
        path = tmp_path / "test.json"
        with open(path, "w") as f:
            json.dump(data, f)

        loaded = load_json_data(path)
        assert loaded == data

    def test_save_json_pretty(self, tmp_path):
        """Test saving pretty JSON."""
        data = {"key": "value"}
        path = tmp_path / "output.json"
        save_json_pretty(data, path)

        with open(path, "r") as f:
            content = f.read()
            assert "\n" in content  # Pretty printed

    def test_save_parquet_and_csv(self, tmp_path):
        """Test saving parquet (CSV is commented out in source)."""
        df = pd.DataFrame({"a": [1, 2]})
        parquet_path = tmp_path / "test.parquet"

        save_parquet_and_csv(df, parquet_path)

        assert parquet_path.exists()
        assert not (tmp_path / "test.csv").exists() # Should not exist as per source code

    def test_write_placeholder_file(self, tmp_path):
        """Test writing placeholder."""
        path = tmp_path / "placeholder.txt"
        write_placeholder_file(path, "content")
        assert path.read_text(encoding="utf-8") == "content"
