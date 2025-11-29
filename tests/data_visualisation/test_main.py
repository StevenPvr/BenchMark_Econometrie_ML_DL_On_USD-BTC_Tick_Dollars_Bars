from unittest.mock import patch
import pytest
from src.data_visualisation.main import main

@patch("src.data_visualisation.main.run_full_analysis")
def test_main(mock_run_analysis):
    """Test the main entry point."""
    main()

    assert mock_run_analysis.called
    args, kwargs = mock_run_analysis.call_args

    # Check that it uses the correct default paths (imported from main)
    # Since we can't easily check the exact values of imported constants without importing them here too,
    # we just check arguments are passed.
    assert "parquet_path" in kwargs
    assert "output_dir" in kwargs
    assert kwargs["show_plots"] is True
    assert kwargs["sample_fraction"] == 0.2
