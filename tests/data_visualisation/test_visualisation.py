import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from src.data_visualisation.visualisation import (
    load_dollar_bars,
    compute_log_returns,
    plot_log_returns_distribution,
    run_stationarity_tests,
    plot_stationarity,
    plot_log_returns_time_series,
    compute_autocorrelation,
    mann_kendall_test,
    compute_trend_statistics,
    plot_trend_extraction,
    plot_trend_analysis,
    run_full_analysis,
    create_figure_canvas,
    save_canvas,
    add_zero_line,
    prepare_temporal_axis,
    plot_histogram_with_normal_overlay,
)

# Mocking constants just in case they are imported and cause issues,
# though they are imported inside functions or at top level.
# We will mock the module imports where necessary.

@pytest.fixture
def mock_dollar_bars_df():
    """Create a sample DataFrame mimicking dollar bars."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
    np.random.seed(42)
    close_prices = 100 * np.exp(np.random.normal(0, 0.01, 100).cumsum())
    df = pd.DataFrame({
        "close": close_prices,
        "datetime_close": dates
    })
    return df

@pytest.fixture
def mock_log_returns(mock_dollar_bars_df):
    """Create sample log returns."""
    return compute_log_returns(mock_dollar_bars_df)

def test_load_dollar_bars(tmp_path):
    """Test loading dollar bars from parquet."""
    # Create a dummy parquet file
    df = pd.DataFrame({"close": [1, 2, 3]})
    file_path = tmp_path / "test_bars.parquet"
    df.to_parquet(file_path)

    loaded_df = load_dollar_bars(file_path)
    pd.testing.assert_frame_equal(df, loaded_df)

    # Test file not found
    with pytest.raises(FileNotFoundError):
        load_dollar_bars(tmp_path / "non_existent.parquet")

def test_compute_log_returns(mock_dollar_bars_df):
    """Test log return calculation."""
    log_rets = compute_log_returns(mock_dollar_bars_df)

    assert isinstance(log_rets, pd.Series)
    assert len(log_rets) == len(mock_dollar_bars_df)
    # First value should be NaN or 0 (implementation gives NaN usually)
    assert np.isnan(log_rets.iloc[0])

    # Manual check
    p0 = mock_dollar_bars_df["close"].iloc[0]
    p1 = mock_dollar_bars_df["close"].iloc[1]
    expected = np.log(p1 / p0)
    assert np.isclose(log_rets.iloc[1], expected)

@patch("src.data_visualisation.visualisation.plt")
def test_plot_log_returns_distribution(mock_plt, mock_log_returns, tmp_path):
    """Test plotting distribution of log returns."""
    mock_fig = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, [MagicMock(), MagicMock()])

    # Run function
    fig = plot_log_returns_distribution(mock_log_returns, output_dir=tmp_path)

    assert fig == mock_fig
    # Check if saved
    assert (tmp_path / "log_returns_distribution.png").exists() or mock_fig.savefig.called

@patch("src.data_visualisation.visualisation.adfuller")
@patch("src.data_visualisation.visualisation.kpss")
def test_run_stationarity_tests(mock_kpss, mock_adfuller, mock_log_returns, tmp_path):
    """Test stationarity tests execution."""
    # Mock return values for adfuller
    # stat, pvalue, usedlag, nobs, critical values, icbest
    mock_adfuller.return_value = (-3.5, 0.01, 1, 98, {"1%": -3.5, "5%": -2.9}, 100.0)

    # Mock return values for kpss
    # stat, pvalue, lags, critical values
    mock_kpss.return_value = (0.1, 0.1, 5, {"1%": 0.7, "5%": 0.4})

    results = run_stationarity_tests(mock_log_returns, output_dir=tmp_path)

    assert "ADF" in results
    assert "KPSS" in results
    assert results["ADF"]["p_value"] == 0.01
    assert results["ADF"]["conclusion"] == "Stationnaire"
    assert results["KPSS"]["p_value"] == 0.1
    assert results["KPSS"]["conclusion"] == "Stationnaire"

    # Check if saved
    assert (tmp_path / "stationarity_tests.json").exists()

@patch("src.data_visualisation.visualisation.plt")
def test_plot_stationarity(mock_plt, mock_log_returns, tmp_path):
    """Test plotting stationarity."""
    mock_fig = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, [MagicMock(), MagicMock()])

    results = {
        "ADF": {"conclusion": "Stationnaire", "p_value": 0.01},
        "KPSS": {"conclusion": "Stationnaire", "p_value": 0.1}
    }

    fig = plot_stationarity(mock_log_returns, results, output_dir=tmp_path)

    assert fig == mock_fig
    assert (tmp_path / "stationarity_plot.png").exists() or mock_fig.savefig.called

@patch("src.data_visualisation.visualisation.plt")
def test_plot_log_returns_time_series(mock_plt, mock_dollar_bars_df, mock_log_returns, tmp_path):
    """Test plotting time series."""
    mock_fig = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, [MagicMock(), MagicMock()])

    fig = plot_log_returns_time_series(mock_dollar_bars_df, mock_log_returns, output_dir=tmp_path)

    assert fig == mock_fig
    assert (tmp_path / "log_returns_time_series.png").exists() or mock_fig.savefig.called

@patch("src.data_visualisation.visualisation.plt")
@patch("statsmodels.graphics.tsaplots.plot_acf")
@patch("statsmodels.graphics.tsaplots.plot_pacf")
def test_compute_autocorrelation(mock_pacf, mock_acf, mock_plt, mock_log_returns, tmp_path):
    """Test autocorrelation plotting."""
    mock_fig = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, [MagicMock(), MagicMock()])

    fig = compute_autocorrelation(mock_log_returns, output_dir=tmp_path)

    assert fig == mock_fig
    assert mock_acf.called
    assert mock_pacf.called
    assert (tmp_path / "log_returns_acf.png").exists() or mock_fig.savefig.called

def test_mann_kendall_test():
    """Test Mann-Kendall test logic."""
    # Monotonically increasing
    inc = np.array([1, 2, 3, 4, 5])
    res_inc = mann_kendall_test(inc)
    assert res_inc["statistic_S"] > 0
    assert "croissante" in res_inc["trend"]

    # Monotonically decreasing
    dec = np.array([5, 4, 3, 2, 1])
    res_dec = mann_kendall_test(dec)
    assert res_dec["statistic_S"] < 0
    assert "decroissante" in res_dec["trend"]

    # No trend (random)
    # This might fail randomly if we are unlucky, but with fixed seed or clear pattern it's fine.
    # 1, 3, 2, 4 is mixed.
    mixed = np.array([1, 3, 2, 4])
    res_mixed = mann_kendall_test(mixed)
    # S = (3-1) + (2-1) + (4-1) + (2-3) + (4-3) + (4-2)
    # Signs: +1, +1, +1, -1, +1, +1 = 4.
    # Let's just check structure.
    assert "p_value" in res_mixed
    assert "trend" in res_mixed

def test_compute_trend_statistics(mock_dollar_bars_df, tmp_path):
    """Test trend statistics computation."""
    series = mock_dollar_bars_df["close"]
    results = compute_trend_statistics(series, output_dir=tmp_path)

    assert "Mann_Kendall" in results
    assert "Linear_Regression" in results
    assert "Change_Statistics" in results

    assert (tmp_path / "trend_analysis.json").exists()

@patch("src.data_visualisation.visualisation.plt")
def test_plot_trend_extraction(mock_plt, mock_dollar_bars_df, tmp_path):
    """Test trend extraction plot."""
    mock_fig = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, [MagicMock(), MagicMock(), MagicMock(), MagicMock()])

    series = mock_dollar_bars_df["close"]
    fig = plot_trend_extraction(series, output_dir=tmp_path)

    assert fig == mock_fig
    assert (tmp_path / "trend_extraction.png").exists() or mock_fig.savefig.called

@patch("src.data_visualisation.visualisation.plt")
def test_plot_trend_analysis(mock_plt, mock_dollar_bars_df, tmp_path):
    """Test trend analysis plot."""
    mock_fig = MagicMock()

    # Creating a numpy array of mocks to simulate subplots(2, 2)
    # We use np.empty with dtype=object to avoid numpy trying to iterate over MagicMocks
    # which can cause issues (creating higher dim arrays)

    axes_array = np.empty((2, 2), dtype=object)
    axes_array[0, 0] = MagicMock()
    axes_array[0, 1] = MagicMock()
    axes_array[1, 0] = MagicMock()
    axes_array[1, 1] = MagicMock()

    mock_plt.subplots.return_value = (mock_fig, axes_array)

    series = mock_dollar_bars_df["close"]
    trend_results = compute_trend_statistics(series)

    fig = plot_trend_analysis(series, trend_results, output_dir=tmp_path)

    assert fig == mock_fig
    assert (tmp_path / "trend_analysis.png").exists() or mock_fig.savefig.called

@patch("src.data_visualisation.visualisation.load_dollar_bars")
@patch("src.data_visualisation.visualisation.plt")
@patch("src.data_visualisation.visualisation.plot_log_returns_distribution")
@patch("src.data_visualisation.visualisation.run_stationarity_tests")
@patch("src.data_visualisation.visualisation.plot_stationarity")
@patch("src.data_visualisation.visualisation.plot_log_returns_time_series")
@patch("src.data_visualisation.visualisation.compute_autocorrelation")
@patch("src.data_visualisation.visualisation.compute_trend_statistics")
@patch("src.data_visualisation.visualisation.plot_trend_analysis")
@patch("src.data_visualisation.visualisation.plot_trend_extraction")
def test_run_full_analysis(
    mock_plot_trend_extraction,
    mock_plot_trend_analysis,
    mock_compute_trend_statistics,
    mock_compute_autocorrelation,
    mock_plot_ts,
    mock_plot_stationarity,
    mock_run_stationarity,
    mock_plot_dist,
    mock_plt,
    mock_load,
    mock_dollar_bars_df,
    tmp_path
):
    """Test the full analysis orchestration."""
    mock_load.return_value = mock_dollar_bars_df

    # Setup mocks to return something valid
    mock_compute_trend_statistics.return_value = {"Linear_Regression": {}, "Mann_Kendall": {}}
    mock_run_stationarity.return_value = {"ADF": {}, "KPSS": {}}

    results = run_full_analysis(
        parquet_path=Path("dummy"),
        output_dir=tmp_path,
        show_plots=False,
        sample_fraction=1.0
    )

    assert results["n_bars"] == len(mock_dollar_bars_df)
    assert mock_load.called
    assert mock_plot_dist.called
    assert mock_run_stationarity.called
    assert mock_plot_stationarity.called
    assert mock_plot_ts.called
    assert mock_compute_autocorrelation.called
    assert mock_compute_trend_statistics.called
    assert mock_plot_trend_analysis.called
    assert mock_plot_trend_extraction.called

    # Test sampling
    run_full_analysis(
        parquet_path=Path("dummy"),
        output_dir=tmp_path,
        show_plots=False,
        sample_fraction=0.5
    )
    # Check that sample was called on df (indirectly by checking n_bars in result if we could,
    # but since we mocked load, the dataframe returned is constant, but the function should handle it).
    # Actually, run_full_analysis splits the DF.

    # If we want to verify sampling actually happened, we'd need to inspect the dataframe passed to functions.
    # But for now, ensuring no crash is good enough.

def test_garch_utils(tmp_path):
    """Test GARCH visualization utilities."""
    # create_figure_canvas
    fig, canvas, axes = create_figure_canvas()
    assert fig is not None

    # save_canvas
    mock_fig = MagicMock()
    save_canvas(mock_fig, tmp_path / "test.png")
    assert mock_fig.savefig.called

    # add_zero_line
    mock_ax = MagicMock()
    add_zero_line(mock_ax)
    assert mock_ax.axhline.called

    # prepare_temporal_axis
    dates = pd.date_range("2023-01-01", periods=10)
    axis = prepare_temporal_axis(dates)
    assert len(axis) == 10

    axis_fixed = prepare_temporal_axis(None, length=5)
    assert len(axis_fixed) == 5

    # plot_histogram_with_normal_overlay
    residuals = np.random.normal(0, 1, 100)

    # With implicit ax (using the imported plt from module if we wanted, but we imported it at top level here)
    # But wait, plot_histogram_with_normal_overlay calls plt.gca() if ax is None.
    # We should provide an ax to avoid relying on global state, or use our imported plt.

    fig, ax = plt.subplots()
    mean, std = plot_histogram_with_normal_overlay(residuals, ax=ax)
    plt.close(fig)

    assert isinstance(mean, float)
    assert isinstance(std, float)
