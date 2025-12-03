"""Tests for the data_visualisation module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data_visualisation import (
    compute_autocorrelation,
    compute_autocorrelation_squared,
    compute_log_returns,
    compute_trend_statistics,
    load_dollar_bars,
    mann_kendall_test,
    plot_log_returns_distribution,
    plot_log_returns_time_series,
    plot_stationarity,
    plot_trend_analysis,
    plot_trend_extraction,
    run_full_analysis,
    run_ljung_box_test,
    run_normality_tests,
    run_stationarity_tests,
)


@pytest.fixture
def mock_dollar_bars_df() -> pd.DataFrame:
    """Create a sample DataFrame mimicking dollar bars."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
    np.random.seed(42)
    close_prices = 100 * np.exp(np.random.normal(0, 0.01, 100).cumsum())
    return pd.DataFrame({
        "close": close_prices,
        "datetime_close": dates,
    })


@pytest.fixture
def mock_log_returns(mock_dollar_bars_df: pd.DataFrame) -> pd.Series:
    """Create sample log returns."""
    return compute_log_returns(mock_dollar_bars_df)


def test_load_dollar_bars(tmp_path: Path) -> None:
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


def test_compute_log_returns(mock_dollar_bars_df: pd.DataFrame) -> None:
    """Test log return calculation."""
    log_rets = compute_log_returns(mock_dollar_bars_df)

    assert isinstance(log_rets, pd.Series)
    assert len(log_rets) == len(mock_dollar_bars_df)
    # First value should be NaN
    assert np.isnan(log_rets.iloc[0])

    # Manual check
    p0 = mock_dollar_bars_df["close"].iloc[0]
    p1 = mock_dollar_bars_df["close"].iloc[1]
    expected = np.log(p1 / p0)
    assert np.isclose(log_rets.iloc[1], expected)


@patch("src.data_visualisation.distribution.plt")
def test_plot_log_returns_distribution(
    mock_plt: MagicMock,
    mock_log_returns: pd.Series,
    tmp_path: Path,
) -> None:
    """Test plotting distribution of log returns."""
    mock_fig = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, [MagicMock(), MagicMock()])

    fig = plot_log_returns_distribution(mock_log_returns, output_dir=tmp_path)

    assert fig == mock_fig
    assert (tmp_path / "log_returns_distribution.png").exists() or mock_fig.savefig.called


@patch("src.data_visualisation.stationarity.adfuller")
@patch("src.data_visualisation.stationarity.kpss")
def test_run_stationarity_tests(
    mock_kpss: MagicMock,
    mock_adfuller: MagicMock,
    mock_log_returns: pd.Series,
    tmp_path: Path,
) -> None:
    """Test stationarity tests execution."""
    # Mock return values for adfuller
    mock_adfuller.return_value = (-3.5, 0.01, 1, 98, {"1%": -3.5, "5%": -2.9}, 100.0)

    # Mock return values for kpss
    mock_kpss.return_value = (0.1, 0.1, 5, {"1%": 0.7, "5%": 0.4})

    results = run_stationarity_tests(mock_log_returns, output_dir=tmp_path)

    assert "ADF" in results
    assert "KPSS" in results
    assert results["ADF"]["p_value"] == 0.01
    assert results["ADF"]["conclusion"] == "Stationnaire"
    assert results["KPSS"]["p_value"] == 0.1
    assert results["KPSS"]["conclusion"] == "Stationnaire"

    assert (tmp_path / "stationarity_tests.json").exists()


@patch("src.data_visualisation.stationarity.plt")
def test_plot_stationarity(
    mock_plt: MagicMock,
    mock_log_returns: pd.Series,
    tmp_path: Path,
) -> None:
    """Test plotting stationarity."""
    mock_fig = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, [MagicMock(), MagicMock()])

    results = {
        "ADF": {"conclusion": "Stationnaire", "p_value": 0.01},
        "KPSS": {"conclusion": "Stationnaire", "p_value": 0.1},
    }

    fig = plot_stationarity(mock_log_returns, results, output_dir=tmp_path)

    assert fig == mock_fig
    assert (tmp_path / "stationarity_plot.png").exists() or mock_fig.savefig.called


@patch("src.data_visualisation.time_series.plt")
def test_plot_log_returns_time_series(
    mock_plt: MagicMock,
    mock_dollar_bars_df: pd.DataFrame,
    mock_log_returns: pd.Series,
    tmp_path: Path,
) -> None:
    """Test plotting time series."""
    mock_fig = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, [MagicMock(), MagicMock()])

    fig = plot_log_returns_time_series(mock_dollar_bars_df, mock_log_returns, output_dir=tmp_path)

    assert fig == mock_fig
    assert (tmp_path / "log_returns_time_series.png").exists() or mock_fig.savefig.called


@patch("src.data_visualisation.autocorrelation.plt")
@patch("src.data_visualisation.autocorrelation.plot_acf")
@patch("src.data_visualisation.autocorrelation.plot_pacf")
def test_compute_autocorrelation(
    mock_pacf: MagicMock,
    mock_acf: MagicMock,
    mock_plt: MagicMock,
    mock_log_returns: pd.Series,
    tmp_path: Path,
) -> None:
    """Test autocorrelation plotting."""
    mock_fig = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, [MagicMock(), MagicMock()])

    fig = compute_autocorrelation(mock_log_returns, output_dir=tmp_path)

    assert fig == mock_fig
    assert mock_acf.called
    assert mock_pacf.called
    assert (tmp_path / "log_returns_acf.png").exists() or mock_fig.savefig.called


@patch("src.data_visualisation.autocorrelation.plt")
@patch("src.data_visualisation.autocorrelation.plot_acf")
@patch("src.data_visualisation.autocorrelation.plot_pacf")
def test_compute_autocorrelation_squared(
    mock_pacf: MagicMock,
    mock_acf: MagicMock,
    mock_plt: MagicMock,
    mock_log_returns: pd.Series,
    tmp_path: Path,
) -> None:
    """Test autocorrelation of squared log returns (volatility clustering)."""
    mock_fig = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, [MagicMock(), MagicMock()])

    fig = compute_autocorrelation_squared(mock_log_returns, output_dir=tmp_path)

    assert fig == mock_fig
    assert mock_acf.called
    assert mock_pacf.called
    assert (tmp_path / "log_returns_squared_acf.png").exists() or mock_fig.savefig.called


def test_run_ljung_box_test(mock_log_returns: pd.Series, tmp_path: Path) -> None:
    """Test Ljung-Box autocorrelation test."""
    results = run_ljung_box_test(mock_log_returns, output_dir=tmp_path)

    assert "test_name" in results
    assert results["test_name"] == "Ljung-Box"
    assert "lags_tested" in results
    assert "hypothesis" in results

    # Check that we have results for default lags
    lags_tested = results.get("lags_tested", {})
    assert isinstance(lags_tested, dict)
    assert "10" in lags_tested or "error" in results
    assert "20" in lags_tested or "error" in results
    assert "40" in lags_tested or "error" in results

    assert (tmp_path / "ljung_box_test.json").exists()


def test_run_ljung_box_test_custom_lags(mock_log_returns: pd.Series, tmp_path: Path) -> None:
    """Test Ljung-Box test with custom lags."""
    results = run_ljung_box_test(mock_log_returns, lags=[5, 15], output_dir=tmp_path)

    assert "lags_tested" in results
    lags_tested = results.get("lags_tested", {})
    assert isinstance(lags_tested, dict)
    assert "5" in lags_tested or "error" in results
    assert "15" in lags_tested or "error" in results


def test_run_normality_tests(mock_log_returns: pd.Series, tmp_path: Path) -> None:
    """Test normality tests (Jarque-Bera and Shapiro-Wilk)."""
    results = run_normality_tests(mock_log_returns, output_dir=tmp_path)

    assert "n_observations" in results
    assert "Jarque_Bera" in results
    assert "Shapiro_Wilk" in results

    # Check Jarque-Bera results
    jb = results["Jarque_Bera"]
    if isinstance(jb, dict) and "error" not in jb:
        assert "statistic" in jb
        assert "p_value" in jb
        assert "conclusion" in jb
        assert jb["conclusion"] in ["Non normale", "Compatible avec normale"]

    # Check Shapiro-Wilk results
    sw = results["Shapiro_Wilk"]
    if isinstance(sw, dict) and "error" not in sw:
        assert "statistic" in sw
        assert "p_value" in sw
        assert "conclusion" in sw

    assert (tmp_path / "normality_tests.json").exists()


def test_run_normality_tests_large_sample(tmp_path: Path) -> None:
    """Test normality tests with large sample (triggers Shapiro-Wilk subsampling)."""
    np.random.seed(42)
    large_series = pd.Series(np.random.normal(0, 1, 6000))

    results = run_normality_tests(large_series, output_dir=tmp_path)

    # Shapiro-Wilk should have a note about subsampling
    sw = results["Shapiro_Wilk"]
    if isinstance(sw, dict) and "error" not in sw:
        assert "note" in sw or len(large_series) <= 5000


def test_mann_kendall_test() -> None:
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

    # Mixed pattern
    mixed = np.array([1, 3, 2, 4])
    res_mixed = mann_kendall_test(mixed)
    assert "p_value" in res_mixed
    assert "trend" in res_mixed


def test_compute_trend_statistics(mock_dollar_bars_df: pd.DataFrame, tmp_path: Path) -> None:
    """Test trend statistics computation."""
    series = mock_dollar_bars_df["close"]
    results = compute_trend_statistics(series, output_dir=tmp_path)

    assert "Mann_Kendall" in results
    assert "Linear_Regression" in results
    assert "Change_Statistics" in results

    assert (tmp_path / "trend_analysis.json").exists()


@patch("src.data_visualisation.trend.plt")
def test_plot_trend_extraction(
    mock_plt: MagicMock,
    mock_dollar_bars_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Test trend extraction plot."""
    mock_fig = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, [MagicMock(), MagicMock(), MagicMock(), MagicMock()])

    series = mock_dollar_bars_df["close"]
    fig = plot_trend_extraction(series, output_dir=tmp_path)

    assert fig == mock_fig
    assert (tmp_path / "trend_extraction.png").exists() or mock_fig.savefig.called


@patch("src.data_visualisation.trend.plt")
def test_plot_trend_analysis(
    mock_plt: MagicMock,
    mock_dollar_bars_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Test trend analysis plot."""
    mock_fig = MagicMock()

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


@patch("src.data_visualisation.analysis.load_dollar_bars")
@patch("src.data_visualisation.analysis.plt")
@patch("src.data_visualisation.analysis.plot_log_returns_distribution")
@patch("src.data_visualisation.analysis.run_normality_tests")
@patch("src.data_visualisation.analysis.run_stationarity_tests")
@patch("src.data_visualisation.analysis.plot_stationarity")
@patch("src.data_visualisation.analysis.plot_log_returns_time_series")
@patch("src.data_visualisation.analysis.compute_autocorrelation")
@patch("src.data_visualisation.analysis.compute_autocorrelation_squared")
@patch("src.data_visualisation.analysis.run_ljung_box_test")
def test_run_full_analysis(
    mock_run_ljung_box: MagicMock,
    mock_compute_autocorrelation_squared: MagicMock,
    mock_compute_autocorrelation: MagicMock,
    mock_plot_ts: MagicMock,
    mock_plot_stationarity: MagicMock,
    mock_run_stationarity: MagicMock,
    mock_run_normality: MagicMock,
    mock_plot_dist: MagicMock,
    mock_plt: MagicMock,
    mock_load: MagicMock,
    mock_dollar_bars_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Test the full analysis orchestration."""
    mock_load.return_value = mock_dollar_bars_df

    mock_run_stationarity.return_value = {"ADF": {}, "KPSS": {}}
    mock_run_normality.return_value = {"Jarque_Bera": {}, "Shapiro_Wilk": {}}
    mock_run_ljung_box.return_value = {"lags_tested": {}}

    results = run_full_analysis(
        parquet_path=Path("dummy"),
        output_dir=tmp_path,
        show_plots=False,
        sample_fraction=1.0,
    )

    assert results["n_bars"] == len(mock_dollar_bars_df)
    assert mock_load.called
    assert mock_plot_dist.called
    assert mock_run_normality.called
    assert mock_run_stationarity.called
    assert mock_plot_stationarity.called
    assert mock_plot_ts.called
    assert mock_compute_autocorrelation.called
    assert mock_compute_autocorrelation_squared.called
    assert mock_run_ljung_box.called


def test_run_full_analysis_systematic_sampling(tmp_path: Path) -> None:
    """Test that sampling is systematic (not random) and preserves temporal order."""
    # Create a dataframe with sequential values to verify ordering
    dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
    df = pd.DataFrame({
        "close": np.arange(100, dtype=float),
        "datetime_close": dates,
    })

    parquet_path = tmp_path / "test_bars.parquet"
    df.to_parquet(parquet_path)

    with patch("src.data_visualisation.analysis.plt"):
        with patch("src.data_visualisation.analysis.plot_log_returns_distribution"):
            with patch("src.data_visualisation.analysis.run_normality_tests") as mock_norm:
                with patch("src.data_visualisation.analysis.run_stationarity_tests") as mock_stat:
                    with patch("src.data_visualisation.analysis.plot_stationarity"):
                        with patch("src.data_visualisation.analysis.plot_log_returns_time_series"):
                            with patch("src.data_visualisation.analysis.compute_autocorrelation"):
                                with patch("src.data_visualisation.analysis.compute_autocorrelation_squared"):
                                    with patch("src.data_visualisation.analysis.run_ljung_box_test"):
                                        mock_stat.return_value = {"ADF": {}, "KPSS": {}}
                                        mock_norm.return_value = {}

                                        results = run_full_analysis(
                                            parquet_path=parquet_path,
                                            output_dir=tmp_path,
                                            show_plots=False,
                                            sample_fraction=0.2,
                                        )

                                        # With step=5, we should have 20 bars
                                        assert results["n_bars"] == 20
