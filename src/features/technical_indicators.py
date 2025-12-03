"""Technical Analysis indicators using the ta library.

This module computes classical technical indicators in bulk:

1. Momentum Indicators:
   - RSI (Relative Strength Index)
   - MACD (Moving Average Convergence Divergence)
   - Stochastic Oscillator
   - Williams %R
   - ROC (Rate of Change)

2. Volatility Indicators:
   - Bollinger Bands (upper, lower, width, %B)
   - ATR (Average True Range)
   - Keltner Channel

3. Trend Indicators:
   - SMA/EMA (various periods)
   - ADX (Average Directional Index)
   - Aroon Indicator
   - CCI (Commodity Channel Index)

4. Volume Indicators:
   - OBV (On-Balance Volume)
   - VWAP (Volume Weighted Average Price)
   - CMF (Chaikin Money Flow)
   - MFI (Money Flow Index)

Reference:
    Murphy, J. J. (1999). Technical Analysis of the Financial Markets.
    Pring, M. J. (2002). Technical Analysis Explained.
"""

from __future__ import annotations

import warnings
from typing import cast

import pandas as pd  # type: ignore[import-untyped]

from src.utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "compute_all_technical_indicators",
    "compute_momentum_indicators",
    "compute_volatility_indicators",
    "compute_trend_indicators",
    "compute_volume_indicators",
]


def _ensure_ta_installed() -> None:
    """Check that ta library is installed."""
    try:
        import ta  # type: ignore[import-untyped]  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "The 'ta' library is required for technical indicators. "
            "Install it with: pip install ta"
        ) from e


def compute_all_technical_indicators(
    df_bars: pd.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
    fillna: bool = False,
) -> pd.DataFrame:
    """Compute all technical indicators using the ta library.

    Uses ta.add_all_ta_features() to compute indicators in bulk.
    This includes momentum, volatility, trend, and volume indicators.

    Args:
        df_bars: DataFrame with OHLCV data.
        open_col: Name of open price column.
        high_col: Name of high price column.
        low_col: Name of low price column.
        close_col: Name of close price column.
        volume_col: Name of volume column.
        fillna: Whether to fill NaN values (default: False to preserve causality).

    Returns:
        DataFrame with all technical indicator columns (original columns excluded).

    Example:
        >>> df_ta = compute_all_technical_indicators(df_bars)
        >>> df_bars = pd.concat([df_bars, df_ta], axis=1)
    """
    _ensure_ta_installed()
    import ta

    # Validate required columns
    required_cols = [open_col, high_col, low_col, close_col, volume_col]
    missing_cols = [c for c in required_cols if c not in df_bars.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    logger.info("Computing all technical indicators with ta library...")

    # Create a copy to avoid modifying original
    df_work = cast(
        pd.DataFrame,
        df_bars[[open_col, high_col, low_col, close_col, volume_col]].copy(),
    )

    # Capture original columns before modification (ta library modifies in-place)
    original_cols = set(df_work.columns)

    # Suppress warnings from ta library (division by zero, etc.)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        df_with_ta = ta.add_all_ta_features(
            df_work,
            open=open_col,
            high=high_col,
            low=low_col,
            close=close_col,
            volume=volume_col,
            fillna=fillna,
        )

    # Extract only the new TA columns (exclude original OHLCV)
    ta_cols = [c for c in df_with_ta.columns if c not in original_cols]

    # Exclude PSAR columns (psar_up/psar_down have structural NaN by design)
    psar_cols = [c for c in ta_cols if "psar" in c.lower()]
    if psar_cols:
        logger.info("Excluding %d PSAR columns (structural NaN): %s", len(psar_cols), psar_cols)
        ta_cols = [c for c in ta_cols if c not in psar_cols]

    result = cast(pd.DataFrame, df_with_ta[ta_cols].copy())
    result.index = df_bars.index

    # Add prefix to all columns for clarity
    result.columns = [f"ta_{c}" for c in result.columns]

    logger.info(
        "Computed %d technical indicators. Categories: momentum, volatility, trend, volume",
        len(result.columns),
    )

    # Log sample of indicators
    sample_indicators = list(result.columns)[:10]
    logger.info("Sample indicators: %s", sample_indicators)

    return result


def compute_momentum_indicators(
    df_bars: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
    fillna: bool = False,
) -> pd.DataFrame:
    """Compute momentum indicators only.

    Includes:
    - RSI (14-period)
    - MACD (12, 26, 9)
    - Stochastic Oscillator
    - Williams %R
    - ROC
    - Awesome Oscillator
    - Ultimate Oscillator

    Args:
        df_bars: DataFrame with OHLCV data.
        high_col: Name of high price column.
        low_col: Name of low price column.
        close_col: Name of close price column.
        volume_col: Name of volume column.
        fillna: Whether to fill NaN values.

    Returns:
        DataFrame with momentum indicator columns.
    """
    _ensure_ta_installed()
    from ta import momentum
    from ta.trend import MACD  # type: ignore[import-untyped]

    logger.info("Computing momentum indicators...")

    result = pd.DataFrame(index=df_bars.index)

    close = cast(pd.Series, df_bars[close_col])
    high = cast(pd.Series, df_bars[high_col])
    low = cast(pd.Series, df_bars[low_col])
    volume = cast(pd.Series, df_bars[volume_col])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # RSI
        result["ta_rsi_14"] = momentum.rsi(close, window=14, fillna=fillna)
        result["ta_rsi_7"] = momentum.rsi(close, window=7, fillna=fillna)

        # MACD
        macd = MACD(close, window_slow=26, window_fast=12, window_sign=9, fillna=fillna)
        result["ta_macd"] = macd.macd()
        result["ta_macd_signal"] = macd.macd_signal()
        result["ta_macd_diff"] = macd.macd_diff()

        # Stochastic
        stoch = momentum.StochasticOscillator(
            high, low, close, window=14, smooth_window=3, fillna=fillna
        )
        result["ta_stoch_k"] = stoch.stoch()
        result["ta_stoch_d"] = stoch.stoch_signal()

        # Williams %R
        result["ta_williams_r"] = momentum.williams_r(high, low, close, lbp=14, fillna=fillna)

        # ROC
        result["ta_roc_10"] = momentum.roc(close, window=10, fillna=fillna)
        result["ta_roc_20"] = momentum.roc(close, window=20, fillna=fillna)

        # Awesome Oscillator
        result["ta_ao"] = momentum.awesome_oscillator(high, low, window1=5, window2=34, fillna=fillna)

        # Ultimate Oscillator
        result["ta_uo"] = momentum.ultimate_oscillator(
            high, low, close, window1=7, window2=14, window3=28, fillna=fillna
        )

        # PPO
        result["ta_ppo"] = momentum.ppo(close, window_slow=26, window_fast=12, fillna=fillna)

        # MFI (Money Flow Index - combines price and volume)
        result["ta_mfi"] = volume_ta_mfi(high, low, close, volume, fillna=fillna)

    logger.info("Computed %d momentum indicators", len(result.columns))
    return result


def volume_ta_mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    fillna: bool = False,
) -> pd.Series:
    """Compute Money Flow Index."""
    from ta import volume as vol

    return vol.money_flow_index(high, low, close, volume, window=14, fillna=fillna)


def compute_volatility_indicators(
    df_bars: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    fillna: bool = False,
) -> pd.DataFrame:
    """Compute volatility indicators only.

    Includes:
    - Bollinger Bands (upper, lower, width, %B)
    - ATR (Average True Range)
    - Keltner Channel
    - Donchian Channel

    Args:
        df_bars: DataFrame with OHLC data.
        high_col: Name of high price column.
        low_col: Name of low price column.
        close_col: Name of close price column.
        fillna: Whether to fill NaN values.

    Returns:
        DataFrame with volatility indicator columns.
    """
    _ensure_ta_installed()
    from ta import volatility

    logger.info("Computing volatility indicators...")

    result = pd.DataFrame(index=df_bars.index)

    close = cast(pd.Series, df_bars[close_col])
    high = cast(pd.Series, df_bars[high_col])
    low = cast(pd.Series, df_bars[low_col])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Bollinger Bands
        bb = volatility.BollingerBands(close, window=20, window_dev=2, fillna=fillna)
        result["ta_bb_upper"] = bb.bollinger_hband()
        result["ta_bb_lower"] = bb.bollinger_lband()
        result["ta_bb_middle"] = bb.bollinger_mavg()
        result["ta_bb_width"] = bb.bollinger_wband()
        result["ta_bb_pband"] = bb.bollinger_pband()  # %B

        # ATR
        result["ta_atr_14"] = volatility.average_true_range(high, low, close, window=14, fillna=fillna)
        result["ta_atr_7"] = volatility.average_true_range(high, low, close, window=7, fillna=fillna)

        # Keltner Channel
        kc = volatility.KeltnerChannel(high, low, close, window=20, window_atr=10, fillna=fillna)
        result["ta_kc_upper"] = kc.keltner_channel_hband()
        result["ta_kc_lower"] = kc.keltner_channel_lband()
        result["ta_kc_middle"] = kc.keltner_channel_mband()
        result["ta_kc_width"] = kc.keltner_channel_wband()
        result["ta_kc_pband"] = kc.keltner_channel_pband()

        # Donchian Channel
        dc = volatility.DonchianChannel(high, low, close, window=20, fillna=fillna)
        result["ta_dc_upper"] = dc.donchian_channel_hband()
        result["ta_dc_lower"] = dc.donchian_channel_lband()
        result["ta_dc_middle"] = dc.donchian_channel_mband()
        result["ta_dc_width"] = dc.donchian_channel_wband()
        result["ta_dc_pband"] = dc.donchian_channel_pband()

        # Ulcer Index
        result["ta_ulcer_idx"] = volatility.ulcer_index(close, window=14, fillna=fillna)

    logger.info("Computed %d volatility indicators", len(result.columns))
    return result


def compute_trend_indicators(
    df_bars: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    fillna: bool = False,
) -> pd.DataFrame:
    """Compute trend indicators only.

    Includes:
    - SMA (various periods)
    - EMA (various periods)
    - ADX
    - Aroon
    - CCI
    - TRIX
    - Vortex Indicator

    Args:
        df_bars: DataFrame with OHLC data.
        high_col: Name of high price column.
        low_col: Name of low price column.
        close_col: Name of close price column.
        fillna: Whether to fill NaN values.

    Returns:
        DataFrame with trend indicator columns.
    """
    _ensure_ta_installed()
    from ta import trend

    logger.info("Computing trend indicators...")

    result = pd.DataFrame(index=df_bars.index)

    close = cast(pd.Series, df_bars[close_col])
    high = cast(pd.Series, df_bars[high_col])
    low = cast(pd.Series, df_bars[low_col])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # SMA
        result["ta_sma_7"] = trend.sma_indicator(close, window=7, fillna=fillna)
        result["ta_sma_14"] = trend.sma_indicator(close, window=14, fillna=fillna)
        result["ta_sma_20"] = trend.sma_indicator(close, window=20, fillna=fillna)
        result["ta_sma_50"] = trend.sma_indicator(close, window=50, fillna=fillna)

        # EMA
        result["ta_ema_7"] = trend.ema_indicator(close, window=7, fillna=fillna)
        result["ta_ema_14"] = trend.ema_indicator(close, window=14, fillna=fillna)
        result["ta_ema_20"] = trend.ema_indicator(close, window=20, fillna=fillna)
        result["ta_ema_50"] = trend.ema_indicator(close, window=50, fillna=fillna)

        # ADX
        adx = trend.ADXIndicator(high, low, close, window=14, fillna=fillna)
        result["ta_adx"] = adx.adx()
        result["ta_adx_pos"] = adx.adx_pos()
        result["ta_adx_neg"] = adx.adx_neg()

        # Aroon
        aroon = trend.AroonIndicator(high, low, window=25, fillna=fillna)
        result["ta_aroon_up"] = aroon.aroon_up()
        result["ta_aroon_down"] = aroon.aroon_down()
        result["ta_aroon_ind"] = aroon.aroon_indicator()

        # CCI
        result["ta_cci_14"] = trend.cci(high, low, close, window=14, fillna=fillna)
        result["ta_cci_20"] = trend.cci(high, low, close, window=20, fillna=fillna)

        # TRIX
        result["ta_trix"] = trend.trix(close, window=15, fillna=fillna)

        # Vortex Indicator
        vortex = trend.VortexIndicator(high, low, close, window=14, fillna=fillna)
        result["ta_vortex_pos"] = vortex.vortex_indicator_pos()
        result["ta_vortex_neg"] = vortex.vortex_indicator_neg()
        result["ta_vortex_diff"] = vortex.vortex_indicator_diff()

        # Mass Index
        result["ta_mass_idx"] = trend.mass_index(high, low, window_fast=9, window_slow=25, fillna=fillna)

        # DPO
        result["ta_dpo"] = trend.dpo(close, window=20, fillna=fillna)

        # Ichimoku
        ich = trend.IchimokuIndicator(high, low, window1=9, window2=26, window3=52, fillna=fillna)
        result["ta_ichimoku_a"] = ich.ichimoku_a()
        result["ta_ichimoku_b"] = ich.ichimoku_b()
        result["ta_ichimoku_base"] = ich.ichimoku_base_line()
        result["ta_ichimoku_conv"] = ich.ichimoku_conversion_line()

    logger.info("Computed %d trend indicators", len(result.columns))
    return result


def compute_volume_indicators(
    df_bars: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
    fillna: bool = False,
) -> pd.DataFrame:
    """Compute volume indicators only.

    Includes:
    - OBV (On-Balance Volume)
    - VWAP
    - CMF (Chaikin Money Flow)
    - ADI (Accumulation/Distribution Index)
    - EMV (Ease of Movement)
    - Force Index

    Args:
        df_bars: DataFrame with OHLCV data.
        high_col: Name of high price column.
        low_col: Name of low price column.
        close_col: Name of close price column.
        volume_col: Name of volume column.
        fillna: Whether to fill NaN values.

    Returns:
        DataFrame with volume indicator columns.
    """
    _ensure_ta_installed()
    from ta import volume

    logger.info("Computing volume indicators...")

    result = pd.DataFrame(index=df_bars.index)

    close = cast(pd.Series, df_bars[close_col])
    high = cast(pd.Series, df_bars[high_col])
    low = cast(pd.Series, df_bars[low_col])
    vol = cast(pd.Series, df_bars[volume_col])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # OBV
        result["ta_obv"] = volume.on_balance_volume(close, vol, fillna=fillna)

        # CMF
        result["ta_cmf"] = volume.chaikin_money_flow(high, low, close, vol, window=20, fillna=fillna)

        # ADI (Accumulation/Distribution Index)
        result["ta_adi"] = volume.acc_dist_index(high, low, close, vol, fillna=fillna)

        # EMV (Ease of Movement)
        emv = volume.EaseOfMovementIndicator(high, low, vol, window=14, fillna=fillna)
        result["ta_emv"] = emv.ease_of_movement()
        result["ta_emv_sma"] = emv.sma_ease_of_movement()

        # Force Index
        result["ta_fi_13"] = volume.force_index(close, vol, window=13, fillna=fillna)

        # MFI
        result["ta_mfi_14"] = volume.money_flow_index(high, low, close, vol, window=14, fillna=fillna)

        # NVI (Negative Volume Index)
        result["ta_nvi"] = volume.negative_volume_index(close, vol, fillna=fillna)

        # VWAP - compute manually since ta doesn't have it directly
        # VWAP = cumsum(price * volume) / cumsum(volume)
        typical_price = (high + low + close) / 3
        result["ta_vwap"] = (typical_price * vol).cumsum() / vol.cumsum()

    logger.info("Computed %d volume indicators", len(result.columns))
    return result
