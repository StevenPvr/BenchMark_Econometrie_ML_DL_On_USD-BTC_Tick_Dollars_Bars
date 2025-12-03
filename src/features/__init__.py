"""Features module for computing trading features from tick data.

Available features:
- Volume Imbalance (VI): Order flow toxicity proxy
- VPIN: Volume-Synchronized Probability of Informed Trading
- Kyle's Lambda: Price impact coefficient
- Temporal Acceleration: Bar formation speed dynamics
- Entropy: Shannon, Approximate, and Sample entropy for regime detection
- Momentum: Cumulative returns and recent extremes
- Realized Volatility: Historical volatility and return-volatility ratio
- Realized Higher Moments: Skewness and kurtosis
- Jump Detection: Bipower variation, jump component (De Prado/Barndorff-Nielsen)
- Trend: Moving averages, z-score, cross MA, return streak
- Range Volatility: Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang
- Microstructure Volatility: Intrabar variance/range from tick data
- Temporal/Calendar: Cyclical time encoding, time since shock, vol regime, drawdown
- Z-score Normalization: Rolling z-score for all features (for linear models)
- Fractional Differentiation: Long memory features (De Prado FFD)
- Lag Generation: Intelligent lag structure by feature type
- Scalers: StandardScaler and MinMaxScaler (fit on train, transform both)
"""

from src.features.entropy import (
    compute_approximate_entropy,
    compute_sample_entropy,
    compute_shannon_entropy,
)
from src.features.fractional_diff import (
    compute_frac_diff,
    compute_frac_diff_features,
    compute_frac_diff_ffd,
    compute_frac_diff_weights,
    find_min_frac_diff_order,
)
from src.features.kyle_lambda import compute_kyle_lambda
from src.features.lag_generator import (
    classify_feature_type,
    compute_lagged_features,
    generate_all_lags,
    get_lags_for_feature,
    summarize_lag_structure,
)
from src.features.microstructure_volatility import (
    compute_intrabar_volatility,
    compute_microstructure_features,
)
from src.features.momentum import (
    compute_cumulative_returns,
    compute_recent_extremes,
)
from src.features.range_volatility import (
    compute_garman_klass_volatility,
    compute_parkinson_volatility,
    compute_range_ratios,
    compute_rogers_satchell_volatility,
    compute_yang_zhang_volatility,
)
from src.features.realized_volatility import (
    compute_local_sharpe,  # Deprecated, use compute_return_volatility_ratio
    compute_return_volatility_ratio,
    compute_realized_volatility,
    compute_realized_skewness,
    compute_realized_kurtosis,
)
from src.features.jump_detection import (
    compute_bipower_variation,
    compute_jump_component,
    compute_all_jump_features,
)
from src.features.scalers import (
    MinMaxScalerCustom,
    ScalerManager,
    StandardScalerCustom,
    fit_and_transform_features,
)
from src.features.temporal_acceleration import (
    compute_temporal_acceleration,
    compute_temporal_acceleration_smoothed,
    compute_temporal_jerk,
)
from src.features.temporal_calendar import (
    compute_all_temporal_features,
    compute_cyclical_time_features,
    compute_drawdown_features,
    compute_time_since_shock,
    compute_volatility_regime,
)
from src.features.trade_classification import (
    classify_trades_direct,
    classify_trades_tick_rule,
)
from src.features.trend import (
    compute_cross_ma,
    compute_moving_averages,
    compute_price_zscore,
    compute_return_streak,
)
from src.features.zscore_normalizer import (
    compute_all_features_zscore,
    compute_rolling_zscore,
    save_zscore_features,
)
from src.features.volume_imbalance import (
    compute_volume_imbalance,
    compute_volume_imbalance_bars,
)
from src.features.vpin import compute_vpin

# Pipeline utilities
from src.features.pipeline import (
    compute_timestamp_features,
    drop_empty_columns,
    drop_initial_nan_rows,
    drop_timestamp_columns,
    get_columns_to_scale,
    interpolate_sporadic_nan,
    shift_target_to_future_return,
    split_train_test,
    TRAIN_RATIO,
)
from src.features.compute import apply_lags, compute_all_features
from src.features.io import (
    fit_and_save_scalers,
    load_input_data,
    save_outputs,
    save_train_test_splits,
)
from src.features.batch import (
    BATCH_OVERLAP,
    DEFAULT_BATCH_SIZE,
    run_batch_pipeline,
)

__all__ = [
    # Trade classification
    "classify_trades_tick_rule",
    "classify_trades_direct",
    # Volume Imbalance
    "compute_volume_imbalance",
    "compute_volume_imbalance_bars",
    # VPIN
    "compute_vpin",
    # Kyle's Lambda
    "compute_kyle_lambda",
    # Temporal Acceleration
    "compute_temporal_acceleration",
    "compute_temporal_acceleration_smoothed",
    "compute_temporal_jerk",
    # Entropy
    "compute_shannon_entropy",
    "compute_approximate_entropy",
    "compute_sample_entropy",
    # Momentum
    "compute_cumulative_returns",
    "compute_recent_extremes",
    # Realized Volatility & Higher Moments
    "compute_realized_volatility",
    "compute_return_volatility_ratio",
    "compute_local_sharpe",  # Deprecated alias
    "compute_realized_skewness",
    "compute_realized_kurtosis",
    # Jump Detection
    "compute_bipower_variation",
    "compute_jump_component",
    "compute_all_jump_features",
    # Trend
    "compute_moving_averages",
    "compute_price_zscore",
    "compute_cross_ma",
    "compute_return_streak",
    # Range Volatility
    "compute_parkinson_volatility",
    "compute_garman_klass_volatility",
    "compute_rogers_satchell_volatility",
    "compute_yang_zhang_volatility",
    "compute_range_ratios",
    # Microstructure Volatility
    "compute_intrabar_volatility",
    "compute_microstructure_features",
    # Temporal/Calendar/Regime
    "compute_cyclical_time_features",
    "compute_time_since_shock",
    "compute_volatility_regime",
    "compute_drawdown_features",
    "compute_all_temporal_features",
    # Z-score Normalization
    "compute_rolling_zscore",
    "compute_all_features_zscore",
    "save_zscore_features",
    # Fractional Differentiation
    "compute_frac_diff_weights",
    "compute_frac_diff",
    "compute_frac_diff_ffd",
    "find_min_frac_diff_order",
    "compute_frac_diff_features",
    # Lag Generation
    "classify_feature_type",
    "get_lags_for_feature",
    "compute_lagged_features",
    "generate_all_lags",
    "summarize_lag_structure",
    # Scalers
    "StandardScalerCustom",
    "MinMaxScalerCustom",
    "ScalerManager",
    "fit_and_transform_features",
    # Pipeline utilities
    "compute_timestamp_features",
    "drop_empty_columns",
    "drop_initial_nan_rows",
    "drop_timestamp_columns",
    "get_columns_to_scale",
    "interpolate_sporadic_nan",
    "shift_target_to_future_return",
    "split_train_test",
    "TRAIN_RATIO",
    # Compute
    "apply_lags",
    "compute_all_features",
    # I/O
    "fit_and_save_scalers",
    "load_input_data",
    "save_outputs",
    "save_train_test_splits",
    # Batch
    "BATCH_OVERLAP",
    "DEFAULT_BATCH_SIZE",
    "run_batch_pipeline",
]
