
import pytest
import pandas as pd
import numpy as np
from src.features.zscore_normalizer import compute_rolling_zscore

def test_compute_rolling_zscore(sample_bars):
    df = pd.DataFrame({
        "A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })

    # window=4
    # 1,2,3,4 -> mean=2.5, std=1.29 -> 4 -> (4-2.5)/1.29 = 1.16
    df_z = compute_rolling_zscore(df, window=4, min_periods=4)

    assert "A_zscore" in df_z.columns
    # First 3 should be NaN
    assert df_z["A_zscore"].iloc[:3].isna().all()
    # 4th should be valid
    assert not np.isnan(df_z["A_zscore"].iloc[3])
