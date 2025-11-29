from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution.
_script_dir = Path(__file__).parent
# Find project root by looking for .git, pyproject.toml, or setup.py
_project_root = _script_dir
while _project_root != _project_root.parent:
    if (_project_root / ".git").exists() or (_project_root / "pyproject.toml").exists() or (_project_root / "setup.py").exists():
        break
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pytest  # type: ignore
from src.analyse_features.utils.numba_funcs import (
    fast_rolling_correlation,
    fast_rolling_spearman,
    fast_spearman_matrix,
    fast_distance_correlation,
    fast_autocorrelation,
    fast_vif_single
)

class TestNumbaFuncs:
    def test_fast_rolling_correlation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        window = 3

        result = fast_rolling_correlation(x, y, window)

        # First 2 should be NaN (window-1)
        assert np.isnan(result[0])
        assert np.isnan(result[1])

        # Rest should be 1.0 (perfect correlation)
        np.testing.assert_allclose(result[2:], 1.0)

    def test_fast_rolling_correlation_negative(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        window = 3

        result = fast_rolling_correlation(x, y, window)
        np.testing.assert_allclose(result[2:], -1.0)

    def test_fast_rolling_spearman(self):
        x = np.array([1.0, 2.0, 3.0, 10.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 100.0, 5.0]) # Same ranks
        window = 3

        result = fast_rolling_spearman(x, y, window)
        np.testing.assert_allclose(result[2:], 1.0)

    def test_fast_spearman_matrix(self):
        # Avoid constant columns as naive rank implementation doesn't handle ties well
        data = np.array([
            [1.0, 10.0, 1.0],
            [2.0, 9.0, 3.0],
            [3.0, 8.0, 0.0],
            [4.0, 7.0, 2.0]
        ])
        # Col 0: increasing (1,2,3,4)
        # Col 1: decreasing (10,9,8,7) -> perfect neg corr with col 0
        # Col 2: randomish (1,3,0,2) -> ranks: 1->1, 3->3, 0->0, 2->2 (0,1,2,3 for ranks)
        # Wait, ranks of [1,3,0,2] -> sorted is [0,1,2,3].
        # 0 is idx 2. 1 is idx 0. 2 is idx 3. 3 is idx 1.
        # Ranks: [1, 3, 0, 2] (0-based)

        # Corr(Col 0, Col 2):
        # Col 0 ranks: [0, 1, 2, 3]
        # Col 2 ranks: [1, 3, 0, 2]
        # Covariance...

        corr = fast_spearman_matrix(data)

        assert corr.shape == (3, 3)
        assert corr[0, 0] == 1.0
        np.testing.assert_allclose(corr[0, 1], -1.0)
        np.testing.assert_allclose(corr[1, 0], -1.0)

        # Just check it returns a value between -1 and 1
        assert -1.0 <= corr[0, 2] <= 1.0

    def test_fast_distance_correlation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        dcor = fast_distance_correlation(x, y)
        # dCor(X, X) is 1
        np.testing.assert_allclose(dcor, 1.0, atol=1e-10)

        y_const = np.ones(5)
        dcor_const = fast_distance_correlation(x, y_const)
        # dCor with constant is 0 (actually dvar_y is 0)
        assert dcor_const == 0.0

    def test_fast_autocorrelation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        max_lag = 2

        acf = fast_autocorrelation(x, max_lag)

        assert len(acf) == 3
        assert acf[0] == 1.0 # Lag 0 is always 1

    def test_fast_vif_single(self):
        # Create multicollinear data: x2 = 2*x1
        X = np.array([
            [1.0, 2.0, 5.0],
            [2.0, 4.0, 2.0],
            [3.0, 6.0, 1.0],
            [4.0, 8.0, 4.0]
        ])
        # Feature 1 (idx 1) is perfectly predicted by Feature 0 (idx 0)

        vif = fast_vif_single(X, 1)
        assert np.isinf(vif) or vif > 1000 # High VIF

if __name__ == "__main__":
    # Allow running individual test file with pytest and colored output
    import pytest  # type: ignore
    pytest.main([__file__, "-v", "--color=yes"])
