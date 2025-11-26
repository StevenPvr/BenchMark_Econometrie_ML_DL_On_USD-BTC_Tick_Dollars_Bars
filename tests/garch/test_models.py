"""Tests for src/garch/garch_params/models.py module."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.garch.garch_params.models import (
    EGARCHParams,
    _build_alpha_dict_entries,
    _build_beta_dict_entries,
    _build_gamma_dict_entries,
    _extract_alpha_from_array,
    _extract_alpha_from_dict,
    _extract_beta_from_array,
    _extract_beta_from_dict,
    _extract_gamma_from_array,
    _extract_gamma_from_dict,
    _extract_params_from_array,
    _validate_order_constraints,
    create_egarch_params_from_array,
    create_egarch_params_from_dict,
)


class TestExtractParamsFromArray:
    """Test cases for _extract_params_from_array function."""

    def test_extracts_single_param_order_1(self):
        """Should extract single parameter for order 1."""
        params = np.array([0.1, 0.2, 0.3, 0.4])
        value, next_idx = _extract_params_from_array(params, idx=0, order=1, max_order=3)

        assert value == 0.1
        assert next_idx == 1

    def test_extracts_tuple_order_2(self):
        """Should extract tuple for order 2."""
        params = np.array([0.1, 0.2, 0.3, 0.4])
        value, next_idx = _extract_params_from_array(params, idx=1, order=2, max_order=3)

        assert value == (0.2, 0.3)
        assert next_idx == 3

    def test_extracts_tuple_order_3(self):
        """Should extract tuple for order 3."""
        params = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        value, next_idx = _extract_params_from_array(params, idx=1, order=3, max_order=3)

        assert value == (0.2, 0.3, 0.4)
        assert next_idx == 4

    def test_raises_on_unsupported_order(self):
        """Should raise ValueError for unsupported order."""
        params = np.array([0.1, 0.2])
        with pytest.raises(ValueError, match="not supported"):
            _extract_params_from_array(params, idx=0, order=4, max_order=3)


class TestExtractAlphaFromArray:
    """Test cases for _extract_alpha_from_array function."""

    def test_extracts_alpha_order_1(self):
        """Should extract single alpha for o=1."""
        params = np.array([0.1, 0.2, 0.3])
        alpha, next_idx = _extract_alpha_from_array(params, idx=0, o=1)

        assert alpha == 0.1
        assert next_idx == 1

    def test_extracts_alpha_order_2(self):
        """Should extract alpha tuple for o=2."""
        params = np.array([0.1, 0.2, 0.3])
        alpha, next_idx = _extract_alpha_from_array(params, idx=0, o=2)

        assert alpha == (0.1, 0.2)
        assert next_idx == 2


class TestExtractGammaFromArray:
    """Test cases for _extract_gamma_from_array function."""

    def test_extracts_gamma_order_1(self):
        """Should extract single gamma for o=1."""
        params = np.array([-0.05, 0.2])
        gamma, next_idx = _extract_gamma_from_array(params, idx=0, o=1)

        assert gamma == -0.05
        assert next_idx == 1

    def test_extracts_gamma_order_2(self):
        """Should extract gamma tuple for o=2."""
        params = np.array([-0.05, -0.03, 0.9])
        gamma, next_idx = _extract_gamma_from_array(params, idx=0, o=2)

        assert gamma == (-0.05, -0.03)
        assert next_idx == 2


class TestExtractBetaFromArray:
    """Test cases for _extract_beta_from_array function."""

    def test_extracts_beta_order_1(self):
        """Should extract single beta for p=1."""
        params = np.array([0.9, 5.0])
        beta, next_idx = _extract_beta_from_array(params, idx=0, p=1)

        assert beta == 0.9
        assert next_idx == 1

    def test_extracts_beta_order_2(self):
        """Should extract beta tuple for p=2."""
        params = np.array([0.7, 0.2, 5.0])
        beta, next_idx = _extract_beta_from_array(params, idx=0, p=2)

        assert beta == (0.7, 0.2)
        assert next_idx == 2

    def test_extracts_beta_order_3(self):
        """Should extract beta tuple for p=3."""
        params = np.array([0.5, 0.3, 0.1, 5.0])
        beta, next_idx = _extract_beta_from_array(params, idx=0, p=3)

        assert beta == (0.5, 0.3, 0.1)
        assert next_idx == 3


class TestExtractFromDict:
    """Test cases for dict extraction functions."""

    def test_extract_alpha_order_1(self):
        """Should extract alpha from dict for o=1."""
        params = {"alpha": 0.15, "gamma": -0.05, "beta": 0.9}
        alpha = _extract_alpha_from_dict(params, o=1)

        assert alpha == 0.15

    def test_extract_alpha_order_2(self):
        """Should extract alpha from dict for o=2."""
        params = {"alpha1": 0.10, "alpha2": 0.05, "gamma1": -0.05, "gamma2": -0.02}
        alpha = _extract_alpha_from_dict(params, o=2)

        assert alpha == (0.10, 0.05)

    def test_extract_gamma_order_1(self):
        """Should extract gamma from dict for o=1."""
        params = {"alpha": 0.15, "gamma": -0.08}
        gamma = _extract_gamma_from_dict(params, o=1)

        assert gamma == -0.08

    def test_extract_beta_order_1(self):
        """Should extract beta from dict for p=1."""
        params = {"beta": 0.92}
        beta = _extract_beta_from_dict(params, p=1)

        assert beta == 0.92

    def test_extract_beta_order_2(self):
        """Should extract beta from dict for p=2."""
        params = {"beta1": 0.7, "beta2": 0.2}
        beta = _extract_beta_from_dict(params, p=2)

        assert beta == (0.7, 0.2)

    def test_raises_on_missing_key(self):
        """Should raise KeyError when required key is missing."""
        params = {"alpha": 0.15}  # Missing alpha2 for o=2
        with pytest.raises(KeyError):
            _extract_alpha_from_dict(params, o=2)


class TestBuildDictEntries:
    """Test cases for dict building functions."""

    def test_build_alpha_order_1(self):
        """Should build alpha dict entry for o=1."""
        result = _build_alpha_dict_entries(0.15, o=1)

        assert result == {"alpha": 0.15}

    def test_build_alpha_order_2(self):
        """Should build alpha dict entries for o=2."""
        result = _build_alpha_dict_entries((0.10, 0.05), o=2)

        assert result == {"alpha1": 0.10, "alpha2": 0.05}

    def test_build_gamma_order_1(self):
        """Should build gamma dict entry for o=1."""
        result = _build_gamma_dict_entries(-0.08, o=1)

        assert result == {"gamma": -0.08}

    def test_build_beta_order_1(self):
        """Should build beta dict entry for p=1."""
        result = _build_beta_dict_entries(0.92, p=1)

        assert result == {"beta": 0.92}

    def test_build_beta_order_2(self):
        """Should build beta dict entries for p=2."""
        result = _build_beta_dict_entries((0.7, 0.2), p=2)

        assert result == {"beta1": 0.7, "beta2": 0.2}

    def test_build_beta_order_3(self):
        """Should build beta dict entries for p=3."""
        result = _build_beta_dict_entries((0.5, 0.3, 0.1), p=3)

        assert result == {"beta1": 0.5, "beta2": 0.3, "beta3": 0.1}


class TestValidateOrderConstraints:
    """Test cases for _validate_order_constraints function."""

    def test_valid_orders(self):
        """Should accept valid order combinations."""
        _validate_order_constraints(o=1, p=1)  # No exception
        _validate_order_constraints(o=1, p=2)
        _validate_order_constraints(o=1, p=3)
        _validate_order_constraints(o=2, p=1)
        _validate_order_constraints(o=2, p=2)
        _validate_order_constraints(o=2, p=3)

    def test_invalid_arch_order(self):
        """Should reject invalid ARCH order."""
        with pytest.raises(ValueError, match="ARCH order o=0"):
            _validate_order_constraints(o=0, p=1)

        with pytest.raises(ValueError, match="ARCH order o=3"):
            _validate_order_constraints(o=3, p=1)

    def test_invalid_garch_order(self):
        """Should reject invalid GARCH order."""
        with pytest.raises(ValueError, match="GARCH order p=0"):
            _validate_order_constraints(o=1, p=0)

        with pytest.raises(ValueError, match="GARCH order p=4"):
            _validate_order_constraints(o=1, p=4)


class TestEGARCHParams:
    """Test cases for EGARCHParams dataclass."""

    def test_creates_valid_egarch11_student(self):
        """Should create valid EGARCH(1,1) with Student-t."""
        params = EGARCHParams(
            omega=-0.1,
            alpha=0.15,
            gamma=-0.08,
            beta=0.92,
            nu=5.0,
            lambda_skew=None,
            o=1,
            p=1,
            dist="student",
        )

        assert params.omega == -0.1
        assert params.alpha == 0.15
        assert params.gamma == -0.08
        assert params.beta == 0.92
        assert params.nu == 5.0
        assert params.o == 1
        assert params.p == 1
        assert params.dist == "student"

    def test_creates_valid_egarch21_student(self):
        """Should create valid EGARCH(2,1) with Student-t."""
        params = EGARCHParams(
            omega=-0.1,
            alpha=(0.10, 0.05),
            gamma=(-0.05, -0.03),
            beta=0.90,
            nu=6.0,
            lambda_skew=None,
            o=2,
            p=1,
            dist="student",
        )

        assert params.alpha == (0.10, 0.05)
        assert params.gamma == (-0.05, -0.03)
        assert params.o == 2

    def test_creates_valid_egarch12_student(self):
        """Should create valid EGARCH(1,2) with Student-t."""
        params = EGARCHParams(
            omega=-0.1,
            alpha=0.15,
            gamma=-0.08,
            beta=(0.7, 0.2),
            nu=5.0,
            lambda_skew=None,
            o=1,
            p=2,
            dist="student",
        )

        assert params.beta == (0.7, 0.2)
        assert params.p == 2

    def test_creates_valid_egarch11_skewt(self):
        """Should create valid EGARCH(1,1) with Skew-t."""
        params = EGARCHParams(
            omega=-0.1,
            alpha=0.15,
            gamma=-0.08,
            beta=0.92,
            nu=5.0,
            lambda_skew=-0.2,
            o=1,
            p=1,
            dist="skewt",
        )

        assert params.lambda_skew == -0.2
        assert params.dist == "skewt"

    def test_rejects_invalid_arch_order(self):
        """Should reject invalid ARCH order."""
        with pytest.raises(ValueError, match="ARCH order"):
            EGARCHParams(
                omega=-0.1,
                alpha=0.15,
                gamma=-0.08,
                beta=0.92,
                nu=5.0,
                lambda_skew=None,
                o=3,  # Invalid
                p=1,
                dist="student",
            )

    def test_rejects_invalid_garch_order(self):
        """Should reject invalid GARCH order."""
        with pytest.raises(ValueError, match="GARCH order"):
            EGARCHParams(
                omega=-0.1,
                alpha=0.15,
                gamma=-0.08,
                beta=0.92,
                nu=5.0,
                lambda_skew=None,
                o=1,
                p=4,  # Invalid
                dist="student",
            )

    def test_rejects_alpha_type_mismatch(self):
        """Should reject alpha type that doesn't match order."""
        with pytest.raises(TypeError, match="alpha"):
            EGARCHParams(
                omega=-0.1,
                alpha=(0.1, 0.05),  # Tuple for o=1 is invalid
                gamma=-0.08,
                beta=0.92,
                nu=5.0,
                lambda_skew=None,
                o=1,
                p=1,
                dist="student",
            )

    def test_rejects_unsupported_distribution(self):
        """Should reject unsupported distribution."""
        with pytest.raises(ValueError, match="not supported"):
            EGARCHParams(
                omega=-0.1,
                alpha=0.15,
                gamma=-0.08,
                beta=0.92,
                nu=5.0,
                lambda_skew=None,
                o=1,
                p=1,
                dist="normal",  # Not supported
            )


class TestEGARCHParamsToDict:
    """Test cases for EGARCHParams.to_dict method."""

    def test_to_dict_egarch11(self):
        """Should convert EGARCH(1,1) to dict."""
        params = EGARCHParams(
            omega=-0.1,
            alpha=0.15,
            gamma=-0.08,
            beta=0.92,
            nu=5.0,
            lambda_skew=None,
            o=1,
            p=1,
            dist="student",
        )

        result = params.to_dict()

        assert result["omega"] == -0.1
        assert result["alpha"] == 0.15
        assert result["gamma"] == -0.08
        assert result["beta"] == 0.92
        assert result["nu"] == 5.0
        assert "lambda" not in result

    def test_to_dict_egarch21(self):
        """Should convert EGARCH(2,1) to dict with numbered params."""
        params = EGARCHParams(
            omega=-0.1,
            alpha=(0.10, 0.05),
            gamma=(-0.05, -0.03),
            beta=0.90,
            nu=6.0,
            lambda_skew=None,
            o=2,
            p=1,
            dist="student",
        )

        result = params.to_dict()

        assert result["alpha1"] == 0.10
        assert result["alpha2"] == 0.05
        assert result["gamma1"] == -0.05
        assert result["gamma2"] == -0.03
        assert result["beta"] == 0.90

    def test_to_dict_includes_loglik(self):
        """Should include loglik if present."""
        params = EGARCHParams(
            omega=-0.1,
            alpha=0.15,
            gamma=-0.08,
            beta=0.92,
            nu=5.0,
            lambda_skew=None,
            o=1,
            p=1,
            dist="student",
            loglik=-1234.56,
        )

        result = params.to_dict()

        assert result["loglik"] == -1234.56


class TestEGARCHParamsToArray:
    """Test cases for EGARCHParams.to_array method."""

    def test_to_array_egarch11(self):
        """Should convert EGARCH(1,1) to array."""
        params = EGARCHParams(
            omega=-0.1,
            alpha=0.15,
            gamma=-0.08,
            beta=0.92,
            nu=5.0,
            lambda_skew=None,
            o=1,
            p=1,
            dist="student",
        )

        result = params.to_array()

        expected = np.array([-0.1, 0.15, -0.08, 0.92, 5.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_to_array_egarch21(self):
        """Should convert EGARCH(2,1) to array."""
        params = EGARCHParams(
            omega=-0.1,
            alpha=(0.10, 0.05),
            gamma=(-0.05, -0.03),
            beta=0.90,
            nu=6.0,
            lambda_skew=None,
            o=2,
            p=1,
            dist="student",
        )

        result = params.to_array()

        # omega, alpha1, alpha2, gamma1, gamma2, beta, nu
        expected = np.array([-0.1, 0.10, 0.05, -0.05, -0.03, 0.90, 6.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_to_array_skewt(self):
        """Should include lambda_skew for Skew-t."""
        params = EGARCHParams(
            omega=-0.1,
            alpha=0.15,
            gamma=-0.08,
            beta=0.92,
            nu=5.0,
            lambda_skew=-0.2,
            o=1,
            p=1,
            dist="skewt",
        )

        result = params.to_array()

        expected = np.array([-0.1, 0.15, -0.08, 0.92, 5.0, -0.2])
        np.testing.assert_array_almost_equal(result, expected)


class TestEGARCHParamsBetaRepresentative:
    """Test cases for get_beta_representative method."""

    def test_beta_representative_p1(self):
        """Should return beta for p=1."""
        params = EGARCHParams(
            omega=-0.1,
            alpha=0.15,
            gamma=-0.08,
            beta=0.92,
            nu=5.0,
            lambda_skew=None,
            o=1,
            p=1,
            dist="student",
        )

        assert params.get_beta_representative() == 0.92

    def test_beta_representative_p2(self):
        """Should return sum of betas for p=2."""
        params = EGARCHParams(
            omega=-0.1,
            alpha=0.15,
            gamma=-0.08,
            beta=(0.7, 0.2),
            nu=5.0,
            lambda_skew=None,
            o=1,
            p=2,
            dist="student",
        )

        assert params.get_beta_representative() == pytest.approx(0.9)


class TestEGARCHParamsExtractForVariance:
    """Test cases for extract_for_variance method."""

    def test_extract_for_variance_egarch11(self):
        """Should extract (alpha, gamma, beta) for EGARCH(1,1)."""
        params = EGARCHParams(
            omega=-0.1,
            alpha=0.15,
            gamma=-0.08,
            beta=0.92,
            nu=5.0,
            lambda_skew=None,
            o=1,
            p=1,
            dist="student",
        )

        alpha, gamma, beta = params.extract_for_variance()

        assert alpha == 0.15
        assert gamma == -0.08
        assert beta == 0.92

    def test_extract_for_variance_egarch22(self):
        """Should extract tuples for EGARCH(2,2)."""
        params = EGARCHParams(
            omega=-0.1,
            alpha=(0.10, 0.05),
            gamma=(-0.05, -0.03),
            beta=(0.7, 0.2),
            nu=5.0,
            lambda_skew=None,
            o=2,
            p=2,
            dist="student",
        )

        alpha, gamma, beta = params.extract_for_variance()

        assert alpha == (0.10, 0.05)
        assert gamma == (-0.05, -0.03)
        assert beta == (0.7, 0.2)


class TestCreateEGARCHParamsFromDict:
    """Test cases for create_egarch_params_from_dict function."""

    def test_creates_from_dict_egarch11(self):
        """Should create EGARCH(1,1) from dict."""
        params_dict = {
            "omega": -0.1,
            "alpha": 0.15,
            "gamma": -0.08,
            "beta": 0.92,
            "nu": 5.0,
        }

        params = create_egarch_params_from_dict(params_dict, o=1, p=1, dist="student")

        assert params.omega == -0.1
        assert params.alpha == 0.15
        assert params.gamma == -0.08
        assert params.beta == 0.92
        assert params.nu == 5.0

    def test_creates_from_dict_egarch21(self):
        """Should create EGARCH(2,1) from dict."""
        params_dict = {
            "omega": -0.1,
            "alpha1": 0.10,
            "alpha2": 0.05,
            "gamma1": -0.05,
            "gamma2": -0.03,
            "beta": 0.90,
            "nu": 6.0,
        }

        params = create_egarch_params_from_dict(params_dict, o=2, p=1, dist="student")

        assert params.alpha == (0.10, 0.05)
        assert params.gamma == (-0.05, -0.03)

    def test_creates_from_dict_skewt(self):
        """Should create Skew-t params from dict."""
        params_dict = {
            "omega": -0.1,
            "alpha": 0.15,
            "gamma": -0.08,
            "beta": 0.92,
            "nu": 5.0,
            "lambda": -0.2,
        }

        params = create_egarch_params_from_dict(params_dict, o=1, p=1, dist="skewt")

        assert params.lambda_skew == -0.2

    def test_raises_on_missing_omega(self):
        """Should raise KeyError when omega is missing."""
        params_dict = {
            "alpha": 0.15,
            "gamma": -0.08,
            "beta": 0.92,
        }

        with pytest.raises(KeyError):
            create_egarch_params_from_dict(params_dict, o=1, p=1, dist="student")


class TestCreateEGARCHParamsFromArray:
    """Test cases for create_egarch_params_from_array function."""

    def test_creates_from_array_egarch11(self):
        """Should create EGARCH(1,1) from array."""
        params_array = np.array([-0.1, 0.15, -0.08, 0.92, 5.0])

        params = create_egarch_params_from_array(params_array, o=1, p=1, dist="student")

        assert params.omega == -0.1
        assert params.alpha == 0.15
        assert params.gamma == -0.08
        assert params.beta == 0.92
        assert params.nu == 5.0

    def test_creates_from_array_egarch21(self):
        """Should create EGARCH(2,1) from array."""
        params_array = np.array([-0.1, 0.10, 0.05, -0.05, -0.03, 0.90, 6.0])

        params = create_egarch_params_from_array(params_array, o=2, p=1, dist="student")

        assert params.alpha == (0.10, 0.05)
        assert params.gamma == (-0.05, -0.03)
        assert params.beta == 0.90

    def test_creates_from_array_skewt(self):
        """Should create Skew-t params from array."""
        params_array = np.array([-0.1, 0.15, -0.08, 0.92, 5.0, -0.2])

        params = create_egarch_params_from_array(params_array, o=1, p=1, dist="skewt")

        assert params.nu == 5.0
        assert params.lambda_skew == -0.2

    def test_includes_loglik_and_converged(self):
        """Should include optional loglik and converged flags."""
        params_array = np.array([-0.1, 0.15, -0.08, 0.92, 5.0])

        params = create_egarch_params_from_array(
            params_array, o=1, p=1, dist="student", loglik=-1234.56, converged=True
        )

        assert params.loglik == -1234.56
        assert params.converged is True


class TestEGARCHParamsRoundTrip:
    """Round-trip tests for EGARCHParams conversion."""

    def test_dict_roundtrip_egarch11(self):
        """Dict -> params -> dict should preserve values."""
        original = {
            "omega": -0.1,
            "alpha": 0.15,
            "gamma": -0.08,
            "beta": 0.92,
            "nu": 5.0,
        }

        params = create_egarch_params_from_dict(original, o=1, p=1, dist="student")
        result = params.to_dict()

        assert abs(result["omega"] - original["omega"]) < 1e-10
        assert abs(result["alpha"] - original["alpha"]) < 1e-10
        assert abs(result["gamma"] - original["gamma"]) < 1e-10
        assert abs(result["beta"] - original["beta"]) < 1e-10
        assert abs(result["nu"] - original["nu"]) < 1e-10

    def test_array_roundtrip_egarch11(self):
        """Array -> params -> array should preserve values."""
        original = np.array([-0.1, 0.15, -0.08, 0.92, 5.0])

        params = create_egarch_params_from_array(original, o=1, p=1, dist="student")
        result = params.to_array()

        np.testing.assert_array_almost_equal(result, original)

    def test_array_roundtrip_egarch22(self):
        """Array -> params -> array for EGARCH(2,2)."""
        # omega, alpha1, alpha2, gamma1, gamma2, beta1, beta2, nu
        original = np.array([-0.1, 0.10, 0.05, -0.05, -0.03, 0.7, 0.2, 6.0])

        params = create_egarch_params_from_array(original, o=2, p=2, dist="student")
        result = params.to_array()

        np.testing.assert_array_almost_equal(result, original)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
