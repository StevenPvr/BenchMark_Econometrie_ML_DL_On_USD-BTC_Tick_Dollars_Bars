"""EGARCH model parameter representations.

This module provides a unified dataclass for EGARCH parameters,
eliminating duplication across estimation and optimization modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from src.garch.garch_params.core import (
    validate_skewt_params,
    validate_student_params,
)
from src.utils import get_logger

logger = get_logger(__name__)


# ============================================================================
# Private helper functions for parameter extraction and building
# ============================================================================


def _extract_params_from_array(
    params: np.ndarray, idx: int, order: int, max_order: int
) -> tuple[float | tuple[float, ...], int]:
    """Extract parameter(s) from array based on order.

    Args:
        params: Parameter array.
        idx: Current index in array.
        order: Parameter order (1, 2, or 3).
        max_order: Maximum allowed order for this parameter type.

    Returns:
        Tuple of (parameter_value, new_index).

    Raises:
        ValueError: If order is not supported.
    """
    if order == 1:
        return float(params[idx]), idx + 1
    if order == 2 and max_order >= 2:
        return (float(params[idx]), float(params[idx + 1])), idx + 2
    if order == 3 and max_order >= 3:
        return (float(params[idx]), float(params[idx + 1]), float(params[idx + 2])), idx + 3
    msg = f"Order {order} not supported (max: {max_order})"
    raise ValueError(msg)


def _extract_alpha_from_array(
    params: np.ndarray, idx: int, o: int
) -> tuple[float | tuple[float, float], int]:
    """Extract alpha parameter(s) from array."""
    result, new_idx = _extract_params_from_array(params, idx, o, 2)
    if o == 1:
        return cast(float, result), new_idx
    else:  # o == 2
        return cast(tuple[float, float], result), new_idx


def _extract_gamma_from_array(
    params: np.ndarray, idx: int, o: int
) -> tuple[float | tuple[float, float], int]:
    """Extract gamma parameter(s) from array."""
    result, new_idx = _extract_params_from_array(params, idx, o, 2)
    if o == 1:
        return cast(float, result), new_idx
    else:  # o == 2
        return cast(tuple[float, float], result), new_idx


def _extract_beta_from_array(
    params: np.ndarray, idx: int, p: int
) -> tuple[float | tuple[float, float] | tuple[float, float, float], int]:
    """Extract beta parameter(s) from array."""
    result, new_idx = _extract_params_from_array(params, idx, p, 3)
    if p == 1:
        return cast(float, result), new_idx
    elif p == 2:
        return cast(tuple[float, float], result), new_idx
    else:  # p == 3
        return cast(tuple[float, float, float], result), new_idx


def _extract_order_1_param(params: dict[str, float], param_name: str) -> float:
    """Extract single parameter value."""
    return params[param_name]


def _extract_order_2_param(params: dict[str, float], param_name: str) -> tuple[float, float]:
    """Extract parameter as tuple of 2 values."""
    key1, key2 = f"{param_name}1", f"{param_name}2"
    if key1 in params and key2 in params:
        return (params[key1], params[key2])
    if param_name in params and isinstance(params[param_name], tuple):
        return cast(tuple[float, float], params[param_name])
    msg = f"EGARCH(o,2) requires {key1} and {key2}, got {list(params.keys())}"
    raise KeyError(msg)


def _extract_order_3_param(params: dict[str, float], param_name: str) -> tuple[float, float, float]:
    """Extract parameter as tuple of 3 values."""
    key1, key2, key3 = f"{param_name}1", f"{param_name}2", f"{param_name}3"
    if key1 in params and key2 in params and key3 in params:
        return (params[key1], params[key2], params[key3])
    if param_name in params and isinstance(params[param_name], tuple):
        return cast(tuple[float, float, float], params[param_name])
    msg = f"EGARCH(o,3) requires {key1}, {key2}, and {key3}, got {list(params.keys())}"
    raise KeyError(msg)


def _extract_param_from_dict(
    params: dict[str, float], param_name: str, order: int, max_order: int
) -> float | tuple[float, ...]:
    """Extract parameter from dictionary based on order.

    Args:
        params: Parameter dictionary.
        param_name: Base parameter name (e.g., 'alpha', 'gamma', 'beta').
        order: Parameter order (1, 2, or 3).
        max_order: Maximum allowed order for this parameter type.

    Returns:
        Parameter value(s) matching the order.

    Raises:
        KeyError: If required keys are missing.
        ValueError: If order is not supported.
    """
    extractors = {
        1: _extract_order_1_param,
        2: _extract_order_2_param,
        3: _extract_order_3_param,
    }

    if order not in extractors:
        msg = f"Order {order} not supported for {param_name} (max: {max_order})"
        raise ValueError(msg)

    if order > max_order:
        msg = f"Order {order} not supported for {param_name} (max: {max_order})"
        raise ValueError(msg)

    return cast(float | tuple[float, ...], extractors[order](params, param_name))


def _extract_alpha_from_dict(params: dict[str, float], o: int) -> float | tuple[float, float]:
    """Extract alpha parameter(s) from dictionary."""
    result = _extract_param_from_dict(params, "alpha", o, 2)
    if o == 1:
        return cast(float, result)
    else:  # o == 2
        return cast(tuple[float, float], result)


def _extract_gamma_from_dict(params: dict[str, float], o: int) -> float | tuple[float, float]:
    """Extract gamma parameter(s) from dictionary."""
    result = _extract_param_from_dict(params, "gamma", o, 2)
    if o == 1:
        return cast(float, result)
    else:  # o == 2
        return cast(tuple[float, float], result)


def _extract_beta_from_dict(
    params: dict[str, float], p: int
) -> float | tuple[float, float] | tuple[float, float, float]:
    """Extract beta parameter(s) from dictionary."""
    result = _extract_param_from_dict(params, "beta", p, 3)
    if p == 1:
        return cast(float, result)
    elif p == 2:
        return cast(tuple[float, float], result)
    else:  # p == 3
        return cast(tuple[float, float, float], result)


def _build_order_1_dict_entry(param_value: float, param_name: str) -> dict[str, float]:
    """Build dictionary entry for single parameter."""
    if isinstance(param_value, tuple):
        msg = f"{param_name} should be float for order=1"
        raise TypeError(msg)
    return {param_name: float(param_value)}


def _build_order_2_dict_entries(
    param_value: tuple[float, float], param_name: str
) -> dict[str, float]:
    """Build dictionary entries for parameter tuple of 2 values."""
    if not isinstance(param_value, tuple) or len(param_value) != 2:
        msg = f"{param_name} should be tuple of 2 floats for order=2"
        raise TypeError(msg)
    return {f"{param_name}1": float(param_value[0]), f"{param_name}2": float(param_value[1])}


def _build_order_3_dict_entries(
    param_value: tuple[float, float, float], param_name: str
) -> dict[str, float]:
    """Build dictionary entries for parameter tuple of 3 values."""
    if not isinstance(param_value, tuple) or len(param_value) != 3:
        msg = f"{param_name} should be tuple of 3 floats for order=3"
        raise TypeError(msg)
    return {
        f"{param_name}1": float(param_value[0]),
        f"{param_name}2": float(param_value[1]),
        f"{param_name}3": float(param_value[2]),
    }


def _build_param_dict_entries(
    param_value: float | tuple[float, ...], param_name: str, order: int, max_order: int
) -> dict[str, float]:
    """Build dictionary entries for parameter based on order.

    Args:
        param_value: Parameter value(s).
        param_name: Base parameter name (e.g., 'alpha', 'gamma', 'beta').
        order: Parameter order (1, 2, or 3).
        max_order: Maximum allowed order for this parameter type.

    Returns:
        Dictionary with parameter entries.

    Raises:
        TypeError: If parameter type doesn't match order.
        ValueError: If order is not supported.
    """
    builders = {
        1: _build_order_1_dict_entry,
        2: _build_order_2_dict_entries,
        3: _build_order_3_dict_entries,
    }

    if order not in builders:
        msg = f"Order {order} not supported for {param_name} (max: {max_order})"
        raise ValueError(msg)

    if order > max_order:
        msg = f"Order {order} not supported for {param_name} (max: {max_order})"
        raise ValueError(msg)

    if order == 1:
        assert isinstance(
            param_value, float
        ), f"Expected float for order 1, got {type(param_value)}"
        return _build_order_1_dict_entry(param_value, param_name)
    elif order == 2:
        assert (
            isinstance(param_value, tuple) and len(param_value) == 2
        ), f"Expected tuple of 2 floats for order 2, got {type(param_value)}"
        return _build_order_2_dict_entries(param_value, param_name)
    else:  # order == 3
        assert (
            isinstance(param_value, tuple) and len(param_value) == 3
        ), f"Expected tuple of 3 floats for order 3, got {type(param_value)}"
        return _build_order_3_dict_entries(param_value, param_name)


def _build_alpha_dict_entries(alpha: float | tuple[float, float], o: int) -> dict[str, float]:
    """Build dictionary entries for alpha parameter(s)."""
    return _build_param_dict_entries(alpha, "alpha", o, 2)


def _build_gamma_dict_entries(gamma: float | tuple[float, float], o: int) -> dict[str, float]:
    """Build dictionary entries for gamma parameter(s)."""
    return _build_param_dict_entries(gamma, "gamma", o, 2)


def _build_beta_dict_entries(
    beta: float | tuple[float, float] | tuple[float, float, float], p: int
) -> dict[str, float]:
    """Build dictionary entries for beta parameter(s)."""
    return _build_param_dict_entries(beta, "beta", p, 3)


def _append_order_1_param(params_list: list[float], param_value: float) -> None:
    """Append single parameter value to list."""
    if isinstance(param_value, tuple):
        msg = "parameter should be float for order=1"
        raise TypeError(msg)
    params_list.append(float(param_value))


def _append_order_2_params(params_list: list[float], param_value: tuple[float, float]) -> None:
    """Append parameter tuple of 2 values to list."""
    if not isinstance(param_value, tuple) or len(param_value) != 2:
        msg = "parameter should be tuple of 2 floats for order=2"
        raise TypeError(msg)
    params_list.extend([float(param_value[0]), float(param_value[1])])


def _append_order_3_params(
    params_list: list[float], param_value: tuple[float, float, float]
) -> None:
    """Append parameter tuple of 3 values to list."""
    if not isinstance(param_value, tuple) or len(param_value) != 3:
        msg = "parameter should be tuple of 3 floats for order=3"
        raise TypeError(msg)
    params_list.extend([float(param_value[0]), float(param_value[1]), float(param_value[2])])


def _append_param_to_list(
    params_list: list[float], param_value: float | tuple[float, ...], order: int, max_order: int
) -> None:
    """Append parameter value(s) to list based on order.

    Args:
        params_list: List to append to.
        param_value: Parameter value(s).
        order: Parameter order (1, 2, or 3).
        max_order: Maximum allowed order for this parameter type.

    Raises:
        TypeError: If parameter type doesn't match order.
        ValueError: If order is not supported.
    """
    appenders = {
        1: _append_order_1_param,
        2: _append_order_2_params,
        3: _append_order_3_params,
    }

    if order not in appenders:
        msg = f"Order {order} not supported (max: {max_order})"
        raise ValueError(msg)

    if order > max_order:
        msg = f"Order {order} not supported (max: {max_order})"
        raise ValueError(msg)

    if order == 1:
        assert isinstance(
            param_value, float
        ), f"Expected float for order 1, got {type(param_value)}"
        _append_order_1_param(params_list, param_value)
    elif order == 2:
        assert (
            isinstance(param_value, tuple) and len(param_value) == 2
        ), f"Expected tuple of 2 floats for order 2, got {type(param_value)}"
        _append_order_2_params(params_list, param_value)
    else:  # order == 3
        assert (
            isinstance(param_value, tuple) and len(param_value) == 3
        ), f"Expected tuple of 3 floats for order 3, got {type(param_value)}"
        _append_order_3_params(params_list, param_value)


def _append_alpha_to_list(
    params_list: list[float], alpha: float | tuple[float, float], o: int
) -> None:
    """Append alpha parameter(s) to list."""
    _append_param_to_list(params_list, alpha, o, 2)


def _append_gamma_to_list(
    params_list: list[float], gamma: float | tuple[float, float], o: int
) -> None:
    """Append gamma parameter(s) to list."""
    _append_param_to_list(params_list, gamma, o, 2)


def _append_beta_to_list(
    params_list: list[float],
    beta: float | tuple[float, float] | tuple[float, float, float],
    p: int,
) -> None:
    """Append beta parameter(s) to list."""
    _append_param_to_list(params_list, beta, p, 3)


def _validate_order_constraints(o: int, p: int) -> None:
    """Validate ARCH and GARCH order constraints.

    Args:
        o: ARCH order.
        p: GARCH order.

    Raises:
        ValueError: If orders are not supported.
    """
    if o not in (1, 2):
        msg = f"ARCH order o={o} not supported (only o=1 or 2)"
        raise ValueError(msg)
    if p not in (1, 2, 3):
        msg = f"GARCH order p={p} not supported (only p=1, 2, or 3)"
        raise ValueError(msg)


def _validate_order_1_param(param_value: float | tuple[float, ...], param_name: str) -> None:
    """Validate single parameter value."""
    if not isinstance(param_value, (int, float)):
        msg = f"{param_name} must be float for order=1, got {type(param_value)}"
        raise TypeError(msg)


def _validate_order_2_param(param_value: float | tuple[float, ...], param_name: str) -> None:
    """Validate parameter tuple of 2 values."""
    if not isinstance(param_value, tuple) or len(param_value) != 2:
        msg = f"{param_name} must be tuple of 2 floats for order=2, got {param_value}"
        raise TypeError(msg)


def _validate_order_3_param(param_value: float | tuple[float, ...], param_name: str) -> None:
    """Validate parameter tuple of 3 values."""
    if not isinstance(param_value, tuple) or len(param_value) != 3:
        msg = f"{param_name} must be tuple of 3 floats for order=3, got {param_value}"
        raise TypeError(msg)


def _validate_param_type(
    param_value: float | tuple[float, ...], param_name: str, order: int, max_order: int
) -> None:
    """Validate parameter type matches the expected order.

    Args:
        param_value: Parameter value(s).
        param_name: Parameter name for error messages.
        order: Expected order (1, 2, or 3).
        max_order: Maximum allowed order for this parameter type.

    Raises:
        TypeError: If parameter type doesn't match order.
        ValueError: If order is not supported.
    """
    validators = {
        1: _validate_order_1_param,
        2: _validate_order_2_param,
        3: _validate_order_3_param,
    }

    if order not in validators:
        msg = f"Order {order} not supported for {param_name} (max: {max_order})"
        raise ValueError(msg)

    if order > max_order:
        msg = f"Order {order} not supported for {param_name} (max: {max_order})"
        raise ValueError(msg)

    validators[order](param_value, param_name)


def _validate_alpha_type(alpha: float | tuple[float, float], o: int) -> None:
    """Validate alpha parameter type matches ARCH order."""
    _validate_param_type(alpha, "alpha", o, 2)


def _validate_gamma_type(gamma: float | tuple[float, float], o: int) -> None:
    """Validate gamma parameter type matches ARCH order."""
    _validate_param_type(gamma, "gamma", o, 2)


def _validate_beta_type(
    beta: float | tuple[float, float] | tuple[float, float, float], p: int
) -> None:
    """Validate beta parameter type matches GARCH order."""
    _validate_param_type(beta, "beta", p, 3)


def _validate_distribution_params(dist: str, nu: float | None, lambda_skew: float | None) -> None:
    """Validate distribution-specific parameters.

    Args:
        dist: Distribution name.
        nu: Degrees of freedom parameter.
        lambda_skew: Skewness parameter.

    Raises:
        ValueError: If distribution is not supported.
    """
    dist_lower = dist.lower()
    validators = {
        "student": lambda nu, ls: None,  # Student validation handled elsewhere
        "skewt": lambda nu, ls: None,  # Skew-t validation handled elsewhere
    }

    if dist_lower not in validators:
        msg = f"Distribution '{dist}' not supported"
        raise ValueError(msg)

    validator = validators[dist_lower]
    validator(nu, lambda_skew)


def _extract_dist_params_from_array(
    params: np.ndarray, idx: int, dist: str
) -> tuple[float | None, float | None]:
    """Extract distribution-specific parameters from array.

    Args:
        params: Parameter array.
        idx: Current index in array.
        dist: Distribution name.

    Returns:
        Tuple of (nu, lambda_skew).
    """
    nu = None
    lambda_skew = None
    dist_lower = dist.lower()

    if dist_lower == "student":
        # For student distribution, nu is optional in the array (uses default if not provided)
        if idx < len(params):
            nu = float(params[idx])
        # If not provided, nu remains None and will use default
    elif dist_lower == "skewt":
        if idx < len(params):
            nu = float(params[idx])
        if idx + 1 < len(params):
            lambda_skew = float(params[idx + 1])

    return nu, lambda_skew


@dataclass(frozen=True)
class EGARCHParams:
    """EGARCH model parameters.

    Attributes:
        omega: Constant term in log-variance equation.
        alpha: ARCH coefficient(s) - float for o=1, tuple for o=2.
        gamma: Asymmetry coefficient(s) - float for o=1, tuple for o=2.
        beta: GARCH coefficient(s) - float for p=1, tuple for p=2 or p=3.
        nu: Degrees of freedom parameter (Student-t, Skew-t).
        lambda_skew: Skewness parameter (Skew-t).
        o: ARCH order (1 or 2).
        p: GARCH order (1, 2, or 3).
        dist: Distribution name ('student', 'skewt').
        loglik: Log-likelihood value (optional).
        converged: Convergence status (optional).
    """

    omega: float
    alpha: float | tuple[float, float]
    gamma: float | tuple[float, float]
    beta: float | tuple[float, float] | tuple[float, float, float]
    nu: float | None
    lambda_skew: float | None
    o: int
    p: int
    dist: str
    loglik: float | None = None
    converged: bool | None = None

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        _validate_order_constraints(self.o, self.p)
        _validate_alpha_type(self.alpha, self.o)
        _validate_gamma_type(self.gamma, self.o)
        _validate_beta_type(self.beta, self.p)
        _validate_distribution_params(self.dist, self.nu, self.lambda_skew)

    def to_dict(self) -> dict[str, float]:
        """Convert parameters to dictionary representation.

        Returns:
            Dictionary with parameter names as keys.
        """
        logger.debug(f"Converting EGARCH({self.o},{self.p}) parameters to dict")

        out: dict[str, float] = {"omega": self.omega}

        # Add alpha/gamma/beta using helpers
        out.update(_build_alpha_dict_entries(self.alpha, self.o))
        out.update(_build_gamma_dict_entries(self.gamma, self.o))
        out.update(_build_beta_dict_entries(self.beta, self.p))

        # Add distribution parameters
        if self.nu is not None:
            out["nu"] = float(self.nu)
        if self.lambda_skew is not None:
            out["lambda"] = float(self.lambda_skew)

        # Add metadata
        if self.loglik is not None:
            out["loglik"] = float(self.loglik)
        if self.converged is not None:
            out["converged"] = float(self.converged)

        return out

    def to_array(self) -> np.ndarray:
        """Convert parameters to array representation for optimization.

        Returns:
            1D numpy array with parameters in standard order.
        """
        logger.debug(f"Converting EGARCH({self.o},{self.p}) parameters to array")

        params_list: list[float] = [self.omega]

        # Add alpha/gamma/beta using helpers
        _append_alpha_to_list(params_list, self.alpha, self.o)
        _append_gamma_to_list(params_list, self.gamma, self.o)
        _append_beta_to_list(params_list, self.beta, self.p)

        # Add distribution parameters
        if self.nu is not None:
            params_list.append(float(self.nu))
        if self.lambda_skew is not None:
            params_list.append(float(self.lambda_skew))

        return np.array(params_list, dtype=float)

    def get_beta_representative(self) -> float:
        """Get representative beta value for stationarity checks.

        Returns:
            Single beta value (for p=1) or sum of betas (for p=2).
        """
        if self.p == 1:
            if isinstance(self.beta, tuple):
                msg = "beta should be float for p=1"
                raise TypeError(msg)
            return float(self.beta)
        # p == 2
        if not isinstance(self.beta, tuple):
            msg = "beta should be tuple for p=2"
            raise TypeError(msg)
        return float(self.beta[0] + self.beta[1])

    def validate_stationarity(self) -> bool:
        """Validate parameter stationarity conditions.

        Returns:
            True if parameters satisfy stationarity constraints.
        """
        beta_val = self.get_beta_representative()

        dist_lower = self.dist.lower()
        if dist_lower == "student":
            if self.nu is None:
                return False
            return validate_student_params(beta_val, self.nu)
        if dist_lower == "skewt":
            if self.nu is None or self.lambda_skew is None:
                return False
            return validate_skewt_params(beta_val, self.nu, self.lambda_skew)
        return False

    def extract_for_variance(
        self,
    ) -> tuple[
        float | tuple[float, float],
        float | tuple[float, float],
        float | tuple[float, float] | tuple[float, float, float],
    ]:
        """Extract (alpha, gamma, beta) tuple for variance computation.

        This method provides the parameters in the exact format required by
        variance filtering and forecasting functions.

        Returns:
            Tuple of (alpha, gamma, beta) where:
            - alpha/gamma: float for o=1, tuple[float, float] for o=2
            - beta: float for p=1, tuple[float, float] for p=2, tuple[float, float, float] for p=3

        Example:
            >>> params = EGARCHParams(omega=0.1, alpha=0.2, gamma=-0.1, beta=0.9,
            ...                       nu=5.0, lambda_skew=None, o=1, p=1, dist="student")
            >>> alpha, gamma, beta = params.extract_for_variance()
            >>> # alpha=0.2, gamma=-0.1, beta=0.9 (all floats)

            >>> params = EGARCHParams(omega=0.1, alpha=(0.2, 0.15), gamma=(-0.1, -0.05),
            ...                       beta=(0.8, 0.1), nu=5.0, lambda_skew=None,
            ...                       o=2, p=2, dist="student")
            >>> alpha, gamma, beta = params.extract_for_variance()
            >>> # alpha=(0.2, 0.15), gamma=(-0.1, -0.05), beta=(0.8, 0.1)
        """
        return self.alpha, self.gamma, self.beta

    def __str__(self) -> str:
        """Return string representation of parameters."""
        parts = [f"EGARCH({self.o},{self.p})", f"dist={self.dist}"]

        params_dict = self.to_dict()
        if self.o == 1:
            parts.append(f"alpha={params_dict['alpha']:.4f}")
            parts.append(f"gamma={params_dict['gamma']:.4f}")
        else:
            parts.append(f"alpha=({params_dict['alpha1']:.4f}, {params_dict['alpha2']:.4f})")
            parts.append(f"gamma=({params_dict['gamma1']:.4f}, {params_dict['gamma2']:.4f})")

        if self.p == 1:
            parts.append(f"beta={params_dict['beta']:.4f}")
        else:
            parts.append(f"beta=({params_dict['beta1']:.4f}, {params_dict['beta2']:.4f})")

        if self.nu is not None:
            parts.append(f"nu={self.nu:.2f}")
        if self.lambda_skew is not None:
            parts.append(f"lambda={self.lambda_skew:.4f}")

        if self.loglik is not None:
            parts.append(f"loglik={self.loglik:.2f}")

        return ", ".join(parts)


def create_egarch_params_from_dict(
    params: dict[str, float],
    o: int,
    p: int,
    dist: str,
) -> EGARCHParams:
    """Create EGARCHParams from parameter dictionary.

    Args:
        params: Parameter dictionary.
        o: ARCH order.
        p: GARCH order.
        dist: Distribution name.

    Returns:
        EGARCHParams instance.

    Raises:
        KeyError: If required parameters are missing.
        ValueError: If parameter types don't match orders.
    """
    logger.debug(f"Creating EGARCH({o},{p}) parameters from dict")

    omega = params["omega"]

    # Extract alpha, gamma, beta using helpers
    alpha = _extract_alpha_from_dict(params, o)
    gamma = _extract_gamma_from_dict(params, o)
    beta = _extract_beta_from_dict(params, p)

    # Extract distribution parameters
    nu = params.get("nu")
    lambda_skew = params.get("lambda")

    # Extract metadata
    loglik = params.get("loglik")
    converged_val = params.get("converged")
    converged = bool(converged_val) if converged_val is not None else None

    return EGARCHParams(
        omega=omega,
        alpha=alpha,
        gamma=gamma,
        beta=beta,
        nu=nu,
        lambda_skew=lambda_skew,
        o=o,
        p=p,
        dist=dist,
        loglik=loglik,
        converged=converged,
    )


def create_egarch_params_from_optimization_result(
    result: Any,
    o: int,
    p: int,
    dist: str,
) -> EGARCHParams:
    """Create EGARCHParams from scipy optimization result.

    Args:
        result: Optimization result object with x, fun, success attributes.
        o: ARCH order.
        p: GARCH order.
        dist: Distribution name.

    Returns:
        EGARCHParams instance.
    """
    loglik = float(-result.fun) if hasattr(result, "fun") else None
    converged = bool(result.success) if hasattr(result, "success") else None

    return create_egarch_params_from_array(
        params=result.x,
        o=o,
        p=p,
        dist=dist,
        loglik=loglik,
        converged=converged,
    )


def create_egarch_params_from_array(
    params: np.ndarray,
    o: int,
    p: int,
    dist: str,
    loglik: float | None = None,
    converged: bool | None = None,
) -> EGARCHParams:
    """Create EGARCHParams from parameter array.

    Args:
        params: Parameter vector.
        o: ARCH order.
        p: GARCH order.
        dist: Distribution name.
        loglik: Optional log-likelihood value.
        converged: Optional convergence status.

    Returns:
        EGARCHParams instance.
    """
    logger.debug(f"Creating EGARCH({o},{p}) parameters from array")

    idx = 0
    omega = float(params[idx])
    idx += 1

    alpha, idx = _extract_alpha_from_array(params, idx, o)
    gamma, idx = _extract_gamma_from_array(params, idx, o)
    beta, idx = _extract_beta_from_array(params, idx, p)
    nu, lambda_skew = _extract_dist_params_from_array(params, idx, dist)

    return EGARCHParams(
        omega=omega,
        alpha=alpha,
        gamma=gamma,
        beta=beta,
        nu=nu,
        lambda_skew=lambda_skew,
        o=o,
        p=p,
        dist=dist,
        loglik=loglik,
        converged=converged,
    )
