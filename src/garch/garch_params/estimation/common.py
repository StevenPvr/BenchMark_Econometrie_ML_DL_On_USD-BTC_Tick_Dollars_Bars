"""Common utilities for EGARCH MLE estimation.

This module consolidates convergence tracking and the thread-safe MLE cache
to reduce file count while keeping related utilities together.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)


# =============================== Convergence ===============================


@dataclass
class ConvergenceResult:
    """Convergence result for a single estimation.

    Attributes:
        converged: Whether optimizer converged successfully.
        n_iterations: Number of optimizer iterations (if available).
        final_loglik: Final log-likelihood value.
        message: Optimizer message (if available).
    """

    converged: bool
    n_iterations: int | None
    final_loglik: float
    message: str | None = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        parts = [f"converged={self.converged}"]
        if self.n_iterations is not None:
            parts.append(f"n_iter={self.n_iterations}")
        parts.append(f"loglik={self.final_loglik:.2f}")
        if self.message:
            parts.append(f"msg='{self.message}'")
        return ", ".join(parts)


class ConvergenceTracker:
    """Track convergence across multiple estimations."""

    def __init__(self) -> None:
        self.results: list[ConvergenceResult] = []

    def add_result(self, result: ConvergenceResult) -> None:
        self.results.append(result)

    def add_from_optimizer_result(self, opt_result: Any) -> None:
        converged = bool(opt_result.success) if hasattr(opt_result, "success") else False
        n_iterations = int(opt_result.nit) if hasattr(opt_result, "nit") else None
        final_loglik = float(-opt_result.fun) if hasattr(opt_result, "fun") else float("nan")
        message = str(opt_result.message) if hasattr(opt_result, "message") else None

        result = ConvergenceResult(
            converged=converged,
            n_iterations=n_iterations,
            final_loglik=final_loglik,
            message=message,
        )
        self.add_result(result)

    def compute_convergence_rate(self) -> float:
        if not self.results:
            return 0.0
        n_converged = sum(1 for r in self.results if r.converged)
        return n_converged / len(self.results)

    def get_summary(self) -> dict[str, Any]:
        if not self.results:
            return {
                "n_estimations": 0,
                "n_converged": 0,
                "convergence_rate": 0.0,
                "mean_iterations": None,
                "mean_loglik": None,
            }

        n_converged = sum(1 for r in self.results if r.converged)
        convergence_rate = n_converged / len(self.results)

        iterations = [r.n_iterations for r in self.results if r.n_iterations is not None]
        mean_iterations = sum(iterations) / len(iterations) if iterations else None

        logliks = [r.final_loglik for r in self.results if r.converged]
        mean_loglik = sum(logliks) / len(logliks) if logliks else None

        return {
            "n_estimations": len(self.results),
            "n_converged": n_converged,
            "convergence_rate": convergence_rate,
            "mean_iterations": mean_iterations,
            "mean_loglik": mean_loglik,
        }

    def log_summary(self, min_rate: float = 0.95) -> None:  # pragma: no cover - logging
        summary = self.get_summary()
        if summary["n_estimations"] == 0:
            logger.warning("No estimations tracked")
            return
        conv_rate = summary["convergence_rate"]
        logger.info(
            "Convergence summary: %d/%d succeeded (%.1f%%)",
            summary["n_converged"],
            summary["n_estimations"],
            conv_rate * 100,
        )
        if summary["mean_iterations"] is not None:
            logger.info("Mean iterations: %.1f", summary["mean_iterations"])
        if conv_rate < min_rate:
            logger.warning(
                "Low convergence rate: %.1f%% < %.1f%% threshold",
                conv_rate * 100,
                min_rate * 100,
            )

    def get_failed_results(self) -> list[ConvergenceResult]:
        return [r for r in self.results if not r.converged]

    def clear(self) -> None:
        self.results.clear()


def extract_convergence_info(opt_result: Any) -> ConvergenceResult:
    converged = bool(opt_result.success) if hasattr(opt_result, "success") else False
    n_iterations = int(opt_result.nit) if hasattr(opt_result, "nit") else None
    final_loglik = float(-opt_result.fun) if hasattr(opt_result, "fun") else float("nan")
    message = str(opt_result.message) if hasattr(opt_result, "message") else None

    return ConvergenceResult(
        converged=converged,
        n_iterations=n_iterations,
        final_loglik=final_loglik,
        message=message,
    )


# ================================ MLE Cache ================================


def _compute_residual_hash(residuals: np.ndarray) -> str:
    """Compute a hash of residuals for cache key generation."""
    residual_bytes = residuals.tobytes()
    return hashlib.sha256(residual_bytes).hexdigest()


def _create_cache_key(residuals: np.ndarray, o: int, p: int, dist: str) -> str:
    """Create cache key for MLE estimation."""
    residual_hash = _compute_residual_hash(residuals)
    return f"{residual_hash}_{o}_{p}_{dist}"


class MLECache:
    """Thread-safe cache for EGARCH MLE estimations."""

    def __init__(self, max_size: int = 1000) -> None:
        if max_size <= 0:
            msg = f"max_size must be positive, got {max_size}"
            raise ValueError(msg)
        self.cache: dict[str, tuple[dict[str, float], Any]] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self._access_order: list[str] = []

    def get(
        self, residuals: np.ndarray, o: int, p: int, dist: str
    ) -> tuple[dict[str, float], Any] | None:
        key = _create_cache_key(residuals, o, p, dist)
        if key in self.cache:
            self.hits += 1
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            logger.debug(
                "Cache hit for key=%s (hits=%d, misses=%d)", key[:16], self.hits, self.misses
            )
            return self.cache[key]

        self.misses += 1
        logger.debug("Cache miss for key=%s (hits=%d, misses=%d)", key[:16], self.hits, self.misses)
        return None

    def put(
        self,
        residuals: np.ndarray,
        o: int,
        p: int,
        dist: str,
        params: dict[str, float],
        convergence: Any,
    ) -> None:
        key = _create_cache_key(residuals, o, p, dist)

        if len(self.cache) >= self.max_size and key not in self.cache:
            if self._access_order:
                lru_key = self._access_order.pop(0)
                del self.cache[lru_key]
                logger.debug("Evicted LRU key=%s from cache", lru_key[:16])

        self.cache[key] = (params, convergence)
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        logger.debug(
            "Stored result in cache for key=%s (size=%d/%d)",
            key[:16],
            len(self.cache),
            self.max_size,
        )

    def clear(self) -> None:
        self.cache.clear()
        self._access_order.clear()
        self.hits = 0
        self.misses = 0
        logger.debug("Cache cleared")

    def get_stats(self) -> dict[str, int | float]:
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": hit_rate,
        }


# Global cache instance
_global_cache: MLECache | None = None


def get_global_cache() -> MLECache:
    global _global_cache  # noqa: PLW0603
    if _global_cache is None:
        _global_cache = MLECache(max_size=1000)
    return _global_cache


def clear_global_cache() -> None:
    global _global_cache  # noqa: PLW0603
    if _global_cache is not None:
        _global_cache.clear()


__all__ = [
    # Convergence
    "ConvergenceResult",
    "ConvergenceTracker",
    "extract_convergence_info",
    # Cache
    "MLECache",
    "get_global_cache",
    "clear_global_cache",
    "_compute_residual_hash",
    "_create_cache_key",
]
