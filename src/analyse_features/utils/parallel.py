"""Parallelization utilities for feature analysis.

This module provides optimized parallel execution wrappers:
- joblib-based parallelization with progress tracking
- Chunked processing for memory efficiency
- Context managers for parallel pools

Performance notes:
- Uses 'loky' backend by default (process-based, avoids GIL)
- Supports 'threading' backend for I/O-bound tasks
- Automatic chunk sizing based on data size
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, TypeVar

from joblib import Parallel, delayed, cpu_count

from src.analyse_features.config import (
    CHUNK_SIZE,
    JOBLIB_BACKEND,
    JOBLIB_VERBOSITY,
    N_JOBS,
)

T = TypeVar("T")
R = TypeVar("R")


def get_n_jobs(n_jobs: int | None = None) -> int:
    """Get the number of parallel jobs to use.

    Args:
        n_jobs: Requested number of jobs.
            - None or -1: Use all available cores
            - Positive int: Use that many cores
            - Negative int: Use (n_cores + 1 + n_jobs) cores

    Returns:
        Number of jobs to use.
    """
    if n_jobs is None:
        n_jobs = N_JOBS

    n_cores = cpu_count() or 1

    if n_jobs == -1:
        return n_cores
    elif n_jobs < -1:
        return max(1, n_cores + 1 + n_jobs)
    else:
        return min(n_jobs, n_cores)


def parallel_map(
    func: Callable[[T], R],
    items: Iterable[T],
    n_jobs: int | None = None,
    backend: str | None = None,
    verbose: int | None = None,
    desc: str | None = None,
) -> list[R]:
    """Apply a function to items in parallel.

    This is a simple wrapper around joblib.Parallel for map-style operations.

    Args:
        func: Function to apply to each item.
        items: Iterable of items to process.
        n_jobs: Number of parallel jobs (default: N_JOBS from config).
        backend: Joblib backend ('loky', 'multiprocessing', 'threading').
        verbose: Verbosity level (0-10).
        desc: Description for logging (unused, for API compatibility).

    Returns:
        List of results in the same order as inputs.

    Example:
        >>> def square(x):
        ...     return x ** 2
        >>> parallel_map(square, [1, 2, 3, 4, 5])
        [1, 4, 9, 16, 25]
    """
    n_jobs = get_n_jobs(n_jobs)
    backend = backend or JOBLIB_BACKEND
    verbose = verbose if verbose is not None else JOBLIB_VERBOSITY

    items_list = list(items)

    if n_jobs == 1 or len(items_list) == 1:
        return [func(item) for item in items_list]

    results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(func)(item) for item in items_list
    )

    return results


def parallel_apply(
    func: Callable[..., R],
    items: Iterable[tuple[Any, ...]],
    n_jobs: int | None = None,
    backend: str | None = None,
    verbose: int | None = None,
) -> list[R]:
    """Apply a function with multiple arguments in parallel.

    Similar to parallel_map but for functions that take multiple arguments.

    Args:
        func: Function to apply (takes multiple args).
        items: Iterable of argument tuples.
        n_jobs: Number of parallel jobs.
        backend: Joblib backend.
        verbose: Verbosity level.

    Returns:
        List of results.

    Example:
        >>> def add(a, b):
        ...     return a + b
        >>> parallel_apply(add, [(1, 2), (3, 4), (5, 6)])
        [3, 7, 11]
    """
    n_jobs = get_n_jobs(n_jobs)
    backend = backend or JOBLIB_BACKEND
    verbose = verbose if verbose is not None else JOBLIB_VERBOSITY

    items_list = list(items)

    if n_jobs == 1 or len(items_list) == 1:
        return [func(*args) for args in items_list]

    results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(func)(*args) for args in items_list
    )

    return results


def chunked_parallel(
    func: Callable[[list[T]], list[R]],
    items: list[T],
    chunk_size: int | None = None,
    n_jobs: int | None = None,
    backend: str | None = None,
    verbose: int | None = None,
) -> list[R]:
    """Process items in chunks using parallel execution.

    Useful for memory-intensive operations where processing all items
    at once would exhaust memory.

    Args:
        func: Function that processes a chunk (list) of items.
        items: Full list of items to process.
        chunk_size: Number of items per chunk (default: CHUNK_SIZE from config).
        n_jobs: Number of parallel jobs.
        backend: Joblib backend.
        verbose: Verbosity level.

    Returns:
        Flattened list of all results.

    Example:
        >>> def process_chunk(chunk):
        ...     return [x * 2 for x in chunk]
        >>> chunked_parallel(process_chunk, [1, 2, 3, 4, 5], chunk_size=2)
        [2, 4, 6, 8, 10]
    """
    chunk_size = chunk_size or CHUNK_SIZE
    n_jobs = get_n_jobs(n_jobs)
    backend = backend or JOBLIB_BACKEND
    verbose = verbose if verbose is not None else JOBLIB_VERBOSITY

    chunks = [
        items[i : i + chunk_size]
        for i in range(0, len(items), chunk_size)
    ]

    if n_jobs == 1 or len(chunks) == 1:
        all_results: list[R] = []
        for chunk in chunks:
            all_results.extend(func(chunk))
        return all_results

    chunk_results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(func)(chunk) for chunk in chunks
    )

    all_results = []
    for result in chunk_results:
        all_results.extend(result)

    return all_results


def parallel_dataframe_apply(
    df_func: Callable[..., Any],
    df_columns: list[str],
    n_jobs: int | None = None,
    **kwargs: Any,
) -> list[Any]:
    """Apply a function to DataFrame columns in parallel.

    Specialized for column-wise operations on DataFrames.

    Args:
        df_func: Function that takes (column_name, **kwargs) and returns result.
        df_columns: List of column names to process.
        n_jobs: Number of parallel jobs.
        **kwargs: Additional arguments passed to df_func.

    Returns:
        List of results for each column.
    """
    n_jobs = get_n_jobs(n_jobs)

    def process_column(col: str) -> Any:
        return df_func(col, **kwargs)

    return parallel_map(process_column, df_columns, n_jobs=n_jobs)


class ParallelContext:
    """Context manager for parallel execution with shared resources.

    Useful when you need to share data across parallel workers.

    Example:
        >>> with ParallelContext(n_jobs=4) as ctx:
        ...     results = ctx.map(my_func, items)
    """

    def __init__(
        self,
        n_jobs: int | None = None,
        backend: str | None = None,
        verbose: int | None = None,
    ) -> None:
        """Initialize parallel context.

        Args:
            n_jobs: Number of parallel jobs.
            backend: Joblib backend.
            verbose: Verbosity level.
        """
        self.n_jobs = get_n_jobs(n_jobs)
        self.backend = backend or JOBLIB_BACKEND
        self.verbose = verbose if verbose is not None else JOBLIB_VERBOSITY
        self._parallel: Parallel | None = None

    def __enter__(self) -> ParallelContext:
        """Enter the context."""
        self._parallel = Parallel(
            n_jobs=self.n_jobs,
            backend=self.backend,
            verbose=self.verbose,
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit the context."""
        self._parallel = None

    def map(self, func: Callable[[T], R], items: Iterable[T]) -> list[R]:
        """Map function over items."""
        if self._parallel is None:
            raise RuntimeError("ParallelContext not entered")
        return self._parallel(delayed(func)(item) for item in items)
