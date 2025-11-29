import pytest
from src.analyse_features.utils.parallel import (
    get_n_jobs,
    parallel_map,
    parallel_apply,
    chunked_parallel,
    ParallelContext
)

def square(x):
    return x * x

def add(x, y):
    return x + y

def process_chunk(chunk):
    return [x * 2 for x in chunk]

class TestParallel:

    def test_get_n_jobs(self):
        assert get_n_jobs(1) == 1
        assert get_n_jobs(-1) > 0 # Should use all cores

    def test_parallel_map(self):
        items = [1, 2, 3, 4]
        results = parallel_map(square, items, n_jobs=2)
        assert results == [1, 4, 9, 16]

    def test_parallel_apply(self):
        items = [(1, 2), (3, 4)]
        results = parallel_apply(add, items, n_jobs=2)
        assert results == [3, 7]

    def test_chunked_parallel(self):
        items = [1, 2, 3, 4, 5]
        results = chunked_parallel(process_chunk, items, chunk_size=2, n_jobs=2)
        assert results == [2, 4, 6, 8, 10]

    def test_parallel_context(self):
        items = [1, 2, 3]
        with ParallelContext(n_jobs=2) as ctx:
            results = ctx.map(square, items)
        assert results == [1, 4, 9]

    def test_sequential_execution(self):
        # n_jobs=1 should run sequentially
        items = [1, 2, 3]
        results = parallel_map(square, items, n_jobs=1)
        assert results == [1, 4, 9]
