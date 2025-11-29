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

import pytest  # type: ignore
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

if __name__ == "__main__":
    # Allow running individual test file with pytest and colored output
    import pytest  # type: ignore
    pytest.main([__file__, "-v", "--color=yes"])
