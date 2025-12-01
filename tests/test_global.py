"""
Global test runner that executes all test files sequentially.

This script runs all test files in the tests/ directory one by one,
providing a comprehensive test suite execution with colored output.

Each test file is run in a separate subprocess to avoid mock pollution
and other side effects between test modules.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Add project root to Python path for direct execution.
_script_dir = Path(__file__).parent
# Find project root by looking for .git, pyproject.toml, or setup.py
_project_root = _script_dir.parent
while _project_root != _project_root.parent:
    if (_project_root / ".git").exists() or (_project_root / "pyproject.toml").exists() or (_project_root / "setup.py").exists():
        break
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def find_all_test_files() -> list[Path]:
    """Find all test_*.py files in the tests directory."""
    tests_dir = Path(__file__).parent
    test_files = list(tests_dir.rglob("test_*.py"))

    # Filter out files in tick/ directory, exclude test_global.py itself, and LSTM tests
    test_files = [
        f for f in test_files
        if "tick" not in str(f) and f.name != "test_global.py" and "lstm" not in f.name.lower()
    ]
    test_files.sort()

    return test_files


def run_all_tests_sequentially() -> int:
    """Run all test files sequentially in separate subprocesses."""
    test_files = find_all_test_files()

    if not test_files:
        print("No test files found!")
        return 1

    print("=" * 80)
    print("GLOBAL TEST RUNNER")
    print("=" * 80)
    print(f"Found {len(test_files)} test file(s) to execute")
    print("=" * 80)
    print()

    total_passed = 0
    total_failed = 0
    failed_files: list[Path] = []

    # Setup environment with PYTHONPATH
    env = os.environ.copy()
    pythonpath = str(_project_root)
    if "PYTHONPATH" in env:
        pythonpath = f"{pythonpath}:{env['PYTHONPATH']}"
    env["PYTHONPATH"] = pythonpath

    for i, test_file in enumerate(test_files, 1):
        relative_path = test_file.relative_to(Path(__file__).parent)
        print(f"\n[{i}/{len(test_files)}] Running: {relative_path}")
        print("-" * 80)

        # Run pytest in a separate subprocess to isolate side effects
        result = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                str(test_file),
                "-v",
                "--color=yes",
                "--tb=short",
            ],
            env=env,
            cwd=str(_project_root),
        )

        if result.returncode == 0:
            total_passed += 1
            print(f"✓ PASSED: {relative_path}")
        else:
            total_failed += 1
            failed_files.append(test_file)
            print(f"✗ FAILED: {relative_path} (exit code: {result.returncode})")

    # Print summary
    print()
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total test files: {len(test_files)}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")

    if failed_files:
        print("\nFailed test files:")
        for failed_file in failed_files:
            print(f"  - {failed_file.relative_to(Path(__file__).parent)}")

    print("=" * 80)

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    exit_code = run_all_tests_sequentially()
    sys.exit(exit_code)

