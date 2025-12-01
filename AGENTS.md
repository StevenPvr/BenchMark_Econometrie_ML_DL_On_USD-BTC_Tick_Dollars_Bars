# Repository Guidelines

## Project Structure & Module Organization

- `src/` holds the pipeline: `data_fetching` (ccxt downloads), `data_cleaning`, `data_preparation`, feature builders (`features/`, `clear_features/`), labeling (`labelling/`), and models (`model/` plus `model/baseline/`). Shared helpers live in `src/utils/`, global constants in `src/constants.py`, paths in `src/path.py`, and logging defaults in `src/config_logging.py`.
- `tests/` mirrors `src/` with pytest suites for each module; use it as a template for new tests.
- `scripts/run_data_fetching_daemon.sh` controls the long-running fetcher; data artifacts land in `data/` and logs in `logs/`. Keep generated Parquet files out of commits.

## Environment Setup

- Python 3.10+ with a virtualenv: `python -m venv venv && source venv/bin/activate`.
- Install dependencies: `pip install -r requirements.txt`.
- Run commands from the repo root so paths in `src/path.py` resolve correctly.

## Build, Test, and Development Commands

- Run all tests: `python -m pytest tests`; keep the suite green before pushing.
- Focused runs: `python -m pytest tests/data_preparation/test_preparation.py` or append `-k <pattern>` for quick loops.
- Lint/format: `ruff check .`, `black src tests`, `mypy src` (prefer fixing types before merge).
- Data fetch daemon: `./scripts/run_data_fetching_daemon.sh start|status|logs|stop` (details in `README_DATA_FETCHING_DAEMON.md`).

## Coding Style & Naming Conventions

- Follow PEP 8 with 4-space indentation; add type hints to public functions/classes.
- snake_case for modules/functions/vars, PascalCase for classes, UPPER_SNAKE_CASE for constants (centralize shared values in `constants.py`).
- Import paths via `src/path.py` instead of hardcoding locations; log through `logging` with `config_logging.py`.
- Document non-obvious logic with short docstrings; prefer descriptive parameter names.

## Testing Guidelines

- Name test files `test_<module>.py` and mirror the structure under `tests/`. Keep tests deterministic; seed randomness when present.
- Add regression tests with each feature or fix. Reuse fixtures from `tests/conftest.py` instead of duplicating setup.
- Before a PR, run `python -m pytest tests` plus targeted suites you touched; maintain current coverage.

## Commit & Pull Request Guidelines

- Use the existing imperative style (`Add documentation README.md`, `Add data cleaning improvements`); scope commits narrowly.
- PRs should include a change summary, rationale, linked issue, test commands/results, and notes on data requirements or breaking changes. Add screenshots or log snippets when altering plots, metrics, or daemon behavior.
- Do not commit large artifacts from `data/` or `logs/`; rely on scripts and instructions to regenerate outputs.
