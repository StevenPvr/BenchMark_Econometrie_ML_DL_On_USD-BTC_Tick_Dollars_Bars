# Repository Guidelines

## Project Structure & Module Organization

- `src/` contains the full pipeline: `data_fetching`, `data_cleaning`, and `data_preparation` for ETL; `arima/` and `garch/` econometrics; ML models in `model/`; visualization helpers in `data_visualisation/` and `visualization/`; shared configuration in `path.py` and `constants.py`.
- Tests live in `tests/`, mirroring the data modules (e.g., `tests/data_preparation/test_preparation.py`); keep new tests alongside the feature you touch.
- Data is split under `data/raw`, `data/cleaned`, and `data/prepared`; output/result/plot folders are defined in `src/path.py`—extend paths there rather than hard-coding.
- `venv/` is a local environment; prefer your own `.venv` to avoid polluting it or committing site packages.

## Build, Test, and Development Commands

- Create/activate an env: `python -m venv .venv && source .venv/bin/activate` (or reuse `source venv/bin/activate`).
- Install Python deps with pip (match imports such as pandas, numpy, statsmodels, lightgbm/xgboost/catboost, torch where used); if you keep a lockfile, use `python -m pip install -r requirements.txt`.
- Run all tests: `python -m pytest tests -q`. Target a module: `python -m pytest tests/data_preparation/test_preparation.py -k adaptive`.
- Execute pipeline stages via module entrypoints, e.g., `python -m src.data_fetching.main` → `python -m src.data_cleaning.main` → `python -m src.data_preparation.main`.

## Coding Style & Naming Conventions

- Follow PEP 8 with 4-space indentation and type hints (`from __future__ import annotations` is standard here).
- Use snake_case for modules/functions/variables and PascalCase for classes. Keep functions pure where possible; pass paths or DataFrames explicitly.
- Centralize file locations in `path.py`/`constants.py`; avoid embedding absolute paths in notebooks or scripts.
- Add docstrings to public functions and brief inline comments only for non-obvious logic.

## Testing Guidelines

- Tests use pytest with lightweight DataFrames and temp directories; avoid touching real `data/` files in unit tests.
- When adding features, assert schema details (columns/index types) in addition to counts, mirroring the dollar-bar coverage.
- Seed randomness for reproducibility, and prefer deterministic fixtures over network/file I/O.

## Commit & Pull Request Guidelines

- Keep commits focused with concise, present-tense messages; the existing history is short and descriptive (French or English is fine).
- PR descriptions should summarize the pipeline step touched, data assumptions, and how to reproduce (commands and expected outputs/plots under `outputs/`).
- Verify `python -m pytest` before requesting review and call out any skipped/expensive cases. Do not commit large raw data, generated artifacts, or virtualenv directories.
