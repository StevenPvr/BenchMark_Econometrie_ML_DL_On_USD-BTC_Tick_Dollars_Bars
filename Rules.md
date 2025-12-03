# LLM Coding Rules

Scope: Python project under `src/` with data, ML, and time-series code.

---

## 0. Global Principles (ALL code)

- Apply KISS and DRY at all times.
- Follow PEP 8.
- Comments explain **WHY**, not **WHAT**.
- Prefer small, composable functions and clear data flow.

---

## 1. Code Standards (ALL files)

- No dead code: remove unused functions, variables, imports.
- No duplicated logic: factor out shared code into helpers.
- No hidden globals: avoid mutable global state.

---

## 2. Python Modules (.py) in `src/`

### 2.1 Type Hints & Functions

- Type hints required for:
  - All function parameters.
  - All return types.
- Max 40 lines per function.
- One task per function (Single Responsibility).
- Docstrings:
  - Mandatory for all public functions/classes.
  - Written in English.
  - Use Google or NumPy style.

### 2.2 Architecture

- Reusable, generic helpers → `src/utils.py`.
- ALL constants (magic numbers, defaults, EPS, thresholds) → `src/constants.py`.
- ALL paths (directories, filenames) → `src/path.py`.
- Never hardcode constants or paths in functions: import from `constants.py` / `path.py`.
- Create only functions explicitly requested by the user (no speculative API).
- Typical module layout:
  - `__init__.py`
  - main implementation file(s)
  - `main.py` for CLI / entry points
  - `test_*.py` for unit tests

### 2.3 Imports

- First line in EVERY module:
  - `from __future__ import annotations`
- Check `requirements.txt` **before** adding a new dependency.
- Never import a package that is not used.
- Prefer explicit imports over `import *`.

### 2.4 Logging & Errors

- Use project logger:
  - `from src.utils import get_logger`
  - `logger = get_logger(__name__)`
- No `print` for runtime information → use `logger.info` / `logger.warning` / `logger.error`.
- Raise explicit exceptions on invalid inputs:
  - `ValueError` for wrong argument values.
  - `TypeError` for wrong types.
  - Custom exceptions only if necessary and defined centrally.
- Validate inputs at the start of each public function.

### 2.5 Code Quality

- Remove unused variables and imports immediately.
- Never use undeclared variables (treat as bug).
- After editing or creating functions:
  - Run `get_errors` (or equivalent static checks).
  - Fix all reported errors before completion.
- Format with Black and lint with Ruff:
  - Tools come from `requirements-dev.txt`.

### 2.6 Testing

- For every non-trivial function, create a corresponding unit test.
- Test files: `test_<module_name>.py`.
- Use pytest:
  - Fixtures for shared setup.
  - `monkeypatch` for external dependencies (I/O, network, randomness, time).
- Never use real data files in tests (only synthetic/mocked data).
- Always test:
  - Normal case.
  - Edge cases (empty inputs, None, extreme values).
  - Invalid inputs (assert that exceptions are raised).

### 2.7 Side Effects & I/O

- Core logic functions must be **pure**:
  - Input arguments → return value.
  - No mutation of global state.
- No file I/O (`read_csv`, `to_csv`, etc.) inside core logic modules.
  - I/O only in `main.py` or dedicated I/O modules.
- No plotting inside model / feature / core logic modules.
  - Plotting only in dedicated plotting modules or notebooks.
- Logging is the only allowed side effect in core logic.

### 2.8 Public API

- Limit the public surface of each module:
  - Use `__all__` to expose only intended public functions/classes.
- Internal helpers:
  - Prefix with `_` (e.g. `_compute_step`).
  - Do not re-export via `__init__.py`.
- Keep imports inside `__init__.py` minimal and deliberate.

---

## 3. Data & Files (ALL code)

### 3.1 Directory Layout

- Input data: `data/`
- Results (generic): `results/`
- Plots: `plots/`
- Models: `results/models/`
- Metrics and evaluation outputs: `results/eval/`

### 3.2 File Formats

- Intermediate data: `.parquet` (performance, typed).
- Final/readable tables: `.csv`.
- Config / metadata / experiment configs: `.json`.
- Models: `joblib` files under `results/models/`.
- Metrics: JSON files under `results/eval/`.

### 3.3 Naming & Versioning

- Include timestamps and/or version tags in experiment artifacts:
  - Example: `results/eval/model_X_2025-11-01T120000Z.json`
- Use consistent naming conventions across code, artifacts, and plots.

---

## 4. Performance & DataFrame Engine

### 4.1 Vectorisation (NumPy / pandas)

- Default strategy for all array / Series / DataFrame operations.
- Vectorisation is **mandatory** when:
  - Applying the same operation to all rows/columns.
  - Creating features from existing columns (ratios, logs, rolling metrics, etc.).
  - Filtering with boolean masks.
  - Using `groupby`, `rolling`, `expanding`, aggregations, and joins.
- Forbidden patterns:
  - `for row in df.itertuples()` / `iterrows()` for large data when a vectorised equivalent exists.
- Loops over rows are acceptable only if:
  - The logic is intrinsically sequential and stateful (depends on previous step).
  - Or the data is **very small** (≈ ≤ 1 000 rows) and vectorisation would severely harm readability.
- When in doubt:
  - Prefer a clear vectorised solution over manually optimized Python loops.

### 4.2 Numba

- Numba is **never** used by default.
- Conditions to use Numba (`@njit(cache=True)`):
  - The function is a numeric hot loop (e.g. simulation, log-likelihood on long arrays, iterative optimisation step).
  - A clean vectorised solution is not available or would be unreadable.
  - The function uses only:
    - `numpy.ndarray`, numeric scalars, and Numba-supported operations.
    - No pandas / Polars / logging / I/O / Python objects.
- If using Numba:
  - Isolate the critical loop in a dedicated small function.
  - Keep the Numba-decorated function’s API simple and typed.
- Never apply Numba to:
  - Functions with side effects (I/O, logging).
  - Functions heavily using pandas, Polars, or complex Python objects.

### 4.3 Polars

- Do **not** introduce Polars unless:
  - It is already present in `requirements.txt`.
  - The new code processes very large tables (≈ ≥ 1–5 million rows) or complex multi-step pipelines where pandas is known to be a bottleneck.
- If using Polars:
  - Prefer staying in Polars from input to output (avoid repeated pandas ↔ Polars conversions).
  - Prefer lazy API (`scan_*`, `.lazy()`) for multi-step pipelines (filters, joins, aggregations).
  - Keep column naming conventions consistent with the rest of the project (snake_case, no spaces).
- Default for small to medium data and existing modules: **pandas**.

---

## 5. pandas / DataFrame / Time Series Conventions

- Time index:
  - Use `DatetimeIndex` for time series.
  - Always sort by time before time-based operations (rolling, resample, diff, lags).
- No implicit reliance on row order:
  - Sorting must be explicit.
- Indexing:
  - No chained indexing (`df[col][mask]`).
  - Use `.loc` / `.iloc` consistently.
- Function behavior:
  - Prefer returning new DataFrame/Series objects.
  - If mutating in-place, state it clearly in the docstring.
- Column naming:
  - Use snake_case: `log_return`, `sigma2`, `volume`, `timestamp`.
  - Avoid spaces and special characters in column names.
- Missing values:
  - Handle `NaN` explicitly (drop, fill, or mask).
  - Never silently ignore `NaN` in critical computations.

---

## 6. Numerical Stability & Optimisation Code

- All small constants (epsilons, min variance, clipping thresholds) must come from `constants.py`:
  - No hardcoded `1e-8`, `1e-12`, etc. directly in code.
- When computing quantities that must remain positive (variance, sigma, probabilities):
  - Clip using named constants from `constants.py`.
- Use log-domain formulations where appropriate:
  - Log-likelihood instead of product of probabilities.
  - `logsumexp` pattern to avoid underflow / overflow.
- Optimisation routines:
  - Centralize default optimizer options (e.g. `maxiter`, `ftol`, `gtol`, `eps`) in `constants.py`.
  - Log:
    - Convergence status.
    - Number of iterations.
    - Final objective value.
- Before optimisation:
  - Validate that inputs contain no `NaN` or `inf`.
  - Raise a clear error if validation fails.

---

## 7. Machine Learning Modules

### 7.1 Reproducibility

- Use `DEFAULT_RANDOM_STATE` from `constants.py` everywhere randomness is used:
  - NumPy.
  - pandas (if applicable).
  - scikit-learn.
  - ML libraries (LightGBM, XGBoost, etc.).
- Document all dataset splits:
  - Train/test.
  - Cross-validation (KFold, TimeSeriesSplit, walk-forward, etc.).
- Save:
  - Trained models under `results/models/`.
  - Metrics and evaluation summaries (JSON) under `results/eval/`.

### 7.2 Data Flow

- Separate:
  - Feature engineering (functions that transform raw data).
  - Model training.
  - Evaluation / metrics computation.
- No training logic inside plotting functions or notebooks-only code.

---

## 8. Notebooks / Experiments

- Notebooks must **not** contain core business logic.
  - Only call functions defined in `src/`.
- Any code used more than once in a notebook:
  - Move it into a `src/` module.
  - Import it back into the notebook.
- Each experiment should:
  - Save its config (hyperparameters, dataset, split) to JSON in `results/`.
  - Save metrics to `results/eval/`.
  - Save plots to `plots/`.
- Experiments must be deterministic:
  - Set seeds consistently (respect `DEFAULT_RANDOM_STATE`).

---

## 9. Interaction with User / Specifications

- Never invent methodology or assumptions not specified by the user.
- When requirements are ambiguous or underspecified:
  - Ask for clarification before designing the API or choosing the approach.
- If multiple design options exist:
  - Briefly state the options.
  - Ask the user to choose or specify preferences (performance vs simplicity, etc.).

---
