"""GARCH package wrapper.

This package exposes submodules under `src.garch.*`. To keep imports
lightweight and avoid optional dependencies at import time, no heavy
re-exports are performed here. Import from the specific subpackages, e.g.:

    from src.garch.garch_diagnostic import diagnostics
    from src.garch.garch_params import estimation

"""

from __future__ import annotations

__all__: list[str] = []
