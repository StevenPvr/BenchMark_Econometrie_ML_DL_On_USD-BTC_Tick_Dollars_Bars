"""Exchange connection and validation utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.constants import EXCHANGE_ID, SYMBOL
from src.utils import get_logger

if TYPE_CHECKING:
    import ccxt as ccxt_module

logger = get_logger(__name__)

# Optional dependency; tests monkeypatch this attribute
try:
    import ccxt as _ccxt
except ImportError:
    _ccxt = None  # type: ignore[assignment]

ccxt: ccxt_module | None = _ccxt  # type: ignore[name-defined]


def build_exchange() -> ccxt_module.Exchange:
    """Create ccxt exchange instance using EXCHANGE_ID.

    Returns:
        Configured ccxt exchange instance.

    Raises:
        ValueError: If exchange ID is not found in ccxt.
    """
    global ccxt
    if ccxt is None:
        import importlib

        ccxt = importlib.import_module("ccxt")
    if not hasattr(ccxt, EXCHANGE_ID):
        msg = f"Exchange {EXCHANGE_ID} not found in ccxt"
        raise ValueError(msg)
    return getattr(ccxt, EXCHANGE_ID)()


def validate_symbol(exchange: ccxt_module.Exchange) -> None:
    """Ensure requested symbol exists and is active on the exchange.

    Args:
        exchange: ccxt exchange instance.

    Raises:
        ValueError: If symbol is not available or inactive.
    """
    markets = exchange.load_markets()
    if SYMBOL not in markets:
        msg = f"Symbole {SYMBOL} non disponible sur {EXCHANGE_ID}"
        raise ValueError(msg)
    market_info = markets.get(SYMBOL, {})
    if isinstance(market_info, dict) and not market_info.get("active", True):
        msg = f"Symbole {SYMBOL} inactif sur {EXCHANGE_ID}"
        raise ValueError(msg)


__all__ = [
    "build_exchange",
    "ccxt",
    "validate_symbol",
]
