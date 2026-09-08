"""Turn SDK market records into backtest inputs.

The engine in :mod:`supagamma.backtest` is deliberately network-free. This module
is the thin, *optional* bridge that pulls resolved markets through a live client
and normalizes them into :class:`~supagamma.backtest.ResolvedMarket` objects.

Two prices you can bet at:

* **metadata** (default, free) -- the probability implied by the market record's
  ``outcome_prices``. Fast, but on a *resolved* market this is often the
  settlement price (~0 or ~1), which is useless for a calibration study. Fine for
  smoke tests; not fine for research.
* **VWAP at a horizon** (:func:`vwap_price_fn`, costs money) -- the volume-weighted
  price from the actual trade tape up to N hours *before* resolution. This is the
  honest entry price for calibration and backtests. Every call debits your balance
  (``download.trades`` is $3.00/MB), so it is opt-in and never automatic.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from ._engine import MarketView, ResolvedMarket

Record = Dict[str, Any]

# A price function maps a raw market record -> the YES probability to enter at,
# or None to skip the market. It may make paid API calls; that is the caller's
# explicit choice when they pass one in.
PriceFn = Callable[["Any", Record], Optional[float]]


# --------------------------------------------------------------------------- #
# Parsing the market record. Defensive: fields arrive as lists, dicts, or JSON
# strings depending on the upstream venue.
# --------------------------------------------------------------------------- #

def _as_sequence(value: Any) -> Optional[Sequence[Any]]:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return None
    if isinstance(value, dict):
        return list(value.values())
    if isinstance(value, (list, tuple)):
        return value
    return None


def yes_probability(market: Record) -> Optional[float]:
    """Implied P(YES) from the record's ``outcome_prices``.

    Treats ``outcomes[0]`` as the YES leg. Returns None when it cannot be read.
    """
    prices = _as_sequence(market.get("outcome_prices"))
    if not prices:
        return None
    try:
        p = float(prices[0])
    except (TypeError, ValueError):
        return None
    return p if 0.0 <= p <= 1.0 else None


def realized_outcome(market: Record) -> Optional[int]:
    """Realized outcome as 1 (YES) or 0 (NO), or None if not cleanly resolved.

    ``outcomes[0]`` is the YES leg. ``winning_outcome`` may be a label string
    ("Yes"/"No"/the outcome text) or an integer index.
    """
    win = market.get("winning_outcome")
    if win is None:
        return None
    outcomes = _as_sequence(market.get("outcomes")) or ["Yes", "No"]

    if isinstance(win, bool):
        return 1 if win else 0
    if isinstance(win, int):
        return 1 if win == 0 else 0                     # index 0 == YES leg
    if isinstance(win, str):
        w = win.strip().lower()
        if w in ("yes", "true", "1"):
            return 1
        if w in ("no", "false", "0"):
            return 0
        if outcomes:
            first = str(outcomes[0]).strip().lower()
            return 1 if w == first else 0
    return None


# --------------------------------------------------------------------------- #
# The paid, honest entry price: VWAP from the trade tape before resolution.
# --------------------------------------------------------------------------- #

def vwap_price_fn(horizon_hours: float = 24.0, *, trade_limit: int = 100_000) -> PriceFn:
    """Build a price function that reads the trade tape (this **costs money**).

    Returns the volume-weighted average trade price over the window ending
    ``horizon_hours`` before the market's ``resolution_date`` -- i.e. the price
    while the outcome was still genuinely uncertain, not the settlement print.

    Each produced call invokes ``client.download.trades`` and debits your
    balance. Use a small ``max_markets`` when exploring.
    """

    def _price(client: Any, market: Record) -> Optional[float]:
        end = _parse_dt(market.get("resolution_date")) or _parse_dt(market.get("end_date"))
        cutoff = (end - timedelta(hours=horizon_hours)) if end else None
        try:
            result = client.download.trades(
                market_id=str(market["id"]),
                end=cutoff,
                format="parquet",
                limit=trade_limit,
            )
        except Exception:
            return None
        return _vwap_from_parquet(result.content)

    return _price


def _vwap_from_parquet(content: bytes) -> Optional[float]:
    """VWAP of (price, size) rows from a Parquet blob. Needs pyarrow if present."""
    try:
        import io

        import pyarrow.parquet as pq
    except Exception:
        return None
    try:
        table = pq.read_table(io.BytesIO(content), columns=["price", "size"])  # type: ignore[no-untyped-call]
    except Exception:
        try:
            table = pq.read_table(io.BytesIO(content), columns=["price"])  # type: ignore[no-untyped-call]
        except Exception:
            return None
    prices = [float(x) for x in table.column("price").to_pylist() if x is not None]
    if not prices:
        return None
    if "size" in table.column_names:
        sizes = [float(x or 0.0) for x in table.column("size").to_pylist()]
        denom = sum(sizes)
        if denom > 0:
            return sum(p * s for p, s in zip(prices, sizes)) / denom
    return sum(prices) / len(prices)


def _parse_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# The one entry point most callers use.
# --------------------------------------------------------------------------- #

def resolved_markets(
    client: Any,
    *,
    max_markets: Optional[int] = 500,
    page_limit: int = 500,
    price_fn: Optional[PriceFn] = None,
    **filters: Any,
) -> Iterator[ResolvedMarket]:
    """Yield resolved markets as backtest inputs.

    Pulls ``resolved=True, has_data=True`` markets (plus any extra ``filters``
    like ``category=`` or ``tag=``) via ``client.markets.auto_paginate`` and
    normalizes each into a :class:`ResolvedMarket`. Markets that cannot be read
    cleanly (missing outcome or price) are skipped.

    ``price_fn`` overrides the entry price -- pass :func:`vwap_price_fn` for a
    real study (it costs money). Without it, the free metadata price is used.
    """
    seen = 0
    for market in client.markets.auto_paginate(
        resolved=True, has_data=True, limit=page_limit, **filters
    ):
        outcome = realized_outcome(market)
        if outcome is None:
            continue
        prob = price_fn(client, market) if price_fn else yes_probability(market)
        if prob is None or not (0.0 <= prob <= 1.0):
            continue
        view = MarketView(
            id=str(market.get("id", "")),
            question=str(market.get("question", "")),
            prob=prob,
            meta={
                "category": market.get("category"),
                "volume": market.get("volume"),
                "resolution_date": market.get("resolution_date"),
            },
        )
        yield ResolvedMarket(view=view, outcome=outcome)
        seen += 1
        if max_markets is not None and seen >= max_markets:
            return


def calibration_pairs(markets: Sequence[ResolvedMarket]) -> List[Tuple[float, int]]:
    """(forecast_yes, outcome) pairs for :func:`supagamma.backtest.calibration`."""
    return [(m.view.prob, m.outcome) for m in markets]
