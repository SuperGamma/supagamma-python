"""Turn SDK market records into backtest inputs.

The engine in :mod:`supagamma.backtest` is deliberately network-free. This module
is the thin, *optional* bridge that pulls resolved markets through a live client
and normalizes them into :class:`~supagamma.backtest.ResolvedMarket` objects.

Two prices you can bet at:

* **metadata** (default, free) -- the probability implied by the market record's
  ``outcome_prices``. Fast, but on a *resolved* market this is often the
  settlement price (~0 or ~1), which is useless for a calibration study. Fine for
  smoke tests; not fine for research.
* **VWAP at a horizon** (:func:`vwap_price_fn`, metered) -- the size-weighted price
  of the YES outcome from the actual trade tape up to N hours *before* resolution.
  This is the honest entry price for calibration and backtests. Every price it
  computes is one paid ``download.trades`` request (it counts toward your plan's
  fair-use volume, or debits your balance on a pay-as-you-go deployment), so it is
  opt-in, never automatic, and :func:`resolved_markets` hard-bounds how many it
  may make.

Why the VWAP is not just ``mean(price)``: a trade file holds the fills of BOTH of
a market's outcome tokens, and the two tokens price each other (YES at 0.80 is NO
at 0.20). Pooling them lands near 0.5 whatever the market thinks, so every row has
to be put on the YES scale first. See :func:`yes_vwap`.
"""

from __future__ import annotations

import io
import json
import math
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .._errors import (
    MissingDependencyError,
    NoDataInRangeError,
    ResponseShapeError,
    SupaGammaConfigError,
)
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
        return 1 if win == 0 else 0  # index 0 == YES leg
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

#: The server's own bounds on ``download.trades(limit=...)``.
_MAX_TRADE_LIMIT = 1_000_000


def _require_pyarrow() -> Any:
    """Import ``pyarrow.parquet`` or raise, **before any request is made**.

    pyarrow is an optional extra. Checking it only after a paid download would pay
    for a file this process then cannot read.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise MissingDependencyError(
            "vwap_price_fn reads Parquet trade files and needs pyarrow, which is not "
            'installed. Install it with `pip install "supagamma[parquet]"` (or '
            "`pip install pyarrow`). No request was made."
        ) from exc
    return pq


def yes_vwap(
    trades: Iterable[Tuple[Any, Any, Any]],
    *,
    yes_token: str,
    no_token: str,
) -> Optional[float]:
    """Size-weighted average price of the YES outcome, from fills of BOTH tokens.

    ``trades`` yields ``(token_id, price, size)`` rows, with ``token_id`` the
    ``market_id`` column of a trade file (the outcome token id). A YES-token row
    contributes its ``price``; a NO-token row contributes ``1 - price``, because a
    NO share at 0.20 is the same view as a YES share at 0.80. A row for any other
    token, or with a missing, non-finite or non-positive price or size, is
    ignored. Weights are ``size``, the collateral value of the fill as delivered.

    Without that conversion a market that is truly 80% YES averages to about 0.5,
    because its NO-token fills sit near 0.20.

    Returns None when no usable row remains.
    """
    yes, no = str(yes_token), str(no_token)
    if yes == no:
        raise ValueError("yes_token and no_token must be different tokens")
    weighted = 0.0
    total = 0.0
    for token, price, size in trades:
        if price is None or size is None:
            continue
        key = str(token)
        if key == yes:
            yes_price = float(price)
        elif key == no:
            yes_price = 1.0 - float(price)
        else:
            continue
        weight = float(size)
        if not (weight > 0.0 and math.isfinite(weight) and math.isfinite(yes_price)):
            continue
        weighted += yes_price * weight
        total += weight
    return weighted / total if total > 0.0 else None


def _vwap_from_parquet(
    content: bytes,
    *,
    yes_token: str,
    no_token: str,
    pq: Any,
    request_id: Optional[str] = None,
) -> Optional[float]:
    """:func:`yes_vwap` over a downloaded Parquet trade file."""
    try:
        table = pq.read_table(io.BytesIO(content), columns=["market_id", "price", "size"])
        tokens = table.column("market_id").to_pylist()
        prices = table.column("price").to_pylist()
        sizes = table.column("size").to_pylist()
    except Exception as exc:
        # pyarrow raises its own types (some of them ValueError subclasses) for a
        # corrupt file or a missing column. The file was delivered and paid for, so
        # say exactly what is wrong with it instead of returning a guess.
        raise ResponseShapeError(
            "the trade file could not be read as Parquet with market_id, price and "
            f"size columns ({type(exc).__name__}: {exc})",
            request_id=request_id,
        ) from exc
    return yes_vwap(zip(tokens, prices, sizes), yes_token=yes_token, no_token=no_token)


def _outcome_tokens(market: Record) -> Optional[Tuple[str, str]]:
    """The ``(yes_token, no_token)`` pair from the market record.

    Returns None for a market that cannot be priced (no tokens, or not binary),
    which skips it without spending anything. Raises if the record has no
    ``outcome_token_ids`` field at all, because then *no* market can be priced and
    quietly skipping them all would look like an empty universe.
    """
    if "outcome_token_ids" not in market:
        raise SupaGammaConfigError(
            "This market record has no `outcome_token_ids`, so a trade file cannot be "
            "split into YES and NO fills (pooling both averages toward 0.5). Records "
            "from an API deployment that predates the field cannot be priced with "
            "vwap_price_fn. No request was made."
        )
    ids = _as_sequence(market["outcome_token_ids"])
    if not ids or len(ids) != 2:
        return None
    yes, no = str(ids[0]), str(ids[1])
    return None if yes == no else (yes, no)


def vwap_price_fn(horizon_hours: float = 24.0, *, trade_limit: int = 100_000) -> PriceFn:
    """Build a price function that reads the trade tape (this is **metered**).

    Returns the size-weighted average price of the YES outcome over the trades up
    to ``horizon_hours`` before the market's ``resolution_date`` (or ``end_date``)
    -- i.e. the price while the outcome was still genuinely uncertain, not the
    settlement print. NO-token fills are converted to the YES scale (``1 - price``)
    first; see :func:`yes_vwap`. The window has no start, and the server returns
    the **oldest** ``trade_limit`` rows, so a market with more trades than that
    before the cutoff is priced from its early trading, not its last hours.

    Each produced call makes ONE paid ``client.download.trades`` request. Failing
    before spending is the rule:

    * pyarrow missing -> :class:`~supagamma.MissingDependencyError`, raised here,
      before the function is even returned;
    * a market with no end date, or without exactly two distinct outcome tokens,
      is skipped (returns None) with no request;
    * a record with no ``outcome_token_ids`` field at all raises
      :class:`~supagamma.SupaGammaConfigError` before anything is spent.

    After the request, only "no trades before the cutoff"
    (:class:`~supagamma.NoDataInRangeError`) skips a market. Every other error
    propagates: an expired key, a spent quota, a rate limit or a server fault is
    not a reason to carry on paying for the next market. Pass the result to
    :func:`resolved_markets`, which caps how many requests it may make.
    """
    pq = _require_pyarrow()
    if horizon_hours < 0:
        raise ValueError(f"horizon_hours must be >= 0, got {horizon_hours!r}")
    if not 1 <= trade_limit <= _MAX_TRADE_LIMIT:
        raise ValueError(f"trade_limit must be between 1 and {_MAX_TRADE_LIMIT}, got {trade_limit}")

    def _price(client: Any, market: Record) -> Optional[float]:
        end = _parse_dt(market.get("resolution_date")) or _parse_dt(market.get("end_date"))
        if end is None:
            # No horizon to price at. An open window would include the settlement
            # prints this function exists to avoid, and cost more.
            return None
        tokens = _outcome_tokens(market)
        if tokens is None:
            return None
        try:
            result = client.download.trades(
                market_id=str(market["id"]),
                end=end - timedelta(hours=horizon_hours),
                format="parquet",
                limit=trade_limit,
            )
        except NoDataInRangeError:
            return None
        return _vwap_from_parquet(
            result.content,
            yes_token=tokens[0],
            no_token=tokens[1],
            pq=pq,
            request_id=getattr(result, "request_id", None),
        )

    return _price


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


def _attempt_cap(
    price_fn: Optional[PriceFn], max_markets: Optional[int], max_price_attempts: Optional[int]
) -> Optional[int]:
    """How many times ``price_fn`` may be called, or None when there is no ``price_fn``."""
    if price_fn is None:
        return None
    cap = max_price_attempts if max_price_attempts is not None else max_markets
    if cap is None:
        raise ValueError(
            "A price_fn makes paid requests, so it needs a bound: pass max_markets or "
            "max_price_attempts. (max_markets=None means 'every resolved market', which "
            "would otherwise be one paid request per market in the catalogue.)"
        )
    if cap < 0:
        raise ValueError(f"the bound on price_fn attempts cannot be negative, got {cap}")
    return cap


def resolved_markets(
    client: Any,
    *,
    max_markets: Optional[int] = 500,
    page_limit: int = 500,
    price_fn: Optional[PriceFn] = None,
    max_price_attempts: Optional[int] = None,
    **filters: Any,
) -> Iterator[ResolvedMarket]:
    """Yield resolved markets as backtest inputs.

    Pulls ``resolved=True, has_data=True`` markets (plus any extra ``filters``
    like ``category=``) via ``client.markets.auto_paginate`` and normalizes each
    into a :class:`ResolvedMarket`. Markets that cannot be read cleanly (missing
    outcome or price) are skipped.

    ``price_fn`` overrides the entry price -- pass :func:`vwap_price_fn` for a
    real study (it is metered). Without it, the free metadata price is used and
    ``max_markets`` simply counts the markets yielded.

    **With a ``price_fn`` the bound is on paid attempts, not on yields.** Every
    call to ``price_fn`` counts, whether or not it produces a market, and the
    ``(N+1)``th call is never made. The bound is ``max_price_attempts`` when given,
    else ``max_markets``; if both are None the call raises ``ValueError`` before
    any request, since "every resolved market" would be one paid request per market
    in the catalogue. A market skipped before ``price_fn`` runs (no clean outcome)
    costs nothing and does not count. So ``max_markets=100`` makes at most 100
    paid requests and may yield fewer than 100 markets; raise ``max_price_attempts``
    to keep going past skipped ones.
    """
    attempt_cap = _attempt_cap(price_fn, max_markets, max_price_attempts)
    return _iter_resolved(client, max_markets, page_limit, price_fn, attempt_cap, filters)


def _iter_resolved(
    client: Any,
    max_markets: Optional[int],
    page_limit: int,
    price_fn: Optional[PriceFn],
    attempt_cap: Optional[int],
    filters: Dict[str, Any],
) -> Iterator[ResolvedMarket]:
    if max_markets is not None and max_markets <= 0:
        return
    if attempt_cap is not None and attempt_cap <= 0:
        return
    yielded = 0
    attempts = 0
    for market in client.markets.auto_paginate(
        resolved=True, has_data=True, limit=page_limit, **filters
    ):
        outcome = realized_outcome(market)
        if outcome is None:
            continue
        if price_fn is None:
            prob = yes_probability(market)
        else:
            if attempt_cap is not None and attempts >= attempt_cap:
                return
            attempts += 1
            prob = price_fn(client, market)
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
        yielded += 1
        if max_markets is not None and yielded >= max_markets:
            return


def calibration_pairs(markets: Sequence[ResolvedMarket]) -> List[Tuple[float, int]]:
    """(forecast_yes, outcome) pairs for :func:`supagamma.backtest.calibration`."""
    return [(m.view.prob, m.outcome) for m in markets]
