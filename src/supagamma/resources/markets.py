"""``client.markets`` — the market catalogue (``/v1/markets*``).

Nothing in this namespace spends credits. ``list``/``get``/``stats`` are plain
reads, and ``estimate`` is a cost *preview* that never charges and never
authenticates. That makes every method here safely retryable, which is the
exception in this SDK rather than the rule — see ``client.download`` for the
money paths.

Three server behaviours are load-bearing enough that they are repeated on every
method that touches them, because none of them are visible in the response:

* **``list()`` hides zero-data markets by default.** For the count-based sorts
  the router silently injects ``trade_count > 0 OR orderbook_count > 0``. Pass
  ``has_data=False`` to switch that filter off.
* **``list()`` is capped at 1000 rows per call and 1,000,000 rows deep.** There
  is no total, no cursor and no ``has_more``; the only end-of-data signal is a
  short page. ``auto_paginate()`` implements that rule and stops at the offset
  ceiling with a warning rather than 422-ing.
* **``stats()`` samples at most 10,000 trades.** Its ``trade_count`` saturates
  at exactly ``10000``; the true figure is ``Market["trade_count"]`` from
  ``get()``.

Responses are returned as parsed JSON (bare arrays / bare objects) — this API
has no ``{"data": ..., "meta": ...}`` envelope on any of these routes.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple, cast

from .._client import SAFE_READ
from ._base import AsyncResource, Call, SyncResource, call

__all__ = [
    "SORT_BY_VALUES",
    "DATA_TYPE_VALUES",
    "TIMEFRAME_VALUES",
    "DEFAULT_LIMIT",
    "MAX_LIMIT",
    "MAX_OFFSET",
    "OffsetCapReachedWarning",
    "build_list",
    "build_get",
    "build_stats",
    "build_estimate",
    "Markets",
    "AsyncMarkets",
]

#: Accepted ``sort_by`` values. The server does NOT validate this parameter — an
#: unrecognised value silently falls back to ``"top"`` and still returns 200, so
#: the SDK validates it client-side to turn a silent wrong answer into an error.
SORT_BY_VALUES: Tuple[str, ...] = (
    "top",
    "newest_data",
    "volume",
    "trades",
    "orderbook",
    "created",
)

#: Accepted ``data_type`` values for the list filter. Deliberately narrower than
#: the vocabularies used by ``series``/``orders``/``download`` — the data-type
#: sets differ per route and must not be shared.
DATA_TYPE_VALUES: Tuple[str, ...] = ("trades", "ohlcv", "orderbook")

#: Accepted ``timeframe`` values for ``estimate()``.
TIMEFRAME_VALUES: Tuple[str, ...] = ("1m", "1h", "1d")

DEFAULT_LIMIT = 100
MAX_LIMIT = 1000
#: ``offset`` above this is a 422 from the server, so deep pagination past a
#: million rows is impossible — narrow with filters instead.
MAX_OFFSET = 1_000_000


class OffsetCapReachedWarning(UserWarning):
    """``auto_paginate()`` stopped at the ``/v1/markets`` offset ceiling.

    Emitted instead of failing, because the alternative — a 422 in the middle of
    a long iteration — loses the rows already yielded. Rows beyond the ceiling
    exist but are unreachable by offset; narrow the query instead.
    """


# --- shared validation / serialization ---------------------------------------


def _drop_none(params: Dict[str, Any]) -> Dict[str, Any]:
    """Drop unset params so the server applies its own defaults.

    ``False`` is kept on purpose: ``has_data=False`` is a meaningful value (it
    disables the implicit has-data filter), not an absent one.
    """
    return {k: v for k, v in params.items() if v is not None}


def _check_market_id(market_id: str) -> str:
    """Reject ids that would silently address a different route."""
    if not market_id or not market_id.strip():
        raise ValueError("market_id must be a non-empty string, e.g. '1254468'.")
    if "/" in market_id:
        raise ValueError(
            f"market_id must not contain '/': {market_id!r} would change which endpoint "
            "is called. Pass the bare markets.id value."
        )
    return market_id


def _to_utc(value: datetime) -> datetime:
    """Normalise to UTC-aware.

    Routers disagree about naive datetimes — some coerce to UTC, some pass them
    through to Postgres raw — so the SDK never sends a naive one.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _window(
    start: Optional[datetime], end: Optional[datetime]
) -> Tuple[Optional[str], Optional[str]]:
    """UTC-normalise a window and reject an inverted one client-side.

    ``POST /v1/markets/{id}/estimate`` does NOT validate ``start > end``: it
    returns ``estimated_rows=0, in_range=false`` with a 200, which reads exactly
    like "this market holds nothing in your window". Failing here keeps that
    ambiguity out of the caller's world.
    """
    utc_start = _to_utc(start) if start is not None else None
    utc_end = _to_utc(end) if end is not None else None
    if utc_start is not None and utc_end is not None and utc_start > utc_end:
        raise ValueError(
            f"start must be on or before end (got start={utc_start.isoformat()}, "
            f"end={utc_end.isoformat()})."
        )
    return (
        utc_start.isoformat() if utc_start is not None else None,
        utc_end.isoformat() if utc_end is not None else None,
    )


# --- request builders ---------------------------------------------------------


def build_list(
    *,
    active: Optional[bool] = None,
    closed: Optional[bool] = None,
    resolved: Optional[bool] = None,
    category: Optional[str] = None,
    tag: Optional[str] = None,
    has_data: Optional[bool] = None,
    data_type: Optional[str] = None,
    search: Optional[str] = None,
    sort_by: str = "top",
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
) -> Call:
    """Build ``GET /v1/markets``. Pure — no I/O."""
    if sort_by not in SORT_BY_VALUES:
        raise ValueError(
            f"sort_by must be one of {SORT_BY_VALUES!r}, got {sort_by!r}. "
            "The server does not validate this and would silently sort by 'top'."
        )
    if data_type is not None and data_type not in DATA_TYPE_VALUES:
        raise ValueError(f"data_type must be one of {DATA_TYPE_VALUES!r}, got {data_type!r}.")
    if not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_LIMIT}, got {limit}.")
    if not 0 <= offset <= MAX_OFFSET:
        raise ValueError(f"offset must be between 0 and {MAX_OFFSET}, got {offset}.")
    if tag is not None and any(ch in tag for ch in ",{}"):
        raise ValueError(
            f"tag must not contain ',', '{{' or '}}': {tag!r} would change the meaning of "
            "the server-side filter, which interpolates the value unescaped."
        )

    params = _drop_none(
        {
            "active": active,
            "closed": closed,
            "resolved": resolved,
            "category": category,
            "tag": tag,
            "has_data": has_data,
            "data_type": data_type,
            "search": search,
            "sort_by": sort_by,
            "limit": limit,
            "offset": offset,
        }
    )
    return call("GET", "/v1/markets", params, SAFE_READ)


def build_get(market_id: str) -> Call:
    """Build ``GET /v1/markets/{market_id}``. Pure — no I/O."""
    return call("GET", f"/v1/markets/{_check_market_id(market_id)}", None, SAFE_READ)


def build_stats(market_id: str) -> Call:
    """Build ``GET /v1/markets/{market_id}/stats``. Pure — no I/O."""
    return call("GET", f"/v1/markets/{_check_market_id(market_id)}/stats", None, SAFE_READ)


def build_estimate(
    market_id: str,
    *,
    data_type: str = "trades",
    timeframe: Optional[str] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> Tuple[Call, Dict[str, Any]]:
    """Build ``POST /v1/markets/{market_id}/estimate``. Pure — no I/O.

    Returns ``(spec, body)``: the estimate is the one method in this namespace
    whose payload travels in the JSON body rather than the query string.
    """
    if data_type not in DATA_TYPE_VALUES:
        raise ValueError(f"data_type must be one of {DATA_TYPE_VALUES!r}, got {data_type!r}.")
    if timeframe is not None and timeframe not in TIMEFRAME_VALUES:
        raise ValueError(
            f"timeframe must be one of {TIMEFRAME_VALUES!r}, got {timeframe!r}. "
            "This route accepts a narrower set than GET /v1/trades/ohlcv."
        )
    iso_start, iso_end = _window(start, end)

    body = _drop_none(
        {
            "data_type": data_type,
            "timeframe": timeframe,
            "start": iso_start,
            "end": iso_end,
        }
    )
    spec = call(
        "POST",
        f"/v1/markets/{_check_market_id(market_id)}/estimate",
        None,
        # A cost preview: it never charges and never mutates, so a replay is
        # free. It does burn the 30/min POST /v1/markets* tier.
        SAFE_READ,
    )
    return spec, body


# --- shared docs --------------------------------------------------------------

_LIST_DOC = """List markets.

    Returns a bare JSON array of market objects — no envelope, no total, no
    ``has_more``. Costs nothing.

    Args:
        active/closed/resolved: exact-match filters on the market's lifecycle
            flags. Unset means "don't filter".
        category: exact match.
        tag: matches markets whose ``tags`` array contains this value. Rejected
            client-side if it contains ``,``, ``{`` or ``}`` — the server
            interpolates it unescaped, so those characters change the filter's
            meaning rather than matching literally.
        has_data: ``True`` restricts to markets with trades or orderbook rows.
            ``False`` applies **no filter at all** — it is the documented opt-out
            from the implicit filter described below, not a "dataless only"
            selector. Fully shadowed by ``data_type``.
        data_type: restrict to markets holding this data type. ``"ohlcv"``
            filters on ``trade_count > 0`` (there is no OHLCV counter column),
            so it is a heuristic, not a guarantee.
        search: substring match on the question. Interpolated into a PostgREST
            ``ilike`` pattern **unescaped**, so ``*`` and ``%`` in your term act
            as wildcards. A term broad enough to time out the query comes back
            as a 422 (``"Search query too broad…"``) that is NOT retryable —
            narrow the term.
        sort_by: one of ``SORT_BY_VALUES``. Every sort is DESC; there is no
            ascending option. Validated client-side because the server silently
            falls back to ``"top"`` on a typo and still returns 200.
        limit: 1..1000. Rows beyond ``limit`` are simply absent — a truncated
            page is indistinguishable from a complete one except by being
            exactly ``limit`` long.
        offset: 0..1,000,000. Above the ceiling the server 422s; use ``search``,
            ``category`` or ``tag`` to narrow instead of paging deeper.

    Hidden filter: when ``sort_by`` is ``top``/``trades``/``orderbook`` and
    ``has_data``, ``data_type`` and ``search`` are all unset, the router injects
    ``trade_count > 0 OR orderbook_count > 0``. The same injection happens on the
    ``search`` branch when ``data_type`` and ``has_data`` are both unset. So a
    bare ``list()`` never returns zero-data markets. Pass ``has_data=False`` to
    see them.

    Raises:
        ValueError: for an out-of-range ``limit``/``offset``, an unknown
            ``sort_by``/``data_type``, or an unescapable ``tag`` — all before any
            request is sent.
        ValidationError: 422, including the non-retryable "search too broad"
            variant.
        ServiceUnavailableError: 503 when the listing query times out. Ships
            ``Retry-After: 5`` and is retried automatically.
    """

_GET_DOC = """Fetch one market by id.

    Costs nothing. Returns a bare market object.

    ``market_id`` is ``markets.id`` — the short Polymarket numeric id such as
    ``"1254468"``. It is NOT a ``condition_id`` and NOT a CLOB token id, and the
    token id you get back from ``client.trades`` responses will 404 here. No
    endpoint in this API exposes ``condition_id``/``clob_token_ids``, so those
    join keys have to come from elsewhere.

    ``volume``, ``volume_24h`` and ``liquidity`` are ``null`` when the underlying
    value is zero OR unknown — the server cannot distinguish the two, so neither
    can you. Do not render ``null`` as "no data".

    Raises:
        NotFoundError: 404 ``"Market not found"``.
    """

_STATS_DOC = """Aggregate trade statistics for one market.

    Costs nothing. **Every figure is computed over at most the 10,000 most
    recent trades**, so for any active market these are sample statistics, not
    totals:

    * ``trade_count`` saturates at exactly ``10000``. The true count is
      ``trade_count`` on the market object from :meth:`get`.
    * ``total_volume`` is the sum of ``size`` (shares) over that sample — it is
      not USD notional.
    * ``min_price``/``max_price`` are the sample's extremes.
    * ``first_trade`` is the oldest trade *in the sample*, not the market's
      first trade ever. Both timestamps are raw strings compared as strings.

    A market that exists but has no trades (or no CLOB token ids) returns a 200
    all-zero object rather than a 404, so zeros do not prove the market is
    unknown.

    Raises:
        NotFoundError: 404 — note the wording differs from :meth:`get`'s. Never
            match on the message.
    """

_ESTIMATE_DOC = """Preview the size and cost of a single market's data.

    **Never charges and never authenticates** — a bad credential is ignored here
    rather than rejected. Rate-limited at 30/min (the ``POST /v1/markets*``
    tier), which a retry also consumes.

    This is a heuristic, not a quote. ``trades``/``orderbook`` pro-rate the
    market's TOTAL row count linearly across the overlap between your window and
    the market's ``[data_from, data_to]``, assuming rows are uniformly
    distributed in time. ``ohlcv`` counts wall-clock buckets and doubles them
    ("two outcomes") without ever consulting an OHLCV table.

    Interpreting the response:

    * ``in_range=False`` means EITHER your window does not overlap coverage OR
      the market simply holds zero rows of that type.
    * ``in_range=True`` does NOT mean your window was honoured — it is silently
      clamped to the market's coverage.
    * If the market's coverage window is null or inverted, proration is skipped
      and the FULL row count comes back with ``in_range=True`` regardless of what
      you asked for (for ``ohlcv`` the same condition yields ``rows=0``).

    Args:
        data_type: one of ``DATA_TYPE_VALUES``.
        timeframe: ``1m``/``1h``/``1d``. Ignored unless ``data_type="ohlcv"``,
            where an unset value means ``"1h"``.
        start/end: naive datetimes are treated as UTC and always sent with an
            explicit offset.

    Raises:
        ValueError: if ``start > end``. The server does not validate this — it
            would return a 200 with zero rows, which is indistinguishable from
            an empty window.
        NotFoundError: 404 ``"Market not found"``.
    """

_AUTO_PAGINATE_DOC = """Iterate every market matching the filters, page by page.

    Yields market objects, transparently advancing ``offset`` by ``limit``.
    Costs nothing beyond the per-request rate limit — one HTTP call per page.

    Stop rules, in order:

    1. A page shorter than ``limit`` is the last page. This is the only
       end-of-data signal the API offers on this route, and it is reliable here
       (unlike ``public_markets``, which post-filters in Python).
    2. Reaching ``offset > MAX_OFFSET`` (1,000,000) stops iteration with an
       :class:`OffsetCapReachedWarning`, because the next request would 422.
       Rows past that depth exist but cannot be reached by offset — narrow the
       query with ``search``/``category``/``tag`` or a ``sort_by`` that puts the
       rows you want first.

    Offset pagination is not stable: the underlying order is a live DESC sort, so
    rows inserted or updated mid-iteration can shift the window and cause a
    duplicate or a skip. For an exact snapshot, prefer a single ``list()`` call
    with a ``limit`` large enough to cover the result.

    Args:
        limit: page size, 1..1000. Larger pages mean fewer requests.
        offset: starting offset, for resuming a previous iteration.
    """


class Markets(SyncResource):
    """Synchronous ``client.markets``."""

    def list(
        self,
        *,
        active: Optional[bool] = None,
        closed: Optional[bool] = None,
        resolved: Optional[bool] = None,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        has_data: Optional[bool] = None,
        data_type: Optional[str] = None,
        search: Optional[str] = None,
        sort_by: str = "top",
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        __doc__ = _LIST_DOC  # noqa: F841
        spec = build_list(
            active=active,
            closed=closed,
            resolved=resolved,
            category=category,
            tag=tag,
            has_data=has_data,
            data_type=data_type,
            search=search,
            sort_by=sort_by,
            limit=limit,
            offset=offset,
        )
        return cast(List[Dict[str, Any]], self._json(spec))

    def get(self, market_id: str) -> Dict[str, Any]:
        return cast(Dict[str, Any], self._json(build_get(market_id)))

    def stats(self, market_id: str) -> Dict[str, Any]:
        return cast(Dict[str, Any], self._json(build_stats(market_id)))

    def estimate(
        self,
        market_id: str,
        *,
        data_type: str = "trades",
        timeframe: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        spec, body = build_estimate(
            market_id,
            data_type=data_type,
            timeframe=timeframe,
            start=start,
            end=end,
        )
        return cast(Dict[str, Any], self._json(spec, json=body))

    def auto_paginate(
        self,
        *,
        active: Optional[bool] = None,
        closed: Optional[bool] = None,
        resolved: Optional[bool] = None,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        has_data: Optional[bool] = None,
        data_type: Optional[str] = None,
        search: Optional[str] = None,
        sort_by: str = "top",
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
    ) -> Iterator[Dict[str, Any]]:
        while True:
            spec = build_list(
                active=active,
                closed=closed,
                resolved=resolved,
                category=category,
                tag=tag,
                has_data=has_data,
                data_type=data_type,
                search=search,
                sort_by=sort_by,
                limit=limit,
                offset=offset,
            )
            rows = cast(List[Dict[str, Any]], self._json(spec))
            yield from rows
            if len(rows) < limit:
                return
            offset += limit
            if offset > MAX_OFFSET:
                warnings.warn(
                    f"Stopped at the /v1/markets offset ceiling ({MAX_OFFSET}); further rows "
                    "are unreachable by offset. Narrow the query instead.",
                    OffsetCapReachedWarning,
                    stacklevel=2,
                )
                return


class AsyncMarkets(AsyncResource):
    """Asynchronous ``client.markets``. Identical surface to :class:`Markets`."""

    async def list(
        self,
        *,
        active: Optional[bool] = None,
        closed: Optional[bool] = None,
        resolved: Optional[bool] = None,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        has_data: Optional[bool] = None,
        data_type: Optional[str] = None,
        search: Optional[str] = None,
        sort_by: str = "top",
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        spec = build_list(
            active=active,
            closed=closed,
            resolved=resolved,
            category=category,
            tag=tag,
            has_data=has_data,
            data_type=data_type,
            search=search,
            sort_by=sort_by,
            limit=limit,
            offset=offset,
        )
        return cast(List[Dict[str, Any]], await self._json(spec))

    async def get(self, market_id: str) -> Dict[str, Any]:
        return cast(Dict[str, Any], await self._json(build_get(market_id)))

    async def stats(self, market_id: str) -> Dict[str, Any]:
        return cast(Dict[str, Any], await self._json(build_stats(market_id)))

    async def estimate(
        self,
        market_id: str,
        *,
        data_type: str = "trades",
        timeframe: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        spec, body = build_estimate(
            market_id,
            data_type=data_type,
            timeframe=timeframe,
            start=start,
            end=end,
        )
        return cast(Dict[str, Any], await self._json(spec, json=body))

    async def auto_paginate(
        self,
        *,
        active: Optional[bool] = None,
        closed: Optional[bool] = None,
        resolved: Optional[bool] = None,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        has_data: Optional[bool] = None,
        data_type: Optional[str] = None,
        search: Optional[str] = None,
        sort_by: str = "top",
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
    ) -> AsyncIterator[Dict[str, Any]]:
        while True:
            spec = build_list(
                active=active,
                closed=closed,
                resolved=resolved,
                category=category,
                tag=tag,
                has_data=has_data,
                data_type=data_type,
                search=search,
                sort_by=sort_by,
                limit=limit,
                offset=offset,
            )
            rows = cast(List[Dict[str, Any]], await self._json(spec))
            for row in rows:
                yield row
            if len(rows) < limit:
                return
            offset += limit
            if offset > MAX_OFFSET:
                warnings.warn(
                    f"Stopped at the /v1/markets offset ceiling ({MAX_OFFSET}); further rows "
                    "are unreachable by offset. Narrow the query instead.",
                    OffsetCapReachedWarning,
                    stacklevel=2,
                )
                return


# Docstrings live in module-level constants so the sync and async surfaces cannot
# drift apart. Assigning them here keeps `help()` and IDE hovers correct on both.
for _cls in (Markets, AsyncMarkets):
    _cls.list.__doc__ = _LIST_DOC
    _cls.get.__doc__ = _GET_DOC
    _cls.stats.__doc__ = _STATS_DOC
    _cls.estimate.__doc__ = _ESTIMATE_DOC
    _cls.auto_paginate.__doc__ = _AUTO_PAGINATE_DOC
del _cls
