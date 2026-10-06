"""``client.trades`` — raw fills and OHLCV bars (``/v1/trades*``).

Nothing here spends credits; these are reads, and all of them are retryable.

Three server behaviours are invisible in the response and are therefore repeated
on the methods they affect:

* **The ``market_id`` you send is not the ``market_id`` you get back.** You pass
  a ``markets.id`` (short, numeric). The server resolves it to CLOB outcome
  token ids, and the rows come back keyed by *token* id. Feeding a response
  ``market_id`` back into a request 404s on most routes (``trades.list`` and
  ``ohlcv`` do accept a token id, to narrow a query to that one outcome). The SDK
  exposes it as ``token_id`` on each row while leaving the raw payload untouched.
* **``side`` is matched case-sensitively** against stored lowercase values, so
  ``side="BUY"`` passes the server's own regex and then silently returns ``[]``.
  This module lowercases it for you.
* **``ohlcv`` has no ``offset``.** Bars come back newest-first and ``limit``
  truncates from the newest end, so older bars are reachable only by moving the
  ``start``/``end`` window — never by paging.

Every ``timeframe`` is a real aggregation. ``5m``/``15m`` are folded from the
1-minute archive and ``4h`` from 1-hour bars, on UTC boundaries. (The API did not
aggregate them before 2026-09-29 and this SDK used to warn about it; that warning
is gone.) ``1m``/``5m``/``15m`` are served at most 7 days per call.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from .._client import SAFE_READ
from ..types import OHLCVBar, Trade
from ._base import AsyncResource, Call, SyncResource, call

__all__ = [
    "TIMEFRAME_VALUES",
    "REAL_TIMEFRAMES",
    "FakeTimeframeWarning",
    "build_list",
    "build_ohlcv",
    "build_recent",
    "Trades",
    "AsyncTrades",
]

#: Values the server accepts.
TIMEFRAME_VALUES = ("1m", "5m", "15m", "1h", "4h", "1d")

#: Timeframes whose bars are exactly as wide as requested. Since the API began
#: aggregating 5m/15m/4h (2026-09-29) that is all six; the name is kept for
#: code that imported it.
REAL_TIMEFRAMES = frozenset(TIMEFRAME_VALUES)

MAX_LIST_LIMIT = 10_000
MAX_OFFSET = 1_000_000
MAX_RECENT_LIMIT = 100


class FakeTimeframeWarning(UserWarning):
    """Deprecated: no longer emitted.

    Before 2026-09-29 the API served ``5m``/``15m`` from the 1-minute view and
    ``4h`` from the 1-hour view without resampling, and this SDK warned about it.
    The API now aggregates them, so there is nothing to warn about. The class
    stays so ``warnings.filterwarnings("ignore", category=FakeTimeframeWarning)``
    in existing code keeps working.
    """


def _iso(value: Optional[datetime]) -> Optional[str]:
    """Serialise a datetime for the query string.

    Naive datetimes are passed through as-is rather than assumed UTC. That is
    deliberate: this router does **not** coerce naive values, while
    ``/v1/series/{id}/estimate`` does, so silently attaching a timezone here
    would make the SDK disagree with the server on what instant you meant.
    Pass tz-aware datetimes if you care.
    """
    return value.isoformat() if value is not None else None


def build_list(
    *,
    market_id: Optional[str] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    outcome: Optional[int] = None,
    side: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> Call:
    if side is not None:
        # The server's regex accepts BUY/SELL but the comparison against stored
        # values is exact, so uppercase silently yields an empty list.
        side = side.lower()
    return call(
        "GET",
        "/v1/trades",
        {
            "market_id": market_id,
            "start": _iso(start),
            "end": _iso(end),
            "outcome": outcome,
            "side": side,
            "limit": limit,
            "offset": offset,
        },
        SAFE_READ,
    )


def build_ohlcv(
    *,
    market_id: str,
    outcome: int = 0,
    timeframe: str = "1h",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: int = 500,
    warn: bool = True,
) -> Call:
    # `warn` is accepted and ignored: it controlled the FakeTimeframeWarning that is
    # no longer emitted. Dropping the parameter would break callers that pass it.
    return call(
        "GET",
        "/v1/trades/ohlcv",
        {
            "market_id": market_id,
            "outcome": outcome,
            "timeframe": timeframe,
            "start": _iso(start),
            "end": _iso(end),
            "limit": limit,
        },
        SAFE_READ,
    )


def build_recent(*, limit: int = 50) -> Call:
    return call("GET", "/v1/trades/recent", {"limit": limit}, SAFE_READ)


_TOKEN_NOTE = """
        The rows' ``market_id`` field is a CLOB **outcome token id**, not the
        ``markets.id`` you passed. Each row also gets a ``token_id`` key added
        client-side so the distinction is explicit; the original payload is
        otherwise untouched.
"""


def _annotate(rows: Any) -> Any:
    """Add ``token_id`` alongside the server's overloaded ``market_id``.

    Non-destructive: the server's own key is left in place so nothing that reads
    the raw payload breaks.
    """
    if not isinstance(rows, list):
        return rows
    for row in rows:
        if isinstance(row, dict) and "market_id" in row and "token_id" not in row:
            row["token_id"] = row["market_id"]
    return rows


class Trades(SyncResource):
    """Raw fills and OHLCV bars. All reads; nothing here charges."""

    def list(
        self,
        *,
        market_id: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        outcome: Optional[int] = None,
        side: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Trade]:
        """One page of trades, newest first.

        A market that exists but has no CLOB tokens returns ``[]``, while an
        unknown ``market_id`` raises :class:`~supagamma.NotFoundError` — empty is
        not the same as missing.
        """
        return _annotate(
            self._json(
                build_list(
                    market_id=market_id,
                    start=start,
                    end=end,
                    outcome=outcome,
                    side=side,
                    limit=limit,
                    offset=offset,
                )
            )
        )

    def auto_paginate(
        self, *, limit: int = 1000, max_rows: Optional[int] = None, **kwargs: Any
    ) -> Iterator[Trade]:
        """Walk pages until the server returns a short one.

        There is no total and no cursor, so a short page is the only
        end-of-data signal. Stops at the server's 1,000,000-row offset ceiling.
        """
        offset = int(kwargs.pop("offset", 0))
        yielded = 0
        while True:
            rows = self.list(limit=limit, offset=offset, **kwargs)
            for row in rows:
                yield row
                yielded += 1
                if max_rows is not None and yielded >= max_rows:
                    return
            if len(rows) < limit:
                return
            offset += limit
            if offset > MAX_OFFSET:
                warnings.warn(
                    f"reached the /v1/trades offset ceiling ({MAX_OFFSET}); narrow the "
                    "start/end window to reach older trades.",
                    UserWarning,
                    stacklevel=2,
                )
                return

    def ohlcv(
        self,
        *,
        market_id: str,
        outcome: int = 0,
        timeframe: str = "1h",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 500,
    ) -> List[OHLCVBar]:
        """OHLCV bars for one outcome, newest first.

        Every ``timeframe`` is genuinely that wide: ``5m``/``15m`` are folded
        from 1-minute bars and ``4h`` from 1-hour bars on UTC boundaries.
        ``1m``/``5m``/``15m`` are served at most 7 days per call (a wider window
        is a 400 that points at ``client.exports``); with no ``start``/``end``
        the window ends when the market does. There is no ``offset``: to reach
        older bars, move ``start``/``end`` rather than paging.

        ``outcome`` is the token's position in the market's outcomes (0 = the
        first, e.g. Yes). A market with no such outcome raises
        :class:`~supagamma.BadRequestError`; it is never widened to every token.
        """
        return _annotate(
            self._json(
                build_ohlcv(
                    market_id=market_id,
                    outcome=outcome,
                    timeframe=timeframe,
                    start=start,
                    end=end,
                    limit=limit,
                )
            )
        )

    def recent(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        """The newest fills across all markets. Cap is 100, and unpageable."""
        return self._json(build_recent(limit=limit))


class AsyncTrades(AsyncResource):
    """Async twin of :class:`Trades`."""

    async def list(
        self,
        *,
        market_id: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        outcome: Optional[int] = None,
        side: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Trade]:
        return _annotate(
            await self._json(
                build_list(
                    market_id=market_id,
                    start=start,
                    end=end,
                    outcome=outcome,
                    side=side,
                    limit=limit,
                    offset=offset,
                )
            )
        )

    async def auto_paginate(
        self, *, limit: int = 1000, max_rows: Optional[int] = None, **kwargs: Any
    ) -> AsyncIterator[Trade]:
        offset = int(kwargs.pop("offset", 0))
        yielded = 0
        while True:
            rows = await self.list(limit=limit, offset=offset, **kwargs)
            for row in rows:
                yield row
                yielded += 1
                if max_rows is not None and yielded >= max_rows:
                    return
            if len(rows) < limit:
                return
            offset += limit
            if offset > MAX_OFFSET:
                warnings.warn(
                    f"reached the /v1/trades offset ceiling ({MAX_OFFSET}); narrow the "
                    "start/end window to reach older trades.",
                    UserWarning,
                    stacklevel=2,
                )
                return

    async def ohlcv(
        self,
        *,
        market_id: str,
        outcome: int = 0,
        timeframe: str = "1h",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 500,
    ) -> List[OHLCVBar]:
        return _annotate(
            await self._json(
                build_ohlcv(
                    market_id=market_id,
                    outcome=outcome,
                    timeframe=timeframe,
                    start=start,
                    end=end,
                    limit=limit,
                )
            )
        )

    async def recent(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        return await self._json(build_recent(limit=limit))


for _cls in (Trades, AsyncTrades):
    if _cls.list.__doc__:
        _cls.list.__doc__ += _TOKEN_NOTE
