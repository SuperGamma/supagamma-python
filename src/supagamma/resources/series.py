"""``client.series`` — the stream catalogue (``/v1/series*``).

All three routes are **unauthenticated**; a key only changes which rate-limit
bucket you land in. Nothing here charges.

Two things worth knowing before you build on this:

* **``get()`` is not a lookup.** The server fetches the entire catalogue and
  linear-scans it, so fetching twenty series one at a time is twenty full
  catalogue reads. Prefer ``list()`` once and filter client-side, or pass
  ``cache=True`` to memoise the catalogue for the process lifetime.
* **``estimate()`` does not always price what you asked for.** For raw capture
  streams the response echoes the stream's own catalog ``data_type`` and prices
  at *that* rate, ignoring your input. Never assert
  ``response["data_type"] == requested``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .._client import SAFE_READ
from ._base import AsyncResource, Call, SyncResource, call

__all__ = [
    "DATA_TYPE_VALUES",
    "build_list",
    "build_get",
    "build_estimate",
    "Series",
    "AsyncSeries",
]

DATA_TYPE_VALUES = ("trades", "ohlcv", "orderbook", "greeks", "snapshots")


def build_list() -> Call:
    return call("GET", "/v1/series", {}, SAFE_READ)


def build_get(series_id: str) -> Call:
    # series ids contain a colon ("polymarket:btc-15m"). A colon is a legal
    # path-segment character; httpx will not double-encode a pre-joined path, so
    # pass it through raw rather than quoting it here.
    return call("GET", f"/v1/series/{series_id}", {}, SAFE_READ)


def build_estimate(
    series_id: str,
    *,
    data_type: str = "trades",
    timeframe: Optional[str] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> Tuple[Call, Dict[str, Any]]:
    """Return ``(call, body)`` — this is the one builder that carries a body."""
    body: Dict[str, Any] = {"data_type": data_type}
    # `timeframe` is declared and validated by the server but never read. Sent
    # for forward-compatibility only; it changes nothing today.
    if timeframe is not None:
        body["timeframe"] = timeframe
    if start is not None:
        body["start"] = start.isoformat()
    if end is not None:
        body["end"] = end.isoformat()
    # A POST, but a pure cost preview that never debits — safe to retry. Each
    # attempt does consume the 30/min estimate tier.
    return call("POST", f"/v1/series/{series_id}/estimate", {}, SAFE_READ), body


class Series(SyncResource):
    """The catalogue of purchasable streams."""

    _cache: Optional[List[Dict[str, Any]]] = None

    def list(self) -> List[Dict[str, Any]]:
        """The whole catalogue. Unpaginated and unfiltered by design.

        Served from a 600-second server-side cache behind a CDN, so a newly
        added stream can take ~10 minutes to appear.
        """
        return self._json(build_list())

    def get(self, series_id: str, *, cache: bool = False) -> Dict[str, Any]:
        """One series.

        The server linear-scans the full catalogue for this, so prefer
        ``list()`` plus a client-side filter when you need more than one. Pass
        ``cache=True`` to reuse a single catalogue read for the process lifetime.

        Kalshi ids resolve only when Kalshi public sale is enabled server-side;
        otherwise this raises :class:`~supagamma.NotFoundError`.
        """
        if cache:
            if type(self)._cache is None:
                type(self)._cache = self.list()
            for row in type(self)._cache or []:
                if row.get("series_id") == series_id or row.get("id") == series_id:
                    return row
        return self._json(build_get(series_id))

    def estimate(
        self,
        series_id: str,
        *,
        data_type: str = "trades",
        timeframe: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Price a pull without buying it. Never charges.

        Caveats that change how you read the result:

        * The echoed ``data_type`` may differ from what you sent (raw streams
          price at their own catalog type).
        * ``markets_in_range`` means three different things depending on the
          branch taken: a count of daily partition files, a 0/1 presence flag,
          or an actual market count.
        * Zero rows is not an error, and a nonsense ``(series, data_type)`` pair
          is not guaranteed to raise — some combinations return an all-zero 200.
        * ``coming_soon=true`` means "not purchasable yet", not "free".
        * Naive ``start``/``end`` are coerced to UTC **here**, unlike the
          ``/v1/trades`` query params. Pass tz-aware datetimes to avoid the
          discrepancy.
        """
        spec, body = build_estimate(
            series_id, data_type=data_type, timeframe=timeframe, start=start, end=end
        )
        return self._json(spec, json=body)


class AsyncSeries(AsyncResource):
    """Async twin of :class:`Series`."""

    _cache: Optional[List[Dict[str, Any]]] = None

    async def list(self) -> List[Dict[str, Any]]:
        return await self._json(build_list())

    async def get(self, series_id: str, *, cache: bool = False) -> Dict[str, Any]:
        if cache:
            if type(self)._cache is None:
                type(self)._cache = await self.list()
            for row in type(self)._cache or []:
                if row.get("series_id") == series_id or row.get("id") == series_id:
                    return row
        return await self._json(build_get(series_id))

    async def estimate(
        self,
        series_id: str,
        *,
        data_type: str = "trades",
        timeframe: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        spec, body = build_estimate(
            series_id, data_type=data_type, timeframe=timeframe, start=start, end=end
        )
        return await self._json(spec, json=body)


for _sync_method, _async_method in (
    (Series.get, AsyncSeries.get),
    (Series.estimate, AsyncSeries.estimate),
    (Series.list, AsyncSeries.list),
):
    _async_method.__doc__ = _sync_method.__doc__
