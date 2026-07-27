"""``client.public_markets`` — unauthenticated market metadata (``/v1/public/markets*``).

**This router is disabled in production today.** It 404s for everyone unless the
server sets ``PUBLIC_MARKET_PAGES=true``, which defaults false and is on a
deliberate hold. The methods ship so the SDK is complete, but expect
:class:`~supagamma.NotFoundError` until that changes.

Two behaviours differ from :mod:`~supagamma.resources.markets` and will bite if
you assume they match:

* **A short page does not mean the end.** The server applies a Python-side
  filter *after* the query, so a page can come back shorter than ``limit`` while
  more results exist. :meth:`PublicMarkets.auto_paginate` therefore stops on an
  **empty** page, never a short one — the opposite rule to ``client.markets``.
* **``offset`` has no upper bound here**, and ``limit`` goes to 5000 rather
  than 1000.

The field set is unrelated to ``Market``: no prices, no liquidity. Dates are raw
strings, not normalised datetimes.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from .._client import SAFE_READ
from ._base import AsyncResource, Call, SyncResource, call

__all__ = [
    "MAX_LIMIT",
    "build_list",
    "build_get",
    "PublicMarkets",
    "AsyncPublicMarkets",
]

MAX_LIMIT = 5000


def build_list(
    *, limit: int = 500, offset: int = 0, resolved: Optional[bool] = None
) -> Call:
    return call(
        "GET",
        "/v1/public/markets",
        {"limit": limit, "offset": offset, "resolved": resolved},
        SAFE_READ,
    )


def build_get(market_id: str) -> Call:
    return call("GET", f"/v1/public/markets/{market_id}", {}, SAFE_READ)


class PublicMarkets(SyncResource):
    """Crawlable market metadata. Free, unauthenticated, usually disabled."""

    def list(
        self, *, limit: int = 500, offset: int = 0, resolved: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """One page, ordered by volume descending (not configurable).

        Only markets with data are eligible — the server always applies
        ``volume > 0 AND (trade_count > 0 OR orderbook_count > 0)``, and
        ``volume`` itself is not in the response. Pass ``resolved=True`` for the
        resolution archive.
        """
        return self._json(build_list(limit=limit, offset=offset, resolved=resolved))

    def auto_paginate(
        self, *, limit: int = 500, resolved: Optional[bool] = None
    ) -> Iterator[Dict[str, Any]]:
        """Walk pages until one comes back **empty**.

        Deliberately not the short-page rule used elsewhere: this endpoint
        re-filters in Python after the query, so a short page is expected and
        stopping on it would silently truncate your results.
        """
        offset = 0
        while True:
            rows = self.list(limit=limit, offset=offset, resolved=resolved)
            if not rows:
                return
            yield from rows
            offset += limit

    def get(self, market_id: str) -> Dict[str, Any]:
        """One market's metadata.

        A 404 here is ambiguous by design: it means the id is unknown, *or* the
        market holds no data, *or* the whole router is disabled. The disabled
        case says ``"Not found"`` while a genuine miss says ``"Market not
        found"``, which is the only way to tell them apart.
        """
        return self._json(build_get(market_id))


class AsyncPublicMarkets(AsyncResource):
    """Async twin of :class:`PublicMarkets`."""

    async def list(
        self, *, limit: int = 500, offset: int = 0, resolved: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        return await self._json(build_list(limit=limit, offset=offset, resolved=resolved))

    async def auto_paginate(
        self, *, limit: int = 500, resolved: Optional[bool] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        offset = 0
        while True:
            rows = await self.list(limit=limit, offset=offset, resolved=resolved)
            if not rows:
                return
            for row in rows:
                yield row
            offset += limit

    async def get(self, market_id: str) -> Dict[str, Any]:
        return await self._json(build_get(market_id))


for _name in ("list", "get"):
    getattr(AsyncPublicMarkets, _name).__doc__ = getattr(PublicMarkets, _name).__doc__
