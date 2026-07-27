"""``client.orders`` — cart checkout (``/v1/orders``).

``create()`` spends money and is the **only** route in the entire API with real
idempotency protection — and it is a **body field**, not an ``Idempotency-Key``
header. Sending the header does nothing at all.

Because of that, this SDK generates a ``uuid4`` key for every order you create
and returns it on the result. If a ``create()`` call fails ambiguously (most
importantly :class:`~supagamma.OrderStatusUnknownError`, a 502 meaning the order
*may* have committed), re-issue it with the **same** key:

    order = client.orders.create(items)              # key generated for you
    ...
    client.orders.create(items, idempotency_key=order["idempotency_key"])

The server namespaces the key per user and returns the original order on a
repeat without charging again. Retrying *without* the key charges twice.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from .._client import NEVER, SAFE_READ
from ._base import AsyncResource, Call, SyncResource, call

__all__ = [
    "OrderItem",
    "MAX_ITEMS",
    "build_create",
    "build_list",
    "build_pricing",
    "Orders",
    "AsyncOrders",
]

MAX_ITEMS = 50


@dataclass
class OrderItem:
    """One line of a cart order. Needs ``market_id`` **or** ``series_id``."""

    data_type: str
    market_id: Optional[str] = None
    series_id: Optional[str] = None
    format: str = "csv"
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    timeframe: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"data_type": self.data_type, "format": self.format}
        for key, value in (
            ("market_id", self.market_id),
            ("series_id", self.series_id),
            ("timeframe", self.timeframe),
            ("start", self.start.isoformat() if self.start else None),
            ("end", self.end.isoformat() if self.end else None),
        ):
            if value is not None:
                payload[key] = value
        return payload


def build_create() -> Call:
    return call("POST", "/v1/orders", {}, NEVER)


def build_list(*, limit: int = 20) -> Call:
    return call("GET", "/v1/orders", {"limit": limit}, SAFE_READ)


def build_pricing() -> Call:
    return call("GET", "/v1/orders/estimate", {}, SAFE_READ)


def _create_payload(
    items: Sequence[OrderItem], idempotency_key: Optional[str]
) -> Dict[str, Any]:
    if not items:
        raise ValueError("create() needs at least one item")
    if len(items) > MAX_ITEMS:
        raise ValueError(f"create() accepts at most {MAX_ITEMS} items, got {len(items)}")
    for index, item in enumerate(items):
        if not item.market_id and not item.series_id:
            raise ValueError(f"items[{index}] needs market_id or series_id")
    key = idempotency_key or uuid.uuid4().hex
    if len(key) > 128:
        raise ValueError("idempotency_key must be at most 128 characters")
    return {"items": [i.to_payload() for i in items], "idempotency_key": key}


def estimate_cost_usd(record_count: int, data_type: str, pricing: Dict[str, Any]) -> float:
    """Local cost estimate using a **fetched** ``pricing()`` payload.

    Both maps in ``pricing()`` are open — keyed by ``data_catalog`` data-type
    strings that grow as streams are added — so unknown types fall back to the
    server's own defaults rather than raising.
    """
    per_row = (pricing.get("bytes_per_row") or {}).get(data_type, 230)
    rate = (pricing.get("rate_per_mb") or {}).get(data_type, 1.00)
    free_mb = float(pricing.get("free_tier_mb") or 0.0)
    return max(0.0, (record_count * per_row) / 1_048_576 - free_mb) * rate


class Orders(SyncResource):
    """Cart checkout and its pricing constants."""

    def create(
        self, items: Sequence[OrderItem], *, idempotency_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Place an order. **Spends money. Never retried automatically.**

        A key is generated for you and echoed back as ``idempotency_key`` on the
        result — keep it. On an ambiguous failure, re-issue with that same key;
        the server returns the original order rather than charging again.

        Notes that surprise people:

        * The $10 minimum applies only when nothing was subscription-covered.
        * A series item is not always one row. Raw, options and single-bucket
          series stay as one row; Polymarket-style series expand to one row per
          underlying market.
        * A zero-cost line records ``status="completed"``, not ``"paid"``, and so
          mints **no** 7-day re-download entitlement.
        * ``record_count``/``size_bytes`` on the response are estimates; the
          download re-counts.
        """
        payload = _create_payload(items, idempotency_key)
        result = self._json(build_create(), json=payload)
        if isinstance(result, dict):
            result.setdefault("idempotency_key", payload["idempotency_key"])
        return result

    def list(self, *, limit: int = 20) -> List[Dict[str, Any]]:
        """Past cart checkouts.

        This is **not** download history: rows without an order id — which is
        every direct ``client.download.*`` delivery — are skipped. Use
        ``client.account.downloads()`` for deliveries.

        There is deliberately no auto-pager. The server over-fetches
        ``limit * 10`` rows and groups them in Python, so an account whose orders
        average more than ten line items silently receives fewer than ``limit``,
        and a short page does **not** mean there are no more.
        """
        return self._json(build_list(limit=limit))

    def pricing(self) -> Dict[str, Any]:
        """``rate_per_mb``, ``bytes_per_row``, ``free_tier_mb``, ``minimum_order_usd``.

        Treat both maps as open dictionaries, never as enums — the key set grows
        with each new stream.
        """
        return self._json(build_pricing())

    def estimate_cost(self, record_count: int, data_type: str) -> float:
        """Estimate a cost from live pricing (one extra request per call)."""
        return estimate_cost_usd(record_count, data_type, self.pricing())


class AsyncOrders(AsyncResource):
    """Async twin of :class:`Orders`."""

    async def create(
        self, items: Sequence[OrderItem], *, idempotency_key: Optional[str] = None
    ) -> Dict[str, Any]:
        payload = _create_payload(items, idempotency_key)
        result = await self._json(build_create(), json=payload)
        if isinstance(result, dict):
            result.setdefault("idempotency_key", payload["idempotency_key"])
        return result

    async def list(self, *, limit: int = 20) -> List[Dict[str, Any]]:
        return await self._json(build_list(limit=limit))

    async def pricing(self) -> Dict[str, Any]:
        return await self._json(build_pricing())

    async def estimate_cost(self, record_count: int, data_type: str) -> float:
        return estimate_cost_usd(record_count, data_type, await self.pricing())


for _name in ("create", "list", "pricing", "estimate_cost"):
    getattr(AsyncOrders, _name).__doc__ = getattr(Orders, _name).__doc__
