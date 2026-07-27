"""Resource namespaces, attached to a client at construction time.

Kept in one place so the sync and async trees can be compared at a glance — if
a namespace exists on one and not the other, it is obvious here rather than at
the call site.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .account import Account, AsyncAccount
from .billing import AsyncBilling, Billing
from .download import AsyncDownload, Download
from .markets import AsyncMarkets, Markets
from .orders import AsyncOrders, Orders
from .public_markets import AsyncPublicMarkets, PublicMarkets
from .series import AsyncSeries, Series
from .system import AsyncSystem, System
from .trades import AsyncTrades, Trades

if TYPE_CHECKING:  # pragma: no cover
    from .._client import AsyncSupaGamma, SupaGamma

__all__ = ["build_sync_namespaces", "build_async_namespaces"]

#: attribute name -> (sync class, async class)
NAMESPACES = {
    "markets": (Markets, AsyncMarkets),
    "public_markets": (PublicMarkets, AsyncPublicMarkets),
    "trades": (Trades, AsyncTrades),
    "series": (Series, AsyncSeries),
    "download": (Download, AsyncDownload),
    "orders": (Orders, AsyncOrders),
    "billing": (Billing, AsyncBilling),
    "account": (Account, AsyncAccount),
    "system": (System, AsyncSystem),
}


def build_sync_namespaces(client: SupaGamma) -> None:
    for name, (sync_cls, _) in NAMESPACES.items():
        setattr(client, name, sync_cls(client))


def build_async_namespaces(client: AsyncSupaGamma) -> None:
    for name, (_, async_cls) in NAMESPACES.items():
        setattr(client, name, async_cls(client))
