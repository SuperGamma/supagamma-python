"""``client.system`` — unauthenticated service endpoints.

``/``, ``/health`` and ``/v1/stats`` are exempt from rate limiting entirely,
which also means responses from them carry no ``X-RateLimit-*`` headers.
"""

from __future__ import annotations

from typing import Any, Dict

from .._client import SAFE_READ
from ._base import AsyncResource, Call, SyncResource, call

__all__ = ["build_root", "build_health", "build_stats", "System", "AsyncSystem"]


def build_root() -> Call:
    return call("GET", "/", {}, SAFE_READ)


def build_health() -> Call:
    return call("GET", "/health", {}, SAFE_READ)


def build_stats() -> Call:
    return call("GET", "/v1/stats", {}, SAFE_READ)


class System(SyncResource):
    def root(self) -> Dict[str, Any]:
        """Service name and version."""
        return self._json(build_root())

    def health(self) -> Dict[str, Any]:
        """Liveness. Returns ``{"status": "healthy"}``.

        A degraded service answers 503, which surfaces as
        :class:`~supagamma.ServiceUnavailableError` rather than a payload — that
        is a status signal, not an outage of this SDK. Supplying a
        ``health_token`` to the client adds a per-check breakdown.
        """
        return self._json(build_health())

    def stats(self) -> Dict[str, Any]:
        """Aggregate platform counters (market/trade totals, capture freshness)."""
        return self._json(build_stats())


class AsyncSystem(AsyncResource):
    async def root(self) -> Dict[str, Any]:
        return await self._json(build_root())

    async def health(self) -> Dict[str, Any]:
        return await self._json(build_health())

    async def stats(self) -> Dict[str, Any]:
        return await self._json(build_stats())


for _name in ("root", "health", "stats"):
    getattr(AsyncSystem, _name).__doc__ = getattr(System, _name).__doc__
