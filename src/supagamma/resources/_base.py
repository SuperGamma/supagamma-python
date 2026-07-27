"""Shared base for resource namespaces.

Each resource module defines two thin classes — a sync one and an async one —
over a single set of *pure* request builders. The builders return
``(method, path, params, policy)`` and never touch the network, which is what
makes the tricky parts (parameter names, row caps, which routes may retry)
unit-testable without HTTP.

Duplicating the sync/async method bodies is deliberate. The alternative —
returning something awaitable-or-not — produces a client where forgetting an
``await`` silently yields a coroutine instead of data, and on this API that
failure mode can cost money.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Tuple

from .._client import NEVER, RetryPolicy

if TYPE_CHECKING:  # pragma: no cover
    from .._client import AsyncSupaGamma, SupaGamma

#: (method, path, params, policy)
Call = Tuple[str, str, Dict[str, Any], RetryPolicy]


def call(
    method: str,
    path: str,
    params: Optional[Mapping[str, Any]] = None,
    policy: RetryPolicy = NEVER,
) -> Call:
    return method, path, dict(params or {}), policy


class SyncResource:
    def __init__(self, client: SupaGamma) -> None:
        self._client = client

    def _json(self, spec: Call, *, json: Any = None) -> Any:
        method, path, params, policy = spec
        return self._client.request(method, path, params=params, json=json, policy=policy).json()

    def _raw(self, spec: Call, *, json: Any = None) -> Any:
        method, path, params, policy = spec
        return self._client.request(method, path, params=params, json=json, policy=policy)


class AsyncResource:
    def __init__(self, client: AsyncSupaGamma) -> None:
        self._client = client

    async def _json(self, spec: Call, *, json: Any = None) -> Any:
        method, path, params, policy = spec
        response = await self._client.request(method, path, params=params, json=json, policy=policy)
        return response.json()

    async def _raw(self, spec: Call, *, json: Any = None) -> Any:
        method, path, params, policy = spec
        return await self._client.request(method, path, params=params, json=json, policy=policy)
