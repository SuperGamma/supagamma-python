"""``client.account`` — identity, API keys, usage and GDPR (``/v1/user*``).

One hazard dominates this namespace: **``keys.provision()`` rotates your keys.**
It revokes the previously auto-provisioned key and mints a new one, so calling
it in a retry loop — or, classically, calling it in response to a 401 — is a key
storm that can invalidate a key another in-flight request is still using. The
server carries a 30-second grace window precisely because this has happened.
It is never retried here, and it must never be your 401 handler.

Key scopes can be **narrowed but never widened**: you may mint a read-only key
from a read+download one, but no request through this API can ever grant
``admin``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .._client import NEVER, SAFE_READ
from ._base import AsyncResource, Call, SyncResource, call

__all__ = [
    "build_me",
    "build_usage",
    "build_downloads",
    "build_list_keys",
    "build_create_key",
    "build_revoke_key",
    "build_rotate_key",
    "build_provision",
    "build_data_export",
    "build_privacy_events",
    "build_delete_request",
    "build_delete_cancel",
    "Account",
    "AsyncAccount",
    "Keys",
    "AsyncKeys",
]


def build_me() -> Call:
    return call("GET", "/v1/user/me", {}, SAFE_READ)


def build_usage() -> Call:
    return call("GET", "/v1/user/usage", {}, SAFE_READ)


def build_downloads(*, limit: int = 50, offset: int = 0) -> Call:
    return call("GET", "/v1/user/downloads", {"limit": limit, "offset": offset}, SAFE_READ)


def build_list_keys() -> Call:
    return call("GET", "/v1/user/keys", {}, SAFE_READ)


def build_create_key() -> Call:
    # Mutating: a retry would mint a second key.
    return call("POST", "/v1/user/keys", {}, NEVER)


def build_revoke_key(key_id: str) -> Call:
    return call("DELETE", f"/v1/user/keys/{key_id}", {}, NEVER)


def build_rotate_key(key_id: str) -> Call:
    return call("POST", f"/v1/user/keys/{key_id}/rotate", {}, NEVER)


def build_provision() -> Call:
    # JWT only — an X-API-Key is not accepted here. Destructive; never retried.
    return call("POST", "/v1/user/provision", {}, NEVER)


def build_data_export() -> Call:
    return call("GET", "/v1/user/data-export", {}, SAFE_READ)


def build_privacy_events() -> Call:
    return call("GET", "/v1/user/privacy-events", {}, SAFE_READ)


def build_delete_request() -> Call:
    return call("POST", "/v1/user/delete-request", {}, NEVER)


def build_delete_cancel() -> Call:
    return call("POST", "/v1/user/delete-cancel", {}, NEVER)


class Keys(SyncResource):
    """API key CRUD. Everything except :meth:`list` mutates and is never retried."""

    def list(self) -> List[Dict[str, Any]]:
        """Your keys. The plaintext value is never returned here — only at creation."""
        return self._json(build_list_keys())

    def create(
        self, *, name: Optional[str] = None, scopes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Mint a key. **The plaintext is returned exactly once — store it now.**

        ``scopes`` may only narrow: a subset of the caller's own permissions.
        ``admin`` can never be granted through this route.
        """
        body: Dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if scopes is not None:
            body["scopes"] = scopes
        return self._json(build_create_key(), json=body or None)

    def revoke(self, key_id: str) -> Dict[str, Any]:
        """Revoke a key. A revoked key can still authenticate for up to 60s
        while the positive auth cache expires — don't assume immediate effect."""
        return self._json(build_revoke_key(key_id))

    def rotate(self, key_id: str) -> Dict[str, Any]:
        """Rotate a key, returning the new plaintext once. Never retried."""
        return self._json(build_rotate_key(key_id))

    def provision(self) -> Dict[str, Any]:
        """Create or rotate the auto-provisioned key. **JWT auth only.**

        Revokes prior auto-provisioned keys and mints a fresh one, so this is
        destructive. Never call it on a 401 and never retry it — that is a key
        storm, and it can invalidate a key another in-flight request is holding.
        """
        return self._json(build_provision())


class AsyncKeys(AsyncResource):
    async def list(self) -> List[Dict[str, Any]]:
        return await self._json(build_list_keys())

    async def create(
        self, *, name: Optional[str] = None, scopes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if scopes is not None:
            body["scopes"] = scopes
        return await self._json(build_create_key(), json=body or None)

    async def revoke(self, key_id: str) -> Dict[str, Any]:
        return await self._json(build_revoke_key(key_id))

    async def rotate(self, key_id: str) -> Dict[str, Any]:
        return await self._json(build_rotate_key(key_id))

    async def provision(self) -> Dict[str, Any]:
        return await self._json(build_provision())


class Account(SyncResource):
    """Identity, usage and delivery history."""

    def __init__(self, client: Any) -> None:
        super().__init__(client)
        self.keys = Keys(client)

    def me(self) -> Dict[str, Any]:
        """Who this credential is: user id, key name, rate limit, permissions."""
        return self._json(build_me())

    def usage(self) -> Dict[str, Any]:
        """Usage counters for the current period."""
        return self._json(build_usage())

    def downloads(self, *, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Delivery history — the authoritative record of what you were charged.

        This is the endpoint to check after an ambiguous download or bulk
        failure, since neither reports its cost inline.
        """
        return self._json(build_downloads(limit=limit, offset=offset))

    def data_export(self) -> Dict[str, Any]:
        """Full GDPR export. Can be a large single body — expect a slow response."""
        return self._json(build_data_export())

    def privacy_events(self) -> List[Dict[str, Any]]:
        return self._json(build_privacy_events())

    def request_deletion(self) -> Dict[str, Any]:
        """Begin account deletion. Returns a pending-confirmation status.

        The confirm step is deliberately not exposed by this SDK: it
        cascade-deletes the account and belongs behind a human, not a script.
        """
        return self._json(build_delete_request())

    def cancel_deletion(self) -> Dict[str, Any]:
        return self._json(build_delete_cancel())


class AsyncAccount(AsyncResource):
    """Async twin of :class:`Account`."""

    def __init__(self, client: Any) -> None:
        super().__init__(client)
        self.keys = AsyncKeys(client)

    async def me(self) -> Dict[str, Any]:
        return await self._json(build_me())

    async def usage(self) -> Dict[str, Any]:
        return await self._json(build_usage())

    async def downloads(self, *, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        return await self._json(build_downloads(limit=limit, offset=offset))

    async def data_export(self) -> Dict[str, Any]:
        return await self._json(build_data_export())

    async def privacy_events(self) -> List[Dict[str, Any]]:
        return await self._json(build_privacy_events())

    async def request_deletion(self) -> Dict[str, Any]:
        return await self._json(build_delete_request())

    async def cancel_deletion(self) -> Dict[str, Any]:
        return await self._json(build_delete_cancel())


for _sync_cls, _async_cls, _names in (
    (Keys, AsyncKeys, ("list", "create", "revoke", "rotate", "provision")),
    (
        Account,
        AsyncAccount,
        ("me", "usage", "downloads", "data_export", "privacy_events", "request_deletion"),
    ),
):
    for _name in _names:
        getattr(_async_cls, _name).__doc__ = getattr(_sync_cls, _name).__doc__
