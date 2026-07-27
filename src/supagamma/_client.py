"""Transport: auth, request construction, error mapping, retry.

The retry policy here is deliberately conservative, because on this API a retry
is not free. Every `GET /v1/download/*` atomically debits the caller's balance,
and the only protection against a double charge is a 7-day entitlement waiver
matched on an EXACT parameter tuple. A retry that re-derives `end="now"` looks
like a different request to that matcher and gets charged again.

So retries are per-request, not global: each resource method declares its own
policy via `RetryPolicy`, and money-spending routes declare `NEVER`.
"""

from __future__ import annotations

import contextlib
import os
import random
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import httpx

from ._errors import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
    ServiceUnavailableError,
    SupaGammaConfigError,
    parse_error,
)

DEFAULT_BASE_URL = "https://api.supagamma.com"

# The server 401s on format alone, with no DB lookup, so validating client-side
# turns a wasted round trip into an immediate, clearer error.
_API_KEY_RE = re.compile(r"^sg_[0-9a-f]{32}$")

_DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)


@dataclass(frozen=True)
class RetryPolicy:
    """What a single request is allowed to retry on."""

    attempts: int = 0
    on_status: Tuple[int, ...] = ()
    on_connection_error: bool = False
    #: Retry a 500 exactly once (GET reads only). Separate from `on_status`
    #: because a 500 is not known-transient the way a 503 is.
    on_server_error_once: bool = False

    @property
    def enabled(self) -> bool:
        return self.attempts > 0


#: Money-spending or destructive. Never retried, whatever the client is configured with.
NEVER = RetryPolicy()

#: Safe idempotent GETs, and the two POST estimate previews that never charge.
SAFE_READ = RetryPolicy(
    attempts=3,
    on_status=(429, 503),
    on_connection_error=True,
    on_server_error_once=True,
)

#: 429/503 only — for reads where a mid-flight connection drop is better surfaced
#: than silently replayed.
THROTTLE_ONLY = RetryPolicy(attempts=3, on_status=(429, 503))


class BaseClient:
    """Shared config + response handling for the sync and async clients."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        jwt: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Union[float, httpx.Timeout, None] = None,
        max_retries: int = 3,
        user_agent: Optional[str] = None,
        origin_secret: Optional[str] = None,
        health_token: Optional[str] = None,
        default_headers: Optional[Mapping[str, str]] = None,
        max_retry_wait_seconds: float = 60.0,
    ) -> None:
        api_key = api_key if api_key is not None else os.environ.get("SUPAGAMMA_API_KEY")
        jwt = jwt if jwt is not None else os.environ.get("SUPAGAMMA_JWT")

        # Sending both credentials is not an error on the wire — it silently
        # resolves to the API KEY and ignores the JWT, because the rate-limit
        # middleware caches the key identity before auth ever sees the Bearer.
        # That surprises people badly (JWT-only routes 403 with a "valid" token),
        # so refuse it here instead of letting it happen quietly.
        if api_key and jwt:
            raise SupaGammaConfigError(
                "Pass either api_key or jwt, not both. Sending both makes the server "
                "silently use the API key and ignore the JWT."
            )
        if api_key and not _API_KEY_RE.match(api_key):
            raise SupaGammaConfigError(
                "api_key must look like 'sg_' followed by 32 hex characters. "
                "The server rejects other shapes on format alone."
            )

        self.api_key = api_key
        self.jwt = jwt
        self.base_url = (
            base_url or os.environ.get("SUPAGAMMA_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.max_retries = max(0, int(max_retries))
        self.max_retry_wait_seconds = max_retry_wait_seconds
        self.origin_secret = origin_secret or os.environ.get("SUPAGAMMA_ORIGIN_SECRET")
        self.health_token = health_token or os.environ.get("SUPAGAMMA_HEALTH_TOKEN")
        self._timeout = _DEFAULT_TIMEOUT if timeout is None else timeout
        self._user_agent = user_agent or f"supagamma-python/{_version()}"
        self._extra_headers = dict(default_headers or {})

        #: Rate-limit state from the most recent response. `remaining` is the
        #: tighter of the global and per-endpoint buckets, so it is the right
        #: number to pace against — but `reset` is unreliable (the server always
        #: sends now+60 even on the hourly download tier), so it is not exposed
        #: as a promise.
        self.rate_limit_limit: Optional[int] = None
        self.rate_limit_remaining: Optional[int] = None

    # --- request construction ------------------------------------------------

    def _headers(self, *, extra: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": self._user_agent,
            # Echoed back on every response and recorded in Sentry — the only
            # correlation handle support can use.
            "X-Request-ID": uuid.uuid4().hex,
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        elif self.jwt:
            headers["Authorization"] = f"Bearer {self.jwt}"
        if self.origin_secret:
            headers["X-Origin-Secret"] = self.origin_secret
        if self.health_token:
            headers["X-Health-Token"] = self.health_token
        headers.update(self._extra_headers)
        headers.update(extra or {})
        return headers

    def _url(self, path: str) -> str:
        # Collection routes are registered without a trailing slash; Starlette
        # would 307 otherwise.
        return f"{self.base_url}/{path.lstrip('/')}"

    @staticmethod
    def _clean_params(params: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        """Drop None (so the server applies its own defaults) and lower bools."""
        if not params:
            return {}
        out: Dict[str, Any] = {}
        for k, v in params.items():
            if v is None:
                continue
            out[k] = "true" if v is True else "false" if v is False else v
        return out

    # --- response handling ---------------------------------------------------

    def _capture_rate_limit(self, response: httpx.Response) -> None:
        for header, attr in (
            ("x-ratelimit-limit", "rate_limit_limit"),
            ("x-ratelimit-remaining", "rate_limit_remaining"),
        ):
            raw = response.headers.get(header)
            if raw is not None:
                with contextlib.suppress(ValueError):
                    setattr(self, attr, int(raw))

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        try:
            body: Any = response.json()
        except Exception:
            # A 500 can arrive as text/plain with no JSON at all.
            body = None
        raw = None
        with contextlib.suppress(Exception):
            raw = response.text
        raise parse_error(
            status_code=response.status_code,
            body=body,
            raw_body=raw,
            headers=response.headers,
            request_id=response.headers.get("x-request-id"),
        )

    # --- retry decisions -----------------------------------------------------

    def _effective_attempts(self, policy: RetryPolicy) -> int:
        if not policy.enabled:
            return 0
        return min(policy.attempts, self.max_retries)

    def _sleep_for(self, exc: APIStatusError, policy: RetryPolicy, attempt: int) -> Optional[float]:
        """How long to wait before retrying `exc`, or None to give up.

        A `QuotaExceededError` never reaches here as retryable: it is a distinct
        class from `RateLimitError` precisely so this function cannot confuse a
        billing cap with a rate limit.
        """
        if exc.status_code not in policy.on_status:
            if exc.status_code >= 500 and policy.on_server_error_once and attempt == 0:
                return _backoff(attempt)
            return None

        if isinstance(exc, RateLimitError):
            # The limiter uses fixed windows, so Retry-After is exact — waiting
            # exactly that long beats guessing with exponential backoff.
            wait = exc.retry_after if exc.retry_after is not None else 60.0
            if wait > self.max_retry_wait_seconds:
                # The hourly download tier can ask for ~an hour. Blocking that
                # long silently is worse than surfacing it.
                return None
            return wait
        if isinstance(exc, ServiceUnavailableError):
            return exc.retry_after if exc.retry_after is not None else _backoff(attempt)
        return _backoff(attempt)


def _backoff(attempt: int) -> float:
    """Full-jitter exponential backoff, base 0.5s, capped at 8s."""
    return random.uniform(0, min(8.0, 0.5 * (2**attempt)))


def _version() -> str:
    from . import __version__

    return __version__


class SupaGamma(BaseClient):
    """Synchronous client.

    >>> from supagamma import SupaGamma
    >>> client = SupaGamma(api_key="sg_...")
    >>> markets = client.markets.list(limit=10)
    """

    def __init__(
        self, *args: Any, http_client: Optional[httpx.Client] = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._http = http_client or httpx.Client(
            timeout=self._timeout,
            follow_redirects=True,
        )
        self._owns_http = http_client is None
        self._attach_resources()

    def _attach_resources(self) -> None:
        from .resources import build_sync_namespaces

        build_sync_namespaces(self)

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        json: Any = None,
        headers: Optional[Mapping[str, str]] = None,
        policy: RetryPolicy = NEVER,
        stream: bool = False,
    ) -> httpx.Response:
        attempts = self._effective_attempts(policy)
        url = self._url(path)
        query = self._clean_params(params)
        last_exc: Optional[Exception] = None

        for attempt in range(attempts + 1):
            request_headers = self._headers(extra=headers)
            try:
                response = self._http.request(
                    method, url, params=query, json=json, headers=request_headers
                )
            except httpx.TimeoutException as exc:
                last_exc = APITimeoutError(str(exc), request_id=request_headers["X-Request-ID"])
                if policy.on_connection_error and attempt < attempts:
                    time.sleep(_backoff(attempt))
                    continue
                raise last_exc from exc
            except httpx.HTTPError as exc:
                last_exc = APIConnectionError(str(exc), request_id=request_headers["X-Request-ID"])
                if policy.on_connection_error and attempt < attempts:
                    time.sleep(_backoff(attempt))
                    continue
                raise last_exc from exc

            self._capture_rate_limit(response)
            if response.is_success:
                return response

            try:
                self._raise_for_status(response)
            except APIStatusError as exc:
                if attempt < attempts:
                    wait = self._sleep_for(exc, policy, attempt)
                    if wait is not None:
                        time.sleep(wait)
                        continue
                raise

        raise last_exc or APIConnectionError("request failed")  # pragma: no cover

    def get_json(self, path: str, **kw: Any) -> Any:
        return self.request("GET", path, **kw).json()

    def post_json(self, path: str, **kw: Any) -> Any:
        return self.request("POST", path, **kw).json()

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> SupaGamma:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class AsyncSupaGamma(BaseClient):
    """Asynchronous client. Identical surface to :class:`SupaGamma`.

    >>> async with AsyncSupaGamma(api_key="sg_...") as client:
    ...     markets = await client.markets.list(limit=10)
    """

    def __init__(
        self, *args: Any, http_client: Optional[httpx.AsyncClient] = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._http = http_client or httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=True,
        )
        self._owns_http = http_client is None
        self._attach_resources()

    def _attach_resources(self) -> None:
        from .resources import build_async_namespaces

        build_async_namespaces(self)

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        json: Any = None,
        headers: Optional[Mapping[str, str]] = None,
        policy: RetryPolicy = NEVER,
        stream: bool = False,
    ) -> httpx.Response:
        import asyncio

        attempts = self._effective_attempts(policy)
        url = self._url(path)
        query = self._clean_params(params)
        last_exc: Optional[Exception] = None

        for attempt in range(attempts + 1):
            request_headers = self._headers(extra=headers)
            try:
                response = await self._http.request(
                    method, url, params=query, json=json, headers=request_headers
                )
            except httpx.TimeoutException as exc:
                last_exc = APITimeoutError(str(exc), request_id=request_headers["X-Request-ID"])
                if policy.on_connection_error and attempt < attempts:
                    await asyncio.sleep(_backoff(attempt))
                    continue
                raise last_exc from exc
            except httpx.HTTPError as exc:
                last_exc = APIConnectionError(str(exc), request_id=request_headers["X-Request-ID"])
                if policy.on_connection_error and attempt < attempts:
                    await asyncio.sleep(_backoff(attempt))
                    continue
                raise last_exc from exc

            self._capture_rate_limit(response)
            if response.is_success:
                return response

            try:
                self._raise_for_status(response)
            except APIStatusError as exc:
                if attempt < attempts:
                    wait = self._sleep_for(exc, policy, attempt)
                    if wait is not None:
                        await asyncio.sleep(wait)
                        continue
                raise

        raise last_exc or APIConnectionError("request failed")  # pragma: no cover

    async def get_json(self, path: str, **kw: Any) -> Any:
        return (await self.request("GET", path, **kw)).json()

    async def post_json(self, path: str, **kw: Any) -> Any:
        return (await self.request("POST", path, **kw)).json()

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> AsyncSupaGamma:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()
