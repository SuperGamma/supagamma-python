"""``client.exports`` — async exports (``/v1/exports``): the way to pull anything large.

A synchronous ``client.download.*`` request has to finish inside the ~100 s an edge
proxy allows, and a wide historical pull cannot. An export keeps every HTTP request
short instead: you submit a job, it is built in the background, you poll it, and a
finished job hands out a short-lived signed URL that you can re-issue for free until
the artifact expires (7 days).

    job = client.exports.create(kind="trades", market_id="1254468", format="parquet")
    job = client.exports.wait(job["id"])                  # polls until it succeeds
    client.exports.download(job["id"], "trades.parquet")  # streams it to disk

Money and retries
-----------------

``create()`` is metered work, so it is **never retried automatically**. It is also
the one paid POST in this SDK that is *safe to replay by hand*, because the server
treats ``idempotency_key`` as the identity of the job: a repeat with the same key
returns the SAME job instead of building and billing it twice. The SDK therefore
always sends one. It generates a ``uuid4`` when you pass none and returns it as
``job["idempotency_key"]``. If the call *fails* instead (a timeout, a 502), the same
key is on the exception as ``exc.idempotency_key``, so you can replay it:

    try:
        job = client.exports.create(kind="trades", market_id="1254468")
    except (supagamma.APIConnectionError, supagamma.OrderStatusUnknownError) as exc:
        # a timeout, a dropped connection or a 502: the job may or may not exist
        job = client.exports.create(
            kind="trades", market_id="1254468", idempotency_key=exc.idempotency_key
        )

Two properties of the key worth knowing. It is scoped to your account, and the
server returns the existing job for a reused key **without comparing the request**,
so a key must never be reused for a different pull. And nothing is charged when you
submit: the worker charges when the build finishes, and a charge it refuses (402,
a spent fair-use allowance) fails the job with the reason in ``job["error"]``.

Everything else here is a read and may be retried.

The job object
--------------

Jobs are plain dicts, exactly as the server sends them: ``id``, ``status``
(``queued``, ``running``, ``succeeded``, ``failed`` or ``expired``), ``spec``,
``file_name``, ``record_count``, ``size_bytes``, ``cost``, ``disposition``,
``error``, ``created_at``, ``completed_at`` and ``expires_at``. They are not typed
because the API publishes no schema for them.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

import httpx

from .._client import NEVER, SAFE_READ
from .._errors import (
    APIStatusError,
    ExportFailedError,
    ExportTimeoutError,
    ResponseShapeError,
    SupaGammaError,
    parse_error,
)
from .._streaming import (
    DownloadedFile,
    PathLike,
    awrite_stream,
    filename_from,
    transport_failure,
    write_stream,
)
from ._base import AsyncResource, Call, SyncResource, call

__all__ = [
    "EXPORT_FORMATS",
    "EXPORT_KINDS",
    "LIVE_STATES",
    "MAX_IDEMPOTENCY_KEY_LENGTH",
    "MAX_LIST_LIMIT",
    "TERMINAL_STATES",
    "URL_TTL_MAX",
    "URL_TTL_MIN",
    "AsyncExports",
    "Exports",
    "build_create",
    "build_get",
    "build_list",
    "build_url",
]

EXPORT_KINDS = ("trades", "raw")
EXPORT_FORMATS = ("parquet", "csv", "json")

#: States a job can still leave.
LIVE_STATES = frozenset({"queued", "running"})
#: States a job never leaves. ``succeeded`` is the only good one.
TERMINAL_STATES = frozenset({"succeeded", "failed", "expired"})

MAX_LIST_LIMIT = 200
MAX_IDEMPOTENCY_KEY_LENGTH = 128
#: Server bounds on the signed URL's lifetime, in seconds.
URL_TTL_MIN = 60
URL_TTL_MAX = 3600

# Seconds. A big export legitimately takes minutes, so poll gently and back off.
_DEFAULT_WAIT_TIMEOUT = 3600.0
_DEFAULT_POLL_INTERVAL = 5.0
_MAX_POLL_INTERVAL = 30.0


# --- clocks, indirected so tests can drive them -------------------------------


def _now() -> float:
    return time.monotonic()


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


async def _asleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


# --- validation ---------------------------------------------------------------


def _as_utc(value: datetime) -> datetime:
    """A naive datetime is read as UTC, as the API documents; an aware one is converted."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _utc_iso(value: datetime) -> str:
    """UTC-normalise a bound for the wire.

    The server compares ``start`` and ``end`` directly, and Python refuses to
    compare an aware datetime with a naive one, so never send a mix.
    """
    return _as_utc(value).isoformat()


def _check_job_id(job_id: str) -> str:
    if not job_id or not job_id.strip():
        raise ValueError("job_id must be a non-empty string")
    if "/" in job_id:
        raise ValueError(
            f"job_id must not contain '/': {job_id!r} would change which endpoint is called"
        )
    return job_id


def _create_payload(
    *,
    kind: str,
    market_id: Optional[str],
    series_id: Optional[str],
    start: Optional[datetime],
    end: Optional[datetime],
    format: str,
    idempotency_key: Optional[str],
) -> Dict[str, Any]:
    if kind not in EXPORT_KINDS:
        raise ValueError(f"kind must be one of {EXPORT_KINDS}, got {kind!r}")
    if format not in EXPORT_FORMATS:
        raise ValueError(f"format must be one of {EXPORT_FORMATS}, got {format!r}")
    if kind == "trades":
        if not market_id:
            raise ValueError("a trades export needs market_id (the numeric markets.id)")
        if series_id:
            raise ValueError("series_id does not apply to a trades export; it takes market_id")
    else:
        if not series_id:
            raise ValueError(
                "a raw export needs series_id: the raw stream's SERIES id, such as "
                "'polymarket:l2-delta-tape' (not the catalog data_type that "
                "download.raw also accepts). See client.series.list()."
            )
        if market_id:
            raise ValueError("market_id does not apply to a raw export; it takes series_id")
    if start is not None and end is not None and _as_utc(end) <= _as_utc(start):
        raise ValueError("end must be after start")
    key = idempotency_key or uuid.uuid4().hex
    if len(key) > MAX_IDEMPOTENCY_KEY_LENGTH:
        raise ValueError(f"idempotency_key must be at most {MAX_IDEMPOTENCY_KEY_LENGTH} characters")
    payload: Dict[str, Any] = {"kind": kind, "format": format, "idempotency_key": key}
    for name, value in (
        ("market_id", market_id),
        ("series_id", series_id),
        ("start", _utc_iso(start) if start is not None else None),
        ("end", _utc_iso(end) if end is not None else None),
    ):
        if value is not None:
            payload[name] = value
    return payload


def _check_limit(limit: int) -> int:
    if not 1 <= limit <= MAX_LIST_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_LIST_LIMIT}, got {limit!r}")
    return limit


def _check_ttl(ttl: int) -> int:
    if not URL_TTL_MIN <= ttl <= URL_TTL_MAX:
        raise ValueError(
            f"ttl must be between {URL_TTL_MIN} and {URL_TTL_MAX} seconds, got {ttl!r}"
        )
    return ttl


# --- request builders ----------------------------------------------------------


def build_create() -> Call:
    """``POST /v1/exports`` — queues metered work. ``NEVER`` retried."""
    return call("POST", "/v1/exports", {}, NEVER)


def build_list(*, limit: int = 50) -> Call:
    """``GET /v1/exports`` — a free read."""
    return call("GET", "/v1/exports", {"limit": _check_limit(limit)}, SAFE_READ)


def build_get(job_id: str) -> Call:
    """``GET /v1/exports/{job_id}`` — a free read."""
    return call("GET", f"/v1/exports/{_check_job_id(job_id)}", {}, SAFE_READ)


def build_url(job_id: str, *, ttl: int = 300) -> Call:
    """``GET /v1/exports/{job_id}/url`` — free, and re-issuable."""
    return call(
        "GET", f"/v1/exports/{_check_job_id(job_id)}/url", {"ttl": _check_ttl(ttl)}, SAFE_READ
    )


# --- response handling ----------------------------------------------------------


def _with_key(job: Any, payload: Dict[str, Any]) -> Any:
    """Echo the idempotency key on the job; the server does not send it back."""
    if isinstance(job, dict):
        job.setdefault("idempotency_key", payload["idempotency_key"])
    return job


def _jobs_of(body: Any) -> List[Dict[str, Any]]:
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list):
        raise ResponseShapeError("GET /v1/exports did not return a {'data': [...]} envelope")
    return data


def _url_info_of(body: Any) -> Dict[str, Any]:
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, dict) or not data.get("download_url"):
        raise ResponseShapeError("the export URL response carried no download_url")
    return data


def _check_terminal(job: Dict[str, Any]) -> None:
    status = job.get("status")
    if status in ("failed", "expired"):
        reason = job.get("error") or (
            "the artifact has expired; submit the export again" if status == "expired" else ""
        )
        raise ExportFailedError(
            f"export {job.get('id')} {status}" + (f": {reason}" if reason else ""), job=job
        )


def _timed_out(job: Dict[str, Any], timeout: float) -> ExportTimeoutError:
    return ExportTimeoutError(
        f"export {job.get('id')} is still {job.get('status')!r} after {timeout:g}s. "
        "It keeps building; poll it again with exports.get().",
        job=job,
    )


def _signed_filename(url: str) -> Optional[str]:
    name = PurePosixPath(urlparse(url).path).name
    return name or None


def _signed_failure(response: httpx.Response) -> APIStatusError:
    """The typed error for a signed-URL fetch that did not return 2xx."""
    return parse_error(
        status_code=response.status_code,
        body=None,
        raw_body=response.text[:500],
        headers=response.headers,
        request_id=None,
    )


# --- sync ------------------------------------------------------------------------


class Exports(SyncResource):
    """Async exports: submit, poll, then fetch a signed URL."""

    def create(
        self,
        *,
        kind: str,
        market_id: Optional[str] = None,
        series_id: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        format: str = "parquet",
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Submit an export. **Metered. Never retried automatically.**

        ``kind="trades"`` needs ``market_id`` (the numeric ``markets.id``) and
        exports both of the market's outcome tokens, in one file, told apart only
        by the ``market_id`` column. ``kind="raw"`` needs ``series_id``, the raw
        stream's *series* id such as ``polymarket:l2-delta-tape``.

        Returns the job (status ``queued``) with ``idempotency_key`` added
        client-side. After an ambiguous failure, replay this call with the SAME key
        (it is on the exception as ``exc.idempotency_key`` too) and the server
        returns the original job instead of building it twice. Never reuse a key for
        a different pull; the server returns the old job for it without comparing
        the request. Needs the ``download`` scope.

        Raises ``ValueError`` before sending for a bad ``kind``/``format``, a
        missing or mismatched id, ``end <= start`` or an over-long key.
        """
        payload = _create_payload(
            kind=kind,
            market_id=market_id,
            series_id=series_id,
            start=start,
            end=end,
            format=format,
            idempotency_key=idempotency_key,
        )
        try:
            job: Dict[str, Any] = _with_key(self._json(build_create(), json=payload), payload)
        except SupaGammaError as exc:
            exc.idempotency_key = payload["idempotency_key"]
            raise
        return job

    def list(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        """Your export jobs, newest first (1..200). A free read.

        Returns the list itself; the server's ``{'data': [...], 'meta': {...}}``
        envelope is unwrapped.
        """
        return _jobs_of(self._json(build_list(limit=limit)))

    def get(self, job_id: str) -> Dict[str, Any]:
        """One job. A free read; a 404 means not found *or* not yours.

        A job that has been ``queued`` or ``running`` far longer than a build can
        take is reported ``failed`` here, with an explanation, not left to look
        alive forever.
        """
        job: Dict[str, Any] = self._json(build_get(job_id))
        return job

    def url(self, job_id: str, *, ttl: int = 300) -> Dict[str, Any]:
        """A signed download URL for a finished export. Free; re-issue it any time.

        ``ttl`` is the URL's lifetime in seconds (60..3600). Returns the job
        merged with ``download_url`` and ``expires_in``. A job that is not
        ``succeeded`` raises :class:`~supagamma.ConflictError` (409); an expired
        one raises :class:`~supagamma.GoneError` (410).

        The URL is already authorised, so fetch it **without** your API key.
        :meth:`download` does exactly that.
        """
        return _url_info_of(self._json(build_url(job_id, ttl=ttl)))

    def wait(
        self,
        job_id: str,
        *,
        timeout: float = _DEFAULT_WAIT_TIMEOUT,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        on_poll: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Poll a job until it succeeds, and return it.

        Polls every ``poll_interval`` seconds, backing off toward 30 s. A big
        export takes minutes; that is normal, not a hang. ``on_poll(job)`` is
        called after every poll, for progress output.

        Raises :class:`~supagamma.ExportFailedError` if the job ends ``failed`` or
        ``expired`` (``.job["error"]`` says why, including whether you were
        charged), and :class:`~supagamma.ExportTimeoutError` if ``timeout``
        seconds pass first. The latter does not cancel anything: the job keeps
        building.
        """
        deadline = _now() + timeout
        interval = poll_interval
        while True:
            job = self.get(job_id)
            if on_poll is not None:
                on_poll(job)
            if job.get("status") == "succeeded":
                return job
            _check_terminal(job)
            remaining = deadline - _now()
            if remaining <= 0:
                raise _timed_out(job, timeout)
            _sleep(min(interval, remaining))
            interval = min(interval * 1.5, _MAX_POLL_INTERVAL)

    def download(self, job_id: str, save_to: PathLike, *, ttl: int = 300) -> DownloadedFile:
        """Stream a finished export to ``save_to``. Free.

        Mints a signed URL and streams it to disk (a path, or an existing
        directory, in which case the artifact's own name is used) with the same
        all-or-nothing write as ``download.*(save_to=...)``. The request to the
        signed URL goes through the client's own HTTP client, so your proxy and TLS
        settings apply, but it carries **no** SupaGamma credentials: they are not
        needed, and must not be sent to another host.
        """
        info = self.url(job_id, ttl=ttl)
        url = str(info["download_url"])
        try:
            # ``stream`` is called on the httpx client directly, not through
            # ``self._client.request``: that is what adds the API key, and the signed
            # URL is on another host and needs none.
            with self._client._http.stream("GET", url, follow_redirects=True) as response:
                if not response.is_success:
                    response.read()
                    raise _signed_failure(response)
                return write_stream(
                    response, save_to, filename=filename_from(response) or _signed_filename(url)
                )
        except httpx.HTTPError as exc:
            # This fetch bypasses ``request``, so its connection errors are not mapped
            # for us: a refused connection must not escape as a bare httpx error.
            raise transport_failure(exc, None) from exc


# --- async -----------------------------------------------------------------------


class AsyncExports(AsyncResource):
    """Async twin of :class:`Exports`."""

    async def create(
        self,
        *,
        kind: str,
        market_id: Optional[str] = None,
        series_id: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        format: str = "parquet",
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = _create_payload(
            kind=kind,
            market_id=market_id,
            series_id=series_id,
            start=start,
            end=end,
            format=format,
            idempotency_key=idempotency_key,
        )
        try:
            job: Dict[str, Any] = _with_key(await self._json(build_create(), json=payload), payload)
        except SupaGammaError as exc:
            exc.idempotency_key = payload["idempotency_key"]
            raise
        return job

    async def list(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        return _jobs_of(await self._json(build_list(limit=limit)))

    async def get(self, job_id: str) -> Dict[str, Any]:
        job: Dict[str, Any] = await self._json(build_get(job_id))
        return job

    async def url(self, job_id: str, *, ttl: int = 300) -> Dict[str, Any]:
        return _url_info_of(await self._json(build_url(job_id, ttl=ttl)))

    async def wait(
        self,
        job_id: str,
        *,
        timeout: float = _DEFAULT_WAIT_TIMEOUT,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        on_poll: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        deadline = _now() + timeout
        interval = poll_interval
        while True:
            job = await self.get(job_id)
            if on_poll is not None:
                on_poll(job)
            if job.get("status") == "succeeded":
                return job
            _check_terminal(job)
            remaining = deadline - _now()
            if remaining <= 0:
                raise _timed_out(job, timeout)
            await _asleep(min(interval, remaining))
            interval = min(interval * 1.5, _MAX_POLL_INTERVAL)

    async def download(self, job_id: str, save_to: PathLike, *, ttl: int = 300) -> DownloadedFile:
        info = await self.url(job_id, ttl=ttl)
        url = str(info["download_url"])
        try:
            async with self._client._http.stream("GET", url, follow_redirects=True) as response:
                if not response.is_success:
                    await response.aread()
                    raise _signed_failure(response)
                return await awrite_stream(
                    response, save_to, filename=filename_from(response) or _signed_filename(url)
                )
        except httpx.HTTPError as exc:
            raise transport_failure(exc, None) from exc


for _name in ("create", "list", "get", "url", "wait", "download"):
    getattr(AsyncExports, _name).__doc__ = getattr(Exports, _name).__doc__
del _name
