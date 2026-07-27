"""``client.download`` — **the money path**.

Every method on this namespace except :meth:`Download.raw_datasets` and
:meth:`Download.raw_estimate` debits your credit balance. Read this before using
it in anything automated.

**Nothing here is ever retried automatically.** Each route passes ``NEVER``,
regardless of the client's ``max_retries``. The reason is specific: the server
debits *after* serialising but *before* the body finishes reaching you, and the
only protection against paying twice is a 7-day entitlement waiver matched on an
exact parameter tuple — user, entitlement key, ``data_type``, ``format``,
``timeframe`` and, for windowed pulls, the same ``[start, end]``. A retry that
re-derives ``end=now()`` looks like a *different* request to that matcher and is
charged again at full price. If you retry a download, replay byte-identical
parameters, and sleep a couple of seconds first — the entitlement row is written
by a background task and may not exist the instant the response returns.

**Truncation is silent.** When a pull hits its row cap the response carries no
flag, no header and no marker; a truncated file looks exactly like a complete
one. When completeness matters, call the matching estimate first and compare.

**``raw_datasets()`` and ``raw_estimate()`` are free but not cheap.** Their paths
begin with ``/v1/download``, so they consume the same 10-per-hour bucket as a
real pull. An estimate-then-download loop halves your effective quota. The
dataset catalogue is cached here for the client's lifetime because of it.
"""

from __future__ import annotations

import json
import logging
import warnings
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import httpx

from .._client import NEVER, SAFE_READ
from ._base import AsyncResource, Call, SyncResource, call

logger = logging.getLogger("supagamma.download")

__all__ = [
    "RAW_DATA_TYPES",
    "NORMALIZABLE_DATA_TYPES",
    "BulkItem",
    "DownloadResult",
    "BulkResult",
    "CostWarning",
    "build_trades",
    "build_ohlcv",
    "build_orderbook",
    "build_top_of_book",
    "build_options",
    "build_series",
    "build_raw",
    "build_raw_datasets",
    "build_raw_estimate",
    "build_bulk",
    "Download",
    "AsyncDownload",
]

#: The only accepted ``data_type`` values on :meth:`Download.raw`.
RAW_DATA_TYPES = (
    "hyperliquid_l2_snapshots",
    "hyperliquid_l2_bbo",
    "hyperliquid_trades",
    "hyperliquid_outcome_lifecycle",
    "polymarket_l2_deltas",
    "polymarket_trades",
    "polymarket_lifecycle",
    "polymarket_us_l2_snapshots",
    "limitless_l2_snapshots",
)

#: The subset for which ``shape="normalized"`` exists.
NORMALIZABLE_DATA_TYPES = frozenset(
    {
        "hyperliquid_l2_snapshots",
        "polymarket_us_l2_snapshots",
        "limitless_l2_snapshots",
        "polymarket_l2_deltas",
    }
)

# Local cost model, mirroring the server's. Used only to warn BEFORE spending —
# never to assert what you were charged, which only the bulk manifest reports.
_BYTES_PER_ROW = {
    "trades": 230,
    "ohlcv": 137,
    "orderbook": 2076,
    "greeks": 400,
    "snapshots": 250,
}
_RATE_PER_MB = {
    "trades": 3.00,
    "ohlcv": 20.00,
    "orderbook": 5.00,
    "greeks": 12.00,
    "snapshots": 8.00,
}

#: Warn above this estimated dollar cost.
COST_WARN_THRESHOLD_USD = 25.0


class CostWarning(UserWarning):
    """A pull is estimated to cost more than :data:`COST_WARN_THRESHOLD_USD`."""


def estimate_cost_usd(rows: int, data_type: str) -> float:
    """Local, approximate cost for ``rows`` of ``data_type``.

    Mirrors the server's formula (there is no free-tier allowance), but the
    server re-counts on delivery, so treat this as an order-of-magnitude guide
    for warnings — not as a quote.
    """
    per_row = _BYTES_PER_ROW.get(data_type, 230)
    rate = _RATE_PER_MB.get(data_type, 1.00)
    return max(0.0, (rows * per_row) / 1_048_576) * rate


def _maybe_warn_cost(rows: int, data_type: str, confirm: Optional[Any] = None) -> None:
    cost = estimate_cost_usd(rows, data_type)
    if cost < COST_WARN_THRESHOLD_USD:
        return
    message = (
        f"This {data_type} pull requests up to {rows:,} rows, roughly ${cost:,.2f} at "
        f"${_RATE_PER_MB.get(data_type, 1.0):.2f}/MB. Lower `limit` or narrow start/end "
        f"if that is more than you intended."
    )
    if confirm is not None and not confirm(cost):
        raise RuntimeError(f"download cancelled by confirm_cost hook: {message}")
    warnings.warn(message, CostWarning, stacklevel=4)
    logger.warning(message)


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value is not None else None


# --- builders ----------------------------------------------------------------
# Every data route is NEVER-retry. That is not a per-call preference; it is the
# property that stops the SDK double-charging a customer.


def build_trades(
    *,
    market_id: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    format: str = "parquet",
    limit: int = 100_000,
) -> Call:
    if format == "json":
        # The server serialises this route's Arrow timestamp column with a bare
        # json.dumps() and no `default=`, so datetimes raise TypeError -> 500.
        # It fires before the charge, so it costs nothing but a wasted call.
        warnings.warn(
            "format='json' on download.trades() is known to fail server-side "
            "(the timestamp column is not JSON-serialisable). Use 'parquet' or 'csv'.",
            UserWarning,
            stacklevel=3,
        )
    return call(
        "GET",
        "/v1/download/trades",
        {
            "market_id": market_id,
            "start": _iso(start),
            "end": _iso(end),
            "format": format,
            "limit": limit,
        },
        NEVER,
    )


def build_ohlcv(
    *,
    market_id: str,
    timeframe: str = "1h",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    format: str = "csv",
    limit: int = 1_000_000,
) -> Call:
    return call(
        "GET",
        "/v1/download/ohlcv",
        {
            "market_id": market_id,
            "timeframe": timeframe,
            "start": _iso(start),
            "end": _iso(end),
            "format": format,
            "limit": limit,
        },
        NEVER,
    )


def build_orderbook(
    *,
    market_id: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    format: str = "parquet",
    limit: int = 100_000,
) -> Call:
    return call(
        "GET",
        "/v1/download/orderbook",
        {
            "market_id": market_id,
            "start": _iso(start),
            "end": _iso(end),
            "format": format,
            "limit": limit,
        },
        NEVER,
    )


def build_top_of_book(
    *,
    market_id: str,
    start: datetime,
    end: datetime,
    format: str = "csv",
    limit: int = 100_000,
) -> Call:
    return call(
        "GET",
        "/v1/download/top_of_book",
        {
            "market_id": market_id,
            "start": _iso(start),
            "end": _iso(end),
            "format": format,
            "limit": limit,
        },
        NEVER,
    )


def build_options(
    *,
    series_id: str,
    data_type: str = "trades",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    format: str = "csv",
    instrument_prefix: Optional[str] = None,
    limit: int = 500_000,
) -> Call:
    return call(
        "GET",
        "/v1/download/options",
        {
            "series_id": series_id,
            "data_type": data_type,
            "start": _iso(start),
            "end": _iso(end),
            "format": format,
            "instrument_prefix": instrument_prefix,
            "limit": limit,
        },
        NEVER,
    )


def build_series(
    *,
    series_id: str,
    data_type: str = "trades",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    format: str = "csv",
    instrument_prefix: Optional[str] = None,
    limit: int = 500_000,
) -> Call:
    return call(
        "GET",
        "/v1/download/series",
        {
            "series_id": series_id,
            "data_type": data_type,
            "start": _iso(start),
            "end": _iso(end),
            "format": format,
            "instrument_prefix": instrument_prefix,
            "limit": limit,
        },
        NEVER,
    )


def build_raw(
    *,
    data_type: str,
    start: datetime,
    end: datetime,
    format: str = "json",
    shape: str = "raw",
    limit: int = 500_000,
) -> Call:
    if shape == "normalized" and data_type not in NORMALIZABLE_DATA_TYPES:
        warnings.warn(
            f"shape='normalized' is not available for {data_type!r}; the server will reject it. "
            f"Normalizable types: {sorted(NORMALIZABLE_DATA_TYPES)}.",
            UserWarning,
            stacklevel=3,
        )
    return call(
        "GET",
        "/v1/download/raw",
        {
            "data_type": data_type,
            "start": _iso(start),
            "end": _iso(end),
            "format": format,
            "shape": shape,
            "limit": limit,
        },
        NEVER,
    )


def build_raw_datasets() -> Call:
    # Free, but on the /v1/download prefix, so it burns the 10/hour bucket.
    return call("GET", "/v1/download/raw/datasets", {}, SAFE_READ)


def build_raw_estimate(
    *,
    data_type: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: int = 500_000,
) -> Call:
    return call(
        "GET",
        "/v1/download/raw/estimate",
        {
            "data_type": data_type,
            "start": _iso(start),
            "end": _iso(end),
            "limit": limit,
        },
        SAFE_READ,
    )


def build_bulk() -> Call:
    return call("POST", "/v1/download/bulk", {}, NEVER)


# --- results -----------------------------------------------------------------


@dataclass
class DownloadResult:
    """Bytes plus enough context to save them under the right name."""

    content: bytes
    content_type: str
    filename: Optional[str] = None
    request_id: Optional[str] = None

    def save_to(self, path: Union[str, Path]) -> Path:
        """Write to ``path``. If ``path`` is a directory, use the server's filename."""
        target = Path(path)
        if target.is_dir():
            target = target / (self.filename or "supagamma-download")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.content)
        return target

    def json(self) -> Any:
        """Parse as JSON. Only meaningful when you requested ``format='json'``."""
        return json.loads(self.content)

    def __len__(self) -> int:
        return len(self.content)


@dataclass
class BulkResult:
    """A bulk ZIP plus its parsed manifest.

    ``manifest`` is the **only** place per-item cost and disposition are
    reported anywhere in the API — single downloads report neither.
    """

    content: bytes
    manifest: Dict[str, Any] = field(default_factory=dict)
    filename: Optional[str] = None
    request_id: Optional[str] = None

    @property
    def total_cost(self) -> Optional[float]:
        value = self.manifest.get("total_cost")
        return float(value) if value is not None else None

    @property
    def items(self) -> List[Dict[str, Any]]:
        return list(self.manifest.get("items") or [])

    def save_to(self, path: Union[str, Path]) -> Path:
        target = Path(path)
        if target.is_dir():
            target = target / (self.filename or "supagamma_bulk.zip")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.content)
        return target

    def extract_to(self, directory: Union[str, Path]) -> List[Path]:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(BytesIO(self.content)) as archive:
            archive.extractall(out)
            return [out / name for name in archive.namelist()]


@dataclass
class BulkItem:
    """One line of a bulk request.

    ``format`` is per item — a single ZIP can mix csv, json and parquet, so read
    each entry's ``file`` from the manifest rather than assuming an extension.
    """

    kind: str
    market_id: Optional[str] = None
    series_id: Optional[str] = None
    data_type: Optional[str] = None
    timeframe: Optional[str] = None
    instrument_prefix: Optional[str] = None
    shape: str = "raw"
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    format: str = "csv"
    limit: int = 100_000

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"kind": self.kind, "format": self.format, "limit": self.limit}
        for key, value in (
            ("market_id", self.market_id),
            ("series_id", self.series_id),
            ("data_type", self.data_type),
            ("timeframe", self.timeframe),
            ("instrument_prefix", self.instrument_prefix),
            ("start", _iso(self.start)),
            ("end", _iso(self.end)),
        ):
            if value is not None:
                payload[key] = value
        if self.kind == "raw":
            payload["shape"] = self.shape
        return payload


def _filename_from(response: httpx.Response) -> Optional[str]:
    disposition = response.headers.get("content-disposition", "")
    if "filename=" not in disposition:
        return None
    return disposition.split("filename=", 1)[1].strip().strip('"') or None


def _to_result(response: httpx.Response) -> DownloadResult:
    return DownloadResult(
        content=response.content,
        content_type=response.headers.get("content-type", ""),
        filename=_filename_from(response),
        request_id=response.headers.get("x-request-id"),
    )


def _to_bulk_result(response: httpx.Response) -> BulkResult:
    content = response.content
    manifest: Dict[str, Any] = {}
    try:
        with zipfile.ZipFile(BytesIO(content)) as archive:
            if "manifest.json" in archive.namelist():
                manifest = json.loads(archive.read("manifest.json"))
    except (zipfile.BadZipFile, ValueError, KeyError):
        # Never let a manifest problem hide a bundle the caller already paid
        # for — hand back the bytes regardless.
        logger.warning("bulk response was not a readable zip; returning raw bytes")
    return BulkResult(
        content=content,
        manifest=manifest,
        filename=_filename_from(response),
        request_id=response.headers.get("x-request-id"),
    )


_MONEY_NOTE = """
        **Spends credits.** Never retried automatically; see the module
        docstring for why replaying a download needs byte-identical parameters.
        Row-cap truncation is silent — call the matching estimate if
        completeness matters.
"""


class Download(SyncResource):
    """Paid data delivery. Every method here except the two estimates charges."""

    #: Optional hook: ``confirm_cost(estimated_usd) -> bool``. Return False to
    #: abort before the request is sent.
    confirm_cost: Optional[Any] = None

    _datasets_cache: Optional[Any] = None

    def trades(
        self,
        *,
        market_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        format: str = "parquet",
        limit: int = 100_000,
    ) -> DownloadResult:
        """Per-market fills. $3.00/MB.

        Defaults to parquet: this route's ``format='json'`` path is broken
        server-side. Zero matching rows raises
        :class:`~supagamma.NoDataInRangeError` rather than returning an empty file.
        """
        _maybe_warn_cost(limit, "trades", self.confirm_cost)
        return _to_result(
            self._raw(
                build_trades(market_id=market_id, start=start, end=end, format=format, limit=limit)
            )
        )

    def ohlcv(
        self,
        *,
        market_id: str,
        timeframe: str = "1h",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        format: str = "csv",
        limit: int = 1_000_000,
    ) -> DownloadResult:
        """OHLCV bars. $20.00/MB.

        The one per-market route with a free disposition: a non-subscriber
        pulling ``timeframe='1d'`` for a window entirely inside the last 30 days
        (and older than 24h) is served free, subject to a monthly cap that
        raises :class:`~supagamma.FreeTierCapError`. ``1m``/``1h`` are charged.
        """
        _maybe_warn_cost(limit, "ohlcv", self.confirm_cost)
        return _to_result(
            self._raw(
                build_ohlcv(
                    market_id=market_id,
                    timeframe=timeframe,
                    start=start,
                    end=end,
                    format=format,
                    limit=limit,
                )
            )
        )

    def orderbook(
        self,
        *,
        market_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        format: str = "parquet",
        limit: int = 100_000,
    ) -> DownloadResult:
        """Full-depth L2 snapshots. $5.00/MB — **the expensive one**.

        At ~2 KB per row the default ``limit`` is on the order of **$990**. This
        method warns loudly above :data:`COST_WARN_THRESHOLD_USD`; set
        ``client.download.confirm_cost`` to gate it programmatically.

        Rows carry nested ``bids``/``asks`` arrays. CSV renders those as Python
        ``repr`` strings rather than JSON, so CSV is lossy here — prefer parquet
        or json.
        """
        _maybe_warn_cost(limit, "orderbook", self.confirm_cost)
        return _to_result(
            self._raw(
                build_orderbook(
                    market_id=market_id, start=start, end=end, format=format, limit=limit
                )
            )
        )

    def top_of_book(
        self,
        *,
        market_id: str,
        start: datetime,
        end: datetime,
        format: str = "csv",
        limit: int = 100_000,
    ) -> DownloadResult:
        """Best bid/ask projection. **Free — this route never charges.**

        ``start`` and ``end`` are required and the whole window must sit inside
        the last 30 days and end at least 24h ago, else
        :class:`~supagamma.FreeTierWindowError`. The monthly cap is measured on a
        fixed per-row estimate, so compressing with parquet buys no extra rows.
        """
        return _to_result(
            self._raw(
                build_top_of_book(
                    market_id=market_id, start=start, end=end, format=format, limit=limit
                )
            )
        )

    def options(
        self,
        *,
        series_id: str,
        data_type: str = "trades",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        format: str = "csv",
        instrument_prefix: Optional[str] = None,
        limit: int = 500_000,
    ) -> DownloadResult:
        """Crypto-derivatives archive. trades $3 / greeks $12 / snapshots $8 per MB.

        Crypto capture is paused as a business decision, so this serves the
        historical archive only.
        """
        _maybe_warn_cost(limit, data_type, self.confirm_cost)
        return _to_result(
            self._raw(
                build_options(
                    series_id=series_id,
                    data_type=data_type,
                    start=start,
                    end=end,
                    format=format,
                    instrument_prefix=instrument_prefix,
                    limit=limit,
                )
            )
        )

    def series(
        self,
        *,
        series_id: str,
        data_type: str = "trades",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        format: str = "csv",
        instrument_prefix: Optional[str] = None,
        limit: int = 500_000,
    ) -> DownloadResult:
        """Catalogue-backed series.

        ``data_type`` here is a **pricing tier** (``trades``/``snapshots``/``ohlcv``),
        not a catalog data type — ``greeks`` is not accepted. Whether a given id
        belongs to this route or to :meth:`options` depends on its series family,
        so prefer :meth:`for_series`, which decides from the catalogue.
        """
        _maybe_warn_cost(limit, data_type, self.confirm_cost)
        return _to_result(
            self._raw(
                build_series(
                    series_id=series_id,
                    data_type=data_type,
                    start=start,
                    end=end,
                    format=format,
                    instrument_prefix=instrument_prefix,
                    limit=limit,
                )
            )
        )

    def raw(
        self,
        data_type: str,
        start: datetime,
        end: datetime,
        *,
        format: str = "json",
        shape: str = "raw",
        limit: int = 500_000,
    ) -> DownloadResult:
        """The event-granular L2 tape.

        ``data_type``, ``start`` and ``end`` are positional and required — the
        server declares the window optional then rejects its absence, so the SDK
        makes it explicit.

        With ``shape='raw'`` each row is ``{"recv_ts": <epoch ms>, "frame": <venue
        frame as text>}`` and you parse ``frame`` yourself. With
        ``shape='normalized'`` rows are flat per-level records — but note the
        server then caps the underlying read at 50,000 *input* frames, and bills
        on real serialised bytes, so :meth:`raw_estimate` (which models the raw
        path) will understate it materially.
        """
        return _to_result(
            self._raw(
                build_raw(
                    data_type=data_type,
                    start=start,
                    end=end,
                    format=format,
                    shape=shape,
                    limit=limit,
                )
            )
        )

    def raw_datasets(self, *, refresh: bool = False) -> Any:
        """The raw stream catalogue. Free — but burns a download-bucket slot.

        Cached for the life of this client because of that. Pass
        ``refresh=True`` to force a re-read.
        """
        if self._datasets_cache is None or refresh:
            body = self._json(build_raw_datasets())
            # The one endpoint in the entire API that uses a {data, meta} envelope.
            self._datasets_cache = body.get("data") if isinstance(body, dict) else body
        return self._datasets_cache

    def raw_estimate(
        self,
        *,
        data_type: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 500_000,
    ) -> Dict[str, Any]:
        """Price a raw pull without buying it.

        Free, but it consumes one of your 10 download calls per hour — an
        estimate-then-download pattern halves your effective quota.

        ``capped_by_limit`` is the only place the API admits truncation. Models
        the ``shape='raw'`` path only.
        """
        return self._json(
            build_raw_estimate(data_type=data_type, start=start, end=end, limit=limit)
        )

    def bulk(self, items: Sequence[BulkItem]) -> BulkResult:
        """Up to 25 items, priced as a union, **one atomic debit**, one ZIP.

        There is **no idempotency mechanism on this route at all**, so it is
        never retried and you should not retry it yourself. If a call fails
        ambiguously, check ``client.account.downloads()`` before re-issuing.

        A valid-but-empty item becomes an ``.EMPTY.txt`` entry with cost 0
        rather than failing; an *invalid* item fails the whole request.
        """
        if not items:
            raise ValueError("bulk() needs at least one item")
        if len(items) > 25:
            raise ValueError(f"bulk() accepts at most 25 items, got {len(items)}")
        payload = {"items": [item.to_payload() for item in items]}
        return _to_bulk_result(self._raw(build_bulk(), json=payload))

    def for_series(self, series_id: str, **kwargs: Any) -> DownloadResult:
        """Route a series id to whichever download endpoint actually serves it.

        The split between :meth:`options` and :meth:`series` is determined by
        the series family, not by caller intent, so picking by hand is a 400
        waiting to happen. This consults the catalogue instead.
        """
        catalogue = self._client.series.list()  # type: ignore[attr-defined]
        row = next(
            (r for r in catalogue if r.get("series_id") == series_id or r.get("id") == series_id),
            None,
        )
        if row is None:
            raise ValueError(f"unknown series_id {series_id!r}; see client.series.list()")
        asset_class = str(row.get("asset_class") or "").lower()
        if asset_class in {"options", "crypto", "derivatives"} or series_id.startswith(
            ("deribit:", "lyra:", "aevo:", "delta:")
        ):
            return self.options(series_id=series_id, **kwargs)
        return self.series(series_id=series_id, **kwargs)


class AsyncDownload(AsyncResource):
    """Async twin of :class:`Download`. Same money semantics, same NEVER-retry."""

    confirm_cost: Optional[Any] = None
    _datasets_cache: Optional[Any] = None

    async def trades(
        self,
        *,
        market_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        format: str = "parquet",
        limit: int = 100_000,
    ) -> DownloadResult:
        _maybe_warn_cost(limit, "trades", self.confirm_cost)
        return _to_result(
            await self._raw(
                build_trades(market_id=market_id, start=start, end=end, format=format, limit=limit)
            )
        )

    async def ohlcv(
        self,
        *,
        market_id: str,
        timeframe: str = "1h",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        format: str = "csv",
        limit: int = 1_000_000,
    ) -> DownloadResult:
        _maybe_warn_cost(limit, "ohlcv", self.confirm_cost)
        return _to_result(
            await self._raw(
                build_ohlcv(
                    market_id=market_id,
                    timeframe=timeframe,
                    start=start,
                    end=end,
                    format=format,
                    limit=limit,
                )
            )
        )

    async def orderbook(
        self,
        *,
        market_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        format: str = "parquet",
        limit: int = 100_000,
    ) -> DownloadResult:
        _maybe_warn_cost(limit, "orderbook", self.confirm_cost)
        return _to_result(
            await self._raw(
                build_orderbook(
                    market_id=market_id, start=start, end=end, format=format, limit=limit
                )
            )
        )

    async def top_of_book(
        self,
        *,
        market_id: str,
        start: datetime,
        end: datetime,
        format: str = "csv",
        limit: int = 100_000,
    ) -> DownloadResult:
        return _to_result(
            await self._raw(
                build_top_of_book(
                    market_id=market_id, start=start, end=end, format=format, limit=limit
                )
            )
        )

    async def options(
        self,
        *,
        series_id: str,
        data_type: str = "trades",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        format: str = "csv",
        instrument_prefix: Optional[str] = None,
        limit: int = 500_000,
    ) -> DownloadResult:
        _maybe_warn_cost(limit, data_type, self.confirm_cost)
        return _to_result(
            await self._raw(
                build_options(
                    series_id=series_id,
                    data_type=data_type,
                    start=start,
                    end=end,
                    format=format,
                    instrument_prefix=instrument_prefix,
                    limit=limit,
                )
            )
        )

    async def series(
        self,
        *,
        series_id: str,
        data_type: str = "trades",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        format: str = "csv",
        instrument_prefix: Optional[str] = None,
        limit: int = 500_000,
    ) -> DownloadResult:
        _maybe_warn_cost(limit, data_type, self.confirm_cost)
        return _to_result(
            await self._raw(
                build_series(
                    series_id=series_id,
                    data_type=data_type,
                    start=start,
                    end=end,
                    format=format,
                    instrument_prefix=instrument_prefix,
                    limit=limit,
                )
            )
        )

    async def raw(
        self,
        data_type: str,
        start: datetime,
        end: datetime,
        *,
        format: str = "json",
        shape: str = "raw",
        limit: int = 500_000,
    ) -> DownloadResult:
        return _to_result(
            await self._raw(
                build_raw(
                    data_type=data_type,
                    start=start,
                    end=end,
                    format=format,
                    shape=shape,
                    limit=limit,
                )
            )
        )

    async def raw_datasets(self, *, refresh: bool = False) -> Any:
        if self._datasets_cache is None or refresh:
            body = await self._json(build_raw_datasets())
            self._datasets_cache = body.get("data") if isinstance(body, dict) else body
        return self._datasets_cache

    async def raw_estimate(
        self,
        *,
        data_type: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 500_000,
    ) -> Dict[str, Any]:
        return await self._json(
            build_raw_estimate(data_type=data_type, start=start, end=end, limit=limit)
        )

    async def bulk(self, items: Sequence[BulkItem]) -> BulkResult:
        if not items:
            raise ValueError("bulk() needs at least one item")
        if len(items) > 25:
            raise ValueError(f"bulk() accepts at most 25 items, got {len(items)}")
        payload = {"items": [item.to_payload() for item in items]}
        return _to_bulk_result(await self._raw(build_bulk(), json=payload))

    async def for_series(self, series_id: str, **kwargs: Any) -> DownloadResult:
        catalogue = await self._client.series.list()  # type: ignore[attr-defined]
        row = next(
            (r for r in catalogue if r.get("series_id") == series_id or r.get("id") == series_id),
            None,
        )
        if row is None:
            raise ValueError(f"unknown series_id {series_id!r}; see client.series.list()")
        asset_class = str(row.get("asset_class") or "").lower()
        if asset_class in {"options", "crypto", "derivatives"} or series_id.startswith(
            ("deribit:", "lyra:", "aevo:", "delta:")
        ):
            return await self.options(series_id=series_id, **kwargs)
        return await self.series(series_id=series_id, **kwargs)


for _name in ("trades", "ohlcv", "orderbook", "options", "series", "raw", "bulk"):
    for _cls in (Download, AsyncDownload):
        _method = getattr(_cls, _name, None)
        if _method is not None and _method.__doc__:
            _method.__doc__ += _MONEY_NOTE
