"""Large downloads stream to disk instead of being buffered, and fail safe.

``client.request`` accepted a ``stream`` argument and ignored it, so every download
was read into memory in full: ``DownloadResult.content`` and ``BulkResult.content``
were the whole body. Streaming (``save_to=``) fixes that, and because the bytes were
already paid for, how it fails matters as much as how it succeeds:

* nothing truncated is ever left under the real name, and an existing file is not
  replaced until the new one is complete;
* a body shorter than the server declared is an error, not a file that looks fine;
* a streamed paid request is still never retried;
* the server's filename can never choose the directory.

Everything runs offline against ``httpx.MockTransport``, which lets a test control
the body chunk by chunk and make it fail part-way.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import tracemalloc
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import httpx
import pytest

from supagamma import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncSupaGamma,
    SubscriptionRequiredError,
    SupaGamma,
)
from supagamma._streaming import CHUNK_SIZE, DownloadedFile, safe_filename
from supagamma.resources.download import BulkFile, BulkItem, BulkResult, DownloadResult

KEY = "sg_" + "b" * 32
UTC = timezone.utc
T0 = datetime(2026, 1, 1, tzinfo=UTC)
T1 = datetime(2026, 1, 2, tzinfo=UTC)

MIB = 1 << 20


# --- streams the tests can control -----------------------------------------------------


class Chunks(httpx.SyncByteStream):
    """A body that yields ``chunks`` and then optionally fails."""

    def __init__(self, chunks: Iterator[bytes], error: Optional[Exception] = None) -> None:
        self._chunks = chunks
        self._error = error
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        yield from self._chunks
        if self._error is not None:
            raise self._error

    def close(self) -> None:
        self.closed = True


class AsyncChunks(httpx.AsyncByteStream):
    def __init__(self, chunks: List[bytes], error: Optional[Exception] = None) -> None:
        self._chunks = chunks
        self._error = error
        self.closed = False

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk
        if self._error is not None:
            raise self._error

    async def aclose(self) -> None:
        self.closed = True


def body(
    chunks: List[bytes],
    *,
    declared: Optional[int] = None,
    error: Optional[Exception] = None,
    disposition: Optional[str] = None,
    status: int = 200,
) -> httpx.Response:
    headers: Dict[str, str] = {"content-type": "application/octet-stream", "x-request-id": "req-1"}
    if declared is not None:
        headers["content-length"] = str(declared)
    if disposition:
        headers["content-disposition"] = disposition
    return httpx.Response(status, headers=headers, stream=Chunks(iter(chunks), error))


def sync_client(handler: Callable[[httpx.Request], httpx.Response], **kw: Any) -> SupaGamma:
    return SupaGamma(
        api_key=KEY, http_client=httpx.Client(transport=httpx.MockTransport(handler)), **kw
    )


def async_client(handler: Callable[[httpx.Request], Any], **kw: Any) -> AsyncSupaGamma:
    return AsyncSupaGamma(
        api_key=KEY, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), **kw
    )


def leftovers(directory: Path) -> List[str]:
    return sorted(p.name for p in directory.iterdir())


# --- the point: it does not buffer ---------------------------------------------------------


def test_a_large_download_streams_to_disk_without_holding_it_in_memory(tmp_path):
    total = 48 * MIB

    def handler(_request):
        chunks = (b"x" * MIB for _ in range(48))
        return body(chunks, declared=total)  # type: ignore[arg-type]

    client = sync_client(handler)
    target = tmp_path / "trades.parquet"

    tracemalloc.start()
    try:
        result = client.download.trades(market_id="1254468", limit=100, save_to=target)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert isinstance(result, DownloadedFile)
    assert result.size == total == target.stat().st_size
    assert peak < 16 * MIB, f"peak {peak / MIB:.1f} MiB: the body was held in memory"
    assert result.path == target and os.fspath(result) == str(target)
    assert result.request_id == "req-1" and len(result) == total
    assert leftovers(tmp_path) == ["trades.parquet"]


def test_without_save_to_nothing_changes_and_the_bytes_are_returned():
    client = sync_client(
        lambda _r: httpx.Response(200, content=b"abc", headers={"x-request-id": "r"})
    )
    result = client.download.trades(market_id="1", limit=100)
    assert isinstance(result, DownloadResult) and result.content == b"abc"


def test_the_chunk_size_is_what_keeps_memory_flat():
    assert 64 * 1024 <= CHUNK_SIZE <= 8 * MIB


# --- failing safe ------------------------------------------------------------------------------


def test_a_connection_dropped_mid_body_leaves_no_file_at_all(tmp_path):
    def handler(_request):
        return body([b"a" * MIB, b"b" * MIB], error=httpx.ReadError("connection reset by peer"))

    target = tmp_path / "out.parquet"
    with pytest.raises(APIConnectionError, match="connection reset") as raised:
        sync_client(handler).download.trades(market_id="1", limit=100, save_to=target)
    assert not isinstance(raised.value, APITimeoutError)
    assert raised.value.request_id == "req-1"
    assert leftovers(tmp_path) == []  # no target, and no .part temp file either


def test_a_stall_mid_body_is_a_timeout_error_and_leaves_nothing(tmp_path):
    def handler(_request):
        return body([b"a" * 1024], error=httpx.ReadTimeout("read timed out"))

    with pytest.raises(APITimeoutError):
        sync_client(handler).download.trades(
            market_id="1", limit=100, save_to=tmp_path / "out.parquet"
        )
    assert leftovers(tmp_path) == []


def test_a_failed_download_does_not_replace_an_existing_file(tmp_path):
    target = tmp_path / "out.parquet"
    target.write_bytes(b"the previous, complete download")

    def handler(_request):
        return body([b"half of the new one"], error=httpx.ReadError("reset"))

    with pytest.raises(APIConnectionError):
        sync_client(handler).download.trades(market_id="1", limit=100, save_to=target)
    assert target.read_bytes() == b"the previous, complete download"
    assert leftovers(tmp_path) == ["out.parquet"]


def test_a_successful_download_replaces_the_old_file_atomically(tmp_path):
    target = tmp_path / "out.parquet"
    target.write_bytes(b"old")
    sync_client(lambda _r: body([b"new content"], declared=11)).download.trades(
        market_id="1", limit=100, save_to=target
    )
    assert target.read_bytes() == b"new content"
    assert leftovers(tmp_path) == ["out.parquet"]


def test_a_body_shorter_than_declared_is_an_error_not_a_complete_looking_file(tmp_path):
    def handler(_request):
        return body([b"x" * 600], declared=1000)

    with pytest.raises(APIConnectionError, match="incomplete download: received 600 of 1000"):
        sync_client(handler).download.trades(
            market_id="1", limit=100, save_to=tmp_path / "out.parquet"
        )
    assert leftovers(tmp_path) == []


def test_an_undeclared_length_is_not_policed(tmp_path):
    # Chunked responses have no Content-Length; there is nothing to compare against.
    result = sync_client(lambda _r: body([b"x" * 600])).download.trades(
        market_id="1", limit=100, save_to=tmp_path / "out.parquet"
    )
    assert result.size == 600


def test_an_error_status_raises_its_typed_error_and_creates_no_file(tmp_path):
    stream_holder: Dict[str, Chunks] = {}

    def handler(_request):
        payload = json.dumps(
            {
                "detail": {
                    "code": "subscription_required",
                    "message": "This dataset requires a plan.",
                }
            }
        ).encode()
        response = httpx.Response(
            402, headers={"content-type": "application/json"}, stream=Chunks(iter([payload]))
        )
        stream_holder["stream"] = response.stream  # type: ignore[assignment]
        return response

    with pytest.raises(SubscriptionRequiredError, match="requires a plan"):
        sync_client(handler).download.trades(
            market_id="1", limit=100, save_to=tmp_path / "out.parquet"
        )
    assert leftovers(tmp_path) == []
    assert stream_holder["stream"].closed, "the connection must be released on an error status"


# --- a response that arrives already read (an injected http_client) -------------------


def test_a_pre_read_response_is_not_mistaken_for_a_short_body(tmp_path):
    """``httpx.Response(content=...)`` is fully read before it reaches the writer, so
    ``num_bytes_downloaded`` is 0. So is the response of any caching or test transport
    behind a caller's own ``http_client``. A complete body must not read as incomplete."""
    client = sync_client(
        lambda _r: httpx.Response(200, content=b"z" * 100, headers={"content-length": "100"})
    )
    result = client.download.trades(market_id="1", limit=1, save_to=tmp_path / "o.parquet")
    assert result.size == 100 and leftovers(tmp_path) == ["o.parquet"]


def test_a_pre_read_short_body_is_still_caught(tmp_path):
    client = sync_client(
        lambda _r: httpx.Response(200, content=b"z" * 40, headers={"content-length": "100"})
    )
    with pytest.raises(APIConnectionError, match="received 40 of 100"):
        client.download.trades(market_id="1", limit=1, save_to=tmp_path / "o.parquet")
    assert leftovers(tmp_path) == []


def test_a_pre_read_empty_body_that_declared_bytes_is_still_caught(tmp_path):
    client = sync_client(lambda _r: httpx.Response(200, headers={"content-length": "100"}))
    with pytest.raises(APIConnectionError, match="received 0 of 100"):
        client.download.trades(market_id="1", limit=1, save_to=tmp_path / "o.parquet")
    assert leftovers(tmp_path) == []


def test_a_pre_read_content_encoded_body_is_not_second_guessed(tmp_path):
    # The declared length is of the ENCODED body and the writer sees decoded bytes, so
    # with no wire count to compare there is nothing sound to compare; do not guess.
    encoded = gzip.compress(b"z" * 100)
    assert len(encoded) != 100
    client = sync_client(
        lambda _r: httpx.Response(
            200,
            content=encoded,
            headers={"content-length": str(len(encoded)), "content-encoding": "gzip"},
        )
    )
    result = client.download.trades(market_id="1", limit=1, save_to=tmp_path / "o.parquet")
    assert result.size == 100
    assert (tmp_path / "o.parquet").read_bytes() == b"z" * 100


async def test_an_async_pre_read_response_is_handled_the_same_way(tmp_path):
    aclient = async_client(
        lambda _r: httpx.Response(200, content=b"z" * 100, headers={"content-length": "100"})
    )
    ok = await aclient.download.trades(market_id="1", limit=1, save_to=tmp_path / "ok.parquet")
    assert ok.size == 100

    short = async_client(
        lambda _r: httpx.Response(200, content=b"z" * 40, headers={"content-length": "100"})
    )
    with pytest.raises(APIConnectionError, match="received 40 of 100"):
        await short.download.trades(market_id="1", limit=1, save_to=tmp_path / "short.parquet")
    assert leftovers(tmp_path) == ["ok.parquet"]


# --- money: a streamed paid request is still never replayed ---------------------------


@pytest.mark.parametrize("status", [503, 429, 500])
def test_a_streamed_paid_request_is_never_retried(tmp_path, status):
    seen: List[str] = []

    def handler(request):
        seen.append(request.url.path)
        return httpx.Response(status, json={"detail": "try later"}, headers={"retry-after": "0"})

    client = sync_client(handler, max_retries=5)
    with pytest.raises(APIStatusError) as raised:
        client.download.trades(market_id="1", limit=100, save_to=tmp_path / "out.parquet")
    assert raised.value.status_code == status
    assert seen == ["/v1/download/trades"], "a paid download must be sent exactly once"


def test_a_dropped_connection_before_the_headers_is_not_retried_either(tmp_path):
    seen: List[int] = []

    def handler(_request):
        seen.append(1)
        raise httpx.ConnectError("refused")

    with pytest.raises(APIConnectionError):
        sync_client(handler, max_retries=5).download.trades(
            market_id="1", limit=100, save_to=tmp_path / "out.parquet"
        )
    assert seen == [1]


# --- the server never chooses the directory -------------------------------------------


@pytest.mark.parametrize(
    "header,expected",
    [
        ('attachment; filename="trades_1254468.parquet"', "trades_1254468.parquet"),
        ('attachment; filename="../../escape.parquet"', "escape.parquet"),
        ('attachment; filename="..\\..\\escape.parquet"', "escape.parquet"),
        ('attachment; filename="/etc/cron.d/job"', "job"),
        ('attachment; filename=".."', "supagamma-download"),
        ("attachment", "supagamma-download"),
        (None, "supagamma-download"),
    ],
)
def test_a_directory_target_uses_a_sanitised_server_filename(tmp_path, header, expected):
    target_dir = tmp_path / "out"
    target_dir.mkdir()
    sync_client(lambda _r: body([b"data"], disposition=header)).download.trades(
        market_id="1", limit=100, save_to=target_dir
    )
    assert leftovers(target_dir) == [expected]
    assert leftovers(tmp_path) == ["out"]  # nothing escaped to the parent


def test_a_trailing_separator_means_a_directory_even_if_it_does_not_exist_yet(tmp_path):
    destination = tmp_path / "new_dir"
    result = sync_client(
        lambda _r: body([b"data"], disposition='attachment; filename="t.parquet"')
    ).download.trades(market_id="1", limit=100, save_to=str(destination) + os.sep)
    assert result.path == destination / "t.parquet"


def test_parent_directories_are_created(tmp_path):
    target = tmp_path / "a" / "b" / "c.parquet"
    sync_client(lambda _r: body([b"data"])).download.trades(
        market_id="1", limit=100, save_to=target
    )
    assert target.read_bytes() == b"data"


def test_safe_filename_never_returns_a_path():
    for hostile in ["../x", "a/b/c", "..", ".", "", "C:\\Windows\\system32\\x", "x\\..\\y"]:
        name = safe_filename(hostile)
        assert name and "/" not in name and "\\" not in name and name not in (".", "..")


def test_the_in_memory_result_also_refuses_a_hostile_filename(tmp_path):
    target_dir = tmp_path / "out"
    target_dir.mkdir()
    DownloadResult(content=b"x", content_type="", filename="../../up.parquet").save_to(target_dir)
    assert leftovers(target_dir) == ["up.parquet"]
    assert leftovers(tmp_path) == ["out"]


# --- every paid route can stream ------------------------------------------------------


ROUTES = [
    ("trades", dict(market_id="1"), "/v1/download/trades"),
    ("ohlcv", dict(market_id="1"), "/v1/download/ohlcv"),
    ("orderbook", dict(market_id="1"), "/v1/download/orderbook"),
    ("top_of_book", dict(market_id="1", start=T0, end=T1), "/v1/download/top_of_book"),
    ("options", dict(series_id="deribit:btc-options"), "/v1/download/options"),
    ("series", dict(series_id="polymarket:ohlcv-1h"), "/v1/download/series"),
]


@pytest.mark.parametrize("method,kwargs,path", ROUTES, ids=[r[0] for r in ROUTES])
def test_every_per_route_download_streams(tmp_path, method, kwargs, path):
    seen: List[httpx.Request] = []

    def handler(request):
        seen.append(request)
        return body([b"payload"], declared=7)

    result = getattr(sync_client(handler).download, method)(
        limit=10, save_to=tmp_path / "f.bin", **kwargs
    )
    assert isinstance(result, DownloadedFile) and result.path.read_bytes() == b"payload"
    assert [r.url.path for r in seen] == [path]


def test_raw_streams(tmp_path):
    seen: List[httpx.Request] = []

    def handler(request):
        seen.append(request)
        return body([b"frames"])

    result = sync_client(handler).download.raw(
        "polymarket_l2_deltas", T0, T1, format="parquet", save_to=tmp_path / "raw.parquet"
    )
    assert result.path.read_bytes() == b"frames"
    assert seen[0].url.params["data_type"] == "polymarket_l2_deltas"


def test_for_series_routes_and_streams(tmp_path):
    seen: List[str] = []

    def handler(request):
        seen.append(request.url.path)
        if request.url.path == "/v1/series":
            return httpx.Response(
                200, json=[{"series_id": "deribit:btc-options", "asset_class": "options"}]
            )
        return body([b"derivs"])

    result = sync_client(handler).download.for_series(
        "deribit:btc-options", limit=10, save_to=tmp_path / "d.bin"
    )
    assert result.path.read_bytes() == b"derivs"
    assert seen == ["/v1/series", "/v1/download/options"]


# --- bulk -----------------------------------------------------------------------------


def make_zip(manifest: Optional[Dict[str, Any]] = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("00_trades.csv", "a,b\n1,2\n")
        if manifest is not None:
            archive.writestr("manifest.json", json.dumps(manifest))
    return buffer.getvalue()


ITEMS = [BulkItem(kind="trades", market_id="1", format="csv")]


def test_bulk_streams_to_disk_and_reads_its_manifest_from_the_saved_archive(tmp_path):
    manifest = {"total_cost": 12.5, "items": [{"file": "00_trades.csv", "status": "charged"}]}
    payload = make_zip(manifest)
    seen: List[Any] = []

    def handler(request):
        seen.append(json.loads(request.content))
        return body([payload], declared=len(payload), disposition='attachment; filename="b.zip"')

    result = sync_client(handler).download.bulk(ITEMS, save_to=tmp_path / "bundle.zip")

    assert isinstance(result, BulkFile)
    assert result.path.read_bytes() == payload
    assert result.manifest == manifest and result.total_cost == 12.5
    assert result.items == manifest["items"]
    assert seen == [
        {"items": [{"kind": "trades", "format": "csv", "limit": 100000, "market_id": "1"}]}
    ]
    extracted = result.extract_to(tmp_path / "unzipped")
    assert [p.name for p in extracted] == ["00_trades.csv", "manifest.json"]
    assert (tmp_path / "unzipped" / "00_trades.csv").read_text() == "a,b\n1,2\n"


def test_bulk_without_save_to_is_unchanged():
    payload = make_zip({"total_cost": 3.0, "items": []})
    result = sync_client(lambda _r: httpx.Response(200, content=payload)).download.bulk(ITEMS)
    assert isinstance(result, BulkResult) and result.total_cost == 3.0


def test_a_bulk_response_that_is_not_a_zip_is_still_kept_when_it_was_paid_for(tmp_path):
    result = sync_client(lambda _r: body([b"definitely not a zip"])).download.bulk(
        ITEMS, save_to=tmp_path / "bundle.zip"
    )
    assert result.path.read_bytes() == b"definitely not a zip" and result.manifest == {}


@pytest.mark.parametrize("items,message", [([], "at least one"), (ITEMS * 26, "at most 25")])
def test_bulk_validates_before_sending_anything(tmp_path, items, message):
    sent: List[int] = []
    client = sync_client(lambda _r: sent.append(1) or body([b"x"]))
    with pytest.raises(ValueError, match=message):
        client.download.bulk(items, save_to=tmp_path / "b.zip")
    assert sent == []


# --- the transport primitive ----------------------------------------------------------


def test_the_stream_context_manager_closes_the_response_even_if_the_body_is_never_read():
    holder: Dict[str, Chunks] = {}

    def handler(_request):
        response = body([b"unread"])
        holder["stream"] = response.stream  # type: ignore[assignment]
        return response

    client = sync_client(handler)
    with client.stream("GET", "/v1/anything") as response:
        assert response.status_code == 200
    assert holder["stream"].closed


def test_request_with_stream_true_returns_an_unread_response():
    client = sync_client(lambda _r: body([b"a", b"b"]))
    response = client.request("GET", "/v1/x", stream=True)
    try:
        assert not response.is_stream_consumed
        assert b"".join(response.iter_bytes()) == b"ab"
    finally:
        response.close()


# --- the async twin -------------------------------------------------------------------


def abody(
    chunks: List[bytes],
    *,
    declared: Optional[int] = None,
    error: Optional[Exception] = None,
    status: int = 200,
) -> httpx.Response:
    headers: Dict[str, str] = {"x-request-id": "req-a"}
    if declared is not None:
        headers["content-length"] = str(declared)
    return httpx.Response(status, headers=headers, stream=AsyncChunks(chunks, error))


async def test_async_streams_to_disk(tmp_path):
    chunks = [b"a" * MIB, b"b" * MIB, b"c" * 10]
    total = sum(len(c) for c in chunks)
    client = async_client(lambda _r: abody(chunks, declared=total))
    result = await client.download.trades(market_id="1", limit=100, save_to=tmp_path / "t.bin")
    assert isinstance(result, DownloadedFile) and result.size == total
    assert (tmp_path / "t.bin").read_bytes() == b"".join(chunks)
    assert leftovers(tmp_path) == ["t.bin"]
    await client.aclose()


async def test_async_without_save_to_returns_bytes():
    client = async_client(lambda _r: httpx.Response(200, content=b"abc"))
    result = await client.download.trades(market_id="1", limit=100)
    assert isinstance(result, DownloadResult) and result.content == b"abc"


async def test_async_mid_body_failure_leaves_nothing_and_keeps_the_old_file(tmp_path):
    target = tmp_path / "t.bin"
    target.write_bytes(b"old")
    client = async_client(lambda _r: abody([b"part"], error=httpx.ReadError("reset")))
    with pytest.raises(APIConnectionError):
        await client.download.trades(market_id="1", limit=100, save_to=target)
    assert target.read_bytes() == b"old" and leftovers(tmp_path) == ["t.bin"]


async def test_async_short_body_is_an_error(tmp_path):
    client = async_client(lambda _r: abody([b"x" * 5], declared=50))
    with pytest.raises(APIConnectionError, match="incomplete download"):
        await client.download.trades(market_id="1", limit=100, save_to=tmp_path / "t.bin")
    assert leftovers(tmp_path) == []


async def test_async_error_status_is_typed_and_never_retried(tmp_path):
    seen: List[int] = []

    def handler(_request):
        seen.append(1)
        return httpx.Response(
            402, json={"detail": {"code": "subscription_required", "message": "needs a plan"}}
        )

    client = async_client(handler, max_retries=5)
    with pytest.raises(SubscriptionRequiredError):
        await client.download.trades(market_id="1", limit=100, save_to=tmp_path / "t.bin")
    assert seen == [1] and leftovers(tmp_path) == []


async def test_async_bulk_streams(tmp_path):
    payload = make_zip({"total_cost": 1.0, "items": []})
    client = async_client(lambda _r: abody([payload], declared=len(payload)))
    result = await client.download.bulk(ITEMS, save_to=tmp_path / "b.zip")
    assert isinstance(result, BulkFile) and result.total_cost == 1.0


async def test_async_stream_context_manager_closes_the_response():
    stream = AsyncChunks([b"unread"])
    client = async_client(lambda _r: httpx.Response(200, stream=stream))
    async with client.stream("GET", "/v1/anything") as response:
        assert response.status_code == 200
    assert stream.closed
