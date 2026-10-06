"""``client.exports``: the documented ``/v1/exports`` workflow.

Create, list, inspect, poll, and obtain a signed URL, with the guarantees a paid
operation needs:

* ``create`` is metered work, so it is never retried automatically, whatever the
  client's ``max_retries``;
* it always carries an idempotency key, in the body (the server ignores a header),
  so a *manual* replay returns the same job instead of building and billing twice;
* the key survives a failed call, because that is when the replay is wanted;
* the signed URL is already authorised, so fetching it sends no SupaGamma
  credentials to another host.

Reads (``list``, ``get``, ``url``) may be retried like any other read.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterator, List

import httpx
import pytest
import respx

from supagamma import (
    APIConnectionError,
    APITimeoutError,
    AsyncSupaGamma,
    ConflictError,
    ExportFailedError,
    ExportTimeoutError,
    GoneError,
    NotFoundError,
    PermissionDeniedError,
    ResponseShapeError,
    ServiceUnavailableError,
    SupaGamma,
)
from supagamma._client import NEVER
from supagamma.resources import exports
from supagamma.resources.exports import (
    URL_TTL_MAX,
    URL_TTL_MIN,
    build_create,
    build_get,
    build_list,
    build_url,
)

KEY = "sg_" + "c" * 32
BASE = "https://api.test"
SIGNED = "https://files.example/exports/u/j-1/trades_1254468.parquet?X-Amz-Signature=abc123"
UTC = timezone.utc


def make_job(status: str = "queued", **over: Any) -> Dict[str, Any]:
    job = {
        "id": "j-1",
        "status": status,
        "spec": {"kind": "trades", "market_id": "1254468", "format": "parquet"},
        "file_name": "trades_1254468",
        "record_count": None,
        "size_bytes": None,
        "cost": None,
        "disposition": None,
        "error": None,
        "created_at": "2026-10-04T10:00:00+00:00",
        "completed_at": None,
        "expires_at": None,
    }
    job.update(over)
    return job


def mock_api() -> respx.MockRouter:
    """The API host, mocked. Several tests assert that a request was *not* made, or
    was made exactly once, so a registered route is allowed to stay uncalled."""
    return respx.mock(base_url=BASE, assert_all_called=False)


def mock_hosts() -> respx.MockRouter:
    """The API host and the storage host the signed URL points at, on ONE router
    (respx routers do not reliably nest), addressed by absolute URL."""
    return respx.mock(assert_all_called=False)


@pytest.fixture(scope="module")
def client() -> Iterator[SupaGamma]:
    # Module-scoped: building an httpx client loads the CA bundle, which is the
    # slowest thing in this file. The client holds no per-test state.
    with SupaGamma(api_key=KEY, base_url=BASE, max_retries=3) as shared:
        yield shared


# --- policies: which calls may be replayed ---------------------------------------------


def test_creating_an_export_is_never_retried_by_the_client():
    assert build_create()[3] is NEVER


@pytest.mark.parametrize(
    "spec", [build_list(), build_get("j-1"), build_url("j-1")], ids=["list", "get", "url"]
)
def test_the_reads_may_be_retried(spec):
    assert spec[3] is not NEVER and spec[3].enabled


def test_builders_target_the_documented_routes():
    assert build_create()[:2] == ("POST", "/v1/exports")
    assert build_list(limit=7)[:3] == ("GET", "/v1/exports", {"limit": 7})
    assert build_get("j-1")[:2] == ("GET", "/v1/exports/j-1")
    assert build_url("j-1", ttl=90)[:3] == ("GET", "/v1/exports/j-1/url", {"ttl": 90})


# --- validation: nothing invalid leaves the process ----------------------------------------


@pytest.mark.parametrize(
    "kwargs,message",
    [
        (dict(kind="options", market_id="1"), "kind must be one of"),
        (dict(kind="trades"), "needs market_id"),
        (dict(kind="trades", market_id="1", series_id="s"), "series_id does not apply"),
        (dict(kind="raw"), "needs series_id"),
        (
            dict(kind="raw", series_id="polymarket:l2-delta-tape", market_id="1"),
            "market_id does not apply",
        ),
        (dict(kind="trades", market_id="1", format="xlsx"), "format must be one of"),
        (dict(kind="trades", market_id="1", idempotency_key="k" * 129), "at most 128"),
        (
            dict(
                kind="trades",
                market_id="1",
                start=datetime(2026, 1, 2, tzinfo=UTC),
                end=datetime(2026, 1, 1, tzinfo=UTC),
            ),
            "end must be after start",
        ),
        (
            dict(
                kind="trades",
                market_id="1",
                start=datetime(2026, 1, 1, tzinfo=UTC),
                end=datetime(2026, 1, 1, tzinfo=UTC),
            ),
            "end must be after start",
        ),
    ],
)
def test_invalid_requests_fail_before_any_request(client, kwargs, message):
    with mock_api() as router:
        route = router.post("/v1/exports")
        with pytest.raises(ValueError, match=message):
            client.exports.create(**kwargs)
        assert route.call_count == 0


def test_a_raw_export_names_the_series_id_not_the_catalog_data_type(client):
    with pytest.raises(ValueError, match="polymarket:l2-delta-tape"):
        client.exports.create(kind="raw")


@pytest.mark.parametrize("bad", ["", "  ", "a/b", "../x"])
def test_a_job_id_cannot_change_which_endpoint_is_called(client, bad):
    with pytest.raises(ValueError, match="job_id"):
        client.exports.get(bad)
    with pytest.raises(ValueError, match="job_id"):
        client.exports.url(bad)


@pytest.mark.parametrize("limit", [0, 201, -1])
def test_list_limit_is_bounded_client_side(client, limit):
    with pytest.raises(ValueError, match="limit"):
        client.exports.list(limit=limit)


@pytest.mark.parametrize("ttl", [URL_TTL_MIN - 1, URL_TTL_MAX + 1, 0])
def test_url_ttl_is_bounded_client_side(client, ttl):
    with pytest.raises(ValueError, match="ttl"):
        client.exports.url("j-1", ttl=ttl)


def test_mixed_naive_and_aware_bounds_are_normalised_not_a_server_error(client):
    # The server compares start and end directly; Python cannot compare a naive
    # datetime with an aware one. The SDK sends both as UTC.
    with mock_api() as router:
        route = router.post("/v1/exports").mock(return_value=httpx.Response(202, json=make_job()))
        client.exports.create(
            kind="trades",
            market_id="1",
            start=datetime(2026, 1, 1),  # naive: read as UTC
            end=datetime(2026, 1, 2, 5, 30, tzinfo=timezone(timedelta(hours=5, minutes=30))),
        )
        import json

        body = json.loads(route.calls.last.request.content)
    assert body["start"] == "2026-01-01T00:00:00+00:00"
    assert body["end"] == "2026-01-02T00:00:00+00:00"


# --- create: the idempotency contract -----------------------------------------------------


def test_create_sends_the_key_in_the_body_and_echoes_it_on_the_job(client):
    with mock_api() as router:
        route = router.post("/v1/exports").mock(return_value=httpx.Response(202, json=make_job()))
        job = client.exports.create(
            kind="trades",
            market_id="1254468",
            start=datetime(2026, 1, 1, tzinfo=UTC),
            end=datetime(2026, 1, 8, tzinfo=UTC),
            format="csv",
        )
        request = route.calls.last.request
    import json

    body = json.loads(request.content)
    assert body == {
        "kind": "trades",
        "market_id": "1254468",
        "format": "csv",
        "start": "2026-01-01T00:00:00+00:00",
        "end": "2026-01-08T00:00:00+00:00",
        "idempotency_key": job["idempotency_key"],
    }
    assert uuid.UUID(job["idempotency_key"]).hex == job["idempotency_key"]  # a uuid4 hex
    assert "idempotency-key" not in {
        k.lower() for k in request.headers
    }  # a body field, not a header
    assert job["id"] == "j-1" and job["status"] == "queued"


def test_a_supplied_key_is_sent_verbatim(client):
    with mock_api() as router:
        route = router.post("/v1/exports").mock(return_value=httpx.Response(202, json=make_job()))
        job = client.exports.create(kind="trades", market_id="1", idempotency_key="my-job-2026-10")
        import json

        assert json.loads(route.calls.last.request.content)["idempotency_key"] == "my-job-2026-10"
    assert job["idempotency_key"] == "my-job-2026-10"


def test_a_key_the_server_already_returns_is_not_overwritten(client):
    with mock_api() as router:
        router.post("/v1/exports").mock(
            return_value=httpx.Response(202, json=make_job(idempotency_key="server-side"))
        )
        job = client.exports.create(kind="trades", market_id="1", idempotency_key="mine")
    assert job["idempotency_key"] == "server-side"


@pytest.mark.parametrize("status", [503, 429, 500, 502])
def test_create_is_sent_once_even_when_the_client_is_allowed_to_retry(client, status):
    with mock_api() as router:
        route = router.post("/v1/exports").mock(
            return_value=httpx.Response(
                status, json={"detail": "try later"}, headers={"retry-after": "0"}
            )
        )
        with pytest.raises(Exception) as raised:
            client.exports.create(kind="trades", market_id="1")
        assert route.call_count == 1, "a metered POST must never be replayed automatically"
    assert raised.value.status_code == status  # type: ignore[attr-defined]


def test_a_dropped_connection_is_not_retried_either(client):
    with mock_api() as router:
        route = router.post("/v1/exports").mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(APIConnectionError):
            client.exports.create(kind="trades", market_id="1")
        assert route.call_count == 1


def test_the_key_is_on_the_exception_so_a_failed_create_can_be_replayed(client):
    """The case the key exists for. A timeout leaves you not knowing whether the job
    was queued; without the key you would have to guess."""
    seen_keys: List[str] = []
    job = make_job()

    def server(request: httpx.Request) -> httpx.Response:
        import json

        key = json.loads(request.content)["idempotency_key"]
        seen_keys.append(key)
        if len(seen_keys) == 1:
            raise httpx.ReadTimeout("read timed out")  # the job WAS queued; we never saw it
        return httpx.Response(202, json=job)  # the server answers a repeat key with the same job

    with mock_api() as router:
        router.post("/v1/exports").mock(side_effect=server)
        with pytest.raises(APITimeoutError) as raised:
            client.exports.create(kind="trades", market_id="1254468")
        key = raised.value.idempotency_key
        assert key == seen_keys[0] and len(key) == 32

        replayed = client.exports.create(kind="trades", market_id="1254468", idempotency_key=key)

    assert seen_keys == [key, key], "the replay must carry the SAME key"
    assert replayed["id"] == "j-1" and replayed["idempotency_key"] == key


def test_an_error_status_carries_the_key_too(client):
    with mock_api() as router:
        router.post("/v1/exports").mock(
            return_value=httpx.Response(503, json={"detail": "queue down"})
        )
        with pytest.raises(ServiceUnavailableError) as raised:
            client.exports.create(kind="trades", market_id="1", idempotency_key="k-1")
    assert raised.value.idempotency_key == "k-1"


def test_errors_from_other_calls_have_no_key():
    from supagamma import SupaGammaError

    assert SupaGammaError("x").idempotency_key is None


# --- list / get / url ----------------------------------------------------------------------


def test_list_unwraps_the_envelope(client):
    envelope = {"data": [make_job("succeeded"), make_job("queued", id="j-2")], "meta": {"count": 2}}
    with mock_api() as router:
        route = router.get("/v1/exports").mock(return_value=httpx.Response(200, json=envelope))
        jobs = client.exports.list(limit=25)
        assert route.calls.last.request.url.params["limit"] == "25"
    assert [j["id"] for j in jobs] == ["j-1", "j-2"]


def test_list_without_an_envelope_is_a_typed_error(client):
    with mock_api() as router:
        router.get("/v1/exports").mock(return_value=httpx.Response(200, json=[make_job()]))
        with pytest.raises(ResponseShapeError):
            client.exports.list()


def test_get_returns_the_job_as_sent(client):
    with mock_api() as router:
        router.get("/v1/exports/j-1").mock(
            return_value=httpx.Response(200, json=make_job("running"))
        )
        assert client.exports.get("j-1")["status"] == "running"


def test_get_says_not_found_for_someone_elses_job(client):
    with mock_api() as router:
        router.get("/v1/exports/other").mock(
            return_value=httpx.Response(404, json={"detail": "Export job not found"})
        )
        with pytest.raises(NotFoundError):
            client.exports.get("other")


def test_reads_are_retried_on_a_transient_503(client):
    with mock_api() as router:
        route = router.get("/v1/exports/j-1").mock(
            side_effect=[
                httpx.Response(503, json={"detail": "busy"}, headers={"retry-after": "0"}),
                httpx.Response(200, json=make_job("running")),
            ]
        )
        assert client.exports.get("j-1")["status"] == "running"
        assert route.call_count == 2


def test_url_returns_the_signed_url_and_the_job(client):
    payload = {"data": {"download_url": SIGNED, "expires_in": 120, **make_job("succeeded")}}
    with mock_api() as router:
        route = router.get("/v1/exports/j-1/url").mock(
            return_value=httpx.Response(200, json=payload)
        )
        info = client.exports.url("j-1", ttl=120)
        assert route.calls.last.request.url.params["ttl"] == "120"
    assert info["download_url"] == SIGNED and info["expires_in"] == 120 and info["id"] == "j-1"


def test_url_without_a_download_url_is_a_typed_error(client):
    with mock_api() as router:
        router.get("/v1/exports/j-1/url").mock(
            return_value=httpx.Response(200, json={"data": make_job("succeeded")})
        )
        with pytest.raises(ResponseShapeError, match="download_url"):
            client.exports.url("j-1")


@pytest.mark.parametrize(
    "status,error",
    [(409, ConflictError), (410, GoneError), (404, NotFoundError)],
    ids=["not-ready", "expired", "not-yours"],
)
def test_url_failures_are_typed(client, status, error):
    with mock_api() as router:
        router.get("/v1/exports/j-1/url").mock(
            return_value=httpx.Response(status, json={"detail": "nope"})
        )
        with pytest.raises(error):
            client.exports.url("j-1")


# --- wait ---------------------------------------------------------------------------------------


class Clock:
    """A fake clock: sleeping advances time, and nothing really waits."""

    def __init__(self) -> None:
        self.t = 1000.0
        self.sleeps: List[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds

    async def asleep(self, seconds: float) -> None:
        self.sleep(seconds)


@pytest.fixture
def clock(monkeypatch) -> Clock:
    fake = Clock()
    monkeypatch.setattr(exports, "_now", fake.now)
    monkeypatch.setattr(exports, "_sleep", fake.sleep)
    monkeypatch.setattr(exports, "_asleep", fake.asleep)
    return fake


def polls(*jobs: Dict[str, Any]) -> List[httpx.Response]:
    return [httpx.Response(200, json=j) for j in jobs]


def test_wait_polls_until_the_job_succeeds_and_backs_off(client, clock):
    seen: List[str] = []
    done = make_job("succeeded", record_count=10, size_bytes=1234)
    with mock_api() as router:
        route = router.get("/v1/exports/j-1").mock(
            side_effect=polls(make_job("queued"), make_job("running"), make_job("running"), done)
        )
        job = client.exports.wait("j-1", on_poll=lambda j: seen.append(j["status"]))
        assert route.call_count == 4
    assert job["status"] == "succeeded" and job["record_count"] == 10
    assert seen == ["queued", "running", "running", "succeeded"]
    assert clock.sleeps == [5.0, 7.5, 11.25]  # backs off; no sleep after success


def test_wait_caps_the_poll_interval(client, clock):
    with mock_api() as router:
        router.get("/v1/exports/j-1").mock(
            side_effect=polls(*([make_job("running")] * 12), make_job("succeeded"))
        )
        client.exports.wait("j-1", poll_interval=10.0, timeout=10_000)
    assert max(clock.sleeps) == 30.0


def test_a_failed_job_raises_with_the_servers_explanation(client, clock):
    failed = make_job(
        "failed",
        error="No data found in the requested range.",
        completed_at="2026-10-04T10:05:00+00:00",
    )
    with mock_api() as router:
        router.get("/v1/exports/j-1").mock(side_effect=polls(make_job("running"), failed))
        with pytest.raises(ExportFailedError, match="No data found") as raised:
            client.exports.wait("j-1")
    assert raised.value.job["status"] == "failed"


def test_an_expired_job_raises_and_says_to_resubmit(client, clock):
    with mock_api() as router:
        router.get("/v1/exports/j-1").mock(side_effect=polls(make_job("expired")))
        with pytest.raises(ExportFailedError, match="submit the export again"):
            client.exports.wait("j-1")


def test_wait_times_out_without_cancelling_anything(client, clock):
    with mock_api() as router:
        router.get("/v1/exports/j-1").mock(
            return_value=httpx.Response(200, json=make_job("running"))
        )
        with pytest.raises(ExportTimeoutError, match="keeps building") as raised:
            client.exports.wait("j-1", timeout=40.0, poll_interval=15.0)
    assert raised.value.job["status"] == "running"
    assert sum(clock.sleeps) <= 40.0  # never sleeps past the deadline


def test_an_unknown_status_is_treated_as_still_running(client, clock):
    with mock_api() as router:
        router.get("/v1/exports/j-1").mock(
            side_effect=polls(make_job("provisioning"), make_job("succeeded"))
        )
        assert client.exports.wait("j-1")["status"] == "succeeded"


# --- download -------------------------------------------------------------------------------------


SIGNED_ROUTE = SIGNED.split("?")[0]  # respx matches the path; the query is checked by hand


def url_payload(**over: Any) -> Dict[str, Any]:
    return {"data": {"download_url": SIGNED, "expires_in": 300, **make_job("succeeded"), **over}}


def test_download_streams_the_signed_url_without_sending_credentials(client, tmp_path):
    artifact = b"PAR1" + b"x" * 5000
    with mock_hosts() as router:
        mint = router.get(f"{BASE}/v1/exports/j-1/url").mock(
            return_value=httpx.Response(200, json=url_payload())
        )
        fetch = router.get(SIGNED_ROUTE).mock(
            return_value=httpx.Response(
                200, content=artifact, headers={"content-length": str(len(artifact))}
            )
        )
        result = client.exports.download("j-1", tmp_path)

    # the artifact's own name was used (the target is a directory), and it was saved whole
    assert result.path == tmp_path / "trades_1254468.parquet"
    assert result.path.read_bytes() == artifact and result.size == len(artifact)
    assert [p.name for p in tmp_path.iterdir()] == ["trades_1254468.parquet"]
    # the call that mints the URL authenticates; the fetch of the URL itself must not
    minted = {k.lower() for k in mint.calls.last.request.headers}
    assert "x-api-key" in minted
    sent = {k.lower() for k in fetch.calls.last.request.headers}
    assert "x-api-key" not in sent and "authorization" not in sent
    assert fetch.calls.last.request.url.params["X-Amz-Signature"] == "abc123"


def test_download_to_a_file_path_and_a_failed_fetch_leaves_nothing(client, tmp_path):
    expired = "<Error><Code>AccessDenied</Code><Message>Request has expired</Message></Error>"
    with mock_hosts() as router:
        router.get(f"{BASE}/v1/exports/j-1/url").mock(
            return_value=httpx.Response(200, json=url_payload())
        )
        router.get(SIGNED_ROUTE).mock(return_value=httpx.Response(403, text=expired))
        with pytest.raises(PermissionDeniedError, match="expired"):
            client.exports.download("j-1", tmp_path / "out.parquet")
    assert list(tmp_path.iterdir()) == []


def test_download_is_all_or_nothing_like_every_other_streamed_write(client, tmp_path):
    target = tmp_path / "out.parquet"
    target.write_bytes(b"previous")
    with mock_hosts() as router:
        router.get(f"{BASE}/v1/exports/j-1/url").mock(
            return_value=httpx.Response(200, json=url_payload())
        )
        router.get(SIGNED_ROUTE).mock(side_effect=httpx.ReadError("connection reset"))
        with pytest.raises(APIConnectionError):
            client.exports.download("j-1", target)
    assert target.read_bytes() == b"previous"
    assert [p.name for p in tmp_path.iterdir()] == ["out.parquet"]


@pytest.mark.parametrize(
    "failure,expected",
    [
        (httpx.ConnectError("refused"), APIConnectionError),
        (httpx.ConnectTimeout("slow"), APITimeoutError),
        (httpx.ReadTimeout("stalled"), APITimeoutError),
    ],
    ids=["refused", "connect-timeout", "read-timeout"],
)
def test_a_signed_url_fetch_failure_is_an_sdk_error_not_a_bare_httpx_one(
    client, tmp_path, failure, expected
):
    # The signed URL is fetched outside the client, so nothing maps its errors
    # unless download() does. Callers catch SupaGammaError; they must not need httpx.
    with mock_hosts() as router:
        router.get(f"{BASE}/v1/exports/j-1/url").mock(
            return_value=httpx.Response(200, json=url_payload())
        )
        router.get(SIGNED_ROUTE).mock(side_effect=failure)
        with pytest.raises(expected) as raised:
            client.exports.download("j-1", tmp_path / "out.parquet")
    assert type(raised.value) is expected
    assert list(tmp_path.iterdir()) == []


def test_a_short_body_is_an_error_not_a_file_that_looks_fine(client, tmp_path):
    with mock_hosts() as router:
        router.get(f"{BASE}/v1/exports/j-1/url").mock(
            return_value=httpx.Response(200, json=url_payload())
        )
        router.get(SIGNED_ROUTE).mock(
            return_value=httpx.Response(200, content=b"x" * 10, headers={"content-length": "999"})
        )
        with pytest.raises(APIConnectionError, match="incomplete download"):
            client.exports.download("j-1", tmp_path / "out.parquet")
    assert list(tmp_path.iterdir()) == []


def test_the_signed_url_is_not_fetched_when_minting_it_fails(client, tmp_path):
    with mock_hosts() as router:
        router.get(f"{BASE}/v1/exports/j-1/url").mock(
            return_value=httpx.Response(409, json={"detail": "Export is not ready"})
        )
        fetch = router.get(SIGNED_ROUTE)
        with pytest.raises(ConflictError):
            client.exports.download("j-1", tmp_path / "out.parquet")
        assert fetch.call_count == 0


def _two_host_handler(artifact: bytes, seen: List[httpx.Request]):
    """One handler for the API and the storage host, recording every request."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.host == "api.test":
            return httpx.Response(200, json=url_payload())
        return httpx.Response(200, content=artifact, headers={"content-length": str(len(artifact))})

    return handler


def test_download_goes_through_the_clients_own_http_client(tmp_path):
    """A caller who injected an ``http_client`` did it for a reason (a corporate proxy, a
    private CA bundle, mTLS). The signed URL fetch must honour it, or `download` fails
    behind exactly the networks institutional users have, while every other call works."""
    seen: List[httpx.Request] = []
    http = httpx.Client(transport=httpx.MockTransport(_two_host_handler(b"x" * 64, seen)))
    try:
        with SupaGamma(api_key=KEY, base_url=BASE, http_client=http) as client:
            result = client.exports.download("j-1", tmp_path / "o.parquet")
    finally:
        http.close()
    assert [r.url.host for r in seen] == ["api.test", "files.example"]
    assert result.path.read_bytes() == b"x" * 64
    # ... and still without a single SupaGamma credential on the second request
    assert "x-api-key" in {k.lower() for k in seen[0].headers}
    assert not {"x-api-key", "authorization"} & {k.lower() for k in seen[1].headers}


# --- async ----------------------------------------------------------------------------


@pytest.fixture
async def aclient():
    client = AsyncSupaGamma(api_key=KEY, base_url=BASE, max_retries=3)
    yield client
    await client.aclose()


async def test_async_create_list_get_url(aclient):
    import json

    with mock_api() as router:
        post = router.post("/v1/exports").mock(return_value=httpx.Response(202, json=make_job()))
        router.get("/v1/exports").mock(
            return_value=httpx.Response(200, json={"data": [make_job()], "meta": {"count": 1}})
        )
        router.get("/v1/exports/j-1").mock(
            return_value=httpx.Response(200, json=make_job("running"))
        )
        router.get("/v1/exports/j-1/url").mock(
            return_value=httpx.Response(
                200, json={"data": {"download_url": SIGNED, "expires_in": 300}}
            )
        )
        job = await aclient.exports.create(kind="trades", market_id="1")
        assert (
            json.loads(post.calls.last.request.content)["idempotency_key"] == job["idempotency_key"]
        )
        assert [j["id"] for j in await aclient.exports.list()] == ["j-1"]
        assert (await aclient.exports.get("j-1"))["status"] == "running"
        assert (await aclient.exports.url("j-1"))["download_url"] == SIGNED


async def test_async_create_is_never_retried_and_carries_the_key(aclient):
    with mock_api() as router:
        route = router.post("/v1/exports").mock(
            return_value=httpx.Response(503, json={"detail": "busy"}, headers={"retry-after": "0"})
        )
        with pytest.raises(ServiceUnavailableError) as raised:
            await aclient.exports.create(kind="trades", market_id="1", idempotency_key="k-async")
        assert route.call_count == 1
    assert raised.value.idempotency_key == "k-async"


async def test_async_wait(aclient, clock):
    with mock_api() as router:
        router.get("/v1/exports/j-1").mock(
            side_effect=polls(make_job("queued"), make_job("succeeded"))
        )
        assert (await aclient.exports.wait("j-1"))["status"] == "succeeded"
    assert clock.sleeps == [5.0]


async def test_async_wait_failure_and_timeout(aclient, clock):
    with mock_api() as router:
        router.get("/v1/exports/j-1").mock(side_effect=polls(make_job("failed", error="boom")))
        with pytest.raises(ExportFailedError, match="boom"):
            await aclient.exports.wait("j-1")
    with mock_api() as router:
        router.get("/v1/exports/j-2").mock(
            return_value=httpx.Response(200, json=make_job("running", id="j-2"))
        )
        with pytest.raises(ExportTimeoutError):
            await aclient.exports.wait("j-2", timeout=12.0)


async def test_async_download_sends_no_credentials_to_the_signed_url(aclient, tmp_path):
    artifact = b"x" * 3000
    with mock_hosts() as router:
        mint = router.get(f"{BASE}/v1/exports/j-1/url").mock(
            return_value=httpx.Response(200, json=url_payload())
        )
        fetch = router.get(SIGNED_ROUTE).mock(
            return_value=httpx.Response(200, content=artifact, headers={"content-length": "3000"})
        )
        result = await aclient.exports.download("j-1", tmp_path / "a.parquet")
    assert result.path.read_bytes() == artifact and result.size == 3000
    assert "x-api-key" in {k.lower() for k in mint.calls.last.request.headers}
    sent = {k.lower() for k in fetch.calls.last.request.headers}
    assert "x-api-key" not in sent and "authorization" not in sent


async def test_async_download_maps_transport_errors_and_leaves_nothing(aclient, tmp_path):
    with mock_hosts() as router:
        router.get(f"{BASE}/v1/exports/j-1/url").mock(
            return_value=httpx.Response(200, json=url_payload())
        )
        router.get(SIGNED_ROUTE).mock(side_effect=httpx.ReadTimeout("stalled"))
        with pytest.raises(APITimeoutError):
            await aclient.exports.download("j-1", tmp_path / "a.parquet")
    assert list(tmp_path.iterdir()) == []


async def test_async_download_signed_url_failure_is_typed(aclient, tmp_path):
    with mock_hosts() as router:
        router.get(f"{BASE}/v1/exports/j-1/url").mock(
            return_value=httpx.Response(200, json=url_payload())
        )
        router.get(SIGNED_ROUTE).mock(return_value=httpx.Response(403, text="Request has expired"))
        with pytest.raises(PermissionDeniedError, match="expired"):
            await aclient.exports.download("j-1", tmp_path / "a.parquet")
    assert list(tmp_path.iterdir()) == []


async def test_async_download_goes_through_the_clients_own_http_client(tmp_path):
    seen: List[httpx.Request] = []
    http = httpx.AsyncClient(transport=httpx.MockTransport(_two_host_handler(b"y" * 32, seen)))
    try:
        async with AsyncSupaGamma(api_key=KEY, base_url=BASE, http_client=http) as client:
            result = await client.exports.download("j-1", tmp_path / "o.parquet")
    finally:
        await http.aclose()
    assert [r.url.host for r in seen] == ["api.test", "files.example"]
    assert result.path.read_bytes() == b"y" * 32
    assert "x-api-key" in {k.lower() for k in seen[0].headers}
    assert not {"x-api-key", "authorization"} & {k.lower() for k in seen[1].headers}
