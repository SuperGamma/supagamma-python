"""``client.replay``: rebuild a market's L2 book from ``/v1/download/orderbook``.

The route has no cursor, only ``start``/``end``/``limit``, and answers with the
earliest ``limit`` rows of the window. These tests run the SDK against a small
fake of exactly that behaviour, so paging, ordering, the ``at()`` boundaries and
the interval grid are checked against how the server actually slices data.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import respx

from supagamma import (
    AsyncSupaGamma,
    BookState,
    FairUseCapError,
    MissingDependencyError,
    NotFoundError,
    QuotaExceededError,
    RateLimitError,
    Replay,
    ResponseShapeError,
    ServerError,
    SupaGamma,
    SupaGammaError,
)
from supagamma._client import SAFE_READ
from supagamma.replay import REPLAY_RETRY, _Grid, parse_row

KEY = "sg_" + "f" * 32
BASE = "https://api.test"
UTC = timezone.utc
T0 = datetime(2026, 11, 1, 12, 0, 0, tzinfo=UTC)
PATH = "/v1/download/orderbook"


def at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


def row(
    seconds: float,
    token: str = "tY",
    outcome: str = "Yes",
    bid: float = 0.40,
    ask: float = 0.60,
    as_string: bool = True,
) -> Dict[str, Any]:
    bids = [{"price": str(bid), "size": "100"}, {"price": str(bid - 0.01), "size": "50"}]
    asks = [{"price": str(ask), "size": "80"}, {"price": str(ask + 0.01), "size": "20"}]
    return {
        "timestamp": at(seconds).isoformat(),
        "market_id": "0xcond",
        "token_id": token,
        "outcome": outcome,
        "bids": json.dumps(bids) if as_string else bids,
        "asks": json.dumps(asks) if as_string else asks,
        "best_bid": bid,
        "best_ask": ask,
    }


def pair(seconds: float, bid: float = 0.40) -> List[Dict[str, Any]]:
    """One capture tick of a binary market: both outcome tokens, same timestamp."""
    return [
        row(seconds, "tY", "Yes", bid=bid, ask=bid + 0.2),
        row(seconds, "tN", "No", bid=round(0.8 - bid - 0.2, 2), ask=round(0.8 - bid, 2)),
    ]


class FakeOrderbookRoute:
    """The server's slicing: rows in file order, ``start <= ts <= end`` compared as
    the stored ISO strings, the first ``limit`` kept, a 404 when none match."""

    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self.rows = rows
        self.calls: List[Dict[str, str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        query = {k: v[0] for k, v in parse_qs(urlparse(str(request.url)).query).items()}
        self.calls.append(query)
        start = datetime.fromisoformat(query["start"]).astimezone(UTC).isoformat()
        end = datetime.fromisoformat(query["end"]).astimezone(UTC).isoformat()
        limit = int(query["limit"])
        hits = [r for r in self.rows if start <= r["timestamp"] <= end][:limit]
        if not hits:
            return httpx.Response(404, json={"detail": "No orderbook snapshots in this range"})
        return httpx.Response(200, json=hits)


@pytest.fixture(scope="module")
def client() -> Iterator[SupaGamma]:
    c = SupaGamma(api_key=KEY, base_url=BASE, max_retries=3, max_retry_wait_seconds=5)
    yield c
    c.close()


def serve(rows: List[Dict[str, Any]]) -> Any:
    fake = FakeOrderbookRoute(rows)
    router = respx.mock(base_url=BASE, assert_all_called=False)
    router.get(PATH).mock(side_effect=fake)
    return router, fake


# --- parsing -----------------------------------------------------------------------


def test_levels_are_sorted_best_first_and_zero_sizes_dropped():
    raw = row(0)
    raw["bids"] = json.dumps(
        [
            {"price": "0.38", "size": "5"},
            {"price": "0.40", "size": "10"},
            {"price": "0.39", "size": "0"},
        ]
    )
    raw["asks"] = [["0.62", "1"], ["0.60", "2"]]
    book = parse_row(raw)
    assert book.bids == ((0.40, 10.0), (0.38, 5.0))
    assert book.asks == ((0.60, 2.0), (0.62, 1.0))
    assert book.best_bid == 0.40 and book.best_ask == 0.60
    assert book.mid == pytest.approx(0.50)
    assert book.spread == pytest.approx(0.20)
    assert book.timestamp == T0 and book.snapshot_time == T0
    assert book.timestamp.tzinfo is not None


def test_an_empty_side_has_no_best_price():
    raw = row(0)
    raw["asks"] = "[]"
    book = parse_row(raw)
    assert book.asks == () and book.best_ask is None and book.mid is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.pop("token_id"),
        lambda r: r.update(timestamp="yesterday"),
        lambda r: r.update(bids="{not json"),
        lambda r: r.update(bids=json.dumps([{"price": "x", "size": "1"}])),
        lambda r: r.update(asks=json.dumps({"price": "1"})),
    ],
)
def test_an_unreadable_row_is_a_shape_error_not_a_guess(mutate):
    raw = row(0)
    mutate(raw)
    with pytest.raises(ResponseShapeError):
        parse_row(raw)


# --- validation: before any request ---------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"start": datetime(2026, 11, 1), "end": at(60)}, ValueError),
        ({"start": "2026-11-01T12:00:00", "end": at(60)}, ValueError),
        ({"start": "not-a-time-sg_secret", "end": at(60)}, ValueError),
        ({"start": at(60), "end": at(60)}, ValueError),
        ({"start": at(120), "end": at(60)}, ValueError),
        ({"start": 1730462400, "end": at(60)}, TypeError),
        ({"page_size": 0}, ValueError),
        ({"page_size": 1_000_001}, ValueError),
        ({"page_size": True}, TypeError),
        ({"interval": 0}, ValueError),
        ({"interval": -5}, ValueError),
        ({"interval": float("nan")}, ValueError),
        ({"outcome": "Yes", "token_id": "tY"}, ValueError),
        ({"outcome": "  "}, ValueError),
        ({"market_id": "1/../admin"}, ValueError),
        ({"market_id": ""}, ValueError),
    ],
)
def test_bad_input_fails_before_anything_is_sent(client, kwargs, error):
    args: Dict[str, Any] = {"market_id": "1254468", "start": T0, "end": at(3600)}
    args.update(kwargs)
    with respx.mock(base_url=BASE, assert_all_called=False) as router:
        route = router.get(PATH)
        with pytest.raises(error) as raised:
            client.replay(args.pop("market_id"), args.pop("start"), args.pop("end"), **args)
        assert route.call_count == 0
    assert "sg_secret" not in str(raised.value)


def test_iso_strings_with_z_and_offsets_are_accepted_and_sent_as_utc(client):
    replay = client.replay("1254468", "2026-11-01T12:00:00Z", "2026-11-01T15:00:00+02:00")
    assert replay.start == T0
    assert replay.end == at(3600)


def test_building_a_replay_sends_nothing(client):
    with respx.mock(base_url=BASE, assert_all_called=False) as router:
        route = router.get(PATH)
        replay = client.replay("1254468", T0, at(60))
        assert isinstance(replay, Replay)
        assert route.call_count == 0


def test_a_large_page_size_warns_like_a_large_download(client):
    from supagamma.resources.download import CostWarning

    router, _ = serve(pair(0))
    with router, pytest.warns(CostWarning):
        list(client.replay("1254468", T0, at(60), page_size=100_000))


def test_the_confirm_cost_hook_is_read_live_and_sees_every_page(client, monkeypatch):
    import warnings

    from supagamma.resources import download

    # Price every page above the warning line, so the hook is consulted on each.
    monkeypatch.setattr(download, "COST_WARN_THRESHOLD_USD", 0.0)
    seen: List[float] = []

    def hook(usd: float) -> bool:
        seen.append(usd)
        return len(seen) < 2  # allow the first page, veto the second

    replay = client.replay("1254468", T0, at(600), page_size=3)
    rows = [r for s in range(0, 300, 30) for r in pair(s)]
    router, fake = serve(rows)
    monkeypatch.setattr(client.download, "confirm_cost", hook, raising=False)
    with router, warnings.catch_warnings():  # set AFTER the replay was built
        warnings.simplefilter("ignore", download.CostWarning)
        with pytest.raises(RuntimeError, match="confirm_cost"):
            list(replay)
    assert len(seen) == 2
    assert len(fake.calls) == 1


def test_a_market_id_error_does_not_echo_the_id(client):
    with pytest.raises(ValueError) as raised:
        client.replay("sg_secret/x", T0, at(60))
    assert "sg_secret" not in str(raised.value)


# --- iteration and paging -------------------------------------------------------------


def test_snapshots_come_oldest_first_with_the_request_the_route_expects(client):
    rows = pair(0) + pair(30) + pair(60)
    router, fake = serve(rows)
    with router:
        books = list(client.replay("1254468", T0, at(120)))
    assert [(b.timestamp, b.token_id) for b in books] == [
        (at(0), "tN"),
        (at(0), "tY"),
        (at(30), "tN"),
        (at(30), "tY"),
        (at(60), "tN"),
        (at(60), "tY"),
    ]
    assert len(fake.calls) == 1
    sent = fake.calls[0]
    assert sent["market_id"] == "1254468"
    assert sent["format"] == "json"
    assert sent["limit"] == "2500"
    assert sent["start"] == "2026-11-01T12:00:00+00:00"
    assert sent["end"] == "2026-11-01T12:02:00+00:00"


@pytest.mark.parametrize("page_size", [3, 4, 5, 7])
def test_paging_neither_skips_nor_repeats_a_snapshot(client, page_size):
    # A page boundary can fall between the two tokens of one tick (odd sizes).
    # The page must hold more than one tick's rows, or it cannot advance.
    rows = [r for s in range(0, 300, 30) for r in pair(s)]
    router, fake = serve(rows)
    with router:
        books = list(client.replay("1254468", T0, at(600), page_size=page_size))
    got = [(b.timestamp, b.token_id) for b in books]
    assert len(got) == len(set(got)) == len(rows)
    assert got == sorted(got)
    assert len(fake.calls) > 1
    # Each page starts at the newest timestamp the previous one held.
    starts = [c["start"] for c in fake.calls]
    assert starts == sorted(starts)


def test_a_row_written_out_of_order_after_the_cut_is_still_fetched(client):
    # Storage order, not time order: the 60s tick was written before the 30s one.
    # With page_size=4 the first page ends on a 60s row, so the next page starts
    # at 60s... unless the cursor follows the LAST row in storage order.
    rows = pair(0) + pair(60) + pair(30) + pair(90)
    router, _ = serve(rows)
    with router:
        books = list(client.replay("1254468", T0, at(120), page_size=5))
    stamps = sorted({b.timestamp for b in books})
    assert stamps == [at(0), at(30), at(60), at(90)]
    got = [(b.timestamp, b.token_id) for b in books]
    assert got == sorted(got) and len(got) == len(set(got)) == 8


def test_duplicate_price_levels_add_up():
    raw = row(0)
    raw["bids"] = json.dumps([{"price": "0.45", "size": "100"}, {"price": "0.450", "size": "50"}])
    assert parse_row(raw).bids == ((0.45, 150.0),)


def test_rows_out_of_order_within_a_page_are_sorted(client):
    rows = pair(60) + pair(0) + pair(30)
    router, _ = serve(rows)
    with router:
        books = list(client.replay("1254468", T0, at(120)))
    assert [b.timestamp for b in books] == sorted(b.timestamp for b in books)


def test_a_full_page_stuck_on_one_timestamp_stops_instead_of_looping(client):
    rows = [row(0, f"t{i}", f"O{i}") for i in range(4)]
    router, fake = serve(rows)
    with router, pytest.raises(SupaGammaError, match="Raise page_size"):
        list(client.replay("1254468", T0, at(60), page_size=2))
    assert len(fake.calls) == 1


def test_an_empty_window_is_an_empty_replay_not_an_error(client):
    router, fake = serve(pair(0))
    with router:
        assert list(client.replay("1254468", at(100), at(200))) == []
    assert len(fake.calls) == 1


def test_a_market_with_no_orderbook_at_all_still_raises(client):
    with respx.mock(base_url=BASE) as router:
        router.get(PATH).mock(
            return_value=httpx.Response(404, json={"detail": "No orderbook data for this market"})
        )
        with pytest.raises(NotFoundError) as raised:
            list(client.replay("1254468", T0, at(60)))
    assert type(raised.value) is NotFoundError


def test_the_outcome_filter_is_case_insensitive(client):
    router, _ = serve(pair(0) + pair(30))
    with router:
        books = list(client.replay("1254468", T0, at(60), outcome="yes"))
    assert [b.token_id for b in books] == ["tY", "tY"]


def test_token_id_filter(client):
    router, _ = serve(pair(0) + pair(30))
    with router:
        books = list(client.replay("1254468", T0, at(60), token_id="tN"))
    assert [b.outcome for b in books] == ["No", "No"]


def test_an_outcome_the_market_does_not_have_names_the_ones_it_does(client):
    router, _ = serve(pair(0))
    with router, pytest.raises(ValueError, match=r"\['No', 'Yes'\]") as raised:
        list(client.replay("1254468", T0, at(60), outcome="Maybe"))
    assert "Maybe" not in str(raised.value)


def test_a_non_list_body_is_a_shape_error(client):
    with respx.mock(base_url=BASE) as router:
        router.get(PATH).mock(return_value=httpx.Response(200, json={"data": []}))
        with pytest.raises(ResponseShapeError):
            list(client.replay("1254468", T0, at(60)))


# --- retries ------------------------------------------------------------------------


def test_the_replay_policy_retries_only_the_rate_limiter():
    assert REPLAY_RETRY.on_status == (429,)
    assert not REPLAY_RETRY.on_connection_error
    assert not REPLAY_RETRY.on_server_error_once


def test_a_rate_limited_page_is_waited_out_and_retried(client):
    with respx.mock(base_url=BASE) as router:
        route = router.get(PATH).mock(
            side_effect=[
                httpx.Response(
                    429, json={"detail": "Rate limit exceeded"}, headers={"retry-after": "0"}
                ),
                httpx.Response(200, json=pair(0)),
            ]
        )
        books = list(client.replay("1254468", T0, at(60)))
    assert route.call_count == 2
    assert len(books) == 2


def test_a_long_rate_limit_wait_is_surfaced_not_slept(client):
    with respx.mock(base_url=BASE) as router:
        route = router.get(PATH).mock(
            return_value=httpx.Response(
                429, json={"detail": "Rate limit exceeded"}, headers={"retry-after": "3000"}
            )
        )
        with pytest.raises(RateLimitError):
            list(client.replay("1254468", T0, at(60)))
    assert route.call_count == 1


def test_a_too_many_concurrent_reads_429_is_transient_and_retried(client):
    busy = {
        "detail": {
            "code": "too_many_concurrent_reads",
            "message": "You already have 2 data reads in progress. You have not been charged.",
        }
    }
    with respx.mock(base_url=BASE) as router:
        route = router.get(PATH).mock(
            side_effect=[
                httpx.Response(429, json=busy, headers={"retry-after": "0"}),
                httpx.Response(200, json=pair(0)),
            ]
        )
        books = list(client.replay("1254468", T0, at(60)))
    assert route.call_count == 2 and len(books) == 2


def test_a_billing_quota_is_never_retried(client):
    with respx.mock(base_url=BASE) as router:
        route = router.get(PATH).mock(
            return_value=httpx.Response(
                429, json={"detail": {"code": "fair_use_cap", "message": "cap reached"}}
            )
        )
        with pytest.raises(QuotaExceededError):
            list(client.replay("1254468", T0, at(60)))
    assert route.call_count == 1


def test_a_server_error_is_not_retried_because_a_page_may_have_been_metered(client):
    with respx.mock(base_url=BASE) as router:
        route = router.get(PATH).mock(return_value=httpx.Response(500, text="boom"))
        with pytest.raises(ServerError):
            list(client.replay("1254468", T0, at(60)))
    assert route.call_count == 1


def test_a_quota_429_is_not_retried_by_any_read_policy(client):
    # The client-wide fix behind the test above: SAFE_READ lists 429 too, and a
    # billing cap used to fall through to the generic backoff and be retried.
    with respx.mock(base_url=BASE) as router:
        route = router.get("/v1/markets/1").mock(
            return_value=httpx.Response(
                429, json={"detail": {"code": "fair_use_cap", "message": "cap reached"}}
            )
        )
        with pytest.raises(FairUseCapError):
            client.request("GET", "/v1/markets/1", policy=SAFE_READ)
    assert route.call_count == 1


# --- at() ---------------------------------------------------------------------------


def _replay(client: SupaGamma, **kw: Any) -> Replay:
    return client.replay("1254468", T0, at(3600), outcome="Yes", **kw)


def test_at_exactly_on_a_snapshot_returns_that_snapshot(client):
    router, _ = serve(pair(0, bid=0.30) + pair(60, bid=0.40) + pair(120, bid=0.50))
    with router:
        book = _replay(client).at(at(60))
    assert book is not None
    assert book.timestamp == at(60) and book.best_bid == 0.40


def test_at_between_snapshots_returns_the_earlier_one(client):
    router, _ = serve(pair(0, bid=0.30) + pair(60, bid=0.40) + pair(120, bid=0.50))
    with router:
        book = _replay(client).at(at(119.999))
    assert book is not None and book.timestamp == at(60)


def test_at_before_the_first_snapshot_is_none(client):
    router, fake = serve(pair(600))
    with router:
        assert _replay(client).at(at(300)) is None
    # Looked back to start, and no further.
    assert all(c["start"] >= "2026-11-01T12:00:00+00:00" for c in fake.calls)


def test_at_outside_the_window_is_an_error_before_any_request(client):
    with respx.mock(base_url=BASE, assert_all_called=False) as router:
        route = router.get(PATH)
        with pytest.raises(ValueError, match="inside the replay"):
            _replay(client).at(at(-1))
        with pytest.raises(ValueError, match="inside the replay"):
            _replay(client).at(at(3601))
        assert route.call_count == 0


def test_at_needs_one_outcome(client):
    with respx.mock(base_url=BASE, assert_all_called=False) as router:
        route = router.get(PATH)
        with pytest.raises(ValueError, match="books_at"):
            client.replay("1254468", T0, at(3600)).at(at(60))
        assert route.call_count == 0


def test_at_widens_its_lookback_without_re_reading_what_it_checked(client):
    replay = client.replay("1254468", T0, T0 + timedelta(hours=10), outcome="Yes")
    router, fake = serve(pair(60, bid=0.33))
    with router:
        book = replay.at(T0 + timedelta(hours=9), lookback=timedelta(hours=1))
    assert book is not None and book.timestamp == at(60)
    # 1h, then 2h, then 4h, then the rest down to start: four windows that never overlap.
    windows = [
        (datetime.fromisoformat(c["start"]), datetime.fromisoformat(c["end"])) for c in fake.calls
    ]
    assert len(windows) == 4
    for (lo, _), (_, hi) in zip(windows, windows[1:]):
        assert hi < lo
    assert windows[-1][0] == T0


def test_at_with_an_outcome_the_market_lacks_stops_after_one_page(client):
    replay = client.replay("1254468", T0, T0 + timedelta(hours=10), outcome="Yess")
    router, fake = serve(pair(60) + [r for s in range(3600, 36000, 600) for r in pair(s)])
    with router, pytest.raises(ValueError, match="Outcomes served"):
        replay.at(T0 + timedelta(hours=9))
    assert len(fake.calls) == 1


def test_books_at_returns_every_outcome(client):
    router, _ = serve(pair(0, bid=0.30) + pair(60, bid=0.40))
    with router:
        books = client.replay("1254468", T0, at(3600)).books_at(at(90))
    assert sorted(books) == ["tN", "tY"]
    assert all(b.timestamp == at(60) for b in books.values())


def test_books_at_before_anything_is_empty(client):
    router, _ = serve(pair(600))
    with router:
        assert client.replay("1254468", T0, at(3600)).books_at(at(10)) == {}


# --- the interval grid -----------------------------------------------------------------


def test_the_grid_carries_each_book_forward_and_says_how_old_it_is(client):
    router, _ = serve(pair(0, bid=0.30) + pair(25, bid=0.40))
    with router:
        books = list(client.replay("1254468", T0, at(40), outcome="Yes", interval=10))
    assert [(b.timestamp, b.snapshot_time, b.best_bid) for b in books] == [
        (at(0), at(0), 0.30),
        (at(10), at(0), 0.30),
        (at(20), at(0), 0.30),
        (at(30), at(25), 0.40),
        (at(40), at(25), 0.40),
    ]


def test_the_grid_starts_at_the_first_snapshot_and_counts_ticks_from_start():
    start, end = T0, at(100)
    grid = _Grid(start, end, timedelta(seconds=30))
    first = parse_row(row(45))
    assert list(grid.feed(first)) == []
    out = list(grid.finish())
    assert [b.timestamp for b in out] == [at(60), at(90)]
    assert all(b.snapshot_time == at(45) for b in out)


def test_a_snapshot_exactly_on_a_tick_is_part_of_that_tick():
    grid = _Grid(T0, at(20), timedelta(seconds=10))
    out = list(grid.feed(parse_row(row(0, bid=0.30))))
    out += list(grid.feed(parse_row(row(10, bid=0.40))))
    out += list(grid.finish())
    assert [(b.timestamp, b.best_bid) for b in out] == [
        (at(0), 0.30),
        (at(10), 0.40),
        (at(20), 0.40),
    ]


def test_an_empty_grid_emits_nothing():
    assert list(_Grid(T0, at(100), timedelta(seconds=1)).finish()) == []


# --- pandas -----------------------------------------------------------------------------


def test_to_pandas_is_long_format_one_row_per_level(client):
    pytest.importorskip("pandas")
    router, _ = serve(pair(0))
    with router:
        frame = client.replay("1254468", T0, at(60), outcome="Yes").to_pandas()
    assert list(frame.columns) == [
        "timestamp",
        "snapshot_time",
        "token_id",
        "outcome",
        "side",
        "level",
        "price",
        "size",
    ]
    assert len(frame) == 4
    best_bid = frame[(frame.side == "bid") & (frame.level == 0)].iloc[0]
    assert best_bid.price == pytest.approx(0.40)


def test_to_pandas_without_pandas_fails_before_any_request(client, monkeypatch):
    monkeypatch.setitem(sys.modules, "pandas", None)
    with respx.mock(base_url=BASE, assert_all_called=False) as router:
        route = router.get(PATH)
        with pytest.raises(MissingDependencyError):
            client.replay("1254468", T0, at(60)).to_pandas()
        assert route.call_count == 0


# --- async ------------------------------------------------------------------------------


@pytest.fixture
async def aclient():
    c = AsyncSupaGamma(api_key=KEY, base_url=BASE, max_retries=3, max_retry_wait_seconds=5)
    yield c
    await c.aclose()


async def test_async_iteration_pages_in_order(aclient):
    rows = [r for s in range(0, 150, 30) for r in pair(s)]
    router, fake = serve(rows)
    with router:
        replay = aclient.replay("1254468", T0, at(600), page_size=3)
        books: List[BookState] = [b async for b in replay]
    got = [(b.timestamp, b.token_id) for b in books]
    assert got == sorted(got) and len(got) == len(set(got)) == len(rows)
    assert len(fake.calls) > 1


async def test_async_at_and_grid(aclient):
    router, _ = serve(pair(0, bid=0.30) + pair(60, bid=0.40))
    with router:
        replay = aclient.replay("1254468", T0, at(120), outcome="Yes", interval=60)
        book: Optional[BookState] = await replay.at(at(61))
        grid = [b async for b in replay]
        empty = await aclient.replay("1254468", at(200), at(300), outcome="Yes").at(at(250))
    assert book is not None and book.best_bid == 0.40
    assert [(b.timestamp, b.snapshot_time) for b in grid] == [
        (at(0), at(0)),
        (at(60), at(60)),
        (at(120), at(60)),
    ]
    assert empty is None


async def test_async_to_pandas(aclient):
    pytest.importorskip("pandas")
    router, _ = serve(pair(0))
    with router:
        frame = await aclient.replay("1254468", T0, at(60)).to_pandas()
    assert set(frame.token_id) == {"tY", "tN"}


# --- review round 2 ----------------------------------------------------------------------


def test_many_small_pages_still_reach_the_cost_warning_once(client):
    from supagamma.resources.download import CostWarning

    # Default pages are each just under the warning line; the running total is not.
    rows = [r for s in range(0, 2_600 * 5, 5) for r in pair(s)]
    router, fake = serve(rows)
    with router, pytest.warns(CostWarning) as record:
        books = list(client.replay("1254468", T0, T0 + timedelta(days=1)))
    assert len(books) == len(rows)
    assert len(fake.calls) == 3
    assert len([w for w in record if issubclass(w.category, CostWarning)]) == 1


def test_a_cut_on_the_cursor_advances_to_the_oldest_later_timestamp():
    from supagamma.replay import _Pager

    pager = _Pager("1", at(0), at(600), page_size=4)
    # Storage order: a late row at the cursor tick ends the page.
    out = pager.accept([row(0), row(30), row(90), row(0, "tN", "No")])
    assert pager.cursor == at(30)
    assert [b.timestamp for b in out] == [at(0), at(0)]


@pytest.mark.parametrize(
    "text, micros",
    [
        ("2026-11-01T12:00:00.5Z", 500_000),
        ("2026-11-01T12:00:00.1234567+00:00", 123_456),
        ("2026-11-01T12:00:00.12Z", 120_000),
    ],
)
def test_any_number_of_fractional_digits_parses_on_every_python(client, text, micros):
    replay = client.replay("1254468", text, at(60))
    assert replay.start == T0.replace(microsecond=micros)


@pytest.mark.parametrize("bad", [0, -1, 1e300, float("inf")])
def test_lookback_errors_name_lookback_and_never_overflow(client, bad):
    replay = _replay(client)
    with pytest.raises(ValueError, match="lookback"):
        replay.at(at(60), lookback=bad)


def test_a_huge_interval_is_a_value_error(client):
    with pytest.raises(ValueError, match="interval"):
        client.replay("1254468", T0, at(60), interval=1e300)


def test_a_capture_gap_for_the_wanted_outcome_widens_instead_of_failing(client):
    # Only "No" was captured in the last few minutes; "Yes" last appears earlier.
    rows = pair(0, bid=0.30) + [row(s, "tN", "No") for s in range(1200, 1800, 60)]
    router, fake = serve(rows)
    with router:
        book = _replay(client).at(at(1790))
    assert book is not None and book.timestamp == at(0) and book.token_id == "tY"
    assert len(fake.calls) > 1
