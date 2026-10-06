"""``client.markets.list``: the series and end-date filters the API gained.

``GET /v1/markets`` takes ``series_id`` (e.g. every BTC 15-minute market),
``ending_after``/``ending_before`` and ``sort_by=end_date``. The SDK could send
none of them, and it always sent ``sort_by="top"``, which on a series query would
override the server's own default (``end_date``, the order the series index
serves) with a trade-count sort.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterator

import httpx
import pytest
import respx

from supagamma import AsyncSupaGamma, SupaGamma
from supagamma.resources import markets
from supagamma.resources.markets import MAX_SERIES_ID_LENGTH, build_list

KEY = "sg_" + "e" * 32
BASE = "https://api.test"
UTC = timezone.utc
AFTER = datetime(2026, 9, 1, tzinfo=UTC)
BEFORE = datetime(2026, 9, 8, tzinfo=UTC)


def params_of(spec) -> dict:
    return spec[2]


# --- the builder ---------------------------------------------------------------------------


def test_an_unset_sort_is_left_to_the_server():
    # The server's default is `top`, or `end_date` with a series_id. Sending `top`
    # regardless overrode the second.
    assert "sort_by" not in params_of(build_list())
    assert "sort_by" not in params_of(build_list(series_id="polymarket:btc-15m"))
    assert params_of(build_list(sort_by="top"))["sort_by"] == "top"


def test_end_date_is_a_sort_the_sdk_accepts_now():
    assert params_of(build_list(sort_by="end_date"))["sort_by"] == "end_date"
    with pytest.raises(ValueError, match="sort_by must be one of"):
        build_list(sort_by="ending")


def test_series_and_window_are_sent_as_the_api_names_them():
    params = params_of(
        build_list(series_id="polymarket:btc-15m", ending_after=AFTER, ending_before=BEFORE)
    )
    assert params["series_id"] == "polymarket:btc-15m"
    assert params["ending_after"] == "2026-09-01T00:00:00+00:00"
    assert params["ending_before"] == "2026-09-08T00:00:00+00:00"


def test_window_bounds_are_always_sent_in_utc_with_an_offset():
    naive = datetime(2026, 9, 1, 12, 0)
    aware = datetime(2026, 9, 8, 5, 30, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    params = params_of(build_list(ending_after=naive, ending_before=aware))
    assert params["ending_after"] == "2026-09-01T12:00:00+00:00"
    assert params["ending_before"] == "2026-09-08T00:00:00+00:00"


def test_one_sided_windows_are_fine():
    assert "ending_before" not in params_of(build_list(ending_after=AFTER))
    assert "ending_after" not in params_of(build_list(ending_before=BEFORE))


def test_an_inverted_window_fails_before_any_request():
    # The server answers it with an empty 200: "no market ends in this window".
    with pytest.raises(ValueError, match="ending_after must be on or before ending_before"):
        build_list(ending_after=BEFORE, ending_before=AFTER)


def test_an_equal_window_is_allowed():
    params = params_of(build_list(ending_after=AFTER, ending_before=AFTER))
    assert params["ending_after"] == params["ending_before"]


@pytest.mark.parametrize("bad", ["", "   ", "x" * (MAX_SERIES_ID_LENGTH + 1)])
def test_a_series_id_the_server_would_reject_fails_here(bad):
    with pytest.raises(ValueError, match="series_id"):
        build_list(series_id=bad)


def test_unset_filters_are_not_sent():
    assert set(params_of(build_list())) == {"limit", "offset"}


# --- through the client --------------------------------------------------------------------


@pytest.fixture(scope="module")
def client() -> Iterator[SupaGamma]:
    with SupaGamma(api_key=KEY, base_url=BASE, max_retries=0) as shared:
        yield shared


def test_list_sends_the_new_filters(client):
    with respx.mock(base_url=BASE, assert_all_called=False) as router:
        route = router.get("/v1/markets").mock(return_value=httpx.Response(200, json=[]))
        assert client.markets.list(series_id="polymarket:btc-15m", ending_after=AFTER) == []
        sent = route.calls.last.request.url.params
    assert sent["series_id"] == "polymarket:btc-15m"
    assert sent["ending_after"] == "2026-09-01T00:00:00+00:00"
    assert "sort_by" not in sent


def test_auto_paginate_keeps_the_filters_on_every_page(client):
    first = [{"id": str(i)} for i in range(2)]
    second = [{"id": "2"}]
    with respx.mock(base_url=BASE, assert_all_called=False) as router:
        route = router.get("/v1/markets").mock(
            side_effect=[httpx.Response(200, json=first), httpx.Response(200, json=second)]
        )
        ids = [
            m["id"]
            for m in client.markets.auto_paginate(
                series_id="polymarket:btc-15m", ending_before=BEFORE, sort_by="end_date", limit=2
            )
        ]
        pages = [call.request.url.params for call in route.calls]
    assert ids == ["0", "1", "2"]
    assert [p["offset"] for p in pages] == ["0", "2"]
    for page in pages:
        assert page["series_id"] == "polymarket:btc-15m"
        assert page["ending_before"] == "2026-09-08T00:00:00+00:00"
        assert page["sort_by"] == "end_date"


def test_an_unknown_series_is_a_typed_not_found(client):
    from supagamma import NotFoundError

    with respx.mock(base_url=BASE, assert_all_called=False) as router:
        router.get("/v1/markets").mock(
            return_value=httpx.Response(404, json={"detail": "Series not found"})
        )
        with pytest.raises(NotFoundError):
            client.markets.list(series_id="polymarket:nope")


async def test_the_async_twin_takes_the_same_filters():
    async with AsyncSupaGamma(api_key=KEY, base_url=BASE, max_retries=0) as aclient:
        with respx.mock(base_url=BASE, assert_all_called=False) as router:
            route = router.get("/v1/markets").mock(return_value=httpx.Response(200, json=[]))
            await aclient.markets.list(series_id="polymarket:btc-15m", ending_before=BEFORE)
            rows = [m async for m in aclient.markets.auto_paginate(ending_after=AFTER)]
            sent = [call.request.url.params for call in route.calls]
    assert rows == []
    assert sent[0]["series_id"] == "polymarket:btc-15m" and "ending_before" in sent[0]
    assert sent[1]["ending_after"] == "2026-09-01T00:00:00+00:00"


def test_the_sync_and_async_signatures_match():
    import inspect

    for name in ("list", "auto_paginate"):
        sync = inspect.signature(getattr(markets.Markets, name)).parameters
        asyn = inspect.signature(getattr(markets.AsyncMarkets, name)).parameters
        assert list(sync) == list(asyn), name
        for new in ("series_id", "ending_after", "ending_before"):
            assert new in sync, (name, new)
