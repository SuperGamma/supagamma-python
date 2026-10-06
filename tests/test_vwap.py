"""The paid VWAP helper: what it pays for, what it refuses to pay for, and the maths.

``vwap_price_fn`` prices a market from the trade tape, and every price is a paid
download. Three things went wrong with it, and each is a test below:

1. **It paid first and checked afterwards.** pyarrow is an optional extra; a plain
   install downloaded the file, found it could not read it, and returned None,
   once per market. Exceptions were swallowed, and ``resolved_markets(max_markets=N)``
   counted *yields*, so a run in which every price failed had no bound at all and
   walked the whole catalogue, paying for each.
2. **It averaged YES and NO together.** A trade file holds the fills of both
   outcome tokens and the two price each other, so the pooled mean sits near 0.5
   whatever the market thinks (a May-2026 archive spot-check pooled to 0.501).
3. **It ignored which token a row was.** The only discriminator is the token id in
   ``market_id``; there is no ``outcome`` column.

Most of this runs without pyarrow, which is how CI runs it: the Parquet layer is
exercised through a fake ``pq``, and the tests that need the real library skip when
it is absent.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx
import pytest
import respx

from supagamma import (
    APITimeoutError,
    AuthenticationError,
    FairUseCapError,
    MissingDependencyError,
    NoDataInRangeError,
    PaymentRequiredError,
    QuotaExceededError,
    RateLimitError,
    ResponseShapeError,
    ServerError,
    SupaGamma,
    SupaGammaConfigError,
)
from supagamma.backtest import data
from supagamma.backtest.data import (
    realized_outcome,
    resolved_markets,
    vwap_price_fn,
    yes_vwap,
)

YES, NO = "7001", "7002"
UTC = timezone.utc


# --- the maths, with no pyarrow and no network ------------------------------------


def test_complementary_yes_and_no_fills_price_the_market_at_the_yes_price_not_half():
    # A market the book thinks is 80% YES. The same economics shows up as YES fills
    # near 0.80 and NO fills near 0.20, so a naive mean of the file is 0.5.
    fills = [(YES, 0.80, 10.0), (NO, 0.20, 10.0)] * 50
    assert sum(p for _t, p, _s in fills) / len(fills) == pytest.approx(0.5)  # the bug
    assert yes_vwap(fills, yes_token=YES, no_token=NO) == pytest.approx(0.80)


def test_a_no_fill_counts_as_one_minus_its_price():
    assert yes_vwap([(NO, 0.25, 4.0)], yes_token=YES, no_token=NO) == pytest.approx(0.75)


def test_fills_are_weighted_by_size_after_conversion():
    # YES 0.90 (size 30) and NO 0.30 (size 10, i.e. YES 0.70): (27 + 7) / 40
    fills = [(YES, 0.90, 30.0), (NO, 0.30, 10.0)]
    assert yes_vwap(fills, yes_token=YES, no_token=NO) == pytest.approx(0.85)


def test_a_yes_only_tape_is_unchanged():
    fills = [(YES, 0.50, 1.0), (YES, 0.70, 3.0)]
    assert yes_vwap(fills, yes_token=YES, no_token=NO) == pytest.approx(0.65)


def test_token_ids_compare_as_text_so_an_integer_column_still_matches():
    assert yes_vwap([(7001, 0.6, 1.0)], yes_token="7001", no_token="7002") == pytest.approx(0.6)


@pytest.mark.parametrize(
    "bad_row",
    [
        ("9999", 0.9, 5.0),  # a token that is neither outcome of this market
        (YES, None, 5.0),  # no price
        (YES, 0.9, None),  # no size
        (YES, 0.9, 0.0),  # weightless
        (YES, 0.9, -3.0),  # negative size
        (YES, 0.9, float("nan")),
        (YES, float("nan"), 5.0),
        (YES, float("inf"), 5.0),
    ],
)
def test_unusable_rows_are_ignored_not_averaged_in(bad_row):
    good = (YES, 0.40, 2.0)
    assert yes_vwap([bad_row, good], yes_token=YES, no_token=NO) == pytest.approx(0.40)


def test_no_usable_rows_is_none_not_zero():
    assert yes_vwap([], yes_token=YES, no_token=NO) is None
    assert yes_vwap([(YES, 0.5, 0.0)], yes_token=YES, no_token=NO) is None


def test_the_two_tokens_must_differ():
    with pytest.raises(ValueError, match="different"):
        yes_vwap([(YES, 0.5, 1.0)], yes_token=YES, no_token=YES)


# --- a fake pq, so the Parquet layer runs without pyarrow ---------------------------


class _Column:
    def __init__(self, values: List[Any]) -> None:
        self._values = values

    def to_pylist(self) -> List[Any]:
        return list(self._values)


class _Table:
    def __init__(self, columns: Dict[str, List[Any]]) -> None:
        self._columns = columns

    def column(self, name: str) -> _Column:
        return _Column(self._columns[name])  # KeyError on a missing column, like pyarrow


class _FakePq:
    """Reads 'Parquet' that is really JSON, and fails the way pyarrow does."""

    def read_table(self, source: Any, columns: Optional[List[str]] = None) -> _Table:
        raw = source.read()
        try:
            decoded = json.loads(raw)
        except ValueError as exc:
            # pyarrow.ArrowInvalid is a ValueError subclass.
            raise ValueError("Parquet magic bytes not found in footer") from exc
        missing = [c for c in (columns or []) if c not in decoded]
        if missing:
            raise ValueError(f"No match for FieldRef.Name({missing[0]})")
        return _Table({c: decoded[c] for c in (columns or decoded)})


def _tape(*fills: Any) -> bytes:
    return json.dumps(
        {
            "market_id": [f[0] for f in fills],
            "price": [f[1] for f in fills],
            "size": [f[2] for f in fills],
            "side": ["buy" for _ in fills],  # a column the reader must not need
        }
    ).encode()


@pytest.fixture
def fake_pq(monkeypatch):
    pq = _FakePq()
    monkeypatch.setattr(data, "_require_pyarrow", lambda: pq)
    return pq


def test_the_parquet_layer_orients_and_weights(fake_pq):
    content = _tape((YES, 0.8, 10.0), (NO, 0.2, 10.0))
    price = data._vwap_from_parquet(content, yes_token=YES, no_token=NO, pq=fake_pq)
    assert price == pytest.approx(0.8)


def test_a_corrupt_file_is_a_typed_error_not_a_bare_arrow_error(fake_pq):
    # ArrowInvalid is a ValueError. Letting it through raw would hide the request id
    # and read like a bad argument rather than "the delivered file is unreadable".
    with pytest.raises(ResponseShapeError, match="could not be read") as raised:
        data._vwap_from_parquet(
            b"not parquet at all", yes_token=YES, no_token=NO, pq=fake_pq, request_id="req-1"
        )
    assert raised.value.request_id == "req-1"
    assert isinstance(raised.value.__cause__, ValueError)


def test_a_file_missing_a_column_is_a_typed_error(fake_pq):
    content = json.dumps({"market_id": [YES], "price": [0.5]}).encode()  # no size
    with pytest.raises(ResponseShapeError, match="market_id, price and size"):
        data._vwap_from_parquet(content, yes_token=YES, no_token=NO, pq=fake_pq)


# --- real pyarrow, where it is installed ---------------------------------------------


def test_real_parquet_round_trip():
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    import io

    table = pa.table(
        {
            "trade_id": ["a", "b", "c"],
            "market_id": [YES, NO, NO],
            "timestamp": ["2026-03-01T00:00:00+00:00"] * 3,
            "side": ["buy", "sell", "buy"],
            "price": [0.8, 0.2, 0.2],
            "size": [10.0, 5.0, 5.0],
        }
    )
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    price = data._vwap_from_parquet(buffer.getvalue(), yes_token=YES, no_token=NO, pq=pq)
    assert price == pytest.approx(0.8)


def test_real_corrupt_parquet_is_wrapped():
    pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    with pytest.raises(ResponseShapeError):
        data._vwap_from_parquet(b"PAR1 definitely not", yes_token=YES, no_token=NO, pq=pq)


# --- failing before any request -------------------------------------------------------


class _Download:
    """Records every paid request; the answer is whatever ``respond`` returns or raises."""

    def __init__(self, respond: Any = None) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._respond = respond or (lambda kw: _Result(_tape((YES, 0.8, 1.0), (NO, 0.2, 1.0))))

    def trades(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._respond(kwargs)


class _Result:
    def __init__(self, content: bytes, request_id: str = "req-x") -> None:
        self.content = content
        self.request_id = request_id


class _Markets:
    def __init__(self, records: List[Dict[str, Any]]) -> None:
        self._records = records
        self.paginated = 0
        self.last_kwargs: Dict[str, Any] = {}

    def auto_paginate(self, **kwargs: Any):
        self.paginated += 1
        self.last_kwargs = kwargs
        yield from self._records


class _Client:
    def __init__(self, records: Optional[List[Dict[str, Any]]] = None, download: Any = None):
        self.markets = _Markets(records or [])
        self.download = download or _Download()


def _market(i: int = 1, **over: Any) -> Dict[str, Any]:
    record = {
        "id": str(1000 + i),
        "question": f"q{i}?",
        "outcomes": ["Yes", "No"],
        "outcome_prices": [1.0, 0.0],
        "outcome_token_ids": [YES, NO],
        "winning_outcome": 0,
        "resolution_date": "2026-03-10T12:00:00Z",
        "end_date": "2026-03-09T00:00:00Z",
        "category": None,
        "volume": None,
    }
    record.update(over)
    return record


def _without_pyarrow(monkeypatch) -> None:
    """Make ``import pyarrow.parquet`` fail, whether or not it is installed."""
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)


def test_missing_pyarrow_fails_before_the_function_is_even_built(monkeypatch):
    _without_pyarrow(monkeypatch)
    with pytest.raises(MissingDependencyError, match="pyarrow") as raised:
        vwap_price_fn()
    assert "No request was made" in str(raised.value)
    assert "supagamma[parquet]" in str(raised.value)


def test_the_missing_dependency_is_an_import_error_and_a_config_error(monkeypatch):
    _without_pyarrow(monkeypatch)
    with pytest.raises(ImportError):
        vwap_price_fn()
    with pytest.raises(SupaGammaConfigError):
        vwap_price_fn()


def test_with_pyarrow_missing_a_study_makes_no_request_at_all(monkeypatch):
    """The scenario that cost money: a plain install, run over many markets."""
    _without_pyarrow(monkeypatch)
    client = _Client([_market(i) for i in range(20)])
    with pytest.raises(MissingDependencyError):
        list(resolved_markets(client, max_markets=20, price_fn=vwap_price_fn()))
    assert client.download.calls == []
    assert client.markets.paginated == 0  # not even a free catalogue page


# --- what _price does, and refuses to do -------------------------------------------------


def test_the_request_is_one_parquet_pull_ending_at_the_horizon(fake_pq):
    client = _Client()
    price = vwap_price_fn(horizon_hours=6, trade_limit=2_000)(client, _market())
    assert price == pytest.approx(0.8)
    [call] = client.download.calls
    assert call == {
        "market_id": "1001",
        "end": datetime(2026, 3, 10, 6, 0, tzinfo=UTC),  # resolution_date - 6h
        "format": "parquet",
        "limit": 2_000,
    }
    assert "start" not in call


def test_resolution_date_wins_over_end_date_and_end_date_is_the_fallback(fake_pq):
    client = _Client()
    fn = vwap_price_fn(horizon_hours=24)
    fn(client, _market(resolution_date="2026-03-10T00:00:00Z", end_date="2026-03-20T00:00:00Z"))
    fn(client, _market(resolution_date=None, end_date="2026-03-20T00:00:00Z"))
    assert [c["end"] for c in client.download.calls] == [
        datetime(2026, 3, 9, tzinfo=UTC),
        datetime(2026, 3, 19, tzinfo=UTC),
    ]


def test_a_market_with_no_end_date_is_skipped_without_a_request(fake_pq):
    # An open window would include the settlement prints this price exists to avoid.
    client = _Client()
    record = _market(resolution_date=None, end_date=None)
    assert vwap_price_fn()(client, record) is None
    assert client.download.calls == []


@pytest.mark.parametrize(
    "tokens",
    [None, [], [YES], [YES, NO, "7003"], [YES, YES]],
    ids=["null", "empty", "one", "three", "same-twice"],
)
def test_a_market_that_cannot_be_split_into_yes_and_no_is_skipped_for_free(fake_pq, tokens):
    client = _Client()
    assert vwap_price_fn()(client, _market(outcome_token_ids=tokens)) is None
    assert client.download.calls == []


def test_a_record_with_no_token_field_is_a_config_error_before_any_spend(fake_pq):
    # The API deployment predates `outcome_token_ids`. Skipping every market would
    # look like an empty universe; paying would mix YES and NO.
    client = _Client()
    record = _market()
    del record["outcome_token_ids"]
    with pytest.raises(SupaGammaConfigError, match="outcome_token_ids"):
        vwap_price_fn()(client, record)
    assert client.download.calls == []


def test_no_trades_before_the_cutoff_skips_the_market(fake_pq):
    def respond(_kw):
        raise NoDataInRangeError("No trades found", status_code=404)

    client = _Client(download=_Download(respond))
    assert vwap_price_fn()(client, _market()) is None
    assert len(client.download.calls) == 1  # it was attempted, and that counts


@pytest.mark.parametrize(
    "error",
    [
        AuthenticationError("bad key", status_code=401),
        PaymentRequiredError("subscribe", status_code=402),
        QuotaExceededError("cap", status_code=429),
        FairUseCapError("cap", status_code=429),
        RateLimitError("slow down", status_code=429),
        ServerError("boom", status_code=500),
        APITimeoutError("timed out"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_every_other_failure_propagates_instead_of_being_swallowed(fake_pq, error):
    def respond(_kw):
        raise error

    client = _Client(download=_Download(respond))
    with pytest.raises(type(error)):
        vwap_price_fn()(client, _market())


def test_arguments_are_checked_before_anything_else(fake_pq):
    with pytest.raises(ValueError, match="horizon_hours"):
        vwap_price_fn(horizon_hours=-1)
    with pytest.raises(ValueError, match="trade_limit"):
        vwap_price_fn(trade_limit=0)
    with pytest.raises(ValueError, match="trade_limit"):
        vwap_price_fn(trade_limit=1_000_001)


# --- the hard bound on paid attempts ---------------------------------------------------------


def _counting_price_fn(answer: Optional[float] = None):
    calls: List[str] = []

    def price_fn(_client: Any, market: Dict[str, Any]) -> Optional[float]:
        calls.append(market["id"])
        return answer

    price_fn.calls = calls  # type: ignore[attr-defined]
    return price_fn


def test_failures_cannot_walk_the_whole_catalogue():
    # Every price fails (None). The old loop only stopped on a yield, so it asked
    # for a price for all 50 markets. Now `max_markets` bounds the attempts.
    client = _Client([_market(i) for i in range(50)])
    price_fn = _counting_price_fn(None)
    assert list(resolved_markets(client, max_markets=5, price_fn=price_fn)) == []
    assert len(price_fn.calls) == 5


def test_max_price_attempts_is_the_tighter_bound_when_it_is_smaller():
    client = _Client([_market(i) for i in range(50)])
    price_fn = _counting_price_fn(None)
    list(resolved_markets(client, max_markets=100, max_price_attempts=3, price_fn=price_fn))
    assert len(price_fn.calls) == 3


def test_max_price_attempts_can_exceed_max_markets_to_keep_going_past_skips():
    client = _Client([_market(i) for i in range(30)])
    answers = iter([None] * 7 + [0.6] * 100)

    def price_fn(_c: Any, _m: Dict[str, Any]) -> Optional[float]:
        return next(answers)

    got = list(resolved_markets(client, max_markets=2, max_price_attempts=20, price_fn=price_fn))
    assert len(got) == 2  # stopped at max_markets, after 7 skips + 2 hits = 9 attempts


def test_the_next_call_is_never_made_once_the_bound_is_reached():
    client = _Client([_market(i) for i in range(10)])
    price_fn = _counting_price_fn(0.5)
    assert (
        len(
            list(
                resolved_markets(client, max_markets=None, max_price_attempts=4, price_fn=price_fn)
            )
        )
        == 4
    )
    assert len(price_fn.calls) == 4


def test_an_unbounded_paid_run_is_refused_before_any_request():
    client = _Client([_market(i) for i in range(10)])
    price_fn = _counting_price_fn(0.5)
    with pytest.raises(ValueError, match="needs a bound"):
        resolved_markets(client, max_markets=None, price_fn=price_fn)  # raises on the call
    assert client.markets.paginated == 0 and price_fn.calls == []


def test_a_negative_bound_is_refused():
    with pytest.raises(ValueError, match="negative"):
        resolved_markets(_Client(), max_price_attempts=-1, price_fn=_counting_price_fn())


@pytest.mark.parametrize("with_price_fn", [False, True])
def test_a_zero_bound_does_nothing_at_all(with_price_fn):
    client = _Client([_market(1)])
    price_fn = _counting_price_fn(0.5) if with_price_fn else None
    assert list(resolved_markets(client, max_markets=0, price_fn=price_fn)) == []
    assert client.markets.paginated == 0


def test_a_price_function_error_stops_the_run_after_one_attempt():
    client = _Client([_market(i) for i in range(10)])
    calls: List[str] = []

    def price_fn(_c: Any, market: Dict[str, Any]) -> Optional[float]:
        calls.append(market["id"])
        raise QuotaExceededError("monthly cap", status_code=429)

    with pytest.raises(QuotaExceededError):
        list(resolved_markets(client, max_markets=10, price_fn=price_fn))
    assert len(calls) == 1


def test_markets_skipped_before_pricing_cost_nothing_and_do_not_count():
    unresolved = [_market(i, winning_outcome=None) for i in range(10)]
    resolved = [_market(100 + i) for i in range(10)]
    client = _Client(unresolved + resolved)
    price_fn = _counting_price_fn(0.5)
    got = list(resolved_markets(client, max_markets=3, price_fn=price_fn))
    assert len(got) == 3 and len(price_fn.calls) == 3


def test_filters_reach_the_catalogue_query():
    client = _Client([_market(1)])
    list(resolved_markets(client, max_markets=1, page_limit=250, category="crypto"))
    assert client.markets.last_kwargs == {
        "resolved": True,
        "has_data": True,
        "limit": 250,
        "category": "crypto",
    }


# --- the free path is exactly what it was ------------------------------------------------------


def test_without_a_price_function_the_metadata_price_is_used_and_yields_are_counted():
    records = [_market(i, outcome_prices=[0.25, 0.75]) for i in range(10)]
    records[0] = _market(0, outcome_prices=None)  # unreadable price: skipped, free
    client = _Client(records)
    got = list(resolved_markets(client, max_markets=4))
    assert len(got) == 4
    assert all(m.view.prob == pytest.approx(0.25) for m in got)
    assert all(m.outcome == realized_outcome(records[1]) for m in got)


def test_a_price_outside_zero_to_one_is_dropped():
    client = _Client([_market(1), _market(2)])
    answers = iter([1.7, 0.4])
    got = list(resolved_markets(client, max_markets=5, price_fn=lambda _c, _m: next(answers)))
    assert [m.view.prob for m in got] == [0.4]


# --- end to end over real HTTP, with every price failing ------------------------------------------


@respx.mock
def test_over_http_a_run_where_every_price_fails_makes_only_max_markets_paid_requests(monkeypatch):
    """The audit's scenario, end to end through the real client.

    50 resolved markets, none with trades before the cutoff. The paid endpoint answers
    404 every time, so no price is ever produced. The number of paid requests must be
    the bound, not the catalogue size.
    """
    monkeypatch.setattr(data, "_require_pyarrow", lambda: _FakePq())
    page = [_market(i) for i in range(50)]
    respx.get("https://api.test/v1/markets").mock(return_value=httpx.Response(200, json=page))
    paid = respx.get("https://api.test/v1/download/trades").mock(
        return_value=httpx.Response(404, json={"detail": "No trades found"})
    )
    client = SupaGamma(api_key="sg_" + "a" * 32, base_url="https://api.test", max_retries=3)

    got = list(
        resolved_markets(
            client, max_markets=4, page_limit=50, price_fn=vwap_price_fn(trade_limit=1_000)
        )
    )

    assert got == []
    assert paid.call_count == 4  # not 50, and not retried despite max_retries=3
    for request in paid.calls:
        assert request.request.url.params["format"] == "parquet"
        assert request.request.url.params["limit"] == "1000"
        assert "start" not in request.request.url.params


@respx.mock
def test_over_http_a_complementary_tape_prices_at_the_yes_price(monkeypatch):
    monkeypatch.setattr(data, "_require_pyarrow", lambda: _FakePq())
    respx.get("https://api.test/v1/markets").mock(
        return_value=httpx.Response(200, json=[_market(1)])
    )
    tape = _tape(*([(YES, 0.8, 10.0), (NO, 0.2, 10.0)] * 20))
    respx.get("https://api.test/v1/download/trades").mock(
        return_value=httpx.Response(200, content=tape, headers={"content-type": "application/json"})
    )
    client = SupaGamma(api_key="sg_" + "a" * 32, base_url="https://api.test")

    [market] = resolved_markets(client, max_markets=1, price_fn=vwap_price_fn(trade_limit=1_000))

    assert math.isclose(market.view.prob, 0.8)  # not 0.5
    assert market.outcome == 1
    assert market.view.meta["resolution_date"] == "2026-03-10T12:00:00Z"
    # sanity on the end the test fixes: the cutoff is 24h before resolution
    assert timedelta(hours=24) == datetime(2026, 3, 10, 12, tzinfo=UTC) - datetime(
        2026, 3, 9, 12, tzinfo=UTC
    )
