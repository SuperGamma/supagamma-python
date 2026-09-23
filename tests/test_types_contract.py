"""The typed models and the request builders match the API's published contract.

``supagamma.types`` hand-writes one TypedDict per response schema the API
publishes. Hand-written types drift, so every model is pinned to a vendored
snapshot of ``https://api.supagamma.com/openapi.json`` in BOTH directions:

* a field the server adds, drops or renames fails here;
* so does a model field the server never sends;
* so does a field whose ``None``-ability disagrees with the spec.

The second half pins every query parameter the SDK puts on the wire to a
parameter the spec declares for that route. The server ignores unknown query
parameters, so a stale one does not error; it silently changes nothing. The
``tag`` filter on ``/v1/markets`` did exactly that after the API removed it, and
the SDK kept sending it.

Refresh the snapshot when the API changes, then fix whatever this flags:

    curl -s https://api.supagamma.com/openapi.json -o tests/fixtures/openapi.json
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Set

import pytest

from supagamma import types as t
from supagamma.resources import (
    account,
    billing,
    download,
    markets,
    orders,
    public_markets,
    series,
    system,
    trades,
)

SPEC: Dict[str, Any] = json.loads(
    (Path(__file__).parent / "fixtures" / "openapi.json").read_text(encoding="utf-8")
)
SCHEMAS: Dict[str, Any] = SPEC["components"]["schemas"]

MODEL_SCHEMAS = {
    t.Market: "MarketResponse",
    t.MarketEstimate: "MarketEstimateResponse",
    t.PublicMarketSummary: "PublicMarketSummary",
    t.PublicMarket: "PublicMarket",
    t.Trade: "TradeResponse",
    t.OHLCVBar: "OHLCVResponse",
    t.Series: "SeriesResponse",
    t.SeriesEstimate: "EstimateResponse",
    t.Order: "OrderResponse",
    t.OrderLine: "OrderItemResponse",
    t.Balance: "BalanceResponse",
    t.Checkout: "CheckoutResponse",
    t.Transaction: "TransactionResponse",
    t.DownloadRecord: "DownloadHistoryResponse",
    t.ApiKey: "ApiKeyResponse",
    t.NewApiKey: "ApiKeyCreatedResponse",
    t.PrivacyEvent: "PrivacyEventResponse",
    t.Usage: "UsageResponse",
}

#: Keys the SDK adds client-side. They are deliberately not in the server contract.
SDK_ADDED: Dict[Any, Set[str]] = {t.Trade: {"token_id"}, t.OHLCVBar: {"token_id"}}


def _nullable(prop: Dict[str, Any]) -> bool:
    return any(branch.get("type") == "null" for branch in prop.get("anyOf", []))


def _annotation_text(annotation: Any) -> str:
    """The source text of a model field's annotation.

    ``types.py`` uses ``from __future__ import annotations``, and TypedDict wraps
    those strings in ``typing.ForwardRef``; unwrap it rather than trusting repr.
    """
    return str(getattr(annotation, "__forward_arg__", annotation))


def test_every_published_model_is_exported():
    assert set(t.__all__) == {model.__name__ for model in MODEL_SCHEMAS}


@pytest.mark.parametrize(
    "model,schema", list(MODEL_SCHEMAS.items()), ids=[m.__name__ for m in MODEL_SCHEMAS]
)
def test_model_fields_match_the_published_schema(model: Any, schema: str) -> None:
    properties = SCHEMAS[schema]["properties"]
    fields = set(model.__annotations__) - SDK_ADDED.get(model, set())
    missing = sorted(set(properties) - fields)
    extra = sorted(fields - set(properties))
    assert not missing and not extra, (
        f"{model.__name__} drifted from {schema}: "
        f"server fields the model lacks={missing}, model fields the server never sends={extra}"
    )


@pytest.mark.parametrize(
    "model,schema", list(MODEL_SCHEMAS.items()), ids=[m.__name__ for m in MODEL_SCHEMAS]
)
def test_optional_means_exactly_what_the_server_says(model: Any, schema: str) -> None:
    """``Optional[...]`` on a model field iff the spec says the value can be null."""
    properties = SCHEMAS[schema]["properties"]
    wrong = []
    for name, prop in properties.items():
        annotation = _annotation_text(model.__annotations__[name])
        if annotation.startswith("Optional[") != _nullable(prop):
            wrong.append(f"{name}: model={annotation!r}, spec nullable={_nullable(prop)}")
    assert not wrong, f"{model.__name__}: " + "; ".join(wrong)


# --- query parameters the SDK sends ---------------------------------------------------

UTC = timezone.utc
START = datetime(2026, 9, 1, tzinfo=UTC)
END = datetime(2026, 9, 2, tzinfo=UTC)

#: Every builder that puts query parameters on the wire, called with every
#: optional parameter set, so each name it can ever send is exercised.
BUILDERS = [
    markets.build_list(
        active=True,
        closed=False,
        resolved=False,
        category="politics",
        has_data=True,
        data_type="trades",
        search="election",
        sort_by="volume",
        limit=10,
        offset=0,
    ),
    markets.build_get("1254468"),
    markets.build_stats("1254468"),
    markets.build_estimate("1254468", data_type="trades", start=START, end=END)[0],
    public_markets.build_list(limit=10, offset=0, resolved=True),
    public_markets.build_get("1254468"),
    trades.build_list(
        market_id="1254468", start=START, end=END, outcome=0, side="buy", limit=10, offset=0
    ),
    trades.build_ohlcv(
        market_id="1254468", outcome=0, timeframe="1h", start=START, end=END, limit=10
    ),
    trades.build_recent(limit=10),
    series.build_list(),
    series.build_get("polymarket:l2-delta-tape"),
    orders.build_list(limit=10),
    orders.build_pricing(),
    billing.build_balance(),
    billing.build_pricing(),
    billing.build_transactions(limit=10, offset=0),
    billing.build_subscription(),
    account.build_me(),
    account.build_usage(),
    account.build_downloads(limit=10, offset=0),
    account.build_list_keys(),
    account.build_privacy_events(),
    download.build_trades(market_id="1254468", start=START, end=END, format="csv", limit=10),
    download.build_ohlcv(market_id="1254468", timeframe="1h", start=START, end=END, limit=10),
    download.build_orderbook(market_id="1254468", start=START, end=END, limit=10),
    download.build_top_of_book(market_id="1254468", start=START, end=END, limit=10),
    download.build_options(
        series_id="deribit:btc-options",
        data_type="trades",
        start=START,
        end=END,
        instrument_prefix="BTC-29MAY26-",
        limit=10,
    ),
    download.build_series(
        series_id="polymarket:wallet-leaderboard",
        data_type="trades",
        start=START,
        end=END,
        instrument_prefix="x",
        limit=10,
    ),
    download.build_raw(
        data_type="polymarket_l2_deltas", start=START, end=END, format="json", shape="raw", limit=10
    ),
    download.build_raw_datasets(),
    download.build_raw_estimate(data_type="polymarket_l2_deltas", start=START, end=END, limit=10),
    system.build_root(),
    system.build_health(),
    system.build_stats(),
]


def _spec_operation(method: str, path: str) -> Dict[str, Any]:
    for template, operations in SPEC["paths"].items():
        pattern = "^" + re.sub(r"\{[^/]+\}", "[^/]+", template) + "$"
        if re.match(pattern, path) and method.lower() in operations:
            return operations[method.lower()]
    raise AssertionError(f"{method} {path} is not a route in the published spec")


def _undeclared_params(spec: Any) -> list:
    method, path, params, _policy = spec
    operation = _spec_operation(method, path)
    declared = {p["name"] for p in operation.get("parameters", []) if p.get("in") == "query"}
    return sorted(set(params) - declared)


@pytest.mark.parametrize("spec", BUILDERS, ids=[f"{b[0]} {b[1]}" for b in BUILDERS])
def test_every_query_parameter_is_one_the_route_declares(spec: Any) -> None:
    undeclared = _undeclared_params(spec)
    assert not undeclared, (
        f"{spec[0]} {spec[1]} sends {undeclared}, which the API does not declare. The "
        "server ignores unknown query parameters, so these silently do nothing."
    )


def test_the_parameter_check_catches_a_stale_parameter():
    """Guard the guard: the exact drift this suite was written for must be caught."""
    stale = ("GET", "/v1/markets", {"tag": "elections", "limit": 10}, None)
    assert _undeclared_params(stale) == ["tag"]


def test_the_removed_tag_filter_fails_loudly_instead_of_returning_everything():
    with pytest.raises(ValueError, match="removed"):
        markets.build_list(tag="elections")


# --- the types have to reach the user's type checker ----------------------------------


def test_the_package_is_marked_typed():
    """PEP 561: without ``py.typed`` mypy/pyright ignore every inline annotation
    in an installed package, so none of these models would reach users. 0.1.0
    shipped without it while advertising ``Typing :: Typed``."""
    import supagamma

    assert (Path(supagamma.__file__).parent / "py.typed").is_file()


def test_every_namespace_is_declared_for_type_checkers():
    """Namespaces are attached with setattr, which type checkers cannot see.

    Each one therefore needs a class-level annotation on both clients. A namespace
    added to NAMESPACES without one would be ``Any`` to every user.
    """
    from supagamma import AsyncSupaGamma, SupaGamma
    from supagamma.resources import NAMESPACES

    for name, (sync_cls, async_cls) in NAMESPACES.items():
        assert SupaGamma.__annotations__.get(name) == sync_cls.__name__, name
        assert AsyncSupaGamma.__annotations__.get(name) == async_cls.__name__, name
