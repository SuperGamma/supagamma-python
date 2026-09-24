"""Typed response shapes for the SupaGamma API.

Every model here is a :class:`typing.TypedDict`. At runtime a response is still
exactly the dict the server sent, so no existing code changes behaviour: indexing,
``.get()``, ``json.dumps`` and ``pandas.DataFrame(rows)`` all work as before. What
changes is that editors and type checkers now know every field and its type.

The models mirror the API's published OpenAPI spec
(``https://api.supagamma.com/openapi.json``). ``tests/test_types_contract.py``
pins each model to a vendored snapshot of that spec in both directions, so a field
added, removed or renamed on the server fails CI here instead of drifting quietly.

Two conventions hold for every model:

* **Every declared field is present.** The API serialises its response models in
  full, so a key is never missing; "no value" is ``None``. That is why these
  are total TypedDicts, and why a field typed ``Optional[...]`` means "may be
  ``None``", never "may be absent".
* **Timestamps are ISO-8601 strings**, exactly as sent, e.g.
  ``"2026-09-23T14:57:35.550683+00:00"``. Parse with
  :func:`datetime.datetime.fromisoformat` when you need a ``datetime``.

Routes whose responses the API does not publish a schema for (for example
``markets.stats()``, ``trades.recent()``, ``billing.pricing()`` and the
subscription endpoints) keep returning ``Dict[str, Any]``. They are typed as soon
as the server publishes a model for them, rather than guessed at here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

__all__ = [
    "Market",
    "MarketEstimate",
    "Trade",
    "OHLCVBar",
    "Series",
    "SeriesEstimate",
    "Order",
    "OrderLine",
    "PublicMarketSummary",
    "PublicMarket",
    "Balance",
    "Checkout",
    "Transaction",
    "DownloadRecord",
    "ApiKey",
    "NewApiKey",
    "PrivacyEvent",
    "Usage",
]


# --- markets -----------------------------------------------------------------------


class Market(TypedDict):
    """One market, from ``markets.list()`` / ``markets.get()``.

    ``id`` is ``markets.id``, the short Polymarket numeric id (``"1254468"``). It
    is not a ``condition_id`` or a CLOB token id.

    ``volume``, ``volume_24h`` and ``liquidity`` are ``None`` when the value is
    zero OR unknown; the server cannot tell the two apart.

    ``trade_count``/``orderbook_count`` and ``data_from``/``data_to`` describe the
    data SupaGamma holds for the market, i.e. what you can buy.
    """

    id: str
    event_id: Optional[str]
    question: str
    description: Optional[str]
    outcomes: List[str]
    outcome_prices: Optional[List[float]]
    active: bool
    closed: bool
    resolved: bool
    winning_outcome: Optional[int]
    resolution_date: Optional[str]
    volume: Optional[float]
    volume_24h: Optional[float]
    liquidity: Optional[float]
    start_date: Optional[str]
    end_date: Optional[str]
    trade_count: int
    orderbook_count: int
    data_from: Optional[str]
    data_to: Optional[str]
    category: Optional[str]


class MarketEstimate(TypedDict):
    """A cost preview from ``markets.estimate()``. It never charges.

    ``in_range`` is ``False`` when the requested window holds no data for this
    market; ``estimated_rows`` is then ``0``.
    """

    market_id: str
    data_type: str
    estimated_rows: int
    estimated_size_bytes: int
    estimated_size_mb: float
    rate_per_mb: float
    estimated_cost_usd: float
    in_range: bool


class PublicMarketSummary(TypedDict):
    """One row of ``public_markets.list()``: the unauthenticated index."""

    id: str
    question: str
    category: Optional[str]
    resolved: bool
    trade_count: int
    orderbook_count: int


class PublicMarket(TypedDict):
    """``public_markets.get()``: resolution plus the data held, and nothing paid."""

    id: str
    question: str
    category: Optional[str]
    outcomes: List[str]
    resolved: bool
    winning_outcome: Optional[int]
    winning_outcome_label: Optional[str]
    resolution_date: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    trade_count: int
    orderbook_count: int
    data_from: Optional[str]
    data_to: Optional[str]


# --- trades --------------------------------------------------------------------------


class Trade(TypedDict):
    """One row of ``trades.list()``.

    ``market_id`` here is the outcome TOKEN id, not ``markets.id``, which is why
    the SDK adds ``token_id`` (the same value, under an honest name). ``side`` is
    ``"buy"`` or ``"sell"``; ``outcome`` is the outcome index.
    """

    id: str
    market_id: str
    #: Added client-side by the SDK; equals ``market_id``. Not part of the API spec.
    token_id: str
    timestamp: str
    block_number: Optional[int]
    transaction_hash: Optional[str]
    outcome: int
    side: str
    price: float
    size: float
    usd_value: Optional[float]
    maker: Optional[str]
    taker: Optional[str]
    fee: Optional[float]


class OHLCVBar(TypedDict):
    """One candle from ``trades.ohlcv()``. ``timestamp`` is the bucket start.

    As on :class:`Trade`, ``market_id`` is an outcome token id and the SDK adds
    ``token_id`` alongside it.
    """

    market_id: str
    #: Added client-side by the SDK; equals ``market_id``. Not part of the API spec.
    token_id: str
    outcome: int
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int


# --- series --------------------------------------------------------------------------


class Series(TypedDict):
    """One entry of the data catalogue, from ``series.list()`` / ``series.get()``.

    ``data_current_through`` is the most recent date the catalogue holds for this
    series. It is conservative: the minimum across the series' streams, so one
    stalled stream cannot hide behind a fresh one. ``coming_soon`` series are
    listed but not yet sold.
    """

    series_id: str
    broker: str
    broker_display: str
    asset_class: str
    category: str
    asset: str
    asset_symbol: str
    timeframe: str
    name: str
    description: str
    catalogued_markets: int
    shoppable_markets: int
    earliest: Optional[str]
    latest: Optional[str]
    data_current_through: Optional[str]
    coming_soon: bool


class SeriesEstimate(TypedDict):
    """A cost preview from ``series.estimate()``. It never charges.

    When ``row_cap_applied`` is ``True`` the window holds more rows than one
    download delivers; ``row_cap`` is that ceiling, and the estimate is priced at
    the cap, not the full window.
    """

    series_id: str
    data_type: str
    markets_in_range: int
    estimated_rows: int
    estimated_size_bytes: int
    estimated_size_mb: float
    rate_per_mb: float
    estimated_cost_usd: float
    coming_soon: bool
    row_cap_applied: bool
    row_cap: Optional[int]


# --- orders --------------------------------------------------------------------------


class OrderLine(TypedDict):
    """One purchased line inside an :class:`Order`.

    Named ``OrderLine`` rather than ``OrderItem`` because ``OrderItem`` is the
    REQUEST type you build to place an order (``supagamma.resources.orders``).
    ``expires_at`` ends the free re-download window for this line.
    """

    id: str
    market_id: str
    series_id: Optional[str]
    data_type: str
    format: str
    timeframe: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    record_count: Optional[int]
    size_bytes: Optional[int]
    cost: float
    status: str
    expires_at: str


class Order(TypedDict):
    """An order from ``orders.create()`` / ``orders.list()``."""

    order_id: str
    created_at: str
    total_cost: float
    items: List[OrderLine]


# --- billing -------------------------------------------------------------------------


class Balance(TypedDict):
    """The caller's credit balance, from ``billing.balance()``.

    ``balance == lifetime_credits - lifetime_usage - lifetime_refunds`` always
    holds; the API's own reconciler checks it daily.
    """

    user_id: str
    balance: float
    currency: str
    lifetime_credits: float
    lifetime_usage: float
    lifetime_refunds: float


class Checkout(TypedDict):
    """A one-time credit checkout, from ``billing.checkout()``.

    While the API runs subscription-only (``billing.payg_enabled()`` is
    ``False``), that call raises ``PaygRetiredError`` instead of returning this.
    """

    checkout_id: str
    checkout_url: str
    amount: float


class Transaction(TypedDict):
    """One ledger row from ``billing.transactions()``.

    ``type`` is one of ``credit_purchase``, ``usage_charge``, ``refund`` or
    ``adjustment``.
    """

    id: str
    type: str
    amount: float
    currency: str
    status: str
    description: str
    created_at: str
    metadata: Optional[Dict[str, Any]]


# --- account -------------------------------------------------------------------------


class DownloadRecord(TypedDict):
    """One row of ``account.downloads()``.

    A checkout receipt and the delivery it paid for are separate rows. ``cost`` is
    ``0`` for downloads covered by a subscription or a paid re-download window.
    """

    id: str
    market_id: str
    data_type: str
    format: str
    file_name: str
    record_count: Optional[int]
    size_bytes: Optional[int]
    cost: float
    status: str
    timeframe: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    created_at: str


class ApiKey(TypedDict):
    """One of the caller's API keys, from ``account.keys.list()``. The secret is
    never returned after creation, only ``key_prefix``.

    ``scheduled_revoke_at`` is set on a key that has been rotated: the key keeps
    working until then and stops working after it. ``permissions`` lists what
    the key may do, for example ``["read"]`` for a browse-only key that cannot
    spend.
    """

    id: str
    name: str
    key_prefix: str
    created_at: str
    last_used_at: Optional[str]
    scheduled_revoke_at: Optional[str]
    expires_at: Optional[str]
    permissions: Optional[List[str]]


class NewApiKey(TypedDict):
    """A freshly minted key, from ``account.keys.create()`` / ``.rotate()`` /
    ``.provision()``.

    ``key`` is the full secret. **This is the only time it is ever returned**, so
    store it now; it cannot be retrieved later.
    """

    id: str
    name: str
    key: str
    key_prefix: str
    created_at: str
    expires_at: Optional[str]


class PrivacyEvent(TypedDict):
    """One entry of the caller's security activity, from ``account.privacy_events()``.

    ``ip_hash`` is a hash, never the raw address.
    """

    id: str
    event: str
    metadata: Dict[str, Any]
    created_at: str
    ip_hash: Optional[str]
    user_agent: Optional[str]


class Usage(TypedDict):
    """The caller's usage for the current period, from ``account.usage()``.

    ``period`` and each ``daily`` entry are free-form objects the API does not
    publish a schema for, so they stay untyped.
    """

    user_id: str
    period: Dict[str, Any]
    total_requests: int
    total_bytes: int
    total_mb: float
    estimated_cost: float
    daily: List[Dict[str, Any]]
