# supagamma

Official Python SDK for the [SupaGamma](https://supagamma.com) API — institutional-grade historical data for prediction markets.

```bash
pip install supagamma
```

```python
from supagamma import SupaGamma

client = SupaGamma(api_key="sg_...")        # or set SUPAGAMMA_API_KEY

for market in client.markets.auto_paginate(limit=500):
    print(market["id"], market["question"])
```

Async works identically:

```python
from supagamma import AsyncSupaGamma

async with AsyncSupaGamma() as client:
    bars = await client.trades.ohlcv(market_id="546814", timeframe="1d")
```

## Getting a key

Create one in your [dashboard](https://supagamma.com/dashboard/api-keys). Keys look like `sg_` followed by 32 hex characters, and carry scopes — `read` for browsing, `download` for anything that spends credits. The SDK validates the format locally so a typo fails immediately instead of costing a round trip.

## What's here

| Namespace | What it does |
| --- | --- |
| `client.markets` | Market catalogue, stats, cost estimates |
| `client.trades` | Raw fills, OHLCV bars, recent trades |
| `client.series` | The stream catalogue and its estimates |
| `client.download` | **Paid data delivery** — see below |
| `client.orders` | Cart checkout with idempotency |
| `client.billing` | Balance, transactions, subscription |
| `client.account` | Identity, usage, API keys, GDPR export |
| `client.system` | `/health`, platform stats |
| `client.public_markets` | Public market metadata (usually disabled) |

## Typed responses

Every response is still the plain dict the server sent. Where the API publishes a
response schema, the SDK also declares its shape as a `TypedDict` in
`supagamma.types`, so your editor and type checker know every field:

```python
from supagamma.types import Market

market: Market = client.markets.get("1254468")
market["trade_count"]      # int: the data you can buy
market["volume"]           # Optional[float]: None when zero OR unknown
```

Two rules hold for every model. A declared key is always present (a missing
value is `None`, never an absent key), and timestamps are ISO-8601 strings, as
sent. The models are pinned to a snapshot of the API's published OpenAPI spec,
so a server-side change fails this SDK's CI instead of drifting. Routes the API
publishes no schema for, such as `markets.stats()` and the subscription
endpoints, still return `Dict[str, Any]`.

## Backtesting

`supagamma.backtest` is a dependency-free harness for scoring prediction-market
strategies against resolved markets — plus the calibration analysis behind the
"does the favourite-longshot bias exist?" study. The engine takes no network, so
it unit-tests offline; the `data` helper bridges it to a live client.

```python
from supagamma import SupaGamma
from supagamma.backtest import Backtest, BetFavourite, calibration
from supagamma.backtest.data import resolved_markets, calibration_pairs

client = SupaGamma(api_key="sg_...")
universe = list(resolved_markets(client, max_markets=1_000))

# Is the market's price an honest probability?
print(calibration(calibration_pairs(universe)).as_table())

# Does backing the favourite beat the book?
result = Backtest(bankroll=1_000).run(universe, BetFavourite(stake=10))
print(result.summary())
```

The strategy only ever sees a `MarketView` with no outcome field, so it *cannot*
peek at the answer — look-ahead safety is enforced, not trusted. Write your own
by returning an `Order(side, stake)` (or `None`) from any callable. A full worked
example lives in [`examples/calibration.py`](examples/calibration.py).

It's a research tool, not investment advice, and makes no performance promise:
it shows what *did* happen in historical data. Fees, liquidity, and slippage make
live results different.

## Four things worth knowing

These are properties of the API, not of this library, and the SDK surfaces them rather than hiding them.

### Downloads spend money, so they are never retried

Every `client.download.*` call except the two estimates debits your balance. The SDK sets a no-retry policy on those routes regardless of how you configure `max_retries`.

The reason is specific. The server debits *after* serialising your data but *before* the body finishes arriving, and the only protection against paying twice is a 7-day entitlement waiver matched on an exact parameter tuple. A retry that re-derives `end=datetime.now()` looks like a *different* request to that matcher and is charged again in full. If you retry a download yourself, freeze your parameters first and replay them byte-identically:

```python
start, end = window()          # compute ONCE
try:
    result = client.download.trades(market_id="546814", start=start, end=end)
except supagamma.APITimeoutError:
    time.sleep(2)              # the entitlement row is written in the background
    result = client.download.trades(market_id="546814", start=start, end=end)
```

### 429 means two different things

```python
try:
    client.download.orderbook(market_id="546814")
except supagamma.RateLimitError as e:
    time.sleep(e.retry_after)   # transient — the limiter
except supagamma.QuotaExceededError:
    ...                         # a billing cap; retrying can never succeed
```

`RateLimitError` clears after `retry_after` seconds. `QuotaExceededError` — your monthly fair-use or free-tier cap — clears on a billing-period boundary, carries no `Retry-After`, and retrying it just burns limiter budget on top. They share a status code and nothing else, which is why they are separate classes.

### Truncation is silent

A download that hits its row cap looks exactly like a complete one: no flag, no header, no marker. When completeness matters, estimate first:

```python
est = client.download.raw_estimate(data_type="polymarket_l2_deltas", start=start, end=end)
if est["capped_by_limit"]:
    ...   # narrow the window; paging cannot reach the rest
```

Downloads have no `offset`. A dataset larger than the cap is reachable only by narrowing `start`/`end`.

### Orderbook data is expensive

At roughly 2 KB per row and \$5/MB, the default 100,000-row orderbook pull is on the order of **\$990**. The SDK warns above \$25 and lets you gate it:

```python
client.download.confirm_cost = lambda usd: usd < 50    # abort anything pricier
```

## Errors

Everything derives from `supagamma.SupaGammaError`. The ones you will actually branch on:

| Exception | Meaning |
| --- | --- |
| `InsufficientCreditsError` | 402 — `.shortfall` is exactly what to top up |
| `SubscriptionRequiredError` / `UpgradeRequiredError` | 402 — plan doesn't cover this |
| `RateLimitError` | 429 — transient, honour `.retry_after` |
| `QuotaExceededError` | 429 — billing cap, do not retry |
| `NoDataInRangeError` | 404 — the id is fine, the window is empty |
| `OrderStatusUnknownError` | 502 on order creation — replay the same `idempotency_key` |
| `OriginBlockedError` | 403 — you pointed `base_url` at the origin, not the API |

Every exception carries `.status_code`, `.code`, `.request_id` and the raw `.detail`. Quote `request_id` to support; it is the only correlation handle.

## Orders and idempotency

`client.orders.create()` is the one route with real idempotency protection, and it is a **body field**, not an `Idempotency-Key` header. The SDK generates a key for you and returns it:

```python
order = client.orders.create([
    supagamma.resources.orders.OrderItem(data_type="trades", market_id="546814"),
])

# On an ambiguous failure, replay with the SAME key — the server returns the
# original order instead of charging again.
client.orders.create(items, idempotency_key=order["idempotency_key"])
```

## Configuration

```python
SupaGamma(
    api_key=None,               # env SUPAGAMMA_API_KEY
    jwt=None,                   # env SUPAGAMMA_JWT — mutually exclusive with api_key
    base_url="https://api.supagamma.com",
    timeout=httpx.Timeout(connect=10, read=300, write=30, pool=10),
    max_retries=3,              # applies only to safe reads
    max_retry_wait_seconds=60,  # refuse to block longer than this on a 429
)
```

Pass `api_key` **or** `jwt`, never both — sending both makes the server silently use the key and ignore the JWT, so the SDK refuses it up front.

## Changes

**0.2.0** (not yet on PyPI)

- `supagamma.backtest`: the prediction-market backtesting harness.
- Typed responses in `supagamma.types`, plus a `py.typed` marker so type
  checkers actually read the SDK's annotations. 0.1.0 advertised
  `Typing :: Typed` without the marker, so mypy and pyright ignored its types
  entirely.
- `client.markets`, `client.trades` and the other namespaces are now visible to
  type checkers. They are attached at runtime, and every call on them used to
  resolve as `Any`.
- `markets.list(tag=...)` now raises `ValueError`. The API removed the tag
  filter on 2026-08-26 and ignores the parameter, so the call had been silently
  returning an unfiltered list.

## Requirements

Python 3.9+. The only runtime dependency is `httpx`.

## Licence

MIT — see [LICENSE](LICENSE).
