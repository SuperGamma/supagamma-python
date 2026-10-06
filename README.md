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

Create one in your [dashboard](https://supagamma.com/dashboard/api-keys). Keys look like `sg_` followed by 32 hex characters, and carry scopes — `read` for browsing, `download` for anything that pulls data (`client.download` and `client.exports`). The SDK validates the format locally so a typo fails immediately instead of costing a round trip.

## What's here

| Namespace | What it does |
| --- | --- |
| `client.markets` | Market catalogue, stats, cost estimates |
| `client.trades` | Raw fills, OHLCV bars, recent trades |
| `client.series` | The stream catalogue and its estimates |
| `client.download` | **Metered data delivery** — see below |
| `client.exports` | Background exports for anything large: submit, poll, signed URL |
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

By default `resolved_markets` prices each market from the free market record, and on
a *resolved* market that is often the settlement price (about 0 or 1), which is
useless for calibration. For a real study, price at a horizon from the trade tape:

```python
from supagamma.backtest.data import vwap_price_fn

# needs pyarrow:  pip install "supagamma[parquet]"
universe = list(resolved_markets(client, max_markets=200, price_fn=vwap_price_fn(horizon_hours=24)))
```

This is **metered**: every priced market is one `download.trades` request, so it
counts toward your plan's fair-use volume (or debits a pay-as-you-go balance).
With a `price_fn`, `max_markets` bounds the *paid attempts*, not the markets
yielded: `max_markets=200` makes at most 200 requests even if some markets are
skipped, and raise `max_price_attempts` to look further. `price_fn` with neither
bound is refused. A missing `pyarrow` raises `MissingDependencyError` before any
request is made, and any error other than "no trades before the cutoff" stops the
run rather than carrying on paying.

A trade file holds the fills of **both** of a market's outcome tokens, and they
price each other (YES at 0.80 is NO at 0.20). The helper converts NO fills to the
YES scale (`1 - price`) before it takes the size-weighted average; averaging the
raw column would put every market near 0.5. It needs the `outcome_token_ids` field
on market records, and raises before spending anything if the API does not send it.
`yes_vwap(trades, yes_token=..., no_token=...)` is the same calculation on rows you
already have.

The strategy only ever sees a `MarketView` with no outcome field, so it *cannot*
peek at the answer — look-ahead safety is enforced, not trusted. Write your own
by returning an `Order(side, stake)` (or `None`) from any callable. A full worked
example lives in [`examples/calibration.py`](examples/calibration.py).

It's a research tool, not investment advice, and makes no performance promise:
it shows what *did* happen in historical data. Fees, liquidity, and slippage make
live results different.

## Big pulls: stream to disk, or export

A download is read into memory by default (`result.content`), which is fine for a day of trades and not for a month of orderbook. Pass `save_to=` and the body is written to disk as it arrives, so memory stays flat whatever the size:

```python
file = client.download.trades(
    market_id="1254468", start=start, end=end, format="parquet", save_to="trades.parquet"
)
print(file.path, file.size)        # a DownloadedFile, which is also os.PathLike
```

The write is all-or-nothing: bytes land in a temporary file that replaces the target only once the body is complete, a body shorter than the server declared is an error and not a file that looks fine, and the filename the server suggests can never choose the directory (`save_to` may be an existing directory). A streamed request that costs something is still never retried.

Anything that could outrun the ~100 seconds a synchronous request gets should be an export. It builds the file in the background and hands back a short-lived signed URL, which you can re-issue free until the file expires (7 days):

```python
job = client.exports.create(kind="trades", market_id="1254468", format="parquet")
job = client.exports.wait(job["id"])                   # polls, backing off, until it succeeds
client.exports.download(job["id"], "trades.parquet")   # streams it to disk
```

`exports.wait` raises `ExportFailedError` if the job ends `failed` or `expired` (`.job["error"]` says why) and `ExportTimeoutError` if you run out of patience first; the latter cancels nothing. The request that fetches the signed URL carries none of your credentials. `exports.list()` and `exports.get(job_id)` are free reads.

`exports.create` is never retried automatically, but it is the one metered POST that is safe to replay *by hand*: it always carries an `idempotency_key` (a body field, generated if you pass none), and a repeat returns the same job instead of building it twice. If the call fails, the key is on the exception:

```python
try:
    job = client.exports.create(kind="trades", market_id="1254468")
except (supagamma.APIConnectionError, supagamma.OrderStatusUnknownError) as exc:
    # a timeout, a dropped connection or a 502: the job may or may not exist
    job = client.exports.create(
        kind="trades", market_id="1254468", idempotency_key=exc.idempotency_key
    )
```

Never reuse a key for a different pull: the server returns the old job for a known key without comparing the request.

## Four things worth knowing

These are properties of the API, not of this library, and the SDK surfaces them rather than hiding them.

### Downloads are metered, so they are never retried

Every `client.download.*` call except the two estimates is metered. On a subscription (the production mode, which `client.billing.payg_enabled()` reports as `False`) a pull counts toward your plan's monthly fair-use volume and fails with a 402 or 429 once you are out of allowance; on a pay-as-you-go deployment it debits your balance. The SDK sets a no-retry policy on those routes regardless of how you configure `max_retries`.

The reason is specific. The server meters *after* serialising your data but *before* the body finishes arriving, and the only protection against paying twice is a 7-day entitlement waiver matched on an exact parameter tuple. A retry that re-derives `end=datetime.now()` looks like a *different* request to that matcher and is metered again in full. If you retry a download yourself, freeze your parameters first and replay them byte-identically:

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

### Orderbook data is large

At roughly 2 KB per row, the default 100,000-row orderbook pull is about **200 MB**. That is a small slice of a subscription's monthly fair-use volume, but on a pay-as-you-go balance it is roughly \$990 at the \$5/MB metering rate (the per-MB rates are reference figures on a subscription, where nothing is billed per MB). The SDK warns when the metered estimate passes \$25 and lets you gate it:

```python
client.download.confirm_cost = lambda usd: usd < 50    # abort anything above $50 of metering
```

## Errors

Everything derives from `supagamma.SupaGammaError`. The ones you will actually branch on:

| Exception | Meaning |
| --- | --- |
| `InsufficientCreditsError` | 402 — pay-as-you-go deployments only; `.shortfall` is how much is missing |
| `SubscriptionRequiredError` / `UpgradeRequiredError` | 402 — plan doesn't cover this |
| `RateLimitError` | 429 — transient, honour `.retry_after` |
| `QuotaExceededError` | 429 — billing cap, do not retry |
| `NoDataInRangeError` | 404 — the id is fine, the window is empty |
| `OrderStatusUnknownError` | 502 on order creation — replay with `exc.idempotency_key` |
| `OriginBlockedError` | 403 — you pointed `base_url` at the origin, not the API |
| `ExportFailedError` / `ExportTimeoutError` | `exports.wait` — the job failed or expired / you stopped waiting (`.job` has the last state) |
| `MissingDependencyError` | an optional dependency (`pyarrow`) is not installed; raised before any request |

Every exception that came from an HTTP response carries `.status_code`, `.code`, `.request_id` and the raw `.detail`. Quote `request_id` to support; it is the only correlation handle. An error from a call that carried an idempotency key (`orders.create`, `exports.create`) also has `.idempotency_key`.

## Orders and idempotency

`client.orders.create()` is the one route with real idempotency protection, and it is a **body field**, not an `Idempotency-Key` header. The SDK generates a key for you and returns it:

```python
order = client.orders.create([
    supagamma.resources.orders.OrderItem(data_type="trades", market_id="546814"),
])

# On an ambiguous failure, replay with the SAME key — the server returns the
# original order instead of charging again. The key is on the result, and, since
# a failed call has no result, on the exception too:
try:
    order = client.orders.create(items)
except (supagamma.APIConnectionError, supagamma.OrderStatusUnknownError) as exc:
    order = client.orders.create(items, idempotency_key=exc.idempotency_key)
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

**0.3.0**

- `client.exports`: the documented `/v1/exports` workflow, which the SDK had no
  way to call. `create` (never retried automatically, always carries an
  `idempotency_key`), `list`, `get`, `url`, `wait` (polls with back-off and
  raises `ExportFailedError` / `ExportTimeoutError`) and `download` (streams the
  signed URL to disk, sending none of your credentials).
- Downloads can stream to disk. Every `client.download.*` method takes
  `save_to=` and returns a `DownloadedFile`; `download.bulk(..., save_to=)`
  returns a `BulkFile`. Memory stays flat, the write is all-or-nothing, and a
  server-supplied filename cannot choose the directory. `client.request` used to
  accept a `stream` argument and ignore it, so every download was buffered whole.
- `orders.create` and `exports.create` put the idempotency key on the exception
  (`exc.idempotency_key`) when the call fails. A key the SDK generated was lost
  with the failed call, so the documented replay was impossible in exactly the
  cases it exists for.
- `backtest.data.vwap_price_fn` and `resolved_markets`:
  - the VWAP is now computed on the YES scale. A trade file holds the fills of
    both outcome tokens, and the old code averaged their prices together, which
    lands every market near 0.5. It reads the new `outcome_token_ids` field of a
    market record, and refuses (before spending) if the API does not send it.
  - `pyarrow` is checked before any request is made. It used to be checked after
    the paid download, so a file was paid for and then could not be read. A new
    `supagamma[parquet]` extra installs it.
  - only "no trades before the cutoff" skips a market. Every other exception was
    swallowed, so an expired key or a spent quota carried on to the next request.
  - with a `price_fn`, `max_markets` bounds the paid *attempts* (and there is a
    new `max_price_attempts`). It counted markets yielded, so markets that were
    skipped could make as many paid requests as the catalogue has resolved markets.
- `markets.list` and `markets.auto_paginate` take `series_id` (every market in
  a series, such as `polymarket:btc-15m`), `ending_after` and `ending_before`,
  and accept `sort_by="end_date"`. `sort_by` is now unset by default, which
  lets the server choose (`top`, or `end_date` for a series); the SDK used to
  send `top` on every call, which would have overridden the series default.
- `billing.subscription.checkout` / `redeem` accept `tier="professional"` only,
  as the API does since Researcher, Academic and Enterprise were retired, and the
  documented prices are current.
- The cost warning states the size in MB and that a subscription counts it
  toward fair use instead of billing per MB.
- Typed responses follow the API again: `Market.outcome_token_ids`,
  `Trade.outcome_label` / `collateral`, the plan fields on balances and
  estimates, and `Trade.outcome` / `OHLCVBar.outcome` are `Optional`.
- `trades.ohlcv` no longer warns about `5m`/`15m`/`4h`: the API aggregates them
  since 2026-09-29. `FakeTimeframeWarning` stays importable but is never emitted.
- The vendored OpenAPI snapshot behind `tests/test_types_contract.py` had fallen
  21 contract differences behind the API. It is refreshed, and
  `tests/spec_drift.py` compares it with the API's current spec so that cannot
  happen silently again (see Development).

**0.2.0**

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

## Development

```bash
pip install -e ".[dev]"
pytest                                   # offline: no key, no network
ruff check . && ruff format --check . && mypy
```

### Keeping the contract tests honest

`tests/test_types_contract.py` pins every typed model and every query parameter
to `tests/fixtures/openapi.json`, a copy of the API's spec. A copy goes stale, and
nothing used to notice, so compare it with the API's current spec:

```bash
python tests/spec_drift.py --url https://api.supagamma.com/openapi.json    # exit 1 on drift
python tests/spec_drift.py --url ... --update                              # rewrite the snapshot
```

Maintainers with a checkout of the API can compare against it instead, which is
offline, deterministic and needs no credentials (it imports the app and calls
`app.openapi()`; the interpreter needs the API's requirements):

```bash
python tests/spec_drift.py --monorepo ../supagamma --python path/to/api-venv/python
SUPAGAMMA_MONOREPO=../supagamma pytest tests/test_spec_drift.py   # the same check, as a test
```

Only the contract is compared (operations, parameters, schemas and their types);
descriptions, titles and examples are ignored. After `--update`, fix whatever
`tests/test_types_contract.py` then flags.

## Requirements

Python 3.9+. The only runtime dependency is `httpx`; `pyarrow` is optional
(`supagamma[parquet]`) and `pandas` is optional (`supagamma[pandas]`).

## Licence

MIT — see [LICENSE](LICENSE).
