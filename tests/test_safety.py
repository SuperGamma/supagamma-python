"""The tests that matter most: the ones where being wrong costs the user money.

Everything here runs offline. The builders are pure functions returning
``(method, path, params, policy)``, so the risky decisions — which routes may be
retried, which parameter names go on the wire — are assertable without a server.
"""

from __future__ import annotations

import inspect

import pytest

from supagamma import SupaGamma, SupaGammaConfigError
from supagamma._client import NEVER
from supagamma._errors import (
    FairUseCapError,
    InsufficientCreditsError,
    NoDataInRangeError,
    OriginBlockedError,
    QuotaExceededError,
    RateLimitError,
    parse_error,
)
from supagamma.resources import NAMESPACES, account, billing, download, exports, orders

# --- the 429 split -----------------------------------------------------------


def test_a_string_detail_429_is_a_retryable_rate_limit():
    exc = parse_error(
        status_code=429,
        body={"detail": "Rate limit exceeded. Please slow down."},
        raw_body=None,
        headers={"retry-after": "60"},
        request_id="abc",
    )
    assert isinstance(exc, RateLimitError)
    assert not isinstance(exc, QuotaExceededError)
    assert exc.retry_after == 60.0


def test_an_object_detail_429_is_a_billing_quota_and_is_a_different_class():
    # Same status code, opposite meaning. Retrying this is guaranteed to fail
    # and burns the limiter bucket on top.
    exc = parse_error(
        status_code=429,
        body={"detail": {"code": "fair_use_cap", "message": "Monthly cap reached"}},
        raw_body=None,
        headers={},
        request_id="abc",
    )
    assert isinstance(exc, FairUseCapError)
    assert isinstance(exc, QuotaExceededError)
    assert not isinstance(exc, RateLimitError)
    assert exc.retry_after is None


# --- retry policy on money paths ---------------------------------------------

_MONEY_BUILDERS = [
    ("trades", dict(market_id="1")),
    ("ohlcv", dict(market_id="1")),
    ("orderbook", dict(market_id="1")),
    ("options", dict(series_id="s")),
    ("series", dict(series_id="s")),
]


@pytest.mark.parametrize("name,kwargs", _MONEY_BUILDERS)
def test_every_paid_download_is_never_retried(name, kwargs):
    _, _, _, policy = getattr(download, f"build_{name}")(**kwargs)
    assert policy is NEVER, f"download.{name} must never auto-retry — it spends money"


def test_raw_and_bulk_are_never_retried():
    from datetime import datetime, timezone

    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert download.build_raw(data_type="polymarket_trades", start=now, end=now)[3] is NEVER
    assert download.build_bulk()[3] is NEVER


def test_order_creation_is_never_retried():
    assert orders.build_create()[3] is NEVER


def test_export_creation_is_never_retried():
    # Queues metered work. A replay is only safe BY HAND, with the same key.
    assert exports.build_create()[3] is NEVER


def test_key_provisioning_is_never_retried():
    # A retry loop here is a key storm that can invalidate a key another
    # in-flight request is using.
    assert account.build_provision()[3] is NEVER
    assert account.build_rotate_key("k")[3] is NEVER
    assert account.build_revoke_key("k")[3] is NEVER


def test_the_export_reads_are_retryable():
    # Polling a job and minting its URL are free, and a poll loop that cannot
    # survive one 503 is no use for a job that takes minutes.
    for spec in (exports.build_list(), exports.build_get("j"), exports.build_url("j")):
        assert spec[3] is not NEVER


def test_the_free_estimates_are_retryable():
    # These do not charge, so they may be retried — but they still consume the
    # 10/hour download bucket, which is why they are cached at the call site.
    assert download.build_raw_datasets()[3] is not NEVER
    assert download.build_raw_estimate(data_type="polymarket_trades")[3] is not NEVER


# --- the cost warning ---------------------------------------------------------------


def test_the_cost_warning_gives_the_size_and_does_not_pretend_a_subscriber_pays_per_mb():
    # 100,000 orderbook rows is ~198 MB. That is ~$990 of metering on a pay-as-you-go
    # balance but only fair-use volume on a subscription, which is the production
    # mode, so a bare "$990" would alarm people who will never be charged it.
    with pytest.warns(download.CostWarning) as record:
        download._maybe_warn_cost(100_000, "orderbook")
    message = str(record[0].message)
    assert "198 MB" in message and "$989.91" in message
    assert "fair-use" in message and "metering" in message


def test_a_small_pull_does_not_warn():
    download._maybe_warn_cost(1_000, "trades")  # filterwarnings=error would raise


def test_the_confirm_hook_can_veto_an_expensive_pull_before_it_is_sent():
    seen = []

    def refuse(usd: float) -> bool:
        seen.append(usd)
        return False

    with pytest.raises(RuntimeError, match="cancelled by confirm_cost"):
        download._maybe_warn_cost(100_000, "orderbook", refuse)
    assert seen and 989 < seen[0] < 991


# --- the plan a checkout may ask for ---------------------------------------------


@pytest.mark.parametrize(
    "tier", ["researcher", "academic", "enterprise", "free", "", "Professional"]
)
def test_a_plan_the_api_no_longer_sells_is_rejected_before_any_request(tier):
    # Researcher, Academic and Enterprise were retired server-side; the SDK kept
    # offering "researcher" until it was told otherwise, so a checkout for it went
    # out and came back 422.
    with pytest.raises(ValueError, match="professional"):
        billing.build_subscription_checkout(
            tier=tier,
            billing_period="monthly",
            success_url="https://supagamma.com/ok",
            cancel_url="https://supagamma.com/no",
        )
    with pytest.raises(ValueError, match="professional"):
        billing.build_subscription_redeem(tier=tier, billing_period="annual")


def test_the_one_self_serve_plan_is_accepted_for_both_periods():
    assert billing.SELF_SERVE_TIERS == ("professional",)
    for period in billing.BILLING_PERIODS:
        _, body = billing.build_subscription_redeem(tier="professional", billing_period=period)
        assert body == {"tier": "professional", "billing_period": period}


# --- idempotency --------------------------------------------------------------


def test_order_payload_always_carries_an_idempotency_key():
    payload = orders._create_payload([orders.OrderItem(data_type="trades", market_id="1")], None)
    assert payload["idempotency_key"]
    assert len(payload["idempotency_key"]) <= 128


def test_a_supplied_idempotency_key_is_preserved_verbatim():
    # Reusing the key is the ONLY thing that stops a retry charging twice.
    payload = orders._create_payload(
        [orders.OrderItem(data_type="trades", market_id="1")], "my-key"
    )
    assert payload["idempotency_key"] == "my-key"


def _export_payload(**over):
    spec = dict(
        kind="trades",
        market_id="1",
        series_id=None,
        start=None,
        end=None,
        format="parquet",
        idempotency_key=None,
    )
    spec.update(over)
    return exports._create_payload(**spec)


def test_export_payload_always_carries_an_idempotency_key():
    payload = _export_payload()
    assert payload["idempotency_key"]
    assert len(payload["idempotency_key"]) <= 128


def test_each_export_without_a_supplied_key_gets_its_own():
    # A constant default key would make every export after the first return the
    # FIRST job (the server never compares the request), silently.
    assert _export_payload()["idempotency_key"] != _export_payload()["idempotency_key"]


def test_a_supplied_export_key_is_preserved_verbatim():
    assert _export_payload(idempotency_key="my-key")["idempotency_key"] == "my-key"


def test_an_order_item_without_an_identifier_is_rejected_before_sending():
    with pytest.raises(ValueError, match="market_id or series_id"):
        orders._create_payload([orders.OrderItem(data_type="trades")], None)


# --- client configuration -----------------------------------------------------


def test_sending_both_credentials_is_refused():
    # On the wire this silently resolves to the API key and ignores the JWT.
    with pytest.raises(SupaGammaConfigError, match="not both"):
        SupaGamma(api_key="sg_" + "0" * 32, jwt="ey.some.jwt")


def test_a_malformed_api_key_fails_fast():
    with pytest.raises(SupaGammaConfigError, match="sg_"):
        SupaGamma(api_key="not-a-key")


def test_only_one_auth_header_is_ever_sent():
    client = SupaGamma(api_key="sg_" + "a" * 32)
    headers = client._headers()
    assert "X-API-Key" in headers
    assert "Authorization" not in headers
    assert headers["X-Request-ID"]


def test_none_params_are_dropped_so_server_defaults_apply():
    assert SupaGamma._clean_params({"a": None, "b": 1, "c": True}) == {"b": 1, "c": "true"}


# --- error classification -----------------------------------------------------


def test_a_text_plain_500_does_not_crash_the_parser():
    exc = parse_error(
        status_code=500, body=None, raw_body="Internal Server Error", headers={}, request_id=None
    )
    assert exc.status_code == 500
    assert "Internal Server Error" in str(exc)


def test_origin_block_is_its_own_error_because_it_looks_like_an_auth_failure():
    exc = parse_error(
        status_code=403,
        body={"detail": "Direct origin access is not allowed."},
        raw_body=None,
        headers={},
        request_id=None,
    )
    assert isinstance(exc, OriginBlockedError)


def test_empty_window_404_is_distinguished_from_an_unknown_id():
    # Different remedy: widen the window vs fix the identifier.
    exc = parse_error(
        status_code=404,
        body={"detail": "No trades found"},
        raw_body=None,
        headers={},
        request_id=None,
    )
    assert isinstance(exc, NoDataInRangeError)


def test_insufficient_credits_lifts_the_shortfall():
    exc = parse_error(
        status_code=402,
        body={
            "detail": {
                "code": "insufficient_credits",
                "message": "Not enough credits",
                "required": 12.5,
                "available": 3.25,
                "shortfall": 9.25,
            }
        },
        raw_body=None,
        headers={},
        request_id=None,
    )
    assert isinstance(exc, InsufficientCreditsError)
    assert exc.shortfall == 9.25


def test_the_three_key_insufficient_credits_variant_still_parses():
    # One server path omits available/shortfall entirely.
    exc = parse_error(
        status_code=402,
        body={"detail": {"code": "insufficient_credits", "message": "no", "required": 5.0}},
        raw_body=None,
        headers={},
        request_id=None,
    )
    assert isinstance(exc, InsufficientCreditsError)
    assert exc.available is None


def test_an_unknown_code_still_yields_a_status_error_not_a_keyerror():
    exc = parse_error(
        status_code=418,
        body={"detail": {"code": "brand_new_code", "message": "hi"}},
        raw_body=None,
        headers={},
        request_id=None,
    )
    assert exc.status_code == 418
    assert exc.code == "brand_new_code"


# --- sync/async parity --------------------------------------------------------


@pytest.mark.parametrize("name", sorted(NAMESPACES))
def test_sync_and_async_namespaces_expose_the_same_methods(name):
    sync_cls, async_cls = NAMESPACES[name]

    def public(cls):
        return {
            n
            for n, v in inspect.getmembers(cls)
            if not n.startswith("_") and (inspect.isfunction(v) or inspect.ismethod(v))
        }

    missing = public(sync_cls) - public(async_cls)
    assert not missing, f"{async_cls.__name__} is missing {sorted(missing)}"
