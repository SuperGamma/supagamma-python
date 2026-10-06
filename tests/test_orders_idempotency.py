"""``orders.create``: the idempotency key survives a failed call.

The key is the only thing that stops a replayed order charging twice, and the SDK
generates one when you pass none. Before, that generated key was returned only on
success, so after the failures where replaying is the question (a timeout, the 502
``OrderStatusUnknownError``) there was nothing to replay *with*: the documented
``client.orders.create(items, idempotency_key=order["idempotency_key"])`` needs an
``order`` that the failed call never returned. It is now on the exception too.
"""

from __future__ import annotations

import json
from typing import Iterator

import httpx
import pytest
import respx

from supagamma import (
    APITimeoutError,
    AsyncSupaGamma,
    OrderStatusUnknownError,
    ServiceUnavailableError,
    SupaGamma,
    SupaGammaError,
)
from supagamma.resources.orders import OrderItem

KEY = "sg_" + "d" * 32
BASE = "https://api.test"
ITEMS = [OrderItem(data_type="trades", market_id="1254468")]
ORDER = {"order_id": "o-1", "status": "paid", "items": [], "total_cost": 0.0}


def mock_api() -> respx.MockRouter:
    return respx.mock(base_url=BASE, assert_all_called=False)


@pytest.fixture(scope="module")
def client() -> Iterator[SupaGamma]:
    with SupaGamma(api_key=KEY, base_url=BASE, max_retries=3) as shared:
        yield shared


def sent_key(call: respx.models.Call) -> str:
    return json.loads(call.request.content)["idempotency_key"]


def test_the_generated_key_is_on_the_result_as_before(client):
    with mock_api() as router:
        route = router.post("/v1/orders").mock(return_value=httpx.Response(200, json=dict(ORDER)))
        order = client.orders.create(ITEMS)
    assert order["idempotency_key"] == sent_key(route.calls.last)


def test_the_generated_key_is_on_the_exception_when_the_call_fails(client):
    with mock_api() as router:
        route = router.post("/v1/orders").mock(
            return_value=httpx.Response(502, json={"detail": "Order status unknown"})
        )
        with pytest.raises(OrderStatusUnknownError) as raised:
            client.orders.create(ITEMS)
        assert route.call_count == 1  # and it was not retried
    assert raised.value.idempotency_key == sent_key(route.calls.last)
    assert len(raised.value.idempotency_key) == 32


def test_a_failed_order_can_be_replayed_with_the_exception_key(client):
    """The documented recovery, end to end: the first attempt dies, the second is the
    same order under the same key, and the server answers a repeat key with the
    ORIGINAL order."""
    seen = []

    def server(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["idempotency_key"])
        if len(seen) == 1:
            raise httpx.ReadTimeout("read timed out")  # committed server-side; we never heard
        return httpx.Response(200, json=dict(ORDER))

    with mock_api() as router:
        router.post("/v1/orders").mock(side_effect=server)
        with pytest.raises(APITimeoutError) as raised:
            client.orders.create(ITEMS)
        replayed = client.orders.create(ITEMS, idempotency_key=raised.value.idempotency_key)
    assert seen[0] == seen[1] == raised.value.idempotency_key
    assert replayed["order_id"] == "o-1"


def test_a_key_you_supplied_is_the_one_reported_back(client):
    with mock_api() as router:
        router.post("/v1/orders").mock(return_value=httpx.Response(503, json={"detail": "down"}))
        with pytest.raises(ServiceUnavailableError) as raised:
            client.orders.create(ITEMS, idempotency_key="mine-2026-10")
    assert raised.value.idempotency_key == "mine-2026-10"


def test_each_order_without_a_key_gets_a_fresh_one(client):
    keys = []
    with mock_api() as router:
        route = router.post("/v1/orders").mock(return_value=httpx.Response(200, json=dict(ORDER)))
        client.orders.create(ITEMS)
        client.orders.create(ITEMS)
        keys = [sent_key(call) for call in route.calls]
    assert len(set(keys)) == 2


def test_validation_errors_are_raised_before_anything_is_sent(client):
    with mock_api() as router:
        route = router.post("/v1/orders")
        with pytest.raises(ValueError):
            client.orders.create([])
        with pytest.raises(ValueError, match="128"):
            client.orders.create(ITEMS, idempotency_key="k" * 129)
        assert route.call_count == 0


def test_an_ordinary_error_has_no_key():
    assert SupaGammaError("boom").idempotency_key is None


# --- async -----------------------------------------------------------------------------


async def test_async_create_puts_the_key_on_the_exception():
    async with AsyncSupaGamma(api_key=KEY, base_url=BASE, max_retries=3) as aclient:
        with mock_api() as router:
            route = router.post("/v1/orders").mock(
                return_value=httpx.Response(502, json={"detail": "Order status unknown"})
            )
            with pytest.raises(OrderStatusUnknownError) as raised:
                await aclient.orders.create(ITEMS)
            assert route.call_count == 1
    assert raised.value.idempotency_key == sent_key(route.calls.last)


async def test_async_create_keeps_the_key_on_success():
    async with AsyncSupaGamma(api_key=KEY, base_url=BASE, max_retries=3) as aclient:
        with mock_api() as router:
            route = router.post("/v1/orders").mock(
                return_value=httpx.Response(200, json=dict(ORDER))
            )
            order = await aclient.orders.create(ITEMS, idempotency_key="abc")
    assert order["idempotency_key"] == "abc" == sent_key(route.calls.last)
