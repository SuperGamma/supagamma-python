"""``client.billing`` — credit balance, checkout, transactions, subscriptions.

Everything under ``/v1/billing``. Four of the eight routes move money or change
account state, so **none of the POSTs on this namespace are ever retried**:

===============================================  =======================
Method                                           Retry policy
===============================================  =======================
``billing.balance()``                            ``SAFE_READ``
``billing.pricing()``                            ``SAFE_READ``
``billing.transactions()``                       ``SAFE_READ``
``billing.subscription.get()``                   ``SAFE_READ``
``billing.checkout()``                           ``NEVER``
``billing.subscription.checkout()``              ``NEVER``
``billing.subscription.redeem()``                ``NEVER``
``billing.subscription.cancel()``                ``NEVER``
===============================================  =======================

None of the four POSTs has an idempotency mechanism — not a body field, not a
header. ``redeem()`` debits the balance atomically the moment it succeeds; the
two ``checkout()`` calls each create a fresh Paddle transaction, so a replay
leaves the customer holding two live payment links.

Two commercial modes
--------------------

The deployment runs in one of two modes, switched by the server-side
``SUBSCRIPTION_ONLY`` env var. **The only runtime signal of which mode you are
in is ``payg_enabled`` on** :meth:`Billing.pricing`, so the SDK exposes it
directly as :meth:`Billing.payg_enabled`.

``payg_enabled=True`` (pay-as-you-go, the default deployment):
    Credit top-ups are on sale. :meth:`Billing.checkout` works, downloads and
    orders debit per-MB from the credit balance, and a subscriber who exceeds
    the soft fair-use threshold falls through to metered overage.

``payg_enabled=False`` (subscription-only):
    * :meth:`Billing.checkout` returns 410 ``payg_retired``
      (:class:`~supagamma.PaygRetiredError`) before it does anything else — it
      is the *only* endpoint in the whole API gated on this flag.
    * Any download or order not covered by an active subscription returns 402
      ``subscription_required`` / ``upgrade_required`` instead of debiting
      credits.
    * A subscriber past the soft fair-use threshold gets 429 ``fair_use_cap``
      (:class:`~supagamma.FairUseCapError`) rather than metered overage.
    * Existing credit balances stay spendable **toward a subscription** via
      :meth:`BillingSubscription.redeem`, which is not gated by the flag.
    * :meth:`BillingSubscription.checkout` is also not gated by the flag and
      works identically in both modes.

The flag is served with ``Cache-Control: public, max-age=300, s-maxage=600``,
so it can read up to ~10 minutes stale through a CDN. Treat it as a UI hint,
not as a guarantee: branch on it to choose what to *offer*, but still handle
``PaygRetiredError`` and ``SubscriptionRequiredError`` at the call site.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
)

from .._client import NEVER, SAFE_READ
from ._base import AsyncResource, Call, SyncResource, call

if TYPE_CHECKING:  # pragma: no cover
    from .._client import AsyncSupaGamma, SupaGamma

__all__ = [
    "BILLING_PERIODS",
    "MAX_CHECKOUT_USD",
    "MIN_CHECKOUT_USD",
    "SELF_SERVE_TIERS",
    "TRANSACTIONS_MAX_LIMIT",
    "TRANSACTIONS_MAX_OFFSET",
    "AsyncBilling",
    "AsyncBillingSubscription",
    "Billing",
    "BillingSubscription",
    "build_balance",
    "build_checkout",
    "build_pricing",
    "build_subscription",
    "build_subscription_cancel",
    "build_subscription_checkout",
    "build_subscription_redeem",
    "build_transactions",
    "payg_enabled",
]

#: The only two tiers accepted as INPUT by ``subscription.checkout()`` and
#: ``subscription.redeem()``. ``subscription.get()`` can still *report* ``free``,
#: ``academic`` or ``enterprise`` — those are granted out of band (academic
#: needs manual verification, enterprise is quote-only), never self-serve.
SELF_SERVE_TIERS: Tuple[str, ...] = ("researcher", "professional")

BILLING_PERIODS: Tuple[str, ...] = ("monthly", "annual")

#: Server bounds on ``POST /v1/billing/checkout``'s ``amount`` (USD).
MIN_CHECKOUT_USD = 10.0
MAX_CHECKOUT_USD = 10_000.0

#: Server bounds on ``GET /v1/billing/transactions``.
TRANSACTIONS_MAX_LIMIT = 500
TRANSACTIONS_MAX_OFFSET = 1_000_000

#: A POST builder returns both the :data:`~supagamma.resources._base.Call` and
#: the JSON body, so a test can assert on the body without a transport.
CallWithBody = Tuple[Call, Dict[str, Any]]


# --- client-side validation ---------------------------------------------------
#
# These mirror server-side validators exactly. They exist to turn a wasted round
# trip (and, for checkout, a wasted slot in the 5/min tier) into an immediate,
# readable error. Where the server's rule is a regex we do NOT mirror it — see
# `build_checkout` on the redirect-URL allow-list.


def _check_tier(tier: str) -> str:
    if tier not in SELF_SERVE_TIERS:
        raise ValueError(
            f"tier must be one of {SELF_SERVE_TIERS}, got {tier!r}. "
            "'academic' and 'enterprise' exist but are not self-serve; the server "
            "rejects them with a 422."
        )
    return tier


def _check_billing_period(billing_period: str) -> str:
    if billing_period not in BILLING_PERIODS:
        raise ValueError(
            f"billing_period must be one of {BILLING_PERIODS}, got {billing_period!r}"
        )
    return billing_period


def _check_amount(amount: float) -> float:
    value = float(amount)
    if not MIN_CHECKOUT_USD <= value <= MAX_CHECKOUT_USD:
        raise ValueError(
            f"amount must be between {MIN_CHECKOUT_USD:.0f} and {MAX_CHECKOUT_USD:.0f} USD, "
            f"got {value!r}"
        )
    return value


def _check_page(limit: int, offset: int) -> Tuple[int, int]:
    if not 1 <= limit <= TRANSACTIONS_MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {TRANSACTIONS_MAX_LIMIT}, got {limit!r}")
    if not 0 <= offset <= TRANSACTIONS_MAX_OFFSET:
        raise ValueError(f"offset must be between 0 and {TRANSACTIONS_MAX_OFFSET}, got {offset!r}")
    return limit, offset


# --- request builders ---------------------------------------------------------


def build_balance() -> Call:
    """``GET /v1/billing/balance`` — free read, JWT or API key."""
    return call("GET", "/v1/billing/balance", None, SAFE_READ)


def build_pricing() -> Call:
    """``GET /v1/billing/pricing`` — free, unauthenticated read."""
    return call("GET", "/v1/billing/pricing", None, SAFE_READ)


def build_transactions(*, limit: int = 50, offset: int = 0) -> Call:
    """``GET /v1/billing/transactions`` — free read.

    Raises ``ValueError`` for a ``limit``/``offset`` the server would 422.
    """
    limit, offset = _check_page(limit, offset)
    return call(
        "GET",
        "/v1/billing/transactions",
        {"limit": limit, "offset": offset},
        SAFE_READ,
    )


def build_checkout(*, amount: float, success_url: str, cancel_url: str) -> CallWithBody:
    """``POST /v1/billing/checkout`` — creates a Paddle transaction. ``NEVER`` retried.

    ``success_url``/``cancel_url`` are checked server-side against an allow-list
    (``http(s)://localhost:<port>``, ``http(s)://<sub>.supagamma.com``,
    ``http(s)://supagamma.com``, each followed by end-of-string, ``/`` or ``?``)
    and 422 otherwise. That regex is deliberately NOT mirrored here: a
    client-side copy that drifted would block a legitimate checkout, which is
    worse than one wasted request.
    """
    body: Dict[str, Any] = {
        "amount": _check_amount(amount),
        "success_url": success_url,
        "cancel_url": cancel_url,
    }
    return call("POST", "/v1/billing/checkout", None, NEVER), body


def build_subscription() -> Call:
    """``GET /v1/billing/subscription`` — free read."""
    return call("GET", "/v1/billing/subscription", None, SAFE_READ)


def build_subscription_checkout(
    *, tier: str, billing_period: str, success_url: str, cancel_url: str
) -> CallWithBody:
    """``POST /v1/billing/subscription/checkout`` — Paddle transaction. ``NEVER`` retried."""
    body: Dict[str, Any] = {
        "tier": _check_tier(tier),
        "billing_period": _check_billing_period(billing_period),
        "success_url": success_url,
        "cancel_url": cancel_url,
    }
    return call("POST", "/v1/billing/subscription/checkout", None, NEVER), body


def build_subscription_redeem(*, tier: str, billing_period: str) -> CallWithBody:
    """``POST /v1/billing/subscription/redeem`` — SPENDS CREDITS. ``NEVER`` retried.

    Requires the ``download`` scope (the only billing route that does).
    """
    body: Dict[str, Any] = {
        "tier": _check_tier(tier),
        "billing_period": _check_billing_period(billing_period),
    }
    return call("POST", "/v1/billing/subscription/redeem", None, NEVER), body


def build_subscription_cancel() -> Call:
    """``POST /v1/billing/subscription/cancel`` — state-changing. ``NEVER`` retried.

    Takes no body; the server reads the caller's identity only.
    """
    return call("POST", "/v1/billing/subscription/cancel", None, NEVER)


# --- helpers ------------------------------------------------------------------


def payg_enabled(pricing: Mapping[str, Any]) -> bool:
    """Read the ``SUBSCRIPTION_ONLY`` discriminator out of a ``pricing()`` body.

    ``False`` means pay-as-you-go is retired on this deployment: credit top-ups
    are gone (410) and uncovered downloads/orders are 402 instead of a debit.
    See the module docstring for the full behavioural diff.

    A body missing the key is read as ``True`` (assume PAYG). The server always
    sends it; guessing "retired" from a malformed body would hide a working
    purchase path from the caller.
    """
    return bool(pricing.get("payg_enabled", True))


# --- sync ---------------------------------------------------------------------


class BillingSubscription(SyncResource):
    """``client.billing.subscription`` — the recurring-plan routes.

    Not gated by ``SUBSCRIPTION_ONLY``: these work identically whether or not
    pay-as-you-go is enabled.
    """

    def get(self) -> Dict[str, Any]:
        """The caller's active subscription, or the free-plan placeholder.

        Free read. Never 404s, and never raises for "no subscription" — the
        inactive branch returns ``{"active": False, "tier": "free",
        "billing_period": None, "status": None, "current_period_end": None}``.

        Two caveats worth reading before you branch on this:

        * **The key set differs between branches.** The active branch adds
          ``cancel_at_period_end``; the inactive branch omits it entirely. Use
          ``result.get("cancel_at_period_end", False)``, never ``[...]``.
        * **``active=False`` is not proof of "no plan".** A subscription-lookup
          failure (DB blip, deploy window) is swallowed server-side and reported
          as no active subscription, because failing that way falls back to the
          paywall rather than granting free access. Only rows with
          ``status='active'`` AND a future ``current_period_end`` are ever
          returned.
        """
        data: Dict[str, Any] = self._json(build_subscription())
        return data

    def checkout(
        self,
        *,
        tier: str,
        billing_period: str,
        success_url: str,
        cancel_url: str,
    ) -> Dict[str, Any]:
        """Start a card checkout for a recurring plan. **Not retried, ever.**

        Money: this creates a live Paddle transaction. Nothing is charged until
        the customer completes the hosted checkout, but there is no idempotency
        key on this route, so calling it twice produces two independent payment
        links and a customer can pay both. Returns ``checkout_id`` (a Paddle
        ``txn_...``), ``checkout_url``, ``tier``, ``billing_period``.

        ``tier`` must be ``"researcher"`` or ``"professional"`` and
        ``billing_period`` ``"monthly"`` or ``"annual"``; anything else raises
        ``ValueError`` here rather than 422-ing at the server.

        Errors: 409 if the caller already has an active subscription (changes go
        through the billing portal, not a second checkout); 400 if the account
        has no email on file; 503 ``"Subscriptions for {tier}/{period} are not
        available yet."`` when that price is not configured — the price map is
        built once at API import, so a tier the pricing page advertises can 503
        here until the API restarts.

        Unlike credit checkout this route is NOT covered by the 5/min checkout
        tier (the prefix match does not reach it); only the global bucket applies.
        """
        spec, body = build_subscription_checkout(
            tier=tier,
            billing_period=billing_period,
            success_url=success_url,
            cancel_url=cancel_url,
        )
        data: Dict[str, Any] = self._json(spec, json=body)
        return data

    def redeem(self, *, tier: str, billing_period: str) -> Dict[str, Any]:
        """Fund a subscription from the existing credit balance. **Not retried, ever.**

        Money: this DEBITS the credit balance immediately and atomically — the
        full local price for the tier (researcher $79/mo, $499/yr; professional
        $399/mo, $2499/yr) comes off the balance in the same transaction that
        creates the subscription. Prices come from the API's own config, not
        from Paddle. The period is a flat 30 or 365 days, not calendar-aligned.

        Requires the ``download`` scope: this is the one billing route a
        narrowed ``["read"]`` API key cannot call (403). A JWT always qualifies.

        The response echoes ``status="active"`` and ``cancel_at_period_end=False``
        unconditionally plus ``funded_by="credits"`` — it reports what was
        requested, not what the database returned.

        Errors: 402 ``insufficient_credits`` whose detail carries only
        ``required`` (no ``available``/``shortfall``, unlike the download path);
        409 (plain string) if a subscription is already active; 422 for a
        non-self-serve tier. A 502 here means the atomic RPC rolled back and
        **you were not charged** — but the SDK's parser maps every 502 to
        ``OrderStatusUnknownError``, so read ``.message`` before deciding: this
        route's 502 says so explicitly and is safe to re-issue by hand.
        """
        spec, body = build_subscription_redeem(tier=tier, billing_period=billing_period)
        data: Dict[str, Any] = self._json(spec, json=body)
        return data

    def cancel(self) -> Dict[str, Any]:
        """Cancel at period end. State-changing, so **not retried, ever**.

        The body is a CONSTANT ``{"status": "canceling", "effective":
        "period_end"}`` and confirms nothing:

        * On a Paddle-funded subscription the cancellation is registered at
          Paddle and our row only flips when the webhook lands, so an immediate
          :meth:`get` still shows ``cancel_at_period_end`` false.
        * On a credit-funded subscription the write's status is not checked, so
          a failed write still returns 200.

        Entitlement runs to ``current_period_end`` either way; nothing is
        refunded. 404 ``"No active subscription."`` if there is nothing to
        cancel. A Paddle SDK failure surfaces here as a raw 500.
        """
        data: Dict[str, Any] = self._json(build_subscription_cancel())
        return data


class Billing(SyncResource):
    """``client.billing`` — balance, credit top-ups, transaction history, plans."""

    def __init__(self, client: SupaGamma) -> None:
        super().__init__(client)
        self.subscription = BillingSubscription(client)
        self._pricing_cache: Optional[Dict[str, Any]] = None

    def balance(self) -> Dict[str, Any]:
        """Current credit balance. Free read.

        Never 404s: an account with no balance row gets an all-zero object, so
        **"new user" and "zero balance" are indistinguishable**. ``currency`` is
        hardcoded ``"USD"`` by the handler and is not a per-account setting.
        """
        data: Dict[str, Any] = self._json(build_balance())
        return data

    def pricing(self, *, refresh: bool = False) -> Dict[str, Any]:
        """Public pricing table + the ``payg_enabled`` mode flag. Free, unauthenticated.

        Cached for the life of this client (pass ``refresh=True`` to re-fetch),
        because it changes only on an API deploy and is served with a 5-minute
        browser / 10-minute CDN cache anyway.

        Returns ``payg_enabled``, ``credit_pricing`` (``minimum_purchase``,
        ``currency``, ``presets``) and ``usage_pricing`` (``rate_per_mb``,
        ``bytes_per_row``, ``free_tier_mb``, ``minimum_order_usd``,
        ``currency``). Treat the two rate maps as OPEN dicts keyed by data-type
        strings — the set grows as streams are added, and unknown keys fall back
        server-side to 230 B/row at $1.00/MB.
        """
        if refresh or self._pricing_cache is None:
            data: Dict[str, Any] = self._json(build_pricing())
            self._pricing_cache = data
        return self._pricing_cache

    def payg_enabled(self, *, refresh: bool = False) -> bool:
        """Whether pay-as-you-go is live on this deployment (cached).

        ``False`` means the server runs in ``SUBSCRIPTION_ONLY`` mode:
        :meth:`checkout` will 410 ``payg_retired``, and any download or order not
        covered by an active subscription will 402 ``subscription_required`` /
        ``upgrade_required`` instead of debiting credits. Existing credits can
        still be spent through :meth:`BillingSubscription.redeem`.

        This is the only runtime signal of that mode. It can be up to ~10 minutes
        stale through a CDN, so branch on it for what you *offer* and still
        handle ``PaygRetiredError``/``SubscriptionRequiredError`` at the call site.
        """
        return payg_enabled(self.pricing(refresh=refresh))

    def checkout(self, *, amount: float, success_url: str, cancel_url: str) -> Dict[str, Any]:
        """Create a Paddle checkout to buy credits. **Not retried, ever.**

        Money: no charge happens here, but this mints a live payment link and the
        route has no idempotency key, so a replay leaves the customer with two
        payable checkouts. That is why this method is ``NEVER`` retried even when
        the client is configured with ``max_retries``.

        ``amount`` is 10..10000 USD (``ValueError`` client-side outside that) and
        the ``amount`` echoed back is your request, not what the customer pays —
        Paddle adds tax on top. ``checkout_id`` is a Paddle *transaction* id
        (``txn_...``), not a subscription id. Credits land only when the
        ``transaction.completed`` webhook is processed, so poll :meth:`balance`
        rather than assuming success at redirect time.

        This is the one endpoint gated by ``SUBSCRIPTION_ONLY``: with
        :meth:`payg_enabled` false it raises
        :class:`~supagamma.PaygRetiredError` (410) before doing anything else.

        Rate limited to 5/min on top of the global bucket.
        """
        spec, body = build_checkout(
            amount=amount, success_url=success_url, cancel_url=cancel_url
        )
        data: Dict[str, Any] = self._json(spec, json=body)
        return data

    def transactions(self, *, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """One page of transaction history, newest first. Free read.

        Returns a bare list — no envelope, no total, no ``has_more``. The only
        end-of-data signal is a short page.

        This endpoint can 500 on legitimate data: ``description``, ``type``,
        ``currency`` and ``status`` are non-optional in the server's response
        model, so one NULL row anywhere in the page fails serialization for the
        whole page. Narrowing the window is not possible here — there is no date
        filter — so a persistent 500 needs support, not a retry.
        """
        rows: List[Dict[str, Any]] = self._json(build_transactions(limit=limit, offset=offset))
        return rows

    def iter_transactions(
        self, *, limit: int = 50, offset: int = 0
    ) -> Iterator[Dict[str, Any]]:
        """Iterate transactions across pages, stopping on the first short page.

        Free reads, but offset pagination over a ``created_at DESC`` list is
        inherently unstable: rows landing mid-iteration shift the window and can
        make you see a row twice or miss one. Stops at the server's offset
        ceiling (1 000 000) rather than letting it 422.
        """
        _check_page(limit, offset)
        while True:
            page = self.transactions(limit=limit, offset=offset)
            yield from page
            if len(page) < limit:
                return
            offset += limit
            if offset > TRANSACTIONS_MAX_OFFSET:
                return


# --- async --------------------------------------------------------------------


class AsyncBillingSubscription(AsyncResource):
    """Async twin of :class:`BillingSubscription`."""

    async def get(self) -> Dict[str, Any]:
        """The caller's active subscription, or the free-plan placeholder.

        See :meth:`BillingSubscription.get` — the inactive branch omits
        ``cancel_at_period_end`` entirely, and ``active=False`` can mean a
        swallowed lookup failure rather than "no plan".
        """
        data: Dict[str, Any] = await self._json(build_subscription())
        return data

    async def checkout(
        self,
        *,
        tier: str,
        billing_period: str,
        success_url: str,
        cancel_url: str,
    ) -> Dict[str, Any]:
        """Start a card checkout for a recurring plan. **Not retried, ever.**

        See :meth:`BillingSubscription.checkout`. Creates a live Paddle
        transaction with no idempotency key; 409 if a subscription is already
        active; 503 when that tier/period price is not configured on the server.
        """
        spec, body = build_subscription_checkout(
            tier=tier,
            billing_period=billing_period,
            success_url=success_url,
            cancel_url=cancel_url,
        )
        data: Dict[str, Any] = await self._json(spec, json=body)
        return data

    async def redeem(self, *, tier: str, billing_period: str) -> Dict[str, Any]:
        """Fund a subscription from the credit balance. **Not retried, ever.**

        See :meth:`BillingSubscription.redeem`. DEBITS the balance atomically,
        needs the ``download`` scope, and its 402 detail carries only
        ``required``.
        """
        spec, body = build_subscription_redeem(tier=tier, billing_period=billing_period)
        data: Dict[str, Any] = await self._json(spec, json=body)
        return data

    async def cancel(self) -> Dict[str, Any]:
        """Cancel at period end. State-changing, so **not retried, ever**.

        See :meth:`BillingSubscription.cancel` — the 200 body is a constant and
        confirms nothing about whether the row actually flipped.
        """
        data: Dict[str, Any] = await self._json(build_subscription_cancel())
        return data


class AsyncBilling(AsyncResource):
    """Async twin of :class:`Billing`."""

    def __init__(self, client: AsyncSupaGamma) -> None:
        super().__init__(client)
        self.subscription = AsyncBillingSubscription(client)
        self._pricing_cache: Optional[Dict[str, Any]] = None

    async def balance(self) -> Dict[str, Any]:
        """Current credit balance. Free read.

        Never 404s — a missing balance row returns all zeros, so "new user" and
        "zero balance" are indistinguishable.
        """
        data: Dict[str, Any] = await self._json(build_balance())
        return data

    async def pricing(self, *, refresh: bool = False) -> Dict[str, Any]:
        """Public pricing table + the ``payg_enabled`` mode flag (cached).

        See :meth:`Billing.pricing`. The two rate maps are open dicts keyed by
        data-type strings, not enums.
        """
        if refresh or self._pricing_cache is None:
            data: Dict[str, Any] = await self._json(build_pricing())
            self._pricing_cache = data
        return self._pricing_cache

    async def payg_enabled(self, *, refresh: bool = False) -> bool:
        """Whether pay-as-you-go is live on this deployment (cached).

        See :meth:`Billing.payg_enabled` for what flips when it is ``False``.
        """
        return payg_enabled(await self.pricing(refresh=refresh))

    async def checkout(
        self, *, amount: float, success_url: str, cancel_url: str
    ) -> Dict[str, Any]:
        """Create a Paddle checkout to buy credits. **Not retried, ever.**

        See :meth:`Billing.checkout`. No idempotency key, 5/min tier, and the
        only route gated by ``SUBSCRIPTION_ONLY`` (410 ``payg_retired``).
        """
        spec, body = build_checkout(
            amount=amount, success_url=success_url, cancel_url=cancel_url
        )
        data: Dict[str, Any] = await self._json(spec, json=body)
        return data

    async def transactions(self, *, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """One page of transaction history, newest first. Free read.

        Bare list, no total. A single NULL in ``description``/``type``/
        ``currency``/``status`` 500s the whole page server-side.
        """
        rows: List[Dict[str, Any]] = await self._json(
            build_transactions(limit=limit, offset=offset)
        )
        return rows

    async def iter_transactions(
        self, *, limit: int = 50, offset: int = 0
    ) -> AsyncIterator[Dict[str, Any]]:
        """Iterate transactions across pages, stopping on the first short page.

        See :meth:`Billing.iter_transactions` — offset pagination over a
        ``created_at DESC`` list is unstable while new rows are landing.
        """
        _check_page(limit, offset)
        while True:
            page = await self.transactions(limit=limit, offset=offset)
            for row in page:
                yield row
            if len(page) < limit:
                return
            offset += limit
            if offset > TRANSACTIONS_MAX_OFFSET:
                return
