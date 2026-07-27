"""Exception hierarchy and the wire-error parser.

The API does not use a uniform error envelope. `detail` arrives as one of four
shapes, and **the shape is not predictable from the status code**:

    1. a string   {"detail": "Market not found"}
    2. an object  {"detail": {"code": "insufficient_credits", "required": 12.5, ...}}
    3. a list     {"detail": [{"type","loc","msg", ...}]}          (FastAPI validation)
    4. not JSON   HTTP 500, text/plain, "Internal Server Error"

So 422 may be a list, a string, or an object; 400 may be a string or an object
for the same logical check; and 429 may be either — which matters more than any
of the rest, because the two 429s mean opposite things:

    string detail  -> the rate limiter. Transient. Retry after `Retry-After`.
    object detail  -> a BILLING QUOTA (fair_use_cap / free_tier_cap). It clears
                      on a billing-period or month boundary, carries no
                      Retry-After, and retrying it is guaranteed to fail while
                      burning the limiter bucket on top.

Conflating those two is the single easiest way to write an SDK that quietly
hammers a capped account, so they are different classes: `RateLimitError` and
`QuotaExceededError`, and only the former is retryable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

__all__ = [
    "SupaGammaError",
    "SupaGammaConfigError",
    "APIConnectionError",
    "APITimeoutError",
    "APIStatusError",
    "BadRequestError",
    "RawWindowRequiredError",
    "FreeTierWindowError",
    "SeriesComingSoonError",
    "WrongDataTypeError",
    "MissingIdentifierError",
    "EmptyLineItemError",
    "BelowMinimumOrderError",
    "AuthenticationError",
    "PermissionDeniedError",
    "OriginBlockedError",
    "PaymentRequiredError",
    "InsufficientCreditsError",
    "SubscriptionRequiredError",
    "UpgradeRequiredError",
    "NotFoundError",
    "NoDataInRangeError",
    "SeriesEmptyRangeError",
    "ConflictError",
    "GoneError",
    "PaygRetiredError",
    "PayloadTooLargeError",
    "BundleTooLargeError",
    "ValidationError",
    "InvalidWindowError",
    "RateLimitError",
    "QuotaExceededError",
    "FairUseCapError",
    "FreeTierCapError",
    "ServerError",
    "OrderStatusUnknownError",
    "ServiceUnavailableError",
]


class SupaGammaError(Exception):
    """Base class for everything this SDK raises."""


class SupaGammaConfigError(SupaGammaError):
    """Client-side misuse, caught before any request leaves the process."""


class APIConnectionError(SupaGammaError):
    """The request never produced an HTTP response (DNS, TLS, refused, reset)."""

    def __init__(self, message: str, *, request_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.request_id = request_id


class APITimeoutError(APIConnectionError):
    """Connect/read/write timeout.

    On a download this is genuinely ambiguous: the debit happens after
    serialization but before the body finishes reaching you, so a read timeout
    mid-transfer can mean you were charged and got nothing. Replaying the
    *byte-identical* request is what recovers it, via the 7-day entitlement
    waiver — see `download` docs.
    """


class APIStatusError(SupaGammaError):
    """The server returned a non-2xx response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        code: Optional[str] = None,
        detail: Any = None,
        request_id: Optional[str] = None,
        raw_body: Optional[str] = None,
        headers: Optional[Mapping[str, str]] = None,
        retry_after: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.detail = detail
        self.request_id = request_id
        self.raw_body = raw_body
        self.headers: Dict[str, str] = dict(headers or {})
        self.retry_after = retry_after

    def __str__(self) -> str:
        bits = [str(self.status_code)]
        if self.code:
            bits.append(self.code)
        out = f"{' '.join(bits)}: {self.message}"
        if self.request_id:
            out += f" (request_id={self.request_id})"
        return out


# --- 400 ---------------------------------------------------------------------


class BadRequestError(APIStatusError):
    pass


class RawWindowRequiredError(BadRequestError):
    """Raw L2 pulls must be explicitly time-bounded; there is no implicit window."""


class FreeTierWindowError(BadRequestError):
    pass


class SeriesComingSoonError(BadRequestError):
    def __init__(self, *args: Any, series_id: Optional[str] = None, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.series_id = series_id


class WrongDataTypeError(BadRequestError):
    """The data type asked for doesn't exist for this market/series/venue."""

    def __init__(self, *args: Any, item_index: Optional[int] = None, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.item_index = item_index


class MissingIdentifierError(BadRequestError):
    def __init__(self, *args: Any, item_index: Optional[int] = None, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.item_index = item_index


class EmptyLineItemError(BadRequestError):
    def __init__(self, *args: Any, empty_items: Optional[List[Any]] = None, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.empty_items = empty_items or []


class BelowMinimumOrderError(BadRequestError):
    def __init__(
        self,
        *args: Any,
        minimum: Optional[float] = None,
        total: Optional[float] = None,
        shortfall: Optional[float] = None,
        **kw: Any,
    ) -> None:
        super().__init__(*args, **kw)
        self.minimum = minimum
        self.total = total
        self.shortfall = shortfall


# --- 401 / 403 ---------------------------------------------------------------


class AuthenticationError(APIStatusError):
    """401. Not retryable — a rejected key stays rejected."""


class PermissionDeniedError(APIStatusError):
    """403. The credential is valid but lacks the scope."""


class OriginBlockedError(PermissionDeniedError):
    """You reached the Railway origin directly instead of api.supagamma.com.

    Raised for the exact "Direct origin access is not allowed." response, which
    is returned before auth or rate limiting and is otherwise very confusing —
    a valid key appears to fail with 403.
    """


# --- 402 ---------------------------------------------------------------------


class PaymentRequiredError(APIStatusError):
    """Never retryable. The caller has to act (top up, subscribe, upgrade)."""


class InsufficientCreditsError(PaymentRequiredError):
    def __init__(
        self,
        *args: Any,
        required: Optional[float] = None,
        available: Optional[float] = None,
        shortfall: Optional[float] = None,
        **kw: Any,
    ) -> None:
        super().__init__(*args, **kw)
        self.required = required
        # Optional on purpose: one of the three server variants omits both.
        self.available = available
        self.shortfall = shortfall


class SubscriptionRequiredError(PaymentRequiredError):
    pass


class UpgradeRequiredError(PaymentRequiredError):
    """Your plan doesn't include this product; a higher tier does."""


# --- 404 / 409 / 410 / 413 ---------------------------------------------------


class NotFoundError(APIStatusError):
    pass


class NoDataInRangeError(NotFoundError):
    """The market/series exists, but holds nothing in the window you asked for.

    Distinguished from a genuinely unknown id because the remedy is different:
    widen the window rather than fix the identifier.
    """


class SeriesEmptyRangeError(NoDataInRangeError):
    def __init__(self, *args: Any, series_id: Optional[str] = None, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.series_id = series_id


class ConflictError(APIStatusError):
    pass


class GoneError(APIStatusError):
    pass


class PaygRetiredError(GoneError):
    """Pay-as-you-go is retired on this deployment; subscribe instead."""


class PayloadTooLargeError(APIStatusError):
    pass


class BundleTooLargeError(PayloadTooLargeError):
    pass


# --- 422 ---------------------------------------------------------------------


class ValidationError(APIStatusError):
    """422. `errors` is populated only for FastAPI's list-shaped body."""

    def __init__(self, *args: Any, errors: Optional[List[Any]] = None, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.errors = errors or []


class InvalidWindowError(ValidationError):
    pass


# --- 429: two different things ------------------------------------------------


class RateLimitError(APIStatusError):
    """The rate limiter. Transient and retryable — honour `retry_after`.

    `retry_after` is precise (the server uses fixed windows), so wait exactly
    that long rather than substituting exponential backoff. It can be up to
    ~3599s on the hourly download tier.
    """


class QuotaExceededError(APIStatusError):
    """A billing quota, not a rate limit. NEVER retry this.

    Shares status 429 with `RateLimitError` but means the opposite: it clears on
    a billing-period or month boundary, not after a delay, and carries no
    Retry-After.
    """


class FairUseCapError(QuotaExceededError):
    """Monthly served-volume cap for your subscription tier."""


class FreeTierCapError(QuotaExceededError):
    """Free-plan monthly export cap. Resets at the start of next month."""


# --- 5xx ---------------------------------------------------------------------


class ServerError(APIStatusError):
    pass


class OrderStatusUnknownError(ServerError):
    """502 from POST /v1/orders — the order MAY have been committed.

    Retry only by replaying the same `idempotency_key`; retrying without one can
    charge twice. Safest is to check `client.orders.list()` first.
    """


class ServiceUnavailableError(ServerError):
    """503. All server sources of this are explicitly transient — retryable."""


# --- parsing ------------------------------------------------------------------

# code -> exception class. Dispatch on `detail.code` FIRST, since it is the only
# stable machine-readable discriminator the API offers.
_BY_CODE: Dict[str, type] = {
    "insufficient_credits": InsufficientCreditsError,
    "subscription_required": SubscriptionRequiredError,
    "upgrade_required": UpgradeRequiredError,
    "fair_use_cap": FairUseCapError,
    "free_tier_cap": FreeTierCapError,
    "payg_retired": PaygRetiredError,
    "raw_window_required": RawWindowRequiredError,
    "invalid_window": InvalidWindowError,
    "bundle_too_large": BundleTooLargeError,
    "series_coming_soon": SeriesComingSoonError,
    "series_empty_range": SeriesEmptyRangeError,
    "free_tier_window": FreeTierWindowError,
    "below_minimum_order": BelowMinimumOrderError,
    "empty_line_item": EmptyLineItemError,
    "wrong_data_type": WrongDataTypeError,
    "greeks_unavailable": WrongDataTypeError,
    "options_data_type_unavailable": WrongDataTypeError,
    "not_options_series": WrongDataTypeError,
    "not_series": WrongDataTypeError,
    "unresolved_data_type": WrongDataTypeError,
    "unknown_raw_data_type": WrongDataTypeError,
    "normalized_unavailable": WrongDataTypeError,
    "invalid_timeframe": WrongDataTypeError,
    "missing_market_id": MissingIdentifierError,
    "missing_series_id": MissingIdentifierError,
}

_BY_STATUS: Dict[int, type] = {
    400: BadRequestError,
    401: AuthenticationError,
    402: PaymentRequiredError,
    403: PermissionDeniedError,
    404: NotFoundError,
    409: ConflictError,
    410: GoneError,
    413: PayloadTooLargeError,
    422: ValidationError,
    500: ServerError,
    501: ServerError,
    502: ServerError,
    503: ServiceUnavailableError,
}

# Substrings that identify a "nothing in this window" 404. The server returns
# several phrasings, all string-detail, none carrying a code.
_NO_DATA_MARKERS = (
    "no trades found",
    "no ohlcv data found",
    "no orderbook snapshots",
    "no priced top-of-book",
    "no normalizable",
    "in this range",
)


def _retry_after(headers: Mapping[str, str]) -> Optional[float]:
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        # RFC 7231 also allows an HTTP-date. We don't parse it; callers fall
        # back to their own backoff rather than crashing on an exotic value.
        return None


def _kwargs_from_detail(code: Optional[str], detail: Any) -> Dict[str, Any]:
    """Lift the extra fields a typed subclass exposes off the detail object."""
    if not isinstance(detail, dict):
        return {}
    fields = {
        "insufficient_credits": ("required", "available", "shortfall"),
        "below_minimum_order": ("minimum", "total", "shortfall"),
        "series_coming_soon": ("series_id",),
        "series_empty_range": ("series_id",),
        "empty_line_item": ("empty_items",),
    }.get(code or "", ())
    out = {k: detail.get(k) for k in fields if k in detail}
    if "item_index" in detail and code in {
        "wrong_data_type",
        "greeks_unavailable",
        "options_data_type_unavailable",
        "not_options_series",
        "not_series",
        "unresolved_data_type",
        "unknown_raw_data_type",
        "normalized_unavailable",
        "invalid_timeframe",
        "missing_market_id",
        "missing_series_id",
    }:
        out["item_index"] = detail["item_index"]
    return out


def parse_error(
    *,
    status_code: int,
    body: Any,
    raw_body: Optional[str],
    headers: Mapping[str, str],
    request_id: Optional[str],
) -> APIStatusError:
    """Turn a non-2xx response into the most specific exception we can justify.

    Never raises. An unrecognised shape still yields an `APIStatusError` rather
    than a KeyError/AttributeError from the parser itself.
    """
    detail = body.get("detail") if isinstance(body, dict) else None
    code = detail.get("code") if isinstance(detail, dict) else None

    if isinstance(detail, dict):
        message = str(detail.get("message") or detail.get("detail") or code or "API error")
    elif isinstance(detail, str):
        message = detail
    elif isinstance(detail, list):
        message = "Request validation failed"
    else:
        # Shape 4, or an unrecognised JSON body.
        message = (raw_body or "").strip() or f"HTTP {status_code}"

    cls: Optional[type] = _BY_CODE.get(code) if code else None

    if cls is None:
        if status_code == 429:
            # THE split. A dict detail is a billing quota; a string is the
            # limiter. Only the latter may be retried.
            cls = QuotaExceededError if isinstance(detail, dict) else RateLimitError
        elif (
            status_code == 403
            and isinstance(detail, str)
            and "direct origin access" in detail.lower()
        ):
            cls = OriginBlockedError
        elif (
            status_code == 404
            and isinstance(detail, str)
            and any(m in detail.lower() for m in _NO_DATA_MARKERS)
        ):
            cls = NoDataInRangeError
        elif status_code == 502:
            cls = OrderStatusUnknownError
        else:
            cls = _BY_STATUS.get(status_code, APIStatusError)

    kwargs: Dict[str, Any] = dict(
        status_code=status_code,
        code=code,
        detail=detail,
        request_id=request_id,
        raw_body=raw_body,
        headers=headers,
        retry_after=_retry_after(headers),
    )
    if cls is ValidationError and isinstance(detail, list):
        kwargs["errors"] = detail
    kwargs.update(_kwargs_from_detail(code, detail))

    try:
        return cls(message, **kwargs)  # type: ignore[no-any-return, call-arg]
    except TypeError:
        # A subclass whose signature we mismatched must never mask the real
        # error — degrade to the generic type instead of raising from here.
        return APIStatusError(
            message,
            status_code=status_code,
            code=code,
            detail=detail,
            request_id=request_id,
            raw_body=raw_body,
            headers=headers,
            retry_after=_retry_after(headers),
        )
