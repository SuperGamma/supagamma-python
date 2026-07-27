"""Official Python SDK for the SupaGamma API.

    from supagamma import SupaGamma

    client = SupaGamma(api_key="sg_...")          # or SUPAGAMMA_API_KEY in the env
    for market in client.markets.auto_paginate(limit=500):
        print(market["question"])

Three things about this API that the SDK surfaces rather than hides:

* **Downloads spend money.** Every ``client.download.*`` call debits your
  balance, so they are never retried automatically. See ``download``'s docstring.
* **429 means two different things.** ``RateLimitError`` is transient and
  retryable; ``QuotaExceededError`` is a billing cap and is not. They are
  separate exception classes for exactly that reason.
* **Row caps are silent.** A download that hits its cap looks identical to a
  complete one, so call the matching ``estimate`` first when completeness
  matters.
"""

from __future__ import annotations

__version__ = "0.1.0"

from ._client import AsyncSupaGamma, SupaGamma  # noqa: E402
from ._errors import (  # noqa: E402
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    BelowMinimumOrderError,
    BundleTooLargeError,
    ConflictError,
    EmptyLineItemError,
    FairUseCapError,
    FreeTierCapError,
    FreeTierWindowError,
    GoneError,
    InsufficientCreditsError,
    InvalidWindowError,
    MissingIdentifierError,
    NoDataInRangeError,
    NotFoundError,
    OrderStatusUnknownError,
    OriginBlockedError,
    PaygRetiredError,
    PayloadTooLargeError,
    PaymentRequiredError,
    PermissionDeniedError,
    QuotaExceededError,
    RateLimitError,
    RawWindowRequiredError,
    SeriesComingSoonError,
    SeriesEmptyRangeError,
    ServerError,
    ServiceUnavailableError,
    SubscriptionRequiredError,
    SupaGammaConfigError,
    SupaGammaError,
    UpgradeRequiredError,
    ValidationError,
    WrongDataTypeError,
)

__all__ = [
    "__version__",
    "SupaGamma",
    "AsyncSupaGamma",
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
