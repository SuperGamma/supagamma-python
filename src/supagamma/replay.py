"""Order-book replay: rebuild a market's L2 book at any moment.

    from datetime import datetime, timezone
    from supagamma import SupaGamma

    client = SupaGamma()
    replay = client.replay(
        "1254468",
        start="2026-11-01T12:00:00Z",
        end="2026-11-01T13:00:00Z",
        outcome="Yes",
    )
    for book in replay:                       # oldest first
        print(book.timestamp, book.best_bid, book.best_ask)

    book = replay.at("2026-11-01T12:30:00Z")  # the book as of that moment

**Where the data comes from.** Replay pages through ``GET /v1/download/orderbook``,
the market's full-depth L2 snapshots (one row per outcome token per capture
tick: every 5 s for a market expiring within the hour, up to every 15 min for
one more than a week out). Each book is the snapshot as captured, not a book
rebuilt from deltas: the per-update delta tape (``polymarket_l2_deltas``) is
only served venue-wide, with no per-market filter, so replaying one market from
it would mean downloading the whole venue.

**Server-side paging and outcome filter, when the API offers them.** Every
request asks for keyset paging (``paginate=true``, then the server's
``next_cursor``) and, when the outcome is known, for that outcome's
``token_id`` only, so the other outcomes are neither served nor metered. An
``outcome`` name is turned into its token id from ``client.markets.get`` (one
free read). An API that predates these parameters ignores them and answers
with a bare row list; replay then pages by time and filters outcomes after
download, as it always has. Both work, and the first answer decides which.

**It is metered.** Every page is an ordinary orderbook download: it counts
toward your plan's fair-use volume (or is billed per MB on a pay-as-you-go
deployment) and toward the download rate limit. Narrow the window, and pick
``page_size`` with that in mind. A page the rate limiter refuses (a plain 429)
is waited out and retried, because the limiter turns a request away before
anything is served or charged. Nothing else is retried, and a billing quota is
never retried.

**Memory stays flat.** Iterating holds one page (``page_size`` rows) at a time.
Only :meth:`Replay.to_pandas` collects everything, because a DataFrame is the
whole window by definition.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from ._client import RetryPolicy
from ._errors import (
    MissingDependencyError,
    NoDataInRangeError,
    NotFoundError,
    ResponseShapeError,
    SupaGammaError,
)
from .resources import download as _download
from .resources._base import Call
from .resources.download import CostWarning, build_orderbook, estimate_cost_usd
from .resources.markets import _check_market_id

if TYPE_CHECKING:  # pragma: no cover
    from ._client import AsyncSupaGamma, SupaGamma

__all__ = [
    "BookState",
    "Level",
    "Replay",
    "AsyncReplay",
    "DEFAULT_PAGE_SIZE",
    "DEFAULT_LOOKBACK",
    "REPLAY_RETRY",
]

#: One price level: ``(price, size)``.
Level = Tuple[float, float]
#: A moment: a timezone-aware ``datetime`` or an ISO-8601 string with an offset.
TimeLike = Union[datetime, str]
#: A grid step: a ``timedelta`` or a number of seconds.
IntervalLike = Union[timedelta, int, float]

#: Rows per request. About 2 KB a row, so ~5 MB a page: under the SDK's cost
#: warning threshold, and an hour of the fastest capture tier fits in one page.
DEFAULT_PAGE_SIZE = 2_500
#: The server's own ceiling on ``limit``.
MAX_PAGE_SIZE = 1_000_000
#: How far back :meth:`Replay.at` looks first; it doubles until a window answers.
#: Short, because the whole window is downloaded: 5 min of the fastest (5 s) tier
#: is ~120 rows for a binary market, and the 15 min tier answers on the third try.
DEFAULT_LOOKBACK = timedelta(minutes=5)

#: Replay's retry policy: the rate limiter only. A string-detail 429 is the
#: limiter turning the request away before the route runs, so nothing was served
#: or charged and a retry cannot double-charge. A billing quota (object-detail
#: 429) is a different class and is never retried; 5xx and dropped connections
#: are not retried either, because a download may already have been metered.
REPLAY_RETRY = RetryPolicy(attempts=3, on_status=(429,))

_UTC = timezone.utc


# --- the book ----------------------------------------------------------------


@dataclass(frozen=True)
class BookState:
    """One outcome's L2 book at one moment.

    ``bids`` run best (highest) first and ``asks`` best (lowest) first; each level
    is ``(price, size)`` with ``size > 0``. ``timestamp`` is the moment this book
    describes. ``snapshot_time`` is when the snapshot behind it was captured: the
    two are equal when iterating snapshots, and differ on an ``interval`` grid,
    where a book is carried forward until the next snapshot arrives.
    """

    timestamp: datetime
    token_id: str
    outcome: Optional[str]
    bids: Tuple[Level, ...]
    asks: Tuple[Level, ...]
    snapshot_time: datetime

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0][0] if self.asks else None

    @property
    def mid(self) -> Optional[float]:
        bid, ask = self.best_bid, self.best_ask
        return None if bid is None or ask is None else (bid + ask) / 2

    @property
    def spread(self) -> Optional[float]:
        bid, ask = self.best_bid, self.best_ask
        return None if bid is None or ask is None else ask - bid


# --- input validation (pure) --------------------------------------------------


_FRACTION = re.compile(r"(?<=\d\d:\d\d:\d\d)\.(\d+)")


def _parse_iso(text: str) -> Optional[datetime]:
    """``datetime.fromisoformat`` that also takes a trailing ``Z``; None if unreadable."""
    text = text.strip()
    if text[-1:] in ("Z", "z"):
        text = text[:-1] + "+00:00"
    # Python < 3.11 accepts only 3 or 6 fractional digits; pad or trim to 6.
    text = _FRACTION.sub(lambda m: "." + (m.group(1) + "000000")[:6], text, count=1)
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _parse_time(value: Any, name: str) -> datetime:
    """A caller's moment, as an aware UTC datetime. Naive input is refused.

    The message never repeats the value: it may have come from anywhere.
    """
    parsed: Optional[datetime]
    if isinstance(value, str):
        parsed = _parse_iso(value)
        if parsed is None:
            raise ValueError(
                f"{name} must be an ISO-8601 timestamp with a UTC offset, "
                "e.g. '2026-11-01T12:00:00Z'."
            )
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise TypeError(f"{name} must be a datetime or an ISO-8601 string.")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(
            f"{name} must be timezone-aware (e.g. datetime(..., tzinfo=timezone.utc) "
            "or a string ending in 'Z'); a naive time is ambiguous."
        )
    return parsed.astimezone(_UTC)


def _parse_interval(value: Optional[IntervalLike], name: str = "interval") -> Optional[timedelta]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a timedelta or a number of seconds.")
    if isinstance(value, timedelta):
        step = value
    elif isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number of seconds.")
        try:
            step = timedelta(seconds=value)
        except OverflowError:
            raise ValueError(f"{name} is too large.") from None
    else:
        raise TypeError(f"{name} must be a timedelta or a number of seconds.")
    if step <= timedelta(0):
        raise ValueError(f"{name} must be positive.")
    return step


def _check_page_size(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("page_size must be an integer.")
    if not 1 <= value <= MAX_PAGE_SIZE:
        raise ValueError(f"page_size must be between 1 and {MAX_PAGE_SIZE:,}.")
    return value


def _check_market(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("market_id must be a string (the market's markets.id).")
    try:
        return _check_market_id(value.strip())
    except ValueError:
        # The shared check quotes the id back; this path does not echo input.
        raise ValueError(
            "market_id must be a non-empty markets.id without '/', e.g. '1254468'."
        ) from None


# --- response parsing (pure) ----------------------------------------------------


def _row_time(value: Any) -> datetime:
    """A stored row timestamp. The capture writes ``isoformat()`` of UTC."""
    parsed: Optional[datetime]
    if isinstance(value, str):
        parsed = _parse_iso(value)
        if parsed is None:
            raise ResponseShapeError("an orderbook row carried an unreadable timestamp")
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise ResponseShapeError("an orderbook row carried no timestamp")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_UTC)  # the server documents naive as UTC
    return parsed.astimezone(_UTC)


def _levels(raw: Any, side: str) -> Tuple[Level, ...]:
    """Normalise one side to sorted ``(price, size)`` pairs, best first.

    The capture stores each side as a JSON string of ``{price, size}`` objects
    (strings, as the venue sends them); some files hold real arrays. Both are
    accepted, as are ``[price, size]`` pairs. Zero-size levels are dropped, and
    two entries at the same price add up.
    """
    if raw is None:
        return ()
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw.strip() else []
        except ValueError:
            raise ResponseShapeError(f"an orderbook row's {side}s were not valid JSON") from None
    if not isinstance(raw, list):
        raise ResponseShapeError(f"an orderbook row's {side}s were not a list of levels")
    book: Dict[float, float] = {}
    price_raw: Any
    size_raw: Any
    for level in raw:
        if isinstance(level, dict):
            price_raw, size_raw = level.get("price"), level.get("size")
        elif isinstance(level, (list, tuple)) and len(level) >= 2:
            price_raw, size_raw = level[0], level[1]
        else:
            raise ResponseShapeError(f"an orderbook row held an unreadable {side} level")
        try:
            price, size = float(price_raw), float(size_raw)
        except (TypeError, ValueError):
            raise ResponseShapeError(f"an orderbook row held a non-numeric {side} level") from None
        if not (math.isfinite(price) and math.isfinite(size)):
            raise ResponseShapeError(f"an orderbook row held a non-finite {side} level")
        if size > 0:
            book[price] = book.get(price, 0.0) + size
    return tuple(sorted(book.items(), reverse=(side == "bid")))


def parse_row(row: Any) -> BookState:
    """One ``/v1/download/orderbook`` JSON row as a :class:`BookState`. Pure."""
    if not isinstance(row, dict):
        raise ResponseShapeError("an orderbook row was not a JSON object")
    token = row.get("token_id")
    if not isinstance(token, str) or not token:
        raise ResponseShapeError("an orderbook row carried no token_id")
    ts = _row_time(row.get("timestamp"))
    outcome = row.get("outcome")
    return BookState(
        timestamp=ts,
        token_id=token,
        outcome=str(outcome) if outcome is not None else None,
        bids=_levels(row.get("bids"), "bid"),
        asks=_levels(row.get("asks"), "ask"),
        snapshot_time=ts,
    )


# --- paging (pure state machine) -----------------------------------------------


_TOKEN_RE = re.compile(r"^[0-9]{1,80}$")


def _is_token_id(value: Any) -> bool:
    """The shape the route accepts for ``token_id`` (decimal digits)."""
    return isinstance(value, str) and _TOKEN_RE.match(value) is not None


class _Pager:
    """Walks ``[start, end]`` one page at a time, oldest first, with no gaps or repeats.

    **Keyset mode** (an API that answers ``paginate=true`` with
    ``{data, meta: {next_cursor, has_more}}``): the server orders rows by
    ``(timestamp, token_id, ...)`` and hands back a signed cursor bound to the
    request's ``start``/``end``/``token_id``. Each next page repeats those and
    adds the cursor, until ``has_more`` is false.

    **Time mode** (an older API that ignores ``paginate`` and returns a bare
    list). The route then has no cursor, only ``start``/``end``/``limit``, and returns the
    *first* ``limit`` matching rows in storage order (day files in date order,
    rows in the order they were written), not sorted by time. So when a page
    comes back full, the next page starts AT the timestamp of the page's LAST
    row in storage order, and every row at or after that timestamp is held back
    for it (the bound is inclusive, so a capture tick split by the cut arrives
    whole on the next page). Using the last row rather than the newest one means
    a row written out of order after the cut is still fetched, unless it is
    older than that last row. Such a row (a late capture write) is dropped
    rather than yielded out of order. Ticks are yielded once each, in
    ``(timestamp, token_id)`` order.
    """

    def __init__(
        self,
        market_id: str,
        start: datetime,
        end: datetime,
        page_size: int,
        token_id: Optional[str] = None,
    ) -> None:
        self.market_id = market_id
        self.cursor = start
        self.end = end
        self.page_size = page_size
        self.token_id = token_id
        self.done = False
        #: None until the first answer shows which paging the server speaks.
        self.keyset: Optional[bool] = None
        self.next_cursor: Optional[str] = None
        self._key_start = start
        self._sent_start = start

    def call(self) -> Call:
        # In keyset mode the cursor is bound to the window it was issued for,
        # so start stays where the keyset walk began; in time mode it moves.
        start = self._key_start if self.keyset else self.cursor
        self._sent_start = start
        method, path, params, _ = build_orderbook(
            market_id=self.market_id,
            start=start,
            end=self.end,
            format="json",
            limit=self.page_size,
        )
        params["paginate"] = True
        if self.token_id is not None:
            params["token_id"] = self.token_id
        if self.keyset and self.next_cursor is not None:
            params["cursor"] = self.next_cursor
        return method, path, params, REPLAY_RETRY

    def empty(self) -> None:
        """The window (from the cursor on) holds no snapshots: the route's 404."""
        self.done = True

    def accept(self, body: Any) -> List[BookState]:
        if isinstance(body, dict):
            return self._accept_keyset(body)
        if not isinstance(body, list):
            raise ResponseShapeError("GET /v1/download/orderbook did not return a JSON list")
        if self.keyset:
            # A bare list mid keyset walk: the server changed under us.
            raise ResponseShapeError(
                "GET /v1/download/orderbook stopped paging by cursor in the middle of a replay"
            )
        self.keyset = False
        states = [parse_row(row) for row in body]
        fresh = [s for s in states if self.cursor <= s.timestamp <= self.end]
        fresh.sort(key=lambda s: (s.timestamp, s.token_id))

        if len(body) < self.page_size or not states:
            self.done = True
            return fresh

        nxt = states[-1].timestamp
        if nxt <= self.cursor:
            # The cut fell on a row at the cursor. Advance as little as possible:
            # to the oldest timestamp past it, so nothing in between is skipped.
            later = [s.timestamp for s in states if s.timestamp > self.cursor]
            nxt = min(later) if later else self.cursor
        if nxt <= self.cursor:
            # A full page, all at the cursor: the next request would be identical.
            raise SupaGammaError(
                f"page_size={self.page_size} is too small: a full page held a single "
                "timestamp, so the replay cannot move past it. Raise page_size."
            )
        self.cursor = nxt
        return [s for s in fresh if s.timestamp < nxt]

    def _accept_keyset(self, body: Dict[str, Any]) -> List[BookState]:
        data, meta = body.get("data"), body.get("meta")
        if not isinstance(data, list) or not isinstance(meta, dict):
            raise ResponseShapeError(
                "GET /v1/download/orderbook returned an object without a data list and meta"
            )
        if not self.keyset:
            # First keyset answer (possibly after time-mode pages, if the API
            # was deployed mid-walk): the cursor is bound to the start just sent.
            self.keyset = True
            self._key_start = self._sent_start
        states = [parse_row(row) for row in data]
        fresh = [s for s in states if self._key_start <= s.timestamp <= self.end]
        fresh.sort(key=lambda s: (s.timestamp, s.token_id))
        nxt = meta.get("next_cursor")
        if meta.get("has_more") is True:
            if not isinstance(nxt, str) or not nxt:
                raise ResponseShapeError(
                    "GET /v1/download/orderbook said has_more but sent no next_cursor"
                )
            if nxt == self.next_cursor:
                raise SupaGammaError(
                    "the server returned the same page cursor twice; stopping the replay"
                )
            self.next_cursor = nxt
        else:
            self.done = True
        return fresh


def _rows_in(body: Any) -> int:
    """How many rows a page carried, in either response shape (for the cost guard)."""
    if isinstance(body, list):
        return len(body)
    if isinstance(body, dict) and isinstance(body.get("data"), list):
        return len(body["data"])
    return 0


# --- the interval grid (pure) ----------------------------------------------------


class _Grid:
    """Resample snapshots onto ``start, start+interval, ... <= end``.

    A tick's book is the newest snapshot at or before the tick, per outcome. An
    outcome with no snapshot yet is left out of that tick. After the last
    snapshot the book is carried forward to ``end``; ``snapshot_time`` says how
    old it is.
    """

    def __init__(self, start: datetime, end: datetime, interval: timedelta) -> None:
        self.start = start
        self.end = end
        self.interval = interval
        self.tick = start
        self.latest: Dict[str, BookState] = {}

    def _emit(self) -> Iterator[BookState]:
        for token in sorted(self.latest):
            yield replace(self.latest[token], timestamp=self.tick)

    def feed(self, state: BookState) -> Iterator[BookState]:
        if not self.latest and state.timestamp > self.tick:
            # Nothing to show before the first snapshot: jump, don't spin.
            steps = math.ceil((state.timestamp - self.start) / self.interval)
            self.tick = self.start + steps * self.interval
        while self.tick <= self.end and state.timestamp > self.tick:
            yield from self._emit()
            self.tick += self.interval
        self.latest[state.token_id] = state

    def finish(self) -> Iterator[BookState]:
        if not self.latest:
            return
        while self.tick <= self.end:
            yield from self._emit()
            self.tick += self.interval


# --- shared configuration ---------------------------------------------------------


class _ReplayBase:
    def __init__(
        self,
        market_id: str,
        start: TimeLike,
        end: TimeLike,
        *,
        outcome: Optional[str],
        token_id: Optional[str],
        interval: Optional[IntervalLike],
        page_size: int,
    ) -> None:
        self.market_id = _check_market(market_id)
        self.start = _parse_time(start, "start")
        self.end = _parse_time(end, "end")
        if self.start >= self.end:
            raise ValueError("start must be before end.")
        if outcome is not None and token_id is not None:
            raise ValueError("pass outcome or token_id, not both.")
        if outcome is not None and (not isinstance(outcome, str) or not outcome.strip()):
            raise ValueError("outcome must be a non-empty outcome name, e.g. 'Yes'.")
        if token_id is not None and (not isinstance(token_id, str) or not token_id.strip()):
            raise ValueError("token_id must be a non-empty string.")
        self.outcome = outcome.strip() if outcome is not None else None
        self.token_id = token_id.strip() if token_id is not None else None
        self.interval = _parse_interval(interval)
        self.page_size = _check_page_size(page_size)
        self._resolved = False
        self._server_token: Optional[str] = None

    def _token_from_market(self, market: Any) -> Optional[str]:
        """The token id of ``self.outcome``, from a ``markets.get`` record.

        ``outcome_token_ids`` is parallel to ``outcomes``. None when the record
        does not carry a usable mapping (then the filter stays client-side). An
        outcome the market does not have raises before anything is downloaded.
        """
        if self.outcome is None or not isinstance(market, dict):
            return None
        outcomes, tokens = market.get("outcomes"), market.get("outcome_token_ids")
        if not isinstance(outcomes, list) or not isinstance(tokens, list):
            return None
        if not outcomes or len(outcomes) != len(tokens):
            return None
        for name, token in zip(outcomes, tokens):
            if str(name).strip().lower() == self.outcome.lower():
                return token if _is_token_id(token) else None
        raise self._no_match({str(name) for name in outcomes})

    def _direct_token(self) -> Optional[str]:
        # A token id the route would refuse (not decimal) is filtered client-side.
        return self.token_id if _is_token_id(self.token_id) else None

    def _before_page(self, download: Any, fetched: int, warned: bool) -> bool:
        """The download cost check, run before EACH page, on the running total.

        Prices the rows fetched so far plus this page, so a long replay of
        individually small pages still reaches the warning and the hook. Reads
        ``client.download.confirm_cost`` each time, so a hook set after the
        replay was built applies. Warns once per walk and asks the hook before
        every page past the line. Returns whether it has warned.
        """
        rows = fetched + self.page_size
        cost = estimate_cost_usd(rows, "orderbook")
        if cost < _download.COST_WARN_THRESHOLD_USD:
            return warned
        megabytes = rows * 2076 / 1_048_576
        message = (
            f"This replay reaches up to {rows:,} orderbook rows with its next page, "
            f"about {megabytes:,.0f} MB: roughly ${cost:,.2f} at the $5.00/MB metering "
            "rate (a subscription counts the MB toward its fair-use volume instead of "
            "billing it). Narrow start/end if that is more than you intended."
        )
        hook = getattr(download, "confirm_cost", None)
        if hook is not None and not hook(cost):
            raise RuntimeError(f"replay cancelled by confirm_cost hook: {message}")
        if not warned:
            warnings.warn(message, CostWarning, stacklevel=4)
        return True

    def _check_page(self, served: Set[str], matched: bool) -> None:
        # Pages carry whole capture ticks, and a tick normally holds every
        # outcome. So a page that served two or more outcomes and matched none
        # means the filter matches nothing: stop now instead of downloading the
        # rest of the window. With one outcome served it may be a capture gap
        # for the wanted one, so keep going.
        if len(served) >= 2 and not matched:
            raise self._no_match(served)

    def _wanted(self, state: BookState) -> bool:
        if self._server_token is not None and state.token_id == self._server_token:
            return True
        if self.token_id is not None:
            return state.token_id == self.token_id
        if self.outcome is not None:
            return (state.outcome or "").strip().lower() == self.outcome.lower()
        return True

    def _no_match(self, served: Set[str]) -> ValueError:
        # Lists what the market HAS, never repeats what the caller passed.
        return ValueError(
            "no snapshot in this window matched the outcome/token_id filter. "
            f"Outcomes served: {sorted(served)}."
        )

    def _target(self, when: TimeLike) -> datetime:
        moment = _parse_time(when, "when")
        if not self.start <= moment <= self.end:
            raise ValueError("when must fall inside the replay's [start, end] window.")
        return moment

    def _check_single(self) -> None:
        if self.outcome is None and self.token_id is None:
            raise ValueError(
                "at() returns one book, so the replay needs one outcome: pass "
                "outcome='Yes' (or token_id=...) to replay(), or use books_at() "
                "for every outcome."
            )

    @staticmethod
    def _lookback(value: Any) -> timedelta:
        step = _parse_interval(value, "lookback")
        if step is None:
            raise ValueError("lookback must be positive.")
        return step

    def _frame(self, states: List[BookState]) -> Any:
        import pandas as pd

        rows: List[Dict[str, Any]] = []
        for s in states:
            for side, levels in (("bid", s.bids), ("ask", s.asks)):
                for level, (price, size) in enumerate(levels):
                    rows.append(
                        {
                            "timestamp": s.timestamp,
                            "snapshot_time": s.snapshot_time,
                            "token_id": s.token_id,
                            "outcome": s.outcome,
                            "side": side,
                            "level": level,
                            "price": price,
                            "size": size,
                        }
                    )
        columns = [
            "timestamp",
            "snapshot_time",
            "token_id",
            "outcome",
            "side",
            "level",
            "price",
            "size",
        ]
        return pd.DataFrame(rows, columns=columns)


def _require_pandas() -> None:
    try:
        import pandas  # noqa: F401
    except ImportError:
        raise MissingDependencyError(
            'Replay.to_pandas() needs pandas: pip install "supagamma[pandas]"'
        ) from None


# --- sync -------------------------------------------------------------------------


class Replay(_ReplayBase):
    """A market's L2 book over ``[start, end]``. Build it with :meth:`SupaGamma.replay`.

    Creating one sends nothing; requests go out as you iterate or call
    :meth:`at`. Iterating twice downloads (and meters) the window twice.
    """

    def __init__(
        self,
        client: SupaGamma,
        market_id: str,
        start: TimeLike,
        end: TimeLike,
        *,
        outcome: Optional[str] = None,
        token_id: Optional[str] = None,
        interval: Optional[IntervalLike] = None,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> None:
        self._client = client
        super().__init__(
            market_id,
            start,
            end,
            outcome=outcome,
            token_id=token_id,
            interval=interval,
            page_size=page_size,
        )

    def _resolve(self) -> Optional[str]:
        """The token id to filter on server-side, worked out once per replay."""
        if not self._resolved:
            token = self._direct_token()
            if self.outcome is not None:
                try:
                    market = self._client.markets.get(self.market_id)
                except NotFoundError:
                    market = None  # e.g. a token id passed as market_id
                token = self._token_from_market(market)
            self._server_token = token
            self._resolved = True
        return self._server_token

    def _fetch(self, start: datetime, end: datetime) -> Iterator[BookState]:
        pager = _Pager(self.market_id, start, end, self.page_size, self._resolve())
        served: Set[str] = set()
        matched = False
        fetched = 0
        warned = False
        while not pager.done:
            method, path, params, policy = pager.call()
            warned = self._before_page(self._client.download, fetched, warned)
            try:
                response = self._client.request(method, path, params=params, policy=policy)
            except NoDataInRangeError:
                pager.empty()
                break
            try:
                body = response.json()
            except ValueError:
                raise ResponseShapeError(
                    "GET /v1/download/orderbook did not return JSON",
                    request_id=response.headers.get("x-request-id"),
                ) from None
            fetched += _rows_in(body)
            for state in pager.accept(body):
                served.add(state.outcome or state.token_id)
                if self._wanted(state):
                    matched = True
                    yield state
            self._check_page(served, matched)

    def snapshots(self) -> Iterator[BookState]:
        """Every snapshot in the window, oldest first (ignores ``interval``)."""
        return self._fetch(self.start, self.end)

    def __iter__(self) -> Iterator[BookState]:
        """Snapshots oldest first, or one book per outcome per tick with ``interval``."""
        if self.interval is None:
            yield from self.snapshots()
            return
        grid = _Grid(self.start, self.end, self.interval)
        for state in self.snapshots():
            yield from grid.feed(state)
        yield from grid.finish()

    def books_at(
        self, when: TimeLike, *, lookback: IntervalLike = DEFAULT_LOOKBACK
    ) -> Dict[str, BookState]:
        """Each outcome's book as of ``when``: its newest snapshot at or before it.

        Looks back ``lookback`` first and doubles the step, never before
        ``start``, until a window holds a snapshot. Returns ``{}`` when nothing
        precedes ``when`` inside the replay window. Keyed by ``token_id``. The
        capture snapshots every outcome of a market on the same tick, so all of
        them normally answer together; an outcome missing from the window that
        answered (a capture gap) is absent rather than searched for further back.
        A filter that matches nothing raises ``ValueError`` after the first page.
        """
        hi = self._target(when)
        span = self._lookback(lookback)
        while True:
            lo = self.start if span >= hi - self.start else hi - span
            latest: Dict[str, BookState] = {}
            for state in self._fetch(lo, hi):
                latest[state.token_id] = state
            if latest or lo <= self.start:
                return latest
            hi = lo - timedelta(microseconds=1)
            span *= 2

    def at(
        self, when: TimeLike, *, lookback: IntervalLike = DEFAULT_LOOKBACK
    ) -> Optional[BookState]:
        """The book as of ``when`` (newest snapshot at or before it), or ``None``.

        ``None`` means no snapshot precedes ``when`` inside ``[start, end]``.
        A ``when`` outside the window raises ``ValueError``. Needs the replay to be
        narrowed to one outcome (``outcome=`` or ``token_id=``).
        """
        self._check_single()
        books = self.books_at(when, lookback=lookback)
        return next(iter(books.values()), None)

    def to_pandas(self) -> Any:
        """The whole replay as one long DataFrame, one row per price level.

        Columns: ``timestamp, snapshot_time, token_id, outcome, side, level,
        price, size`` (``level`` 0 is the best). Holds the entire window in
        memory. Needs the ``pandas`` extra; checked before any request.
        """
        _require_pandas()
        return self._frame(list(self))


# --- async ------------------------------------------------------------------------


class AsyncReplay(_ReplayBase):
    """Async twin of :class:`Replay`: ``async for``, ``await replay.at(...)``."""

    def __init__(
        self,
        client: AsyncSupaGamma,
        market_id: str,
        start: TimeLike,
        end: TimeLike,
        *,
        outcome: Optional[str] = None,
        token_id: Optional[str] = None,
        interval: Optional[IntervalLike] = None,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> None:
        self._client = client
        super().__init__(
            market_id,
            start,
            end,
            outcome=outcome,
            token_id=token_id,
            interval=interval,
            page_size=page_size,
        )

    async def _resolve(self) -> Optional[str]:
        """See :meth:`Replay._resolve`."""
        if not self._resolved:
            token = self._direct_token()
            if self.outcome is not None:
                try:
                    market = await self._client.markets.get(self.market_id)
                except NotFoundError:
                    market = None
                token = self._token_from_market(market)
            self._server_token = token
            self._resolved = True
        return self._server_token

    async def _fetch(self, start: datetime, end: datetime) -> AsyncIterator[BookState]:
        pager = _Pager(self.market_id, start, end, self.page_size, await self._resolve())
        served: Set[str] = set()
        matched = False
        fetched = 0
        warned = False
        while not pager.done:
            method, path, params, policy = pager.call()
            warned = self._before_page(self._client.download, fetched, warned)
            try:
                response = await self._client.request(method, path, params=params, policy=policy)
            except NoDataInRangeError:
                pager.empty()
                break
            try:
                body = response.json()
            except ValueError:
                raise ResponseShapeError(
                    "GET /v1/download/orderbook did not return JSON",
                    request_id=response.headers.get("x-request-id"),
                ) from None
            fetched += _rows_in(body)
            for state in pager.accept(body):
                served.add(state.outcome or state.token_id)
                if self._wanted(state):
                    matched = True
                    yield state
            self._check_page(served, matched)

    def snapshots(self) -> AsyncIterator[BookState]:
        """Every snapshot in the window, oldest first (ignores ``interval``)."""
        return self._fetch(self.start, self.end)

    async def __aiter__(self) -> AsyncIterator[BookState]:
        if self.interval is None:
            async for state in self.snapshots():
                yield state
            return
        grid = _Grid(self.start, self.end, self.interval)
        async for state in self.snapshots():
            for book in grid.feed(state):
                yield book
        for book in grid.finish():
            yield book

    async def books_at(
        self, when: TimeLike, *, lookback: IntervalLike = DEFAULT_LOOKBACK
    ) -> Dict[str, BookState]:
        """See :meth:`Replay.books_at`."""
        hi = self._target(when)
        span = self._lookback(lookback)
        while True:
            lo = self.start if span >= hi - self.start else hi - span
            latest: Dict[str, BookState] = {}
            async for state in self._fetch(lo, hi):
                latest[state.token_id] = state
            if latest or lo <= self.start:
                return latest
            hi = lo - timedelta(microseconds=1)
            span *= 2

    async def at(
        self, when: TimeLike, *, lookback: IntervalLike = DEFAULT_LOOKBACK
    ) -> Optional[BookState]:
        """See :meth:`Replay.at`."""
        self._check_single()
        books = await self.books_at(when, lookback=lookback)
        return next(iter(books.values()), None)

    async def to_pandas(self) -> Any:
        """See :meth:`Replay.to_pandas`."""
        _require_pandas()
        return self._frame([state async for state in self])
