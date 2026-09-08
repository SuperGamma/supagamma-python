"""The backtest engine — data-source-agnostic so it unit-tests with no network.

Domain model (binary prediction markets):

* A market resolves to **YES (1)** or **NO (0)**.
* ``MarketView.prob`` is the market's implied probability of YES *at entry* --
  the only price the strategy is allowed to see.
* An :class:`Order` buys ``stake`` dollars of one side. Contracts cost their
  price and pay one dollar on the winning side, so a winning bet returns
  ``stake * (1 - price) / price`` and a loser returns ``-stake``.

**Look-ahead safety is enforced, not trusted:** the engine hands the strategy a
:class:`MarketView` that has no outcome field. The realized outcome lives on the
separate :class:`ResolvedMarket` and is used only to settle after the order is
placed. A strategy therefore *cannot* peek at the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Mapping, Optional, Protocol, Tuple

from ._metrics import (
    CalibrationResult,
    calibration,
    hit_rate,
    max_drawdown,
    sharpe,
    total_return,
)

YES = 1
NO = 0


@dataclass(frozen=True)
class MarketView:
    """What a strategy sees at decision time. No outcome -- by construction."""

    id: str
    question: str
    prob: float                                   # implied P(YES) at entry
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Order:
    """A strategy's decision: buy ``stake`` dollars of ``side`` (YES/NO)."""

    side: int
    stake: float


@dataclass(frozen=True)
class ResolvedMarket:
    """A market plus its realized outcome, used to settle bets."""

    view: MarketView
    outcome: int                                  # realized 1 (YES) or 0 (NO)


class Strategy(Protocol):
    """Anything callable ``(MarketView) -> Optional[Order]``. None = no bet."""

    def __call__(self, market: MarketView) -> Optional[Order]: ...


def settle(price: float, stake: float, won: bool) -> float:
    """PnL of a stake bought at ``price`` (the cost per one-dollar contract).

    A degenerate or already-settled price (<= 0 or >= 1) carries no tradeable
    edge, so it is treated as a no-op bet worth zero.
    """
    if price <= 0.0 or price >= 1.0:
        return 0.0
    return stake * (1.0 - price) / price if won else -stake


@dataclass(frozen=True)
class BetRecord:
    market_id: str
    side: int
    price: float                                  # price of the side we bought
    stake: float
    outcome: int
    pnl: float

    @property
    def won(self) -> bool:
        return self.side == self.outcome

    @property
    def ret(self) -> float:
        return self.pnl / self.stake if self.stake else 0.0


@dataclass
class BacktestResult:
    bankroll: float
    equity_curve: List[float]
    bets: List[BetRecord]
    markets_seen: int

    # ---- headline metrics -------------------------------------------------- #
    @property
    def final_equity(self) -> float:
        return self.equity_curve[-1] if self.equity_curve else self.bankroll

    @property
    def total_return(self) -> float:
        return total_return(self.equity_curve)

    @property
    def max_drawdown(self) -> float:
        return max_drawdown(self.equity_curve)

    @property
    def sharpe(self) -> float:
        return sharpe([b.ret for b in self.bets])

    @property
    def n_bets(self) -> int:
        return len(self.bets)

    @property
    def wins(self) -> int:
        return sum(1 for b in self.bets if b.won)

    @property
    def hit_rate(self) -> float:
        return hit_rate(self.wins, self.n_bets)

    def calibration(self, n_bins: int = 10) -> CalibrationResult:
        """Calibration over the bet sample.

        For calibration across the *whole* universe (not just the markets you
        bet), call ``supagamma.backtest.calibration`` directly on your resolved
        set of ``(forecast_yes, outcome)`` pairs.
        """
        pairs: List[Tuple[float, int]] = []
        for b in self.bets:
            forecast_yes = b.price if b.side == YES else 1.0 - b.price
            pairs.append((forecast_yes, b.outcome))
        return calibration(pairs, n_bins=n_bins)

    def summary(self) -> str:
        return (
            f"markets seen : {self.markets_seen}\n"
            f"bets placed  : {self.n_bets}\n"
            f"hit rate     : {self.hit_rate:.1%}\n"
            f"total return : {self.total_return:+.1%}\n"
            f"max drawdown : {self.max_drawdown:.1%}\n"
            f"per-bet Sharpe: {self.sharpe:.2f}\n"
            f"final equity : ${self.final_equity:,.2f}  (from ${self.bankroll:,.2f})"
        )


class Backtest:
    """Run a :class:`Strategy` over resolved markets and score it.

    >>> from supagamma.backtest import Backtest, BetFavourite
    >>> bt = Backtest(bankroll=1_000)
    >>> result = bt.run(resolved_markets, BetFavourite(stake=10))
    >>> print(result.summary())
    """

    def __init__(self, bankroll: float = 1_000.0) -> None:
        self.bankroll = float(bankroll)

    def run(self, markets: Iterable[ResolvedMarket], strategy: Strategy) -> BacktestResult:
        equity = self.bankroll
        curve: List[float] = [equity]
        bets: List[BetRecord] = []
        seen = 0

        for rm in markets:
            seen += 1
            order = strategy(rm.view)
            if order is None or order.stake <= 0:
                continue
            price = rm.view.prob if order.side == YES else 1.0 - rm.view.prob
            won = order.side == rm.outcome
            pnl = settle(price, order.stake, won)
            equity += pnl
            curve.append(equity)
            bets.append(
                BetRecord(
                    market_id=rm.view.id,
                    side=order.side,
                    price=price,
                    stake=order.stake,
                    outcome=rm.outcome,
                    pnl=pnl,
                )
            )

        return BacktestResult(
            bankroll=self.bankroll,
            equity_curve=curve,
            bets=bets,
            markets_seen=seen,
        )


# --------------------------------------------------------------------------- #
# A couple of illustrative strategies for the tutorial. Not advice -- teaching
# tools that show the API and the documented favourite-longshot effect.
# --------------------------------------------------------------------------- #

@dataclass
class BetFavourite:
    """Back the side the market thinks is more likely, if it is confident enough.

    The favourite-longshot bias says favourites are mildly *under*-priced, so
    systematically backing them is the textbook demonstration strategy.
    """

    min_confidence: float = 0.60      # only bet when the favourite >= this
    stake: float = 10.0

    def __call__(self, m: MarketView) -> Optional[Order]:
        if m.prob >= self.min_confidence:
            return Order(side=YES, stake=self.stake)
        if (1.0 - m.prob) >= self.min_confidence:
            return Order(side=NO, stake=self.stake)
        return None


@dataclass
class FadeLongshot:
    """Bet *against* longshots -- sell the over-priced tail.

    If YES is a longshot (prob <= threshold), buy NO; if NO is the longshot,
    buy YES. The mirror image of :class:`BetFavourite`, framed as fading.
    """

    threshold: float = 0.15
    stake: float = 10.0

    def __call__(self, m: MarketView) -> Optional[Order]:
        if m.prob <= self.threshold:
            return Order(side=NO, stake=self.stake)
        if (1.0 - m.prob) <= self.threshold:
            return Order(side=YES, stake=self.stake)
        return None
