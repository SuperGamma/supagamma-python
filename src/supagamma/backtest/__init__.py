"""Backtest prediction-market strategies on SupaGamma data.

A dependency-free (pure-standard-library) harness for scoring strategies against
resolved prediction markets, plus the calibration analysis behind the
"does the favourite-longshot bias exist?" study.

    from supagamma import SupaGamma
    from supagamma.backtest import Backtest, BetFavourite, calibration
    from supagamma.backtest.data import resolved_markets, calibration_pairs

    client = SupaGamma(api_key="sg_...")
    universe = list(resolved_markets(client, max_markets=1_000))

    # 1) Is the market well calibrated?
    print(calibration(calibration_pairs(universe)).as_table())

    # 2) Does a simple strategy beat it?
    result = Backtest(bankroll=1_000).run(universe, BetFavourite(stake=10))
    print(result.summary())

This is a research tool, not investment advice, and it makes no performance
promise. It shows what *did* happen in historical data; markets, fees, and
liquidity make live results different.
"""

from __future__ import annotations

from ._engine import (
    NO,
    YES,
    Backtest,
    BacktestResult,
    BetFavourite,
    BetRecord,
    FadeLongshot,
    MarketView,
    Order,
    ResolvedMarket,
    Strategy,
    settle,
)
from ._metrics import (
    CalibrationBin,
    CalibrationResult,
    brier_score,
    calibration,
    hit_rate,
    max_drawdown,
    sharpe,
    total_return,
)

__all__ = [
    # engine
    "Backtest",
    "BacktestResult",
    "BetRecord",
    "MarketView",
    "Order",
    "ResolvedMarket",
    "Strategy",
    "settle",
    "YES",
    "NO",
    # strategies
    "BetFavourite",
    "FadeLongshot",
    # metrics
    "calibration",
    "CalibrationResult",
    "CalibrationBin",
    "brier_score",
    "sharpe",
    "max_drawdown",
    "total_return",
    "hit_rate",
]
