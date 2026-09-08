"""Pure-Python performance metrics — no numpy, no pandas.

The SDK is httpx-only by design, so the backtest engine stays dependency-free:
everything here runs on the standard library. Charts in the examples pull in
matplotlib lazily and degrade to text if it is missing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import List, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Return / risk metrics over a series of per-bet returns or an equity curve.
# --------------------------------------------------------------------------- #

def total_return(equity_curve: Sequence[float]) -> float:
    """End-to-start return of an equity curve. ``[100, 110] -> 0.10``."""
    if len(equity_curve) < 2 or equity_curve[0] == 0:
        return 0.0
    return equity_curve[-1] / equity_curve[0] - 1.0


def max_drawdown(equity_curve: Sequence[float]) -> float:
    """Largest peak-to-trough drop as a fraction of the peak (>= 0)."""
    peak = -math.inf
    worst = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        if peak > 0:
            worst = max(worst, (peak - v) / peak)
    return worst


def sharpe(returns: Sequence[float]) -> float:
    """Per-bet Sharpe: mean / stdev of the return series.

    Deliberately *not* annualised — prediction-market bets do not arrive on a
    fixed calendar, so an annualisation factor would be made up. Read this as
    "reward per unit of bet-to-bet volatility". Returns 0.0 when there is no
    dispersion or fewer than two bets.
    """
    if len(returns) < 2:
        return 0.0
    sd = pstdev(returns)
    if sd == 0:
        return 0.0
    return mean(returns) / sd


def hit_rate(wins: int, total: int) -> float:
    return wins / total if total else 0.0


# --------------------------------------------------------------------------- #
# Calibration — the flagship. Are the market's prices honest probabilities?
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CalibrationBin:
    lo: float
    hi: float
    n: int
    predicted: float   # mean forecast probability in the bin
    actual: float      # empirical outcome rate in the bin

    @property
    def edge(self) -> float:
        """actual - predicted. Positive = the market under-priced YES here."""
        return self.actual - self.predicted


@dataclass(frozen=True)
class CalibrationResult:
    bins: List[CalibrationBin]
    brier: float
    n: int

    def longshot_summary(self) -> str:
        """One-line read on the favourite-longshot bias.

        Longshots (low forecast) tend to be *over*-priced and favourites
        (high forecast) *under*-priced. We surface the two tails.
        """
        if not self.bins:
            return "no data"
        low = self.bins[0]
        high = self.bins[-1]
        return (
            f"longshots [{low.lo:.0%}-{low.hi:.0%}]: priced {low.predicted:.1%}, "
            f"resolved {low.actual:.1%} ({low.edge:+.1%}) | "
            f"favourites [{high.lo:.0%}-{high.hi:.0%}]: priced {high.predicted:.1%}, "
            f"resolved {high.actual:.1%} ({high.edge:+.1%})"
        )

    def as_table(self) -> str:
        rows = ["bucket        n   priced  resolved   edge"]
        for b in self.bins:
            rows.append(
                f"{b.lo:>4.0%}-{b.hi:<4.0%} {b.n:>4}  {b.predicted:>6.1%}  "
                f"{b.actual:>7.1%}  {b.edge:>+6.1%}"
            )
        rows.append(f"Brier score: {self.brier:.4f}  (lower is better; n={self.n})")
        return "\n".join(rows)


def brier_score(pairs: Sequence[Tuple[float, int]]) -> float:
    """Mean squared error of probabilistic forecasts. 0 = perfect, 0.25 = coin flip."""
    if not pairs:
        return 0.0
    return mean((p - o) ** 2 for p, o in pairs)


def calibration(pairs: Sequence[Tuple[float, int]], n_bins: int = 10) -> CalibrationResult:
    """Bucket ``(forecast_probability, outcome)`` pairs and compare.

    ``outcome`` is 1 if the event happened else 0. Bins are equal-width over
    ``[0, 1]``. Empty bins are dropped. This is the analysis behind the
    "does the longshot bias exist?" study.
    """
    clean = [(float(p), int(o)) for p, o in pairs if p is not None and o is not None]
    if not clean:
        return CalibrationResult(bins=[], brier=0.0, n=0)

    width = 1.0 / n_bins
    buckets: List[List[Tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, o in clean:
        idx = min(int(p / width), n_bins - 1)   # p == 1.0 lands in the last bin
        buckets[idx].append((p, o))

    bins: List[CalibrationBin] = []
    for i, bucket in enumerate(buckets):
        if not bucket:
            continue
        bins.append(
            CalibrationBin(
                lo=i * width,
                hi=(i + 1) * width,
                n=len(bucket),
                predicted=mean(p for p, _ in bucket),
                actual=mean(o for _, o in bucket),
            )
        )
    return CalibrationResult(bins=bins, brier=brier_score(clean), n=len(clean))
