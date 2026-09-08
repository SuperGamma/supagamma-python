"""Does the favourite-longshot bias exist in prediction markets?

A worked example that answers a real research question with the SupaGamma SDK:
pull resolved markets, bucket them by the price the market quoted, and compare
that price to how often the event *actually* happened. Then check whether a
dead-simple "back the favourite" strategy would have beaten the book.

Run it:

    export SUPAGAMMA_API_KEY=sg_...
    python examples/calibration.py

Notes
-----
* By default this uses the free *metadata* price on each market record. On a
  resolved market that is often the settlement price, so treat the default run
  as a smoke test. For a real study, switch to the trade-tape VWAP (see below) --
  that call reads the fills and **costs money**.
* This is a research tool, not investment advice, and makes no performance
  promise. Fees, liquidity, and slippage make live results different.
"""

from __future__ import annotations

import os
import sys

from supagamma import SupaGamma
from supagamma.backtest import Backtest, BetFavourite, calibration
from supagamma.backtest.data import (
    calibration_pairs,
    resolved_markets,
    vwap_price_fn,  # noqa: F401  (used when you opt into the paid, honest price)
)

MAX_MARKETS = int(os.environ.get("SG_MAX_MARKETS", "1000"))


def main() -> int:
    api_key = os.environ.get("SUPAGAMMA_API_KEY")
    if not api_key:
        print("Set SUPAGAMMA_API_KEY first (get one at https://supagamma.com).")
        return 2

    client = SupaGamma(api_key=api_key)

    # For a rigorous study, uncomment the price_fn to price each market at the
    # trade-tape VWAP 24h before it resolved (this spends credits):
    #   price_fn = vwap_price_fn(horizon_hours=24)
    price_fn = None

    print(f"Pulling up to {MAX_MARKETS} resolved markets with data...")
    universe = list(
        resolved_markets(client, max_markets=MAX_MARKETS, price_fn=price_fn)
    )
    if not universe:
        print("No resolved markets came back. Widen the filters or check the key.")
        return 1
    print(f"Loaded {len(universe)} markets.\n")

    # 1) Calibration: is the market's price an honest probability?
    result = calibration(calibration_pairs(universe), n_bins=10)
    print("CALIBRATION")
    print(result.as_table())
    print("\n" + result.longshot_summary() + "\n")

    # 2) Does backing the favourite beat the book?
    bt = Backtest(bankroll=1_000.0).run(universe, BetFavourite(min_confidence=0.60, stake=10.0))
    print("STRATEGY -- back the favourite (>= 60%), $10 a market")
    print(bt.summary())

    _maybe_plot(result)
    return 0


def _maybe_plot(result) -> None:
    """Save a calibration chart if matplotlib is around; otherwise skip quietly."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("\n(install matplotlib to also save a calibration chart)")
        return
    xs = [b.predicted for b in result.bins]
    ys = [b.actual for b in result.bins]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", label="perfect calibration")
    ax.plot(xs, ys, "o-", color="#ef4444", label="market")
    ax.set_xlabel("price the market quoted")
    ax.set_ylabel("how often it actually happened")
    ax.set_title("Prediction-market calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig("calibration.png", dpi=144)
    print("\nSaved calibration.png")


if __name__ == "__main__":
    sys.exit(main())
