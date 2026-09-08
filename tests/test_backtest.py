"""Backtest engine + metrics tests. All offline: the engine takes no network.

The risky things to get wrong here are money-shaped: the sign and size of a
bet's PnL, look-ahead leakage, and the calibration bucketing that the flagship
study leans on.
"""

from __future__ import annotations

import pytest

from supagamma.backtest import (
    NO,
    YES,
    Backtest,
    BetFavourite,
    FadeLongshot,
    MarketView,
    Order,
    ResolvedMarket,
    brier_score,
    calibration,
    max_drawdown,
    settle,
    sharpe,
    total_return,
)
from supagamma.backtest.data import realized_outcome, yes_probability

# --- settle(): the money math ------------------------------------------------

def test_winning_bet_pays_the_inverse_price():
    # $10 at price 0.25, win -> shares = 40, payout 40, pnl = 30
    assert settle(0.25, 10.0, won=True) == pytest.approx(30.0)


def test_losing_bet_loses_the_stake():
    assert settle(0.25, 10.0, won=False) == pytest.approx(-10.0)


def test_degenerate_prices_are_no_ops():
    assert settle(0.0, 10.0, won=True) == 0.0
    assert settle(1.0, 10.0, won=False) == 0.0


def test_fair_coin_is_symmetric():
    assert settle(0.5, 10.0, won=True) == pytest.approx(10.0)
    assert settle(0.5, 10.0, won=False) == pytest.approx(-10.0)


# --- the engine: side selection + settlement --------------------------------

def _market(id_, prob, outcome):
    return ResolvedMarket(MarketView(id=id_, question="q?", prob=prob), outcome=outcome)


def test_engine_buys_yes_and_settles_a_win():
    # prob(YES)=0.4, we buy YES, outcome YES -> price 0.4, win -> pnl = 10*0.6/0.4 = 15
    m = _market("a", 0.4, outcome=YES)
    result = Backtest(bankroll=100).run([m], lambda v: Order(YES, 10.0))
    assert result.n_bets == 1
    assert result.bets[0].pnl == pytest.approx(15.0)
    assert result.final_equity == pytest.approx(115.0)


def test_engine_buys_no_at_the_complement_price():
    # prob(YES)=0.4 -> price(NO)=0.6, buy NO, outcome NO -> win, pnl = 10*0.4/0.6
    m = _market("a", 0.4, outcome=NO)
    result = Backtest(bankroll=100).run([m], lambda v: Order(NO, 10.0))
    assert result.bets[0].price == pytest.approx(0.6)
    assert result.bets[0].pnl == pytest.approx(10.0 * 0.4 / 0.6)


def test_none_order_places_no_bet():
    m = _market("a", 0.4, outcome=YES)
    result = Backtest().run([m], lambda v: None)
    assert result.n_bets == 0
    assert result.markets_seen == 1


def test_strategy_cannot_see_the_outcome():
    # The view handed to a strategy must not carry the outcome.
    m = _market("a", 0.4, outcome=YES)
    seen = {}

    def spy(view):
        seen["fields"] = view.__dataclass_fields__.keys()
        return None

    Backtest().run([m], spy)
    assert "outcome" not in seen["fields"]


# --- BetFavourite: only bets when confident, on the right side --------------

def test_bet_favourite_backs_yes_when_yes_is_the_favourite():
    order = BetFavourite(min_confidence=0.6, stake=5)(MarketView("a", "q", prob=0.8))
    assert order == Order(YES, 5)


def test_bet_favourite_backs_no_when_no_is_the_favourite():
    order = BetFavourite(min_confidence=0.6, stake=5)(MarketView("a", "q", prob=0.2))
    assert order == Order(NO, 5)


def test_bet_favourite_abstains_near_a_coin_flip():
    assert BetFavourite(min_confidence=0.6)(MarketView("a", "q", prob=0.55)) is None


def test_fade_longshot_sells_the_low_tail():
    assert FadeLongshot(threshold=0.15)(MarketView("a", "q", prob=0.1)) == Order(NO, 10.0)
    assert FadeLongshot(threshold=0.15)(MarketView("a", "q", prob=0.95)) == Order(YES, 10.0)
    assert FadeLongshot(threshold=0.15)(MarketView("a", "q", prob=0.5)) is None


# --- metrics ----------------------------------------------------------------

def test_total_return_and_drawdown():
    assert total_return([100, 150]) == pytest.approx(0.5)
    # peak 150 then 90 -> dd = 60/150 = 0.4
    assert max_drawdown([100, 150, 90, 120]) == pytest.approx(0.4)


def test_sharpe_zero_without_dispersion():
    assert sharpe([0.1, 0.1, 0.1]) == 0.0
    assert sharpe([0.1]) == 0.0


def test_sharpe_positive_for_a_good_series():
    assert sharpe([0.2, 0.1, 0.15, 0.05]) > 0


# --- calibration: the flagship ----------------------------------------------

def test_perfectly_calibrated_market_has_matching_bins():
    # forecast 0.1 -> 10% actually happen; forecast 0.9 -> 90% happen
    pairs = [(0.1, 1)] + [(0.1, 0)] * 9        # 10% YES at forecast 0.1
    pairs += [(0.9, 1)] * 9 + [(0.9, 0)]       # 90% YES at forecast 0.9
    result = calibration(pairs, n_bins=10)
    low = result.bins[0]
    high = result.bins[-1]
    assert low.predicted == pytest.approx(0.1)
    assert low.actual == pytest.approx(0.1)
    assert high.actual == pytest.approx(0.9)
    assert abs(low.edge) < 1e-9


def test_longshot_bias_shows_as_negative_edge_in_the_low_bin():
    # Longshots priced at 0.1 but only 3% actually happen -> over-priced tail.
    pairs = [(0.1, 1)] * 3 + [(0.1, 0)] * 97
    result = calibration(pairs, n_bins=10)
    assert result.bins[0].edge < 0        # actual < predicted


def test_brier_bounds():
    assert brier_score([(1.0, 1), (0.0, 0)]) == 0.0
    assert brier_score([(0.5, 1), (0.5, 0)]) == pytest.approx(0.25)


def test_calibration_p_equals_one_lands_in_last_bin():
    result = calibration([(1.0, 1), (1.0, 1)], n_bins=10)
    assert len(result.bins) == 1
    assert result.bins[0].hi == pytest.approx(1.0)


# --- record parsing ---------------------------------------------------------

@pytest.mark.parametrize(
    "record, expected",
    [
        ({"winning_outcome": "Yes", "outcomes": ["Yes", "No"]}, 1),
        ({"winning_outcome": "No", "outcomes": ["Yes", "No"]}, 0),
        ({"winning_outcome": 0, "outcomes": ["Yes", "No"]}, 1),
        ({"winning_outcome": 1, "outcomes": ["Yes", "No"]}, 0),
        ({"winning_outcome": None}, None),
        ({"winning_outcome": "Trump", "outcomes": ["Trump", "Biden"]}, 1),
    ],
)
def test_realized_outcome_parsing(record, expected):
    assert realized_outcome(record) == expected


def test_yes_probability_handles_list_and_json_string():
    assert yes_probability({"outcome_prices": [0.62, 0.38]}) == pytest.approx(0.62)
    assert yes_probability({"outcome_prices": "[0.62, 0.38]"}) == pytest.approx(0.62)
    assert yes_probability({"outcome_prices": None}) is None
    assert yes_probability({"outcome_prices": [1.4, -0.4]}) is None
