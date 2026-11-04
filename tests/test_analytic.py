"""Closed-form results against QuantLib, the reference and known limits."""

import numpy as np
import pytest

from asianrisk.analytic import (
    conditional_lower_bound_call,
    geometric_asian_call,
    moment_matched_call,
)
from asianrisk.contract import Contract, Market
from asianrisk.reference import SimulationConfig, price_reference

# (spot, vol, maturity) over a spread of total vol. Each maturity puts the 12
# fixings on whole days of a 360-day year, which QuantLib dates require.
CASES = [
    (80.0, 0.3, 1.0),
    (100.0, 0.1, 0.5),
    (100.0, 0.5, 2.0),
    (125.0, 0.6, 1.0),
    (100.0, 0.08, 1.0 / 3.0),
]


@pytest.mark.parametrize("spot, vol, maturity", CASES)
def test_moment_matching_matches_quantlib_turnbull_wakeman(spot, vol, maturity):
    crosscheck = pytest.importorskip("asianrisk.crosscheck")
    contract = Contract("c", "X", 100.0, maturity, 12)
    theirs = crosscheck.quantlib_turnbull_wakeman(contract, Market(spot, vol, 0.03))
    ours, _ = moment_matched_call(spot, 100.0, vol, 0.03, maturity, 12)
    assert ours == pytest.approx(theirs, abs=1e-8)


@pytest.mark.parametrize("spot, vol, maturity", CASES)
def test_lower_bound_stays_below_the_reference_and_close_to_it(spot, vol, maturity):
    contract = Contract("c", "X", 100.0, maturity, 12)
    reference = price_reference(
        contract, Market(spot, vol, 0.03), SimulationConfig(n_paths=2**18, seed=3)
    )
    bound, delta = conditional_lower_bound_call(spot, 100.0, vol, 0.03, maturity, 12)
    assert bound <= reference.price + 4 * reference.price_se
    assert reference.price - bound < 0.05
    assert abs(delta - reference.delta) < 2e-3 + 4 * reference.delta_se


def test_lower_bound_delta_is_the_derivative_of_the_bound():
    spot, h = 100.0, 1e-4
    up, _ = conditional_lower_bound_call(spot + h, 100.0, 0.4, 0.03, 1.0, 12)
    down, _ = conditional_lower_bound_call(spot - h, 100.0, 0.4, 0.03, 1.0, 12)
    _, delta = conditional_lower_bound_call(spot, 100.0, 0.4, 0.03, 1.0, 12)
    assert delta == pytest.approx((up - down) / (2 * h), abs=1e-7)


@pytest.mark.parametrize("formula", [geometric_asian_call, moment_matched_call])
def test_black_delta_is_the_derivative_of_the_price(formula):
    spot, h = 90.0, 1e-4
    up, _ = formula(spot + h, 100.0, 0.3, 0.03, 1.0, 12)
    down, _ = formula(spot - h, 100.0, 0.3, 0.03, 1.0, 12)
    _, delta = formula(spot, 100.0, 0.3, 0.03, 1.0, 12)
    assert delta == pytest.approx((up - down) / (2 * h), abs=1e-7)


@pytest.mark.parametrize(
    "formula", [geometric_asian_call, moment_matched_call, conditional_lower_bound_call]
)
@pytest.mark.parametrize("spot", [90.0, 110.0])
def test_zero_vol_limit_is_the_discounted_intrinsic_value(formula, spot):
    times = np.arange(1, 13) / 12
    growth = np.exp(0.03 * times)
    average = growth.mean() if formula is not geometric_asian_call else np.exp(0.03 * times.mean())
    expected = np.exp(-0.03) * max(spot * average - 100.0, 0.0)
    price, _ = formula(spot, 100.0, 0.0, 0.03, 1.0, 12)
    assert price == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("strike, rate", [(101.0, 0.05), (100.0, 0.05), (99.0, -0.05)])
def test_lower_bound_at_zero_vol_uses_the_forward_not_the_spot(strike, rate):
    """Regression: the zero-vol branch took its sign from ln(K / S) and ignored growth."""
    times = np.arange(1, 13) / 12
    average = 100.0 * np.exp(rate * times).mean()
    expected = np.exp(-rate) * max(average - strike, 0.0)
    expected_delta = np.exp(-rate) * average / 100.0 if average > strike else 0.0
    price, delta = conditional_lower_bound_call(100.0, strike, 0.0, rate, 1.0, 12)
    assert price == pytest.approx(expected, abs=1e-10)
    assert delta == pytest.approx(expected_delta, abs=1e-10)


def test_arrays_give_the_same_answers_as_scalars():
    spot = np.array([90.0, 100.0, 120.0])
    vol = np.array([0.1, 0.2, 0.6])
    maturity = np.array([0.1, 1.0, 2.5])
    for formula in (geometric_asian_call, moment_matched_call, conditional_lower_bound_call):
        prices, deltas = formula(spot, 100.0, vol, 0.03, maturity, 12)
        for i in range(3):
            price, delta = formula(spot[i], 100.0, vol[i], 0.03, maturity[i], 12)
            assert prices[i] == pytest.approx(float(price), abs=1e-12)
            assert deltas[i] == pytest.approx(float(delta), abs=1e-12)
