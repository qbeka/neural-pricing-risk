"""Correctness checks for the Monte Carlo reference.

Statistical tolerances are multiples of the estimated standard error, and exact
identities use a tolerance from double precision rounding. Seeds are fixed, so a
failure is reproducible rather than random.
"""

import numpy as np
import pytest

from asianrisk.analytic import conditional_lower_bound_call, geometric_asian_call
from asianrisk.contract import Contract, Market, PriceResult, Scenario
from asianrisk.reference import SimulationConfig, _sample_units, price_reference

CONTRACT = Contract("c", "X", strike=100.0, maturity=1.0, n_fixings=12)
MARKET = Market(spot=100.0, vol=0.20, rate=0.03)
PLAIN = dict(antithetic=False, control="none")


@pytest.mark.parametrize("spot", [90.0, 110.0])
def test_zero_vol_follows_the_deterministic_path(spot):
    times = np.arange(1, 13) / 12
    average = spot * np.exp(0.03 * times).mean()
    result = price_reference(
        CONTRACT, Market(spot, 0.0, 0.03), SimulationConfig(n_paths=256, seed=1)
    )
    assert result.price == pytest.approx(np.exp(-0.03) * max(average - 100.0, 0.0), abs=1e-12)
    expected_delta = np.exp(-0.03) * average / spot if average > 100.0 else 0.0
    assert result.delta == pytest.approx(expected_delta, abs=1e-12)
    # All samples are equal, so the variance is rounding error from the running sums.
    assert result.price_se < 1e-7


def test_price_increases_with_spot_on_common_random_numbers():
    config = SimulationConfig(n_paths=2**14, seed=2, **PLAIN)
    prices = [
        price_reference(CONTRACT, Market(s, 0.2, 0.03), config).price
        for s in np.linspace(60.0, 140.0, 17)
    ]
    assert np.all(np.diff(prices) >= 0.0)


def test_price_scales_linearly_and_delta_is_invariant_with_spot_and_strike():
    config = SimulationConfig(n_paths=2**14, seed=3, control="none")
    base = price_reference(CONTRACT, MARKET, config)
    scale = 7.5
    scaled = price_reference(
        Contract("c", "X", 100.0 * scale, 1.0, 12), Market(100.0 * scale, 0.2, 0.03), config
    )
    assert scaled.price == pytest.approx(scale * base.price, rel=1e-12)
    assert scaled.delta == pytest.approx(base.delta, rel=1e-12)


@pytest.mark.parametrize("spot, vol", [(100.0, 0.2), (85.0, 0.5), (120.0, 0.1)])
def test_simulated_geometric_payoff_matches_its_analytic_price(spot, vol):
    market = Market(spot, vol, 0.03)
    config = SimulationConfig(seed=4)
    units = _sample_units(np.random.default_rng(4), config, 2**19, CONTRACT, market)
    discount = np.exp(-0.03)
    exact_price, exact_delta = geometric_asian_call(spot, 100.0, vol, 0.03, 1.0, 12)
    for column, exact in ((1, exact_price), (3, exact_delta)):
        mean = discount * units[column].mean()
        se = discount * units[column].std() / np.sqrt(units.shape[1])
        assert abs(mean - exact) < 4 * se


@pytest.mark.parametrize("spot, vol", [(100.0, 0.2), (85.0, 0.5), (120.0, 0.7)])
def test_simulated_conditional_payoff_matches_the_lower_bound(spot, vol):
    market = Market(spot, vol, 0.03)
    config = SimulationConfig(control="conditional")
    units = _sample_units(np.random.default_rng(8), config, 2**19, CONTRACT, market)
    discount = np.exp(-0.03)
    exact_price, exact_delta = conditional_lower_bound_call(spot, 100.0, vol, 0.03, 1.0, 12)
    for column, exact in ((1, exact_price), (3, exact_delta)):
        mean = discount * units[column].mean()
        se = discount * units[column].std() / np.sqrt(units.shape[1])
        assert abs(mean - exact) < 4 * se


def test_conditional_control_beats_the_geometric_control_at_high_total_vol():
    contract = Contract("c", "X", 100.0, 2.5, 12)
    market = Market(100.0, 0.7, 0.03)
    geometric = price_reference(contract, market, SimulationConfig(n_paths=2**16, seed=9))
    conditional = price_reference(
        contract, market, SimulationConfig(n_paths=2**16, seed=9, control="conditional")
    )
    assert conditional.price_se < geometric.price_se / 3


def test_each_variance_reduction_step_lowers_the_standard_error():
    n = 2**16
    plain = price_reference(CONTRACT, MARKET, SimulationConfig(n_paths=n, seed=5, **PLAIN))
    antithetic = price_reference(
        CONTRACT, MARKET, SimulationConfig(n_paths=n, seed=5, control="none")
    )
    control = price_reference(CONTRACT, MARKET, SimulationConfig(n_paths=n, seed=5))
    assert control.price_se < antithetic.price_se < plain.price_se
    assert control.delta_se < antithetic.delta_se < plain.delta_se
    assert control.price_se < plain.price_se / 20


@pytest.mark.parametrize(
    "spot, vol, maturity",
    [(100.0, 0.2, 1.0), (80.0, 0.3, 0.1), (120.0, 0.6, 2.0), (100.0, 0.08, 0.1)],
)
def test_pathwise_delta_matches_a_bumped_price_on_common_random_numbers(spot, vol, maturity):
    contract = Contract("c", "X", 100.0, maturity, 12)
    config = SimulationConfig(n_paths=2**19, seed=6, control="none")
    h = 0.05
    up = price_reference(contract, Market(spot + h, vol, 0.03), config).price
    down = price_reference(contract, Market(spot - h, vol, 0.03), config).price
    pathwise = price_reference(contract, Market(spot, vol, 0.03), config)
    assert abs((up - down) / (2 * h) - pathwise.delta) < 4 * pathwise.delta_se + 1e-4


@pytest.mark.slow
@pytest.mark.parametrize("control, n_paths", [("geometric", 2**10), ("conditional", 2**14)])
def test_standard_errors_are_calibrated_and_the_estimator_is_unbiased(control, n_paths):
    """Repeat a cheap estimate many times against one very precise value.

    z-scores should have mean 0 and standard deviation 1. A pilot coefficient
    reused in the main run would show up here as a shifted mean.

    The conditional control needs more units: its residual is nonzero only on
    rare paths, so below about 2**14 units the sample standard error is itself
    unreliable (measured z standard deviation 2.9 at 2**10, 1.06 at 2**14).
    """
    precise = price_reference(
        CONTRACT, MARKET, SimulationConfig(n_paths=2**23, seed=7, batch_paths=2**16, control=control)
    )
    repeats = 300
    runs = [
        price_reference(
            CONTRACT,
            MARKET,
            SimulationConfig(n_paths=n_paths, pilot_paths=2**9, seed=1000 + i, control=control),
        )
        for i in range(repeats)
    ]
    for value, se, target in (
        ([r.price for r in runs], [r.price_se for r in runs], precise.price),
        ([r.delta for r in runs], [r.delta_se for r in runs], precise.delta),
    ):
        z = (np.array(value) - target) / np.array(se)
        assert abs(z.mean()) < 4 / np.sqrt(repeats)
        assert 0.85 < z.std() < 1.2


def test_records_round_trip_and_reject_other_schema_versions():
    for record in (
        CONTRACT,
        MARKET,
        Scenario("s", -0.1, 0.25),
        PriceResult(1.0, 0.5, 0.1, 0.01, "reference_mc", 0.002),
    ):
        assert type(record).from_record(record.to_record()) == record
    stale = {**MARKET.to_record(), "schema_version": 0}
    with pytest.raises(ValueError):
        Market.from_record(stale)


def test_invalid_configs_are_refused():
    with pytest.raises(ValueError):
        SimulationConfig(n_paths=1)
    with pytest.raises(ValueError):
        SimulationConfig(control="Conditional")


def test_shock_applies_relative_moves_to_spot_and_vol():
    shocked = Scenario("s", -0.1, 0.5).apply(MARKET)
    assert shocked == Market(90.0, pytest.approx(0.3), 0.03)
