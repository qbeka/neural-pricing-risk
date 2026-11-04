"""The one fixed contract priced through the reference and through QuantLib."""

import pytest

from asianrisk.analytic import geometric_asian_call
from asianrisk.contract import Contract, Market
from asianrisk.reference import SimulationConfig, price_reference

CONTRACT = Contract("fixture", "X", strike=100.0, maturity=1.0, n_fixings=12)
MARKET = Market(spot=100.0, vol=0.20, rate=0.03)


def test_reference_agrees_with_quantlib_monte_carlo():
    crosscheck = pytest.importorskip("asianrisk.crosscheck")
    ours = price_reference(CONTRACT, MARKET, SimulationConfig(n_paths=2**18, seed=1))
    theirs, theirs_se = crosscheck.quantlib_arithmetic_mc(CONTRACT, MARKET, 2**18, seed=99)
    # Four standard errors of the difference: independent seeds, so variances add.
    tolerance = 4 * (ours.price_se**2 + theirs_se**2) ** 0.5
    assert abs(ours.price - theirs) < tolerance


def test_geometric_formula_matches_quantlib_analytic_engine():
    crosscheck = pytest.importorskip("asianrisk.crosscheck")
    ours, _ = geometric_asian_call(100.0, 100.0, 0.20, 0.03, 1.0, 12)
    assert ours == pytest.approx(crosscheck.quantlib_geometric_price(CONTRACT, MARKET), abs=1e-10)
