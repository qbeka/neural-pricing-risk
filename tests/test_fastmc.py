"""The batched engine must equal the plain estimator and the reference."""

import numpy as np
import pytest

from asianrisk.contract import Contract, Market
from asianrisk.fastmc import PathEngine
from asianrisk.reference import SimulationConfig, fixing_averages, price_reference, conditional_averages

POINTS = np.array(
    [
        [0.00, 0.20, 1.0],
        [0.10, 0.20, 1.0],
        [-0.20, 0.20, 1.0],
        [-0.30, 0.45, 2.5],
        [0.25, 0.60, 0.5],
        [0.01, 0.10, 0.1],
    ]
)


def test_cached_averages_match_the_reference_path_formulas():
    engine = PathEngine(0.03, 12, n_units=256, seed=1)
    rng = np.random.default_rng(1)
    normals = rng.standard_normal((256, 12))
    a, e = engine.averages(0.3, 0.7)
    expected_a, _ = fixing_averages(normals, 0.3, 0.03, 0.7)
    expected_e = conditional_averages(normals, 0.3, 0.03, 0.7)
    assert np.allclose(a[:256].numpy(), expected_a, atol=1e-12)
    assert np.allclose(e[:256].numpy(), expected_e, atol=1e-12)
    antithetic_a, _ = fixing_averages(-normals, 0.3, 0.03, 0.7)
    assert np.allclose(a[256:].numpy(), antithetic_a, atol=1e-12)


def test_sorted_lookup_equals_direct_sums_on_the_same_paths():
    engine = PathEngine(0.03, 12, n_units=2**12, seed=2)
    result = engine.price(POINTS)
    price, delta = engine.price_without_sorting(POINTS)
    assert np.allclose(result["price"], price, atol=1e-12)
    assert np.allclose(result["delta"], delta, atol=1e-12)


@pytest.mark.parametrize("index", range(len(POINTS)))
def test_engine_agrees_with_the_reference(index):
    log_moneyness, vol, maturity = POINTS[index]
    engine = PathEngine(0.03, 12, n_units=2**16, seed=3)
    result = engine.price(POINTS[[index]])
    reference = price_reference(
        Contract("c", "X", 1.0, float(maturity), 12),
        Market(float(np.exp(log_moneyness)), float(vol), 0.03),
        SimulationConfig(n_paths=2**20, control="conditional", seed=4),
    )
    for name, se_name, value, value_se in (
        ("price", "price_se", reference.price, reference.price_se),
        ("delta", "delta_se", reference.delta, reference.delta_se),
    ):
        combined = np.hypot(result[se_name][0], value_se)
        assert abs(result[name][0] - value) < 5 * combined


def test_unit_count_must_divide_into_batches():
    with pytest.raises(ValueError):
        PathEngine(0.03, 12, n_units=17, seed=1)


def test_groups_share_one_average_per_distinct_vol_and_maturity():
    engine = PathEngine(0.03, 12, n_units=2**10, seed=5)
    assert engine.price(POINTS)["distinct_vol_maturity_pairs"] == 4
