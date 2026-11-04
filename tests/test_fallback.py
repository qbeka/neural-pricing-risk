"""The fallback rule accepts only what calibration supports."""

import json
from pathlib import Path

import numpy as np

from asianrisk.domain import load_domain
from asianrisk.fallback import Tolerance, calibrate_rule

CONFIG = json.loads((Path(__file__).parent.parent / "configs" / "smoke.json").read_text())
DOMAIN = load_domain(CONFIG)
RATE = CONFIG["rate"]
TOLERANCE = Tolerance(price_bp=1.0, delta=1e-3)


def _ordinary_points(count, seed):
    points = DOMAIN.from_unit(np.random.default_rng(seed).random((count, 3)))
    return points[DOMAIN.supported(points, RATE)]


def _rule(price_error, delta_error=0.0, count=40000, seed=0):
    points = _ordinary_points(count, seed)
    zeros = np.zeros(len(points))
    return points, calibrate_rule(
        DOMAIN, RATE, points, zeros + price_error, zeros + delta_error, zeros, zeros, TOLERANCE
    )


def test_a_perfect_method_is_accepted_on_supported_points():
    points, rule = _rule(0.0)
    assert rule.accept(points).all()


def test_a_method_outside_tolerance_is_rejected_everywhere():
    points, rule = _rule(2e-4)
    assert not rule.accept(points).any()
    points, rule = _rule(0.0, delta_error=2e-3)
    assert not rule.accept(points).any()


def test_unsupported_points_are_rejected_for_a_perfect_method():
    _, rule = _rule(0.0)
    outside_box = np.array([[0.6, 0.3, 1.0], [0.0, 0.9, 1.0], [0.0, 0.3, 0.02]])
    in_kink = np.array([[-0.003, 0.1, 0.1]])
    assert not rule.accept(outside_box).any()
    assert not rule.accept(in_kink).any()


def test_a_bin_too_small_for_the_coverage_rank_is_rejected():
    points, rule = _rule(0.0, count=2000)  # about 9 points per bin, below the 99% rank
    assert not rule.accept(points).any()


def test_bad_bins_are_isolated():
    points = _ordinary_points(40000, 1)
    errors = np.where(points[:, 1] > 0.6, 5e-4, 0.0)  # fails only at high vol
    zeros = np.zeros(len(points))
    rule = calibrate_rule(DOMAIN, RATE, points, errors, zeros, zeros, zeros, TOLERANCE)
    query = _ordinary_points(5000, 2)
    accepted = rule.accept(query)
    assert not accepted[query[:, 1] > 0.6].any()
    assert accepted[query[:, 1] < 0.5].mean() > 0.9


def test_unsupported_calibration_points_cannot_rescue_a_bin():
    """Regression: out-of-box points were clipped into edge bins and outvoted real misses."""
    points = _ordinary_points(40000, 5)
    errors = np.full(len(points), 5e-4)
    outside = np.tile([[0.9, 0.2, 1.0]], (4000, 1))
    all_points = np.concatenate([points, outside])
    all_errors = np.concatenate([errors, np.zeros(len(outside))])
    zeros = np.zeros(len(all_points))
    rule = calibrate_rule(DOMAIN, RATE, all_points, all_errors, zeros, zeros, zeros, TOLERANCE)
    assert not rule.accepted_bins.any()


def test_joint_miss_rate_stays_below_one_minus_coverage():
    """P(accepted and outside tolerance) is what the rule bounds, on noisy errors."""
    rng = np.random.default_rng(3)
    points = _ordinary_points(60000, 3)
    scale = 3e-5 * (1.0 + 2.0 * DOMAIN.to_unit(points)[:, 1])
    zeros = np.zeros(len(points))
    errors = rng.normal(0.0, scale)
    rule = calibrate_rule(DOMAIN, RATE, points, errors, zeros, zeros, zeros, TOLERANCE)
    query = _ordinary_points(200000, 4)
    fresh = rng.normal(0.0, 3e-5 * (1.0 + 2.0 * DOMAIN.to_unit(query)[:, 1]))
    accepted = rule.accept(query)
    assert accepted.mean() > 0.05  # low-vol bins pass, high-vol bins do not
    assert np.mean(accepted & (np.abs(fresh) > 1e-4)) <= 0.01
