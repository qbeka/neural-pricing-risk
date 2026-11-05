"""The decision rule is code, so its branches get table tests."""

import pytest

from asianrisk.report import aggregate_book, break_even, verdict

TOLERANCE = {"price_bp": 1.0, "delta": 0.001}


def route(method, kind, seconds, sd=0.0, price=0.5, delta=0.0005):
    return {
        "method": method, "route": kind, "seconds_mean": seconds, "seconds_sd": sd,
        "price_p99_bp": price, "delta_p99": delta,
    }


def job(*extra):
    """A complete summary: both hypothesis routes plus whatever a test adds."""
    return [
        route("nn_diff", "hybrid", 5.0, 0.1),
        route("nn_residual", "hybrid", 6.0, 0.1),
        *extra,
    ]


def test_comparator_is_the_fastest_passing_route_with_no_network():
    summary = job(
        route("direct_mc", "simulation", 9.0),
        route("corrected_grid", "fast_only", 7.0),
        route("grid_linear", "fast_only", 0.5, price=3.0),  # faster, but misses the gate
        route("nn_price", "fast_only", 0.1),  # faster, but has a network
    )
    assert verdict(summary, TOLERANCE)["comparator"] == "corrected_grid"


def test_a_clear_win_is_supported():
    outcome = verdict(job(route("corrected_grid", "fast_only", 7.0, 0.1)), TOLERANCE)
    assert outcome["primary"]["supported"] and outcome["secondary"]["supported"]
    assert outcome["primary"]["speedup"] == pytest.approx(7.0 / 5.0)
    assert not outcome["primary"]["stretch"]


def test_a_win_inside_the_run_to_run_spread_is_not_supported():
    outcome = verdict(job(route("corrected_grid", "fast_only", 5.15, 0.1)), TOLERANCE)
    assert not outcome["primary"]["supported"]


def test_faster_but_outside_the_gate_is_not_supported():
    summary = [
        route("nn_diff", "hybrid", 1.0, price=1.2),
        route("nn_residual", "hybrid", 1.0, delta=0.0011),
        route("corrected_grid", "fast_only", 7.0),
    ]
    outcome = verdict(summary, TOLERANCE)
    assert not outcome["primary"]["supported"] and not outcome["secondary"]["supported"]


def test_tenfold_win_meets_the_stretch_target():
    outcome = verdict(job(route("corrected_grid", "fast_only", 50.5)), TOLERANCE)
    assert outcome["primary"]["stretch"]


def test_no_comparator_supports_nothing():
    """Regression: a passing network used to win by default when no baseline passed."""
    outcome = verdict(job(route("direct_mc", "simulation", 9.0, delta=0.002)), TOLERANCE)
    assert outcome["comparator"] is None
    assert not outcome["primary"]["supported"] and not outcome["primary"]["stretch"]


def test_direct_simulation_can_be_the_comparator():
    outcome = verdict(job(route("direct_mc", "simulation", 9.0)), TOLERANCE)
    assert outcome["comparator"] == "direct_mc"


COSTS = {
    "nn_diff": {"label_wall": 80.0, "train_wall": 20.0, "calibration_wall": 10.0,
                "label_cpu": 0.0, "train_cpu_upper": 0.0, "calibration_cpu": 0.0},
    "grid_cubic": {"label_wall": 40.0, "calibration_wall": 10.0,
                   "label_cpu": 0.0, "calibration_cpu": 0.0},
}


@pytest.mark.parametrize(
    "method, kind, seconds, reference_method, reference_kind, reference_seconds, expected",
    [
        ("nn_diff", "fast_only", 1.0, "direct_mc", "simulation", 11.0, 10.0),  # 100 s at 10 s a job
        ("nn_diff", "hybrid", 1.0, "direct_mc", "simulation", 11.0, 11.0),  # adds calibration
        ("nn_diff", "fast_only", 1.0, "grid_cubic", "fast_only", 3.0, 30.0),  # 60 s at 2 s a job
        ("nn_diff", "fast_only", 3.0, "direct_mc", "simulation", 3.0, None),  # no savings
        ("nn_diff", "fast_only", 4.0, "direct_mc", "simulation", 3.0, None),  # costs more per job
        ("direct_mc", "simulation", 1.0, "nn_diff", "fast_only", 3.0, 0.0),  # cheaper and faster
    ],
)
def test_break_even(method, kind, seconds, reference_method, reference_kind, reference_seconds, expected):
    row = {"method": method, "route": kind, "seconds_mean": seconds}
    reference = {
        "method": reference_method, "route": reference_kind, "seconds_mean": reference_seconds,
    }
    assert break_even(row, reference, COSTS)["jobs_to_recover"] == expected


def test_aggregate_keeps_the_worst_request_from_any_run():
    rows = [
        {
            "run": run, "method": "grid_cubic", "route": "fast_only", "seconds": 1.0 + run,
            "cpu_seconds": 1.0, "price_p99_bp": 0.5, "price_median_bp": 0.1, "price_max_bp": 2.0 + run,
            "delta_p99": 0.0005, "delta_median": 0.0001, "delta_max": 0.002,
            "book_value_error_max_bp_of_gross": 0.0, "fallback_rate": 0.0, "simulated_requests": 0,
        }
        for run in range(3)
    ]
    summary = aggregate_book({"runs": rows, "requests": 10})[0]
    assert summary["price_max_bp"] == 4.0 and summary["runs"] == 3
    assert summary["seconds_mean"] == 2.0 and summary["seconds_sd"] == pytest.approx(1.0)
