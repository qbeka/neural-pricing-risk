"""The saved book and shock grid are the workload definition."""

import numpy as np
import pytest

from asianrisk.contract import Scenario
from asianrisk.domain import load_domain
from asianrisk.fastmc import PathEngine
from asianrisk.portfolio import (
    Book,
    all_requests,
    generate_book,
    load_book,
    request_points,
    save_book,
    shock_grid,
)


def _book(bespoke=False, count=50):
    return generate_book("t", 7, 0.03, 12, count, 4, bespoke)


def test_generation_is_deterministic_and_saved_books_round_trip(tmp_path):
    book = _book()
    assert book == _book()
    path = tmp_path / "book.json"
    save_book(book, path)
    assert load_book(path) == book
    assert Book.from_record(book.to_record()) == book


def test_the_base_scenario_leaves_the_market_unchanged():
    book = _book()
    points = request_points(book, Scenario("base", 0.0, 0.0))
    for row, contract in zip(points, book.contracts):
        market = book.markets[contract.underlying]
        assert row[0] == pytest.approx(np.log(market.spot / contract.strike))
        assert row[1] == market.vol


def test_shock_grid_has_each_named_combination_once():
    grid = shock_grid()
    assert len(grid) == 42
    assert len({scenario.scenario_id for scenario in grid}) == 42
    assert sum(s.spot_shift == 0.0 and s.vol_shift == 0.0 for s in grid) == 1


def test_standard_tenors_share_vol_maturity_pairs_and_bespoke_ones_do_not():
    scenarios = shock_grid()
    standard, _, _ = all_requests(_book(False, 200), scenarios)
    bespoke, _, _ = all_requests(_book(True, 200), scenarios)
    assert len(np.unique(standard[:, 1:], axis=0)) < 0.25 * len(np.unique(bespoke[:, 1:], axis=0))


def test_portfolio_value_equals_a_contract_level_sum():
    book = _book(count=8)
    scenarios = shock_grid()[:3]
    points, strikes, quantities = all_requests(book, scenarios)
    engine = PathEngine(0.03, 12, n_units=2**10, seed=1)
    result = engine.price(points)
    total = strikes * quantities * result["price"]
    by_hand = 0.0
    for offset, scenario in enumerate(scenarios):
        rows = request_points(book, scenario)
        for i, contract in enumerate(book.contracts):
            index = offset * len(book.contracts) + i
            assert np.allclose(points[index], rows[i])
            by_hand += contract.quantity * contract.strike * result["price"][index]
    assert total.sum() == pytest.approx(by_hand)


def test_shocked_book_stays_near_the_supported_box():
    import json
    from pathlib import Path

    config = json.loads(Path("configs/experiment.json").read_text())
    domain = load_domain(config)
    points, _, _ = all_requests(_book(False, 300), shock_grid())
    assert domain.in_box(points).mean() > 0.95
