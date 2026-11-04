"""Surrogate mechanics: derivatives, unit conversion and interpolation identities."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from asianrisk.domain import load_domain
from asianrisk.grid import GridSurface, node_points
from asianrisk.network import NeuralSurface, PriceNet, TrainConfig, train_surface

CONFIG = json.loads((Path(__file__).parent.parent / "configs" / "smoke.json").read_text())
DOMAIN = load_domain(CONFIG)
POINTS = np.array([[0.05, 0.3, 1.0], [-0.3, 0.2, 0.4], [0.2, 0.6, 2.0]])


def _smooth_function(points):
    """A price-like test function of spot, with its exact delta at K = 1."""
    x, vol, maturity = points.T
    spot = np.exp(x)
    return (spot**2) * (1 + vol) * (1 + maturity), 2 * spot * (1 + vol) * (1 + maturity)


def test_network_slope_matches_autograd():
    torch.manual_seed(0)
    net = PriceNet(16, 3).to(torch.float64)
    unit = torch.rand(32, 3, dtype=torch.float64, requires_grad=True)
    value, slope = net(unit)
    gradient = torch.autograd.grad(value.sum(), unit)[0][:, 0]
    assert torch.allclose(slope, gradient, atol=1e-12)


def test_network_delta_is_the_currency_derivative_of_price():
    """Bump the actual currency spot, with strike 100, against the reported delta."""
    torch.manual_seed(1)
    surface = NeuralSurface(DOMAIN, PriceNet(16, 2).to(torch.float64))
    strike, spot, h = 100.0, 104.0, 1e-4
    vol, maturity = 0.3, 1.0

    def price_in_currency(s):
        points = np.array([[np.log(s / strike), vol, maturity]])
        return strike * surface(points)[0][0]

    bumped = (price_in_currency(spot + h) - price_in_currency(spot - h)) / (2 * h)
    delta = surface(np.array([[np.log(spot / strike), vol, maturity]]))[1][0]
    assert delta == pytest.approx(bumped, abs=1e-9)


def test_base_is_added_back_to_price_and_delta():
    torch.manual_seed(2)
    net = PriceNet(8, 2).to(torch.float64)
    plain = NeuralSurface(DOMAIN, net)(POINTS)
    with_base = NeuralSurface(DOMAIN, net, base=_smooth_function)(POINTS)
    expected = _smooth_function(POINTS)
    assert np.allclose(with_base[0], plain[0] + expected[0])
    assert np.allclose(with_base[1], plain[1] + expected[1])


def test_training_beats_a_constant_predictor_on_a_smooth_function():
    points = DOMAIN.from_unit(np.random.default_rng(0).random((2000, 3)))
    price, delta = _smooth_function(points)
    scale = 1e-3
    config = TrainConfig(width=32, depth=2, epochs=300, batch_size=256, seed=0)
    surface, record = train_surface(DOMAIN, points, scale * price, scale * delta, config)
    fitted, _ = surface(points)
    assert record["parameters"] > 0
    assert np.std(fitted - scale * price) < 0.2 * np.std(scale * price)


@pytest.mark.parametrize("order", [1, 3])
def test_grid_returns_node_values_at_nodes(order):
    shape = (9, 7, 7)
    nodes = node_points(DOMAIN, shape)
    price, delta = _smooth_function(nodes)
    surface = GridSurface.from_labels(DOMAIN, shape, price, delta, order)
    fitted_price, fitted_delta = surface(nodes)
    assert np.allclose(fitted_price, price, atol=1e-10)
    assert np.allclose(fitted_delta, delta, atol=1e-10)


def test_grid_error_falls_at_the_expected_order_for_each_scheme():
    query = DOMAIN.from_unit(np.random.default_rng(3).random((2000, 3)))
    exact, _ = _smooth_function(query)
    errors = {}
    for order in (1, 3):
        for n in (17, 33):
            nodes = node_points(DOMAIN, (n, n, n))
            price, delta = _smooth_function(nodes)
            fitted, _ = GridSurface.from_labels(DOMAIN, (n, n, n), price, delta, order)(query)
            errors[order, n] = np.max(np.abs(fitted - exact))
    assert 3.0 < errors[1, 17] / errors[1, 33] < 5.0  # second order
    assert errors[3, 17] / errors[3, 33] > 10.0  # fourth order, up to the box edge
