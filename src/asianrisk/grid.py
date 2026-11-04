"""Interpolation baselines on a regular lattice in (log_moneyness, vol, sqrt(T)).

The table stores price and delta at the nodes and interpolates each separately.
Differentiating the price interpolant would lose an order of accuracy, and the
nodes already have a pathwise delta, so a delta table costs no extra simulation.

With `base`, the table stores label minus base(points) and adds the base back at
query time. That turns the table into a correction to a closed form, so it only
has to resolve a small smooth gap.
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import NdBSpline, make_interp_spline

from asianrisk.domain import Domain


def node_points(domain: Domain, shape: tuple[int, int, int]) -> np.ndarray:
    axes = [np.linspace(0.0, 1.0, n) for n in shape]
    mesh = np.meshgrid(*axes, indexing="ij")
    return domain.from_unit(np.column_stack([axis.ravel() for axis in mesh]))


def _multilinear(table, position):
    """Multilinear interpolation at fractional node positions with shape (n, 3)."""
    shape = np.array(table.shape)
    lower = np.clip(np.floor(position).astype(np.int64), 0, shape - 2)
    weight = position - lower
    result = 0.0
    for corner in range(8):
        offset = np.array([(corner >> axis) & 1 for axis in range(3)])
        factor = np.prod(np.where(offset, weight, 1.0 - weight), axis=1)
        index = lower + offset
        result = result + factor * table[index[:, 0], index[:, 1], index[:, 2]]
    return result


def _cubic_spline(table):
    """Not-a-knot tensor-product cubic spline, with node index as the coordinate.

    Not-a-knot keeps the spline fourth order up to the box edge. A padded edge
    such as `nearest` drops it to first order there, which sets the worst error.
    """
    coefficients, knots = table, []
    for axis, size in enumerate(table.shape):
        along_axis = np.moveaxis(coefficients, axis, 0)
        spline = make_interp_spline(np.arange(size, dtype=float), along_axis, k=3)
        knots.append(spline.t)
        coefficients = np.moveaxis(spline.c, 0, axis)
    return NdBSpline(tuple(knots), coefficients, 3)


@dataclass
class GridSurface:
    """A table over the domain box. Outside the box it clamps to the edge."""

    domain: Domain
    shape: tuple[int, int, int]
    price_table: object  # an array when order is 1, a spline when order is 3
    delta_table: object
    order: int
    base: object = None  # callable(points) -> (price, delta), or None

    @classmethod
    def from_labels(cls, domain, shape, price, delta, order, base=None):
        points = node_points(domain, shape)
        if base is not None:
            base_price, base_delta = base(points)
            price, delta = price - base_price, delta - base_delta
        price_table = np.reshape(price, shape)
        delta_table = np.reshape(delta, shape)
        if order == 3:
            price_table, delta_table = _cubic_spline(price_table), _cubic_spline(delta_table)
        return cls(domain, shape, price_table, delta_table, order, base)

    def __call__(self, points):
        scale = np.array(self.shape) - 1
        position = np.clip(self.domain.to_unit(points), 0.0, 1.0) * scale
        if self.order == 1:
            price = _multilinear(self.price_table, position)
            delta = _multilinear(self.delta_table, position)
        else:
            price, delta = self.price_table(position), self.delta_table(position)
        if self.base is not None:
            base_price, base_delta = self.base(points)
            price, delta = price + base_price, delta + base_delta
        return price, delta
