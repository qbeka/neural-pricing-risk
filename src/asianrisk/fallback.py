"""Decide which requests a fast method may answer and which go to simulation.

A request is accepted only if both hold:

1. Support. The point is inside the domain box and outside the reserved stress
   regions. No fast method is calibrated there, so it goes to simulation.
2. Observed error. The calibration residuals in the point's bin stay within
   tolerance at the required coverage.

For a bin with n calibration points, the score is the larger of the price and
delta errors as a fraction of tolerance, and the bin is accepted when the order
statistic of rank ceil((n + 1) * coverage) is at most 1. A bin too small to have
that rank is rejected.

What this guarantees, and what it does not. For a new point exchangeable with the
bin's calibration points, P(bin accepted and point outside tolerance) is at most
1 - coverage. That is a joint statement. It is not a promise of `coverage` among
accepted points, because acceptance selects bins whose calibration sample looked
good: with a true miss rate of 3% in every bin, the accepted bins still miss 3%.
So the rule is a screen, and accepted-case error must be measured on held-out
data. Nothing here covers stress regions or a book that concentrates in a hard
corner of a bin.
"""

from dataclasses import dataclass

import numpy as np

from asianrisk.domain import Domain
from asianrisk.metrics import BASIS_POINT


@dataclass(frozen=True)
class Tolerance:
    price_bp: float  # absolute price error in basis points of strike
    delta: float  # absolute delta error


@dataclass
class FallbackRule:
    domain: Domain
    rate: float
    accepted_bins: np.ndarray  # boolean array with shape (bins, bins, bins)

    def _bins(self, points):
        bins = self.accepted_bins.shape[0]
        return np.clip((self.domain.to_unit(points) * bins).astype(np.int64), 0, bins - 1)

    def accept(self, points):
        supported = self.domain.supported(points, self.rate)
        cell = self._bins(points)
        return supported & self.accepted_bins[cell[:, 0], cell[:, 1], cell[:, 2]]


def calibrate_rule(
    domain, rate, points, price, delta, reference_price, reference_delta,
    tolerance: Tolerance, coverage=0.99, bins=6,
) -> FallbackRule:
    """Build a rule from a fast method's answers on the calibration set."""
    score = np.maximum(
        np.abs(price - reference_price) / (tolerance.price_bp * BASIS_POINT),
        np.abs(delta - reference_delta) / tolerance.delta,
    )
    # Unsupported points must not vote for a bin that ordinary traffic will use.
    supported = domain.supported(points, rate)
    points, score = points[supported], score[supported]
    cells = np.clip((domain.to_unit(points) * bins).astype(np.int64), 0, bins - 1)
    flat = cells[:, 0] * bins**2 + cells[:, 1] * bins + cells[:, 2]
    accepted = np.zeros(bins**3, dtype=bool)
    for cell in range(bins**3):
        scores = np.sort(score[flat == cell])
        n = len(scores)
        rank = int(np.ceil((n + 1) * coverage))
        accepted[cell] = rank <= n and scores[rank - 1] <= 1.0
    return FallbackRule(domain, rate, accepted.reshape(bins, bins, bins))
