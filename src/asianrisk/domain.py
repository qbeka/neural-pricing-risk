"""Parameter domain, reserved stress regions and the split by parameter region.

A point is a row (log_moneyness, vol, maturity), where log_moneyness = ln(S / K).
Because price is homogeneous of degree one in (S, K),
    price(S, K, vol, T) = K * f(ln(S / K), vol, T)
and delta is the same at any scale, so three inputs are enough once the rate
and the fixing count are fixed by the experiment config.

Every split is decided from the parameters alone, before any label exists.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

SPLITS = ("train", "validation", "calibration", "test")


@dataclass(frozen=True)
class Domain:
    log_moneyness: tuple[float, float]
    vol: tuple[float, float]
    maturity: tuple[float, float]
    near_expiry_maturity: tuple[float, float]
    kink_max_average_vol: float
    kink_width_in_average_vols: float

    def to_unit(self, points):
        """Map points to the unit cube, using sqrt(T) so short maturities get more room."""
        low = np.array([self.log_moneyness[0], self.vol[0], np.sqrt(self.maturity[0])])
        high = np.array([self.log_moneyness[1], self.vol[1], np.sqrt(self.maturity[1])])
        return (_warp(points) - low) / (high - low)

    def from_unit(self, unit):
        low = np.array([self.log_moneyness[0], self.vol[0], np.sqrt(self.maturity[0])])
        high = np.array([self.log_moneyness[1], self.vol[1], np.sqrt(self.maturity[1])])
        warped = low + np.asarray(unit) * (high - low)
        return np.column_stack([warped[:, 0], warped[:, 1], warped[:, 2] ** 2])

    def in_box(self, points):
        unit = self.to_unit(points)
        return np.all((unit >= 0.0) & (unit <= 1.0), axis=1)

    def in_kink_region(self, points, rate: float):
        """Near the money with little volatility left in the average.

        The average of a Brownian path over [0, T] has standard deviation about
        sqrt(T / 3), and the mean of the average sits near S exp(r T / 2).
        """
        log_moneyness, vol, maturity = points.T
        average_vol = vol * np.sqrt(maturity / 3.0)
        distance = np.abs(log_moneyness + 0.5 * rate * maturity)
        return (average_vol < self.kink_max_average_vol) & (
            distance < self.kink_width_in_average_vols * average_vol
        )

    def supported(self, points, rate: float):
        """True for ordinary traffic: inside the box and outside the reserved regions."""
        return self.in_box(points) & ~self.in_kink_region(points, rate)


def _warp(points):
    points = np.asarray(points, dtype=float)
    return np.column_stack([points[:, 0], points[:, 1], np.sqrt(points[:, 2])])


def load_domain(config: dict) -> Domain:
    section = config["domain"]
    return Domain(
        log_moneyness=tuple(section["log_moneyness"]),
        vol=tuple(section["vol"]),
        maturity=tuple(section["maturity"]),
        near_expiry_maturity=tuple(section["near_expiry_maturity"]),
        kink_max_average_vol=section["kink_max_average_vol"],
        kink_width_in_average_vols=section["kink_width_in_average_vols"],
    )


def block_splits(config: dict) -> np.ndarray:
    """Assign every block of the unit-cube lattice to one split, seeded and fixed."""
    section = config["split"]
    n_blocks = section["blocks_per_axis"] ** 3
    order = np.random.default_rng(section["seed"]).permutation(n_blocks)
    assignment = np.empty(n_blocks, dtype=np.int64)
    start = 0
    for split_index, split in enumerate(SPLITS):
        stop = n_blocks if split == SPLITS[-1] else start + round(
            section["fractions"][split] * n_blocks
        )
        assignment[order[start:stop]] = split_index
        start = stop
    return assignment


def split_of(points, domain: Domain, config: dict) -> np.ndarray:
    """Split index of each point, from the block it falls in."""
    per_axis = config["split"]["blocks_per_axis"]
    cells = np.clip((domain.to_unit(points) * per_axis).astype(np.int64), 0, per_axis - 1)
    flat = cells[:, 0] * per_axis**2 + cells[:, 1] * per_axis + cells[:, 2]
    return block_splits(config)[flat]


def _sobol_unit(log2_points: int, seed: int) -> np.ndarray:
    return qmc.Sobol(d=3, scramble=True, seed=seed).random_base2(log2_points)


def sample_split(name: str, domain: Domain, config: dict) -> np.ndarray:
    """Ordinary points that fall in the blocks of one split, without labels."""
    spec = config["datasets"][name]
    points = domain.from_unit(_sobol_unit(spec["log2_pool"], spec["seed"]))
    keep = domain.supported(points, config["rate"]) & (
        split_of(points, domain, config) == SPLITS.index(name)
    )
    return points[keep][: spec["max_points"]]


def sample_near_expiry(domain: Domain, config: dict) -> np.ndarray:
    """Reserved stress set: maturity below the supported range, near the money."""
    spec = config["datasets"]["stress_near_expiry"]
    unit = _sobol_unit(spec["log2_pool"], spec["seed"])[: spec["max_points"]]
    low, high = np.sqrt(domain.near_expiry_maturity)
    maturity = (low + unit[:, 2] * (high - low)) ** 2
    vol = domain.vol[0] + unit[:, 1] * (domain.vol[1] - domain.vol[0])
    width = 4.0 * vol * np.sqrt(maturity / 3.0)
    log_moneyness = np.clip((2.0 * unit[:, 0] - 1.0) * width, *domain.log_moneyness)
    return np.column_stack([log_moneyness, vol, maturity])


def sample_kink(domain: Domain, config: dict) -> np.ndarray:
    """Reserved stress set: inside the box, in the kink region, by rejection."""
    spec = config["datasets"]["stress_kink"]
    unit = _sobol_unit(spec["log2_pool"] + 8, spec["seed"])
    # Shrink the box toward the kink so rejection keeps enough points.
    low = np.array([0.4, 0.0, 0.0])
    high = np.array([0.6, 0.4, 0.6])
    points = domain.from_unit(low + unit * (high - low))
    return points[domain.in_kink_region(points, config["rate"])][: spec["max_points"]]


def sample_dataset(name: str, domain: Domain, config: dict) -> np.ndarray:
    if name == "stress_near_expiry":
        return sample_near_expiry(domain, config)
    if name == "stress_kink":
        return sample_kink(domain, config)
    return sample_split(name, domain, config)
