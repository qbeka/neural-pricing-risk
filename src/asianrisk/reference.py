"""Transparent CPU Monte Carlo reference for the discrete arithmetic Asian call.

Paths use the exact lognormal transition between fixing dates, so there is no
time discretization error. Only the sampling error remains, and that error is
what `price_se` and `delta_se` measure.

Variance reduction has independent switches so each can be tested alone: plain
sampling, antithetic pairs, and a control variate. Two controls are available.
"geometric" is the standard discounted geometric payoff. "conditional" is
    max(E - K, 0) + 1{E > K} (A - E),  with E = E[A | Z] (all per the same spot).
The second term has zero mean given Z, so the expectation is the closed-form
lower bound. Any control that is a function of Z alone, the geometric payoff
included, leaves E[Var(payoff | Z)] behind. The tangent term removes that part,
because the payoff minus this control is nonzero only on paths where A and E
fall on opposite sides of the strike. docs/research-notes.md has the measurement.
"""

import time
from dataclasses import dataclass

import numpy as np

from asianrisk.analytic import (
    conditional_lower_bound_call,
    conditional_terms,
    geometric_asian_call,
)
from asianrisk.contract import Contract, Market, PriceResult


@dataclass(frozen=True)
class SimulationConfig:
    n_paths: int = 2**18  # number of (possibly paired) sample units
    antithetic: bool = True
    control: str = "geometric"  # "geometric", "conditional" or "none"
    pilot_paths: int = 2**12  # independent pilot sample for the coefficient
    batch_paths: int = 2**15  # memory bound: about 8 * 2 * n_fixings * batch bytes
    seed: int | tuple[int, ...] = 0

    def __post_init__(self):
        if self.n_paths < 2:
            raise ValueError("a standard error needs at least 2 sample units")
        if self.control not in ("geometric", "conditional", "none"):
            raise ValueError(f"unknown control {self.control!r}")


def fixing_averages(normals, vol, rate, maturity):
    """Arithmetic and geometric fixing averages per unit of spot.

    `normals` has shape (n_paths, n_fixings). Returning averages per unit of
    spot is what makes spot shocks free once a path set exists: the average
    for spot S is S times this number.
    """
    n_fixings = normals.shape[-1]
    dt = maturity / n_fixings
    steps = (rate - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * normals
    log_ratio = np.cumsum(steps, axis=-1)
    arithmetic = np.mean(np.exp(log_ratio), axis=-1)
    geometric = np.exp(np.mean(log_ratio, axis=-1))
    return arithmetic, geometric


def conditional_averages(normals, vol, rate, maturity):
    """E[A | Z] / S for each path, where Z is computed from the path's normals."""
    n_fixings = normals.shape[-1]
    loadings, log_weights, z_stdev = conditional_terms(vol, rate, maturity, n_fixings)
    dt = maturity / n_fixings
    # mean(W_ti) = sqrt(dt) * sum_i normals_i * (n - i + 1) / n
    path_weights = np.arange(n_fixings, 0, -1) / n_fixings
    z = np.sqrt(dt) * (normals @ path_weights) / z_stdev
    return np.mean(np.exp(log_weights + loadings * z[..., None]), axis=-1)


def _sample_units(rng, config: SimulationConfig, n_units, contract, market):
    """Undiscounted payoff and pathwise delta samples, one value per unit.

    A unit is one path, or an antithetic pair averaged. Units are independent
    of each other, so their sample variance gives a valid standard error.
    The output columns are arithmetic payoff, control payoff, arithmetic delta
    and control delta. Delta uses
    d/dS max(S a - K, 0) = a 1{S a > K}, which is valid almost surely because
    the payoff is Lipschitz in spot.
    """
    normals = rng.standard_normal((n_units, contract.n_fixings))
    signs = (1.0, -1.0) if config.antithetic else (1.0,)
    total = np.zeros((4, n_units))
    for sign in signs:
        arithmetic, geometric = fixing_averages(
            sign * normals, market.vol, market.rate, contract.maturity
        )
        total[0] += np.maximum(market.spot * arithmetic - contract.strike, 0.0)
        if config.control == "conditional":
            conditional = conditional_averages(
                sign * normals, market.vol, market.rate, contract.maturity
            )
            in_money = market.spot * conditional > contract.strike
            total[1] += in_money * (market.spot * arithmetic - contract.strike)
            total[3] += in_money * arithmetic
        else:
            total[1] += np.maximum(market.spot * geometric - contract.strike, 0.0)
            total[3] += geometric * (market.spot * geometric > contract.strike)
        total[2] += arithmetic * (market.spot * arithmetic > contract.strike)
    return total / len(signs)


def _coefficient(target, control):
    """Least-squares control coefficient, zero when the control does not vary."""
    variance = np.var(control)
    if variance == 0.0:
        return 0.0
    return float(np.mean((target - target.mean()) * (control - control.mean())) / variance)


def price_reference(contract: Contract, market: Market, config: SimulationConfig) -> PriceResult:
    """Price and delta with standard errors from independent sample units.

    The control coefficients come from a separate pilot sample and stay fixed
    during the main run, so the estimator is unbiased for any coefficient value.
    """
    started = time.perf_counter()
    rng = np.random.default_rng(config.seed)
    discount = np.exp(-market.rate * contract.maturity)

    price_beta, delta_beta = 0.0, 0.0
    control_price, control_delta = 0.0, 0.0
    if config.control != "none":
        pilot = _sample_units(rng, config, config.pilot_paths, contract, market)
        price_beta = _coefficient(pilot[0], pilot[1])
        delta_beta = _coefficient(pilot[2], pilot[3])
        formula = (
            conditional_lower_bound_call if config.control == "conditional" else geometric_asian_call
        )
        exact = formula(
            market.spot, contract.strike, market.vol, market.rate,
            contract.maturity, contract.n_fixings,
        )
        # Undiscounted expectations of the controls, to match the samples.
        control_price, control_delta = float(exact[0]) / discount, float(exact[1]) / discount

    # Running sums keep memory bounded by batch_paths for any n_paths.
    count, price_sum, price_sq, delta_sum, delta_sq = 0, 0.0, 0.0, 0.0, 0.0
    while count < config.n_paths:
        size = min(config.batch_paths, config.n_paths - count)
        units = _sample_units(rng, config, size, contract, market)
        price_y = units[0] - price_beta * (units[1] - control_price)
        delta_y = units[2] - delta_beta * (units[3] - control_delta)
        price_sum += price_y.sum()
        price_sq += np.square(price_y).sum()
        delta_sum += delta_y.sum()
        delta_sq += np.square(delta_y).sum()
        count += size

    def mean_and_se(total, squares):
        mean = total / count
        variance = max(squares / count - mean**2, 0.0) * count / (count - 1)
        return mean, np.sqrt(variance / count)

    price, price_se = mean_and_se(price_sum, price_sq)
    delta, delta_se = mean_and_se(delta_sum, delta_sq)
    return PriceResult(
        price=float(discount * price),
        delta=float(discount * delta),
        price_se=float(discount * price_se),
        delta_se=float(discount * delta_se),
        route="reference_mc",
        runtime_seconds=time.perf_counter() - started,
    )
