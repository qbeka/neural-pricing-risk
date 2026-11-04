"""Closed-form results for the contract family in `contract.py`.

Every function takes scalars or broadcastable NumPy arrays and returns
(price, delta) with discounting to today. Maturity must be positive.
"""

import numpy as np
from scipy.special import ndtr


def _fixing_times(maturity, n_fixings: int):
    """Fixing times with shape (*maturity.shape, n_fixings)."""
    fractions = np.arange(1, n_fixings + 1) / n_fixings
    return np.asarray(maturity, dtype=float)[..., None] * fractions


def geometric_moments(spot, vol, rate, maturity, n_fixings: int):
    """Mean and variance of ln G, where G is the geometric mean of the fixings.

    ln G = ln S0 + (r - vol^2 / 2) * mean(t_i) + vol * mean(W_ti), and
    Var(mean(W_ti)) = T (n + 1)(2n + 1) / (6 n^2) from summing min(t_i, t_j).
    """
    n = n_fixings
    mean_time = np.asarray(maturity, dtype=float) * (n + 1) / (2 * n)
    mean = np.log(spot) + (rate - 0.5 * np.square(vol)) * mean_time
    variance = np.square(vol) * maturity * (n + 1) * (2 * n + 1) / (6 * n**2)
    return mean, variance


def _black_call(forward, strike, total_variance, discount):
    """Black call on a lognormal with the given forward and total variance.

    Returns price and d(price)/d(forward). Zero variance gives the intrinsic
    value, which is the limit of the formula.
    """
    forward, strike, total_variance = np.broadcast_arrays(
        np.asarray(forward, dtype=float),
        np.asarray(strike, dtype=float),
        np.asarray(total_variance, dtype=float),
    )
    stdev = np.sqrt(total_variance)
    in_the_money = (forward > strike).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(forward / strike) + 0.5 * total_variance) / stdev
    d1 = np.where(stdev > 0, d1, np.nan)
    n1 = np.where(stdev > 0, ndtr(d1), in_the_money)
    n2 = np.where(stdev > 0, ndtr(d1 - stdev), in_the_money)
    return discount * (forward * n1 - strike * n2), discount * n1


def geometric_asian_call(spot, strike, vol, rate, maturity, n_fixings: int):
    """Exact price and delta of the call on the discrete geometric average."""
    mean, variance = geometric_moments(spot, vol, rate, maturity, n_fixings)
    forward = np.exp(mean + 0.5 * variance)
    discount = np.exp(-rate * np.asarray(maturity, dtype=float))
    price, dprice_dforward = _black_call(forward, strike, variance, discount)
    # The forward is proportional to spot, so dF/dS = F/S.
    return price, dprice_dforward * forward / spot
