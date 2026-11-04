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


def moment_matched_call(spot, strike, vol, rate, maturity, n_fixings: int):
    """Turnbull-Wakeman style approximation: treat A as lognormal.

    The first two moments of A are exact:
        E[A]   = S / n * sum_i exp(r t_i)
        E[A^2] = S^2 / n^2 * sum_ij exp(r (t_i + t_j) + vol^2 min(t_i, t_j))
    The approximation is matching a lognormal to them. It is not a bound and
    its error grows with total volatility.
    """
    spot, strike, vol, maturity = np.broadcast_arrays(
        *(np.asarray(a, dtype=float) for a in (spot, strike, vol, maturity))
    )
    times = _fixing_times(maturity, n_fixings)
    growth = np.exp(rate * times)
    forward = spot * growth.mean(axis=-1)
    earlier_time = np.minimum(times[..., :, None], times[..., None, :])
    pair_growth = growth[..., :, None] * growth[..., None, :]
    second_moment = np.mean(
        pair_growth * np.exp(vol[..., None, None] ** 2 * earlier_time), axis=(-2, -1)
    )
    total_variance = np.log(second_moment / growth.mean(axis=-1) ** 2)
    discount = np.exp(-rate * maturity)
    price, dprice_dforward = _black_call(forward, strike, total_variance, discount)
    return price, dprice_dforward * forward / spot


def conditional_lower_bound_call(spot, strike, vol, rate, maturity, n_fixings: int):
    """Lower bound from conditioning on the geometric average (Curran; Rogers-Shi).

    Let Z be the standardized mean of W_ti and b_i = Cov(W_ti, Z). Jensen gives
        price >= exp(-rT) E[max(E[A | Z] - K, 0)],
    where E[A | Z = z] = S / n * sum_i exp(r t_i + vol b_i z - vol^2 b_i^2 / 2).
    That expression increases in z, so with z* solving E[A | z*] = K the
    expectation is closed form:
        bound = exp(-rT) (S / n * sum_i exp(r t_i) N(vol b_i - z*) - K N(-z*))
    and the terms from differentiating z* cancel, leaving
        delta = exp(-rT) / n * sum_i exp(r t_i) N(vol b_i - z*).
    The bound gap is the only error, and it grows with total volatility.
    """
    spot, strike, vol, maturity = np.broadcast_arrays(
        *(np.asarray(a, dtype=float) for a in (spot, strike, vol, maturity))
    )
    n = n_fixings
    index = np.arange(1, n + 1)
    times = _fixing_times(maturity, n)
    z_stdev = np.sqrt(maturity * (n + 1) * (2 * n + 1) / (6 * n**2))
    # Cov(W_ti, mean W) = T / n^2 * (sum_{j <= i} j + i (n - i))
    b = (maturity[..., None] / n**2) * (index * (index + 1) / 2 + index * (n - index))
    b = b / z_stdev[..., None]
    loadings = vol[..., None] * b
    growth = np.exp(rate * times)
    log_weights = rate * times - 0.5 * loadings**2

    # Solve ln(mean_i exp(log_w_i + loading_i z)) = ln(K / S) by Newton. The left
    # side is convex and increasing, so Newton settles quickly from a linear guess.
    log_target = np.log(strike / spot)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.nan_to_num(log_target / loadings[..., -1], nan=0.0)
        for _ in range(50):
            exponents = log_weights + loadings * z[..., None]
            top = exponents.max(axis=-1, keepdims=True)
            scaled = np.exp(exponents - top)
            total = scaled.sum(axis=-1)
            residual = top[..., 0] + np.log(total / n) - log_target
            slope = (loadings * scaled).sum(axis=-1) / total
            step = np.where(slope > 0, residual / slope, 0.0)
            z = z - step
            if np.max(np.abs(np.nan_to_num(step, posinf=0.0, neginf=0.0))) < 1e-12:
                break

    discount = np.exp(-rate * maturity)
    delta = discount * np.mean(growth * ndtr(loadings - z[..., None]), axis=-1)
    price = spot * delta - discount * strike * ndtr(-z)
    # Far out of the money the two terms cancel to rounding error; the bound is nonnegative.
    return np.maximum(price, 0.0), delta
