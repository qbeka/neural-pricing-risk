"""Batched simulation for repricing many requests on one path set.

Work reused between requests:

1. One set of normals serves every request. Under the exact transition, a path
   for (vol, T) is drift + vol sqrt(T / n) * walk, so the walk is computed once.
2. The fixing average per unit of spot is computed once per distinct (vol, T),
   so all spot and strike queries at that (vol, T) share it.
3. With strike 1 and spot S the estimator needs only sums over {a > 1 / S} and
   {e > 1 / S}, where a is the arithmetic average and e = E[A | Z], both per unit
   of spot. After sorting, those sums are prefix-sum lookups, so a query costs
   O(log paths) rather than O(paths).

The estimator is the conditional control with coefficient 1. It needs no pilot
because the coefficient is fixed in advance, so it stays unbiased:
    price = lower bound + exp(-rT) mean[(S a - 1) (1{S a > 1} - 1{S e > 1})]
    delta = bound delta + exp(-rT) mean[a (1{S a > 1} - 1{S e > 1})]

Paths are divided into independent batches of antithetic pairs, and the standard
error comes from the spread of the batch estimates. Far from the money the
correction is a rare event, so that standard error can read near zero while the
true noise is small but not zero (about 0.001 bp of strike in the worst measured
case). Treat it as an estimate with that known weakness.
"""

from dataclasses import dataclass

import numpy as np
import torch

from asianrisk.analytic import conditional_lower_bound_call, conditional_terms


@dataclass
class PathEngine:
    rate: float
    n_fixings: int
    n_units: int  # antithetic pairs
    seed: int
    device: str = "cpu"
    dtype: torch.dtype = torch.float64
    batches: int = 16

    def __post_init__(self):
        if self.n_units % self.batches != 0:
            raise ValueError("n_units must be a multiple of batches")
        n = self.n_fixings
        rng = np.random.default_rng(self.seed)
        normals = rng.standard_normal((self.n_units, n))
        walk = np.cumsum(normals, axis=1)
        # z = mean(W_ti) / stdev(mean(W_ti)) depends on the normals only, not on T.
        weights = np.arange(n, 0, -1) / n
        z = (normals @ weights) / np.sqrt(np.sum(weights**2))
        # Put each antithetic partner in the same batch, as [path, partner].
        self._walk = self._tensor(np.concatenate([walk, -walk]))
        self._z = self._tensor(np.concatenate([z, -z]))

    def _tensor(self, array):
        return torch.as_tensor(array, dtype=self.dtype, device=self.device)

    def averages(self, vol: float, maturity: float):
        """Arithmetic average a and conditional average e, per unit of spot."""
        n = self.n_fixings
        dt = maturity / n
        times = dt * np.arange(1, n + 1)
        drift = self._tensor((self.rate - 0.5 * vol**2) * times)
        arithmetic = torch.exp(drift + vol * dt**0.5 * self._walk).mean(dim=1)
        loadings, log_weights, _ = conditional_terms(vol, self.rate, maturity, n)
        conditional = torch.exp(
            self._tensor(log_weights) + self._tensor(loadings) * self._z[:, None]
        ).mean(dim=1)
        return arithmetic, conditional

    def _group(self, vol, maturity, spots):
        """Batch estimates of the correction for spots sharing (vol, T)."""
        arithmetic, conditional = self.averages(vol, maturity)
        # Each batch holds a run of paths and their antithetic partners.
        pair_batches = self.batches
        batch_size = arithmetic.shape[0] // pair_batches
        a = arithmetic.reshape(2, pair_batches, -1).transpose(0, 1).reshape(pair_batches, batch_size)
        e = conditional.reshape(2, pair_batches, -1).transpose(0, 1).reshape(pair_batches, batch_size)

        sorted_a, _ = torch.sort(a, dim=1)
        e_sorted, e_order = torch.sort(e, dim=1)
        a_by_e = torch.gather(a, 1, e_order)
        zeros = torch.zeros((pair_batches, 1), dtype=a.dtype, device=a.device)
        # tail_sum[b, i] is the sum of entries i onward.
        tail_a = torch.cat([sorted_a.flip(1).cumsum(1).flip(1), zeros], dim=1)
        tail_a_by_e = torch.cat([a_by_e.flip(1).cumsum(1).flip(1), zeros], dim=1)

        spot = self._tensor(spots)
        threshold = (1.0 / spot).expand(pair_batches, -1)
        above_a = torch.searchsorted(sorted_a, threshold, right=True)
        above_e = torch.searchsorted(e_sorted, threshold, right=True)
        sum_a = torch.gather(tail_a, 1, above_a)
        sum_e = torch.gather(tail_a_by_e, 1, above_e)
        count_a = (batch_size - above_a).to(a.dtype)
        count_e = (batch_size - above_e).to(a.dtype)
        delta_correction = (sum_a - sum_e) / batch_size
        price_correction = spot * delta_correction - (count_a - count_e) / batch_size
        return price_correction, delta_correction

    def price(self, points):
        """Price, delta and standard errors at strike 1 for (n, 3) points."""
        points = np.asarray(points, dtype=float)
        spots = np.exp(points[:, 0])
        vols, maturities = points[:, 1], points[:, 2]
        bound, bound_delta = conditional_lower_bound_call(
            spots, 1.0, vols, self.rate, maturities, self.n_fixings
        )
        discount = np.exp(-self.rate * maturities)
        price, delta = bound.copy(), bound_delta.copy()
        price_se, delta_se = np.zeros(len(points)), np.zeros(len(points))

        keys = np.stack([vols, maturities], axis=1)
        unique_keys, group = np.unique(keys, axis=0, return_inverse=True)
        members = np.argsort(group, kind="stable")
        starts = np.searchsorted(group[members], np.arange(len(unique_keys) + 1))
        for index, (vol, maturity) in enumerate(unique_keys):
            chosen = members[starts[index] : starts[index + 1]]
            price_batches, delta_batches = self._group(vol, maturity, spots[chosen])
            scale = self._tensor(discount[chosen])
            price_batches, delta_batches = scale * price_batches, scale * delta_batches
            root = self.batches**0.5
            price[chosen] += price_batches.mean(dim=0).double().cpu().numpy()
            delta[chosen] += delta_batches.mean(dim=0).double().cpu().numpy()
            price_se[chosen] = (price_batches.std(dim=0) / root).double().cpu().numpy()
            delta_se[chosen] = (delta_batches.std(dim=0) / root).double().cpu().numpy()
        return {
            "price": price,
            "delta": delta,
            "price_se": price_se,
            "delta_se": delta_se,
            "distinct_vol_maturity_pairs": len(unique_keys),
        }

    def price_without_sorting(self, points):
        """The same estimator by direct sums. Used to test and to time the lookup."""
        points = np.asarray(points, dtype=float)
        price, delta = conditional_lower_bound_call(
            np.exp(points[:, 0]), 1.0, points[:, 1], self.rate, points[:, 2], self.n_fixings
        )
        for i, (log_moneyness, vol, maturity) in enumerate(points):
            spot = float(np.exp(log_moneyness))
            a, e = self.averages(float(vol), float(maturity))
            mismatch = (a > 1.0 / spot).to(a.dtype) - (e > 1.0 / spot).to(a.dtype)
            discount = float(np.exp(-self.rate * maturity))
            price[i] += discount * float(((spot * a - 1.0) * mismatch).mean())
            delta[i] += discount * float((a * mismatch).mean())
        return price, delta
