# Research notes

Sources read, what was adopted, and what differs. Versions are pinned so a reader can check the same text.

## Sources

| Source | Version | Adopted | Different here |
|---|---|---|---|
| Horvath, Muguruza, Tomas, *Deep Learning Volatility*, [arXiv:1901.09647](https://arxiv.org/abs/1901.09647) | v2, 22 Aug 2019 | Learn the pricing map offline, then evaluate it online | Their task is rough-vol calibration on their hardware. Their timings say nothing about this contract, so none are quoted. Only the abstract was read, through a web summary. |
| Huge, Savine, *Differential Machine Learning*, [arXiv:2005.02347](https://arxiv.org/abs/2005.02347) | v4, 30 Sep 2020 | Pathwise derivatives supervise the derivative of the network | They train on single-path noisy labels with AAD. This project uses a closed-form pathwise delta and averages many paths per label. Only the abstract was read, through a web summary. |
| QuantLib `Examples/AsianOption/AsianOption.cpp` | commit `09edbae052edffbde9393eb097c21ba0b27c3e63`, wheel 1.44 | Geometric versus arithmetic distinction, engine choices for independent checks | The example is a put with dividend yield 3%, Actual/365, and a fixing on settlement. None of that matches this contract, so its printed numbers are not compared. |

Beyond the brief: the conditional lower bound (Curran 1994; Rogers and Shi 1995) and Turnbull-Wakeman moment matching (1991). The research question asks for the fastest credible numerical baseline, and for this contract those closed forms are the honest candidates.

## How the QuantLib comparison is matched

- Actual/360 day count on both curves, with fixings every whole number of days, so t_i = i T / n exactly.
- No dividend yield, no past fixings, running accumulator 0 for arithmetic and 1 for geometric.
- The last fixing is on the expiry date. There is no fixing today.
- `BlackScholesMertonProcess` takes the exact lognormal step for constant vol, so neither side has time discretization error.
- Independent seeds, and QuantLib has its own generator.

## Decisions and failed attempts

**FD disagreed with the reference, and the FD grid was the cause (resolved).** On the fixed contract `FdBlackScholesAsianEngine` at (t, x, a) = (400, 400, 200) gave 5.63343, about 12 standard errors from the reference at 5.63145. Changing the grids one at a time showed the sensitivity is in the average grid: a = 200 gives 5.63343 and a = 800 gives 5.63159. Refining t and x barely moves the answer. QuantLib's own control-variate Monte Carlo gave 5.63155 +/- 0.00009 and the reference gave 5.631726 +/- 0.000002, which differ by 1.9 standard errors. The error was in the interpolation of the average state, not in the reference. FD is therefore not used as a correctness oracle unless its average grid is refined.

**The geometric control runs out of power at high total vol.** At vol 70%, T = 2.5, standard error was 0.58 bp of strike at 2^21 antithetic pairs, against an intended tolerance of 1 bp. That fails the requirement that reference noise be materially smaller than the accepted error.

**First fix failed.** Replacing the geometric payoff with max(E[A|Z] - K, 0) lowered the standard error only about 10%. The geometric payoff and this payoff are both functions of Z alone, so both leave E[Var(payoff | Z)] untouched. Any control built only from Z has that floor.

**Second fix worked.** Adding the tangent term 1{E > K} (A - E), which has mean zero given Z, removes that part. The remaining residual is nonzero only where A and E straddle the strike.

| Spot, vol, T | Geometric SE (bp of K) | Conditional SE (bp of K) | Variance ratio |
|---|---|---|---|
| 100, 0.20, 1.0 | 0.0445 | 0.0023 | 390 |
| 100, 0.60, 1.0 | 0.469 | 0.035 | 178 |
| 100, 0.70, 2.5 | 2.47 | 0.158 | 244 |
| 140, 0.70, 2.5 | 3.67 | 0.211 | 302 |

Measured with 2^17 antithetic pairs on the same seed. It costs about 45% more per path. Delta standard error improves only 20 to 40%, because the delta residual is an indicator mismatch.

**The price of that control is a slower-settling standard error.** Its residual is a rare positive event. Repeating the estimate 1000 times, the z-score standard deviation was 2.95 at 2^10 units, 1.27 at 2^12, 1.06 at 2^14 and 1.08 at 2^16. The conditional control is used only at 2^14 units or more, and its standard errors are read as about 6% optimistic.

**Delta estimator: pathwise, not bumped.** The payoff is Lipschitz in spot, so the pathwise derivative is unbiased. A bumped price on common random numbers agrees within statistical error in `tests/test_reference.py`.

## First independent review (commit 4635f21)

A separate agent, a fresh Claude instance with only the acceptance criteria and the commit hash, reproduced the suite from a clean clone and tried to break the numerics. It is independent of this session's reasoning but not of its maker.

Confirmed by its own derivations and runs: geometric formula against QuantLib to 1e-13, lower bound against independent quadrature to 2e-14, n = 1 equals Black-Scholes, 40,000 fuzzed Newton solves with no failure, a second QuantLib cross-check at S = 87, K = 95, vol 37%, T = 0.75, n = 9.

Found and fixed: sealed sets opened with an empty freeze file; the fallback docstring claimed coverage among accepted points, which is false because acceptance is a selection (it now states the joint bound); the lower bound was wrong at exactly zero vol; unsupported calibration points could rescue a bin; several unvalidated inputs.

Found and accepted, with the reason:
- **The holdout is weak by design.** Blocks are 1/16 of each axis and every test block touches a train block. The sealed test measures filling gaps about one block wide, which is what production queries look like. It does not measure extrapolation. The reserved stress sets are the real out-of-sample check.
- **Grid nodes sit inside test and calibration blocks.** A lattice cannot skip regions. Calibration and test points are equally far from nodes, so calibration stays valid for grids, but grids do see those regions while networks do not. That asymmetry favors grids slightly and is stated in the report.
- **Standard errors go unreliable far from the money.** With rare control mismatches the sample error can read near zero (at (x, vol, T) = (0.45, 0.2, 1), about 26% coverage from nominal 95%). The true noise there is about 0.001 bp of strike, far below tolerance. Sealed labels get a second-seed replication, so label noise is measured directly rather than assumed.
