# Neural pricing and risk

A research tool that reprices a book of arithmetic-average Asian calls under spot and volatility shocks. It tests one question: can a small neural network with Monte Carlo fallback beat the fastest credible numerical baseline at the same price and delta error, on the whole portfolio job?

**The answer is no, and the reason is the useful part.** In a frozen, sealed test, a plain cubic grid beat the best neural route with fallback by 24 times. That closest network learns the error of a closed-form bound. Without fallback, it was the most accurate fast route in the study, an observation outside the frozen hypotheses.

| | |
|---|---|
| **Fastest route at the accuracy gate** | Cubic grid: 21,000 repricings in 11.0 ms, 311 times faster than tuned simulation (3.42 s), at p99 0.17 bp and 0.00076 delta |
| **Primary hypothesis** | Not supported. Neural with fallback took 3.11 s, with 74.8% to 100% of requests falling back across seeds |
| **Secondary hypothesis** | Not supported. Neural correction with fallback took 264 ms, 24 times slower than the cubic grid |
| **Most accurate fast route** | An observation, not a hypothesis. Neural correction with no fallback: p99 0.069 bp and 0.00011 delta, worst of 21,000 requests 0.19 bp and 0.0004 delta, in 30.1 ms |
| **Stress** | Near expiry, the cubic grid's p99 was 228 bp. The lower bound alone was 0.033 bp, and the neural correction kept that at 0.040 bp |
| **Simulation** | Control variates and antithetic pairs cut variance 5,723 to 476,425 times per unit of compute, against plain Monte Carlo |
| **Differential training** | On equal paths, delta labels cut p99 price error about 2.5 times and delta error about 2.8 times |
| **Correctness** | 110 tests, 92% line coverage. 24 random contracts agree with QuantLib within 2.36 standard errors |

Measured on one Apple M5 Pro with other work on the machine. These are synthetic contracts under geometric Brownian motion. [What the numbers leave out](#what-the-numbers-leave-out) has the details, and [docs/benchmark-report.md](docs/benchmark-report.md) has every result.

## Run it

You need [uv](https://docs.astral.sh/uv/). It installs Python 3.12 and the locked dependencies.

```bash
make setup
```

Price one contract through simulation and two closed forms.

```bash
uv run python -m asianrisk price --spot 100 --strike 100 --vol 0.2 --maturity 1
```

[Open the live demo](https://www.qendrimbeka.com/projects/neural-pricing-risk/index.html). Pick a recorded shock to compare the Neural and Cubic grid values, errors and routes. `site/index.html` is the same page offline.

## Check it

Run the correctness suite. It takes about one minute, and it includes a run of every stage on a tiny config.

```bash
make test
```

Run that tiny pipeline alone. It proves the pipeline works, and it proves no result.

```bash
make smoke
```

| Command | What it does | Time |
|---|---|---:|
| `make coverage` | Runs the tests and prints line coverage | 1 min |
| `make report` | Rebuilds tables, plots and the verdict from recorded raw data. Runs no pricer | Seconds |
| `make demo` | Rebuilds the page from the recorded run | Seconds |
| `make fit` | Labels development data, fits every surface and picks the simulation size | A few hours |
| `make freeze` | Pins the plan, code, models and data | Seconds |
| `make evaluate` | Labels the sealed sets and takes every measurement. Close other work first | About 1.5 hours |

`make freeze` and `make evaluate` refuse to replace the recorded run, and they need a clean tree. To repeat the experiment, use a fresh clone with no sealed labels, delete `results/freeze.json` and `results/run`, and commit that first.

## How it works

Each request is one contract under one scenario. Homogeneity reduces it to three inputs: log moneyness, volatility and maturity. Rate and the 12 fixings are fixed.

1. **Simulation** uses exact lognormal steps, antithetic pairs, a control variate from the conditional lower bound, and pathwise delta. The batched engine sorts once for each vol and maturity pair, then answers each strike with a prefix-sum lookup.
2. **Closed forms** are the geometric formula, Turnbull-Wakeman moment matching, and the Curran and Rogers-Shi conditional lower bound.
3. **Surfaces** are linear and cubic grids on price, a small grid on the correction to the lower bound, and three 64 by 3 networks: price only, price and delta, and the correction with delta.
4. **Fallback** accepts a request only inside the supported region and only in a cell whose calibration errors pass a conformal-style rank test. Everything else goes to simulation.
5. **The benchmark** times each route on the whole job, including fallback and path building, against a reference from 8 independent high-precision replicates.

## Why you can trust the numbers

| Control | How it works |
|---|---|
| Frozen plan | The hypotheses, baselines, metrics and rule were committed before any sealed label existed. `results/freeze.json` pins the hash of 79 files |
| Sealed labels | The command refuses to label sealed data without a matching freeze |
| Decision in code | `report.py` applies the rule to raw JSON. Nobody picks the verdict by hand |
| Tamper checks | `evaluate` refuses changed models, data, books, code or a dirty tree, and it never replaces a recorded run |
| Whole-job accounting | Hybrid wall time includes fallback and building simulation paths. Upfront cost includes labels, training and calibration |
| Independent check | QuantLib prices the same contracts on separate seeds |
| Stress kept apart | Near-expiry and kink sets are reported separately from ordinary traffic |

## What the numbers leave out

- One machine ran everything, with other work on it. Load average was 9 to 16 on 18 cores during the primary book, so trust ratios over absolute times.
- The gate reads the p99, so a route can hide 1% of requests. The cubic grid's worst delta error is 0.0053, five times the tolerance.
- Everything runs in Python. Compiled code would change the constants, and single-request latency most of all.
- Closed forms and grids use one thread, and PyTorch uses six. This helps networks and simulation, not grids or closed forms.
- The freeze says route order is shuffled, but direct simulation ran first in every run.
- The fallback maps show one seed, and fallback rates moved across seeds.
- The test has three input dimensions. Networks may win with more, such as varying rate, fixing history or a local volatility surface. That is a future target, and nothing here measures it.
- The claim is numerical accuracy under geometric Brownian motion at inception. It says nothing about market prices, skew, jumps or moving rates.
- Upfront cost omits validation labels, tuning and discarded experiments.

## Where to look

| Path | What it holds |
|---|---|
| [docs/benchmark-report.md](docs/benchmark-report.md) | Every result, and what it means |
| [docs/research-notes.md](docs/research-notes.md) | What was tried, including what failed |
| [docs/environment.md](docs/environment.md) | The machine and locked versions |
| `results/freeze.json` | The frozen plan and hashes |
| `results/run` | The sealed run: `manifest.json`, `raw/`, `tables.md`, `plots/` |
| `configs/` | The experiment, the plan text and the three books |
| `src/asianrisk/analytic.py` | Closed forms |
| `src/asianrisk/reference.py` | The reference simulator |
| `src/asianrisk/fastmc.py` | The batched simulation engine |
| `src/asianrisk/network.py` | Networks, with delta supervision |
| `src/asianrisk/fallback.py` | The fallback rule, and exactly what it guarantees |
| `src/asianrisk/evaluate.py` | Every sealed measurement |
| `src/asianrisk/report.py` | The verdict and the tables |

Built with PyTorch, NumPy, SciPy and QuantLib. Geist and Geist Serif are under the SIL Open Font License. MIT licensed.
