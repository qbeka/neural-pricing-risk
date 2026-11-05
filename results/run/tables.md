# Generated tables

Built by `python -m asianrisk report` from `raw/`. Do not edit by hand.

## Primary book

Denominator: 21000 requests (500 contracts by 42 scenarios), 576 distinct vol and maturity pairs, 5 runs. Errors are medians across runs.

| Route | Wall time (s, mean +/- sd) | CPU (s) | Requests per second | p99 price (bp) | Max price (bp) | p99 delta | Max delta | Passes gate | Sent to simulation |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| grid_linear | 0.007803 +/- 0.0015 | 0.00754 | 2,691,353 | 1.386 | 3.91 | 0.00128 | 0.00812 | no | 0.0% |
| grid_cubic | 0.011 +/- 0.0005 | 0.011 | 1,909,648 | 0.168 | 0.50 | 0.00076 | 0.00534 | yes | 0.0% |
| lower_bound | 0.01323 +/- 0.0015 | 0.013 | 1,586,710 | 3.713 | 8.65 | 0.00051 | 0.00104 | no | 0.0% |
| nn_price | 0.01426 +/- 0.0024 | 0.0492 | 1,473,118 | 4.674 | 49.25 | 0.01258 | 0.10890 | no | 0.0% |
| nn_diff | 0.01495 +/- 0.0033 | 0.0503 | 1,404,395 | 2.132 | 16.54 | 0.00404 | 0.04227 | no | 0.0% |
| moment_matched | 0.01533 +/- 0.0019 | 0.0147 | 1,369,632 | 79.092 | 163.41 | 0.01740 | 0.02540 | no | 0.0% |
| corrected_grid | 0.02053 +/- 0.0023 | 0.0195 | 1,023,013 | 0.126 | 0.33 | 0.00051 | 0.00104 | yes | 0.0% |
| nn_residual | 0.0301 +/- 0.008 | 0.0645 | 697,773 | 0.069 | 0.19 | 0.00011 | 0.00041 | yes | 0.0% |
| nn_residual + fallback | 0.2642 +/- 0.19 | 0.432 | 79,478 | 0.069 | 0.19 | 0.00013 | 0.00041 | yes | 1.4% |
| corrected_grid + fallback | 0.324 +/- 0.04 | 0.686 | 64,806 | 0.130 | 0.60 | 0.00052 | 0.00164 | yes | 2.0% |
| lower_bound + fallback | 1.155 +/- 0.16 | 2.55 | 18,179 | 0.876 | 1.75 | 0.00084 | 0.00164 | yes | 24.8% |
| grid_cubic + fallback | 1.329 +/- 0.087 | 3.15 | 15,799 | 0.216 | 0.95 | 0.00082 | 0.00164 | yes | 21.6% |
| grid_linear + fallback | 2.141 +/- 0.46 | 4.65 | 9,811 | 0.867 | 1.09 | 0.00068 | 0.00164 | yes | 41.9% |
| nn_price + fallback | 3.043 +/- 0.25 | 6.99 | 6,901 | 0.275 | 1.64 | 0.00087 | 0.00164 | yes | 99.4% (97.9% to 100.0%) |
| nn_diff + fallback | 3.109 +/- 0.22 | 7.1 | 6,753 | 0.632 | 1.57 | 0.00087 | 0.00280 | yes | 87.8% (74.8% to 100.0%) |
| moment_matched + fallback | 3.35 +/- 0.78 | 7.21 | 6,269 | 0.293 | 6.15 | 0.00094 | 0.00241 | yes | 98.2% |
| direct_mc | 3.419 +/- 0.45 | 7.32 | 6,142 | 0.244 | 0.95 | 0.00087 | 0.00164 | yes | 100.0% |

## Bespoke book

Denominator: 21000 requests (500 contracts by 42 scenarios), 2898 distinct vol and maturity pairs, 5 runs. Errors are medians across runs.

| Route | Wall time (s, mean +/- sd) | CPU (s) | Requests per second | p99 price (bp) | Max price (bp) | p99 delta | Max delta | Passes gate | Sent to simulation |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| grid_linear | 0.006939 +/- 0.00082 | 0.00668 | 3,026,300 | 1.394 | 3.76 | 0.00109 | 0.00457 | no | 0.0% |
| grid_cubic | 0.009691 +/- 0.00044 | 0.00947 | 2,166,955 | 0.149 | 0.42 | 0.00073 | 0.00408 | yes | 0.0% |
| lower_bound | 0.01027 +/- 0.00076 | 0.00976 | 2,045,289 | 2.744 | 5.89 | 0.00038 | 0.00070 | no | 0.0% |
| nn_diff | 0.01188 +/- 0.00062 | 0.0441 | 1,768,301 | 1.709 | 13.40 | 0.00240 | 0.03564 | no | 0.0% |
| moment_matched | 0.01361 +/- 0.0014 | 0.0134 | 1,543,310 | 59.737 | 111.68 | 0.01552 | 0.02026 | no | 0.0% |
| nn_price | 0.01394 +/- 0.004 | 0.0436 | 1,506,903 | 3.148 | 48.79 | 0.00757 | 0.09094 | no | 0.0% |
| corrected_grid | 0.01474 +/- 0.00053 | 0.0146 | 1,425,138 | 0.112 | 0.19 | 0.00050 | 0.00089 | yes | 0.0% |
| nn_residual | 0.02134 +/- 0.0018 | 0.0503 | 984,265 | 0.056 | 0.17 | 0.00011 | 0.00028 | yes | 0.0% |
| nn_residual + fallback | 0.4312 +/- 0.14 | 0.806 | 48,698 | 0.056 | 0.17 | 0.00012 | 0.00045 | yes | 0.8% |
| corrected_grid + fallback | 0.4604 +/- 0.11 | 0.962 | 45,612 | 0.114 | 0.34 | 0.00050 | 0.00123 | yes | 1.0% |
| lower_bound + fallback | 4.275 +/- 0.52 | 9.84 | 4,913 | 0.949 | 1.52 | 0.00078 | 0.00180 | yes | 18.6% |
| grid_cubic + fallback | 6.344 +/- 0.1 | 15 | 3,310 | 0.160 | 0.58 | 0.00076 | 0.00180 | yes | 28.5% |
| grid_linear + fallback | 7.858 +/- 0.72 | 18.3 | 2,673 | 0.873 | 1.16 | 0.00066 | 0.00180 | yes | 43.2% |
| nn_diff + fallback | 12.61 +/- 0.67 | 30.5 | 1,665 | 0.566 | 1.36 | 0.00082 | 0.00180 | yes | 88.8% (76.6% to 100.0%) |
| direct_mc | 12.82 +/- 0.65 | 31 | 1,639 | 0.194 | 0.58 | 0.00083 | 0.00180 | yes | 100.0% |
| moment_matched + fallback | 12.93 +/- 0.56 | 31 | 1,624 | 0.264 | 6.12 | 0.00092 | 0.00247 | yes | 98.0% |
| nn_price + fallback | 14.03 +/- 1.5 | 32 | 1,497 | 0.198 | 1.29 | 0.00083 | 0.00180 | yes | 99.9% (99.5% to 100.0%) |

## Upfront cost and recovery

One job is 21,000 queries.

| Route | Label wall (s) | Train wall (s) | Calibrate wall (s) | Upfront wall (s) | Upfront CPU (s, upper) | Saves per job vs direct (s) | Jobs to recover vs direct | Queries to recover vs direct | Saves per job vs comparator (s) | Jobs to recover vs comparator |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| grid_cubic | 39 | 0 | 0 | 39 | 579 | 3.408 | 11.6 | 243,043 | comparator | comparator |
| corrected_grid | 7 | 0 | 0 | 7 | 31 | 3.399 | 2.0 | 41,514 | -0.0095 | never |
| nn_residual | 4 | 80 | 0 | 84 | 554 | 3.389 | 24.8 | 520,231 | -0.0191 | never |
| nn_residual + fallback | 4 | 80 | 186 | 270 | 3738 | 3.155 | 85.7 | 1,799,778 | -0.2532 | never |
| corrected_grid + fallback | 7 | 0 | 186 | 193 | 3215 | 3.095 | 62.4 | 1,310,510 | -0.3130 | never |
| lower_bound + fallback | 0 | 0 | 186 | 186 | 3184 | 2.264 | 82.3 | 1,729,259 | -1.1442 | never |
| grid_cubic + fallback | 39 | 0 | 186 | 226 | 3762 | 2.090 | 108.1 | 2,269,593 | -1.3182 | never |
| grid_linear + fallback | 343 | 0 | 186 | 530 | 9245 | 1.279 | 414.2 | 8,698,040 | -2.1295 | never |
| nn_price + fallback | 21 | 346 | 186 | 554 | 5638 | 0.376 | 1,472.4 | 30,920,171 | -3.0321 | never |
| nn_diff + fallback | 21 | 457 | 186 | 665 | 6307 | 0.310 | 2,147.9 | 45,105,213 | -3.0985 | never |
| moment_matched + fallback | 0 | 0 | 186 | 186 | 3184 | 0.069 | 2,685.2 | 56,389,459 | -3.3388 | never |

## Simulation ladder, primary book, one run

| Antithetic pairs | Wall time (s) | p99 price (bp) | p99 delta |
|---:|---:|---:|---:|
| 2^10 | 0.202 | 0.955 | 0.00541 |
| 2^11 | 0.330 | 0.560 | 0.00372 |
| 2^12 | 0.525 | 0.533 | 0.00283 |
| 2^13 | 1.016 | 0.322 | 0.00176 |
| 2^14 | 2.063 | 0.245 | 0.00111 |
| 2^15 | 2.889 | 0.225 | 0.00082 |
| 2^16 | 3.579 | 0.153 | 0.00060 |

## Single-request latency

Median over 300 requests, one at a time, routing decision included. direct_mc uses 30 requests with paths already built.

| Route | Latency |
|---|---:|
| grid_cubic | 29.0 us |
| moment_matched | 41.2 us |
| nn_price | 49.1 us |
| nn_diff | 49.5 us |
| lower_bound | 84.6 us |
| grid_linear | 85.0 us |
| nn_residual | 120.9 us |
| corrected_grid | 157.6 us |
| direct_mc_path_build | 3.16 ms |
| direct_mc | 4.93 ms |

## Sealed ordinary test

`fast_only` is the method alone on every point. `accepted` is the same method on the points it kept. `hybrid_all` adds simulation on the rest. Values are means across seeds, and a seed that accepted nothing has no `accepted` row.

| Route | Scope | Seeds | n | Median price (bp) | p99 price (bp) | Median delta | p99 delta | Sent to simulation |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| corrected_grid | accepted | 5 | 19603 | 0.010 | 0.192 | 0.00008 | 0.00056 | 2.0% |
| corrected_grid | fast_only | 5 | 20000 | 0.011 | 0.200 | 0.00008 | 0.00059 | 0.0% |
| corrected_grid | hybrid_all | 5 | 20000 | 0.011 | 0.230 | 0.00008 | 0.00063 | 2.0% |
| direct_mc | all | 5 | 20000 | 0.010 | 0.357 | 0.00014 | 0.00102 | 100.0% |
| grid_cubic | accepted | 5 | 14583 | 0.010 | 0.218 | 0.00010 | 0.00074 | 27.1% |
| grid_cubic | fast_only | 5 | 20000 | 0.015 | 0.310 | 0.00013 | 0.00087 | 0.0% |
| grid_cubic | hybrid_all | 5 | 20000 | 0.014 | 0.341 | 0.00013 | 0.00099 | 27.1% |
| grid_linear | accepted | 5 | 12838 | 0.323 | 0.886 | 0.00013 | 0.00060 | 35.8% |
| grid_linear | fast_only | 5 | 20000 | 0.341 | 1.231 | 0.00015 | 0.00096 | 0.0% |
| grid_linear | hybrid_all | 5 | 20000 | 0.175 | 0.860 | 0.00013 | 0.00088 | 35.8% |
| lower_bound | accepted | 5 | 13393 | 0.053 | 0.813 | 0.00003 | 0.00023 | 33.0% |
| lower_bound | fast_only | 5 | 20000 | 0.190 | 5.686 | 0.00005 | 0.00079 | 0.0% |
| lower_bound | hybrid_all | 5 | 20000 | 0.058 | 0.771 | 0.00006 | 0.00099 | 33.0% |
| moment_matched | accepted | 5 | 447 | 0.000 | 5.238 | 0.00000 | 0.00233 | 97.8% |
| moment_matched | fast_only | 5 | 20000 | 4.053 | 115.891 | 0.00208 | 0.02306 | 0.0% |
| moment_matched | hybrid_all | 5 | 20000 | 0.010 | 0.434 | 0.00015 | 0.00109 | 97.8% |
| nn_diff | accepted | 4 | 2024 to 4149 | 0.272 | 0.941 | 0.00026 | 0.00102 | 85.7% |
| nn_diff | fast_only | 5 | 20000 | 0.293 | 1.757 | 0.00041 | 0.00288 | 0.0% |
| nn_diff | hybrid_all | 5 | 20000 | 0.013 | 0.599 | 0.00015 | 0.00102 | 88.5% |
| nn_price | accepted | 2 | 245 to 788 | 0.355 | 1.015 | 0.00041 | 0.00126 | 97.4% |
| nn_price | fast_only | 5 | 20000 | 0.530 | 3.932 | 0.00086 | 0.00780 | 0.0% |
| nn_price | hybrid_all | 5 | 20000 | 0.010 | 0.481 | 0.00014 | 0.00104 | 99.0% |
| nn_residual | accepted | 5 | 20000 | 0.014 | 0.125 | 0.00004 | 0.00031 | 0.0% |
| nn_residual | fast_only | 5 | 20000 | 0.014 | 0.125 | 0.00004 | 0.00031 | 0.0% |
| nn_residual | hybrid_all | 5 | 20000 | 0.014 | 0.125 | 0.00004 | 0.00031 | 0.0% |

## Stress: near expiry

`fast_only` is the method alone on every point. `accepted` is the same method on the points it kept. `hybrid_all` adds simulation on the rest. Values are means across seeds, and a seed that accepted nothing has no `accepted` row.

| Route | Scope | Seeds | n | Median price (bp) | p99 price (bp) | Median delta | p99 delta | Sent to simulation |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| corrected_grid | fast_only | 5 | 2500 | 0.004 | 0.050 | 0.00005 | 0.00028 | 0.0% |
| corrected_grid | hybrid_all | 5 | 2500 | 0.000 | 0.005 | 0.00004 | 0.00037 | 100.0% |
| direct_mc | all | 5 | 2500 | 0.000 | 0.005 | 0.00004 | 0.00037 | 100.0% |
| grid_cubic | fast_only | 5 | 2500 | 23.872 | 227.524 | 0.04335 | 0.21868 | 0.0% |
| grid_cubic | hybrid_all | 5 | 2500 | 0.000 | 0.005 | 0.00004 | 0.00037 | 100.0% |
| grid_linear | fast_only | 5 | 2500 | 24.992 | 228.114 | 0.04407 | 0.21539 | 0.0% |
| grid_linear | hybrid_all | 5 | 2500 | 0.000 | 0.005 | 0.00004 | 0.00037 | 100.0% |
| lower_bound | fast_only | 5 | 2500 | 0.001 | 0.033 | 0.00001 | 0.00009 | 0.0% |
| lower_bound | hybrid_all | 5 | 2500 | 0.000 | 0.005 | 0.00004 | 0.00037 | 100.0% |
| moment_matched | fast_only | 5 | 2500 | 0.078 | 1.649 | 0.00029 | 0.00249 | 0.0% |
| moment_matched | hybrid_all | 5 | 2500 | 0.000 | 0.005 | 0.00004 | 0.00037 | 100.0% |
| nn_diff | fast_only | 5 | 2500 | 5.150 | 50.298 | 0.01081 | 0.20309 | 0.0% |
| nn_diff | hybrid_all | 5 | 2500 | 0.000 | 0.005 | 0.00004 | 0.00037 | 100.0% |
| nn_price | fast_only | 5 | 2500 | 9.269 | 76.193 | 0.02034 | 0.26925 | 0.0% |
| nn_price | hybrid_all | 5 | 2500 | 0.000 | 0.005 | 0.00004 | 0.00037 | 100.0% |
| nn_residual | fast_only | 5 | 2500 | 0.006 | 0.040 | 0.00001 | 0.00010 | 0.0% |
| nn_residual | hybrid_all | 5 | 2500 | 0.000 | 0.005 | 0.00004 | 0.00037 | 100.0% |

## Stress: payoff kink

`fast_only` is the method alone on every point. `accepted` is the same method on the points it kept. `hybrid_all` adds simulation on the rest. Values are means across seeds, and a seed that accepted nothing has no `accepted` row.

| Route | Scope | Seeds | n | Median price (bp) | p99 price (bp) | Median delta | p99 delta | Sent to simulation |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| corrected_grid | fast_only | 5 | 2500 | 0.001 | 0.002 | 0.00005 | 0.00020 | 0.0% |
| corrected_grid | hybrid_all | 5 | 2500 | 0.000 | 0.001 | 0.00009 | 0.00038 | 100.0% |
| direct_mc | all | 5 | 2500 | 0.000 | 0.001 | 0.00009 | 0.00038 | 100.0% |
| grid_cubic | fast_only | 5 | 2500 | 0.046 | 0.614 | 0.00026 | 0.01492 | 0.0% |
| grid_cubic | hybrid_all | 5 | 2500 | 0.000 | 0.001 | 0.00009 | 0.00038 | 100.0% |
| grid_linear | fast_only | 5 | 2500 | 1.263 | 3.647 | 0.00136 | 0.01016 | 0.0% |
| grid_linear | hybrid_all | 5 | 2500 | 0.000 | 0.001 | 0.00009 | 0.00038 | 100.0% |
| lower_bound | fast_only | 5 | 2500 | 0.002 | 0.006 | 0.00002 | 0.00009 | 0.0% |
| lower_bound | hybrid_all | 5 | 2500 | 0.000 | 0.001 | 0.00009 | 0.00038 | 100.0% |
| moment_matched | fast_only | 5 | 2500 | 0.230 | 0.543 | 0.00053 | 0.00157 | 0.0% |
| moment_matched | hybrid_all | 5 | 2500 | 0.000 | 0.001 | 0.00009 | 0.00038 | 100.0% |
| nn_diff | fast_only | 5 | 2500 | 1.848 | 16.197 | 0.00538 | 0.05751 | 0.0% |
| nn_diff | hybrid_all | 5 | 2500 | 0.000 | 0.001 | 0.00009 | 0.00038 | 100.0% |
| nn_price | fast_only | 5 | 2500 | 8.038 | 48.754 | 0.01959 | 0.13473 | 0.0% |
| nn_price | hybrid_all | 5 | 2500 | 0.000 | 0.001 | 0.00009 | 0.00038 | 100.0% |
| nn_residual | fast_only | 5 | 2500 | 0.002 | 0.007 | 0.00002 | 0.00009 | 0.0% |
| nn_residual | hybrid_all | 5 | 2500 | 0.000 | 0.001 | 0.00009 | 0.00038 | 100.0% |

## Differential comparison, sealed test

| Train points | Paths per point | Arm | Seeds | p99 price (bp) | p99 delta |
|---:|---:|---|---:|---:|---:|
| 1500 | 32768 | price_delta | 3 | 1.91 | 0.00315 |
| 1500 | 32768 | price_only | 3 | 4.47 | 0.00884 |
| 6000 | 32768 | price_delta | 3 | 1.71 | 0.00310 |
| 6000 | 32768 | price_only | 3 | 4.57 | 0.00884 |
| 24000 | 32768 | price_delta | 3 | 1.79 | 0.00311 |
| 24000 | 32768 | price_only | 3 | 4.25 | 0.00883 |

## Accelerator sensitivity, direct simulation

| Device | Precision | Wall (s) | Max price change vs 64-bit CPU (bp) | Max delta change |
|---|---|---:|---:|---:|
| cpu | torch.float64 | 2.561 | 0.0000 | 0.00000 |
| cpu | torch.float32 | 2.350 | 0.0038 | 0.00003 |
| mps | torch.float32 | 1.429 | 0.0044 | 0.00003 |

## Reference noise

Sealed test labels relabeled with new seeds on 2000 points: price gap median 0.0033 bp, p99 0.1200 bp, max 0.2459 bp; delta gap median 0.000048, p99 0.000343, max 0.000458.

Book reference: 8 independent replicates of 2^19 antithetic pairs. Price standard error median 0.0012 bp, max 0.0391 bp. Delta standard error median 0.000021, max 0.000080.

## Failure specimens

### test

Worst answer of `nn_diff` on this set (among accepted points). ln(S/K) = -0.16666, vol = 0.24455, T = 1.00243. Error is 1.24 tolerances.

Reference: price/K = 0.0116888 +/- 0.0000002, delta = 0.175615 +/- 0.000072.

| Method | price/K | Error (bp) | delta | Error | Accepted |
|---|---:|---:|---:|---:|---|
| lower_bound | 0.0116779 | -0.109 | 0.175766 | +0.000151 | True |
| moment_matched | 0.0113991 | -2.897 | 0.176326 | +0.000711 | False |
| grid_linear | 0.0117412 | +0.524 | 0.175483 | -0.000132 | True |
| grid_cubic | 0.0116888 | -0.000 | 0.175706 | +0.000091 | True |
| corrected_grid | 0.0116877 | -0.011 | 0.175676 | +0.000060 | True |
| nn_price | 0.0116443 | -0.445 | 0.176341 | +0.000726 | False |
| nn_diff | 0.0117126 | +0.238 | 0.176857 | +0.001242 | True |
| nn_residual | 0.0116869 | -0.019 | 0.175753 | +0.000138 | True |
| direct_mc | 0.0116885 | -0.003 | 0.175149 | -0.000466 | n/a |

### stress_near_expiry

Worst answer of `nn_diff` on this set (nothing was accepted, so among all points). ln(S/K) = 0.01104, vol = 0.08676, T = 0.01320. Error is 382.43 tolerances.

Reference: price/K = 0.0113882 +/- 0.0000000, delta = 0.967252 +/- 0.000009.

| Method | price/K | Error (bp) | delta | Error | Accepted |
|---|---:|---:|---:|---:|---|
| lower_bound | 0.0113882 | -0.000 | 0.967250 | -0.000002 | False |
| moment_matched | 0.0113886 | +0.004 | 0.967166 | -0.000085 | False |
| grid_linear | 0.0145191 | +31.309 | 0.781342 | -0.185910 | False |
| grid_cubic | 0.0141293 | +27.411 | 0.749975 | -0.217277 | False |
| corrected_grid | 0.0113882 | +0.000 | 0.967177 | -0.000075 | False |
| nn_price | 0.0163690 | +49.808 | 0.569098 | -0.398153 | False |
| nn_diff | 0.0101736 | -12.146 | 0.584820 | -0.382431 | False |
| nn_residual | 0.0113889 | +0.007 | 0.967250 | -0.000002 | False |
| direct_mc | 0.0113882 | +0.000 | 0.967250 | -0.000002 | n/a |

### stress_kink

Worst answer of `nn_diff` on this set (nothing was accepted, so among all points). ln(S/K) = 0.01674, vol = 0.08733, T = 0.08816. Error is 103.67 tolerances.

Reference: price/K = 0.0192987 +/- 0.0000000, delta = 0.874038 +/- 0.000019.

| Method | price/K | Error (bp) | delta | Error | Accepted |
|---|---:|---:|---:|---:|---|
| lower_bound | 0.0192987 | -0.000 | 0.874055 | +0.000018 | False |
| moment_matched | 0.0193035 | +0.048 | 0.873972 | -0.000065 | False |
| grid_linear | 0.0195074 | +2.087 | 0.862550 | -0.011487 | False |
| grid_cubic | 0.0193043 | +0.056 | 0.838071 | -0.035967 | False |
| corrected_grid | 0.0192987 | -0.000 | 0.873996 | -0.000042 | False |
| nn_price | 0.0222402 | +29.415 | 0.677370 | -0.196668 | False |
| nn_diff | 0.0192976 | -0.011 | 0.770365 | -0.103673 | False |
| nn_residual | 0.0192991 | +0.004 | 0.874056 | +0.000018 | False |
| direct_mc | 0.0192987 | +0.000 | 0.874070 | +0.000033 | n/a |

