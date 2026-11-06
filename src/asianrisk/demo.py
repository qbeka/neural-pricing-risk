"""Generate the offline demo page from a recorded run. Runs no pricer.

Every number on the page comes from the run's raw JSON, and every sentence that
states a result is built from those numbers, so the page cannot drift from the run.
"""

import base64
import json
from importlib.resources import files
from pathlib import Path

from asianrisk.report import load_run, passes, write_report

ROUTE_NAMES = {
    "direct_mc": "Tuned simulation",
    "nn_price": "Neural, price only",
    "nn_diff": "Neural",
    "nn_residual": "Neural correction",
    "corrected_grid": "Corrected grid",
    "lower_bound": "Lower bound",
    "moment_matched": "Moment matching",
    "grid_linear": "Linear grid",
    "grid_cubic": "Cubic grid",
}
# The two routes the map compares. Scenario rows exist only with fallback on.
COMPARED = {"neural": "nn_diff", "grid": "grid_cubic"}
# Edges of the map's colour steps, in book value units.
GAP_STEPS = (1, 2, 3, 5)
# Largest gap, in standard errors, still called agreement across the panel.
QUANTLIB_AGREEMENT_Z = 3.5
ASSUMPTIONS = (
    "Pricing model: risk-neutral geometric Brownian motion with constant rate and constant volatility, dividend yield zero.",
    "Contract: arithmetic-average Asian call at inception, 12 equally spaced future fixings with the last at maturity, no fixing history, payoff max(average - strike, 0) discounted to today.",
    "Simulation: exact lognormal transitions on fixing dates, antithetic pairs, a control variate with a closed-form mean, and pathwise delta.",
    "A shock moves spot and volatility and leaves time unchanged, so every contract stays at inception.",
    "The claim is numerical accuracy under this model. It says nothing about market prices or about risks the model leaves out, such as skew, jumps or moving rates.",
    "The fallback rule was calibrated on ordinary traffic, so it gives no guarantee in stress regions. All inputs are synthetic, and the timings come from one machine.",
)


def _seconds(value: float) -> str:
    if value >= 1:
        return f"{value:.2f} s"
    if value >= 1e-3:
        return f"{value * 1e3:.1f} ms"
    return f"{value * 1e6:.0f} us"


def _multiple(value: float) -> str:
    return f"{value:,.0f}x" if value >= 10 else f"{value:.1f}x"


def _find(summary, method, route):
    return next(row for row in summary if row["method"] == method and row["route"] == route)


def _scenarios(book) -> list[dict]:
    scenarios = {}
    for row in book["scenarios_run0"]:
        entry = scenarios.setdefault(row["scenario_id"], {
            "spot": row["spot_shift"], "vol": row["vol_shift"],
        })
        for key, method in COMPARED.items():
            if row["method"] == method:
                entry[key] = {
                    "value": row["value"], "fast": row["accepted"], "simulated": row["rejected"],
                }
    reference = book["reference"]
    for index, scenario_id in enumerate(reference["scenario_ids"]):
        scenarios[scenario_id]["reference"] = reference["value"][index]
        scenarios[scenario_id]["reference_se"] = reference["value_se"][index]
    return list(scenarios.values())


def _comparator_name(comparator: str) -> str:
    method = comparator.removesuffix(" + fallback")
    suffix = " + fallback" if comparator.endswith(" + fallback") else ""
    return ROUTE_NAMES[method] + suffix


def _headline_rows(summary) -> list[dict]:
    """The fastest route that met the gate, named beside its multiple and its baseline."""
    outcome = summary["verdict"]
    direct = _find(summary["primary"], "direct_mc", "simulation")
    grid = _find(summary["primary"], COMPARED["grid"], "hybrid")
    rows = []
    comparator = outcome["comparator"]
    if comparator:
        rows.append({
            "label": "Fastest at the gate",
            "value": (
                f"{_comparator_name(comparator)}, "
                f"{_multiple(direct['seconds_mean'] / outcome['comparator_seconds'])} faster than tuned simulation"
            ),
            "scope": (
                f"{_seconds(outcome['comparator_seconds'])} against {_seconds(direct['seconds_mean'])} "
                f"for the whole job of {direct['requests']:,} repricings, with no fallback. "
                f"Mean of {direct['runs']} runs on one machine, with synthetic contracts."
            ),
        })
    rows.append({
        "label": f"{ROUTE_NAMES[grid['method']]} in the map",
        "value": f"{_seconds(grid['seconds_mean'])} with fallback on",
        "scope": (
            f"It sent {grid['fallback_rate']:.1%} of requests to simulation, "
            "so the multiple above does not describe the route in the map."
        ),
    })
    return rows


def _hypothesis_rows(summary) -> list[dict]:
    outcome = summary["verdict"]
    against = (
        f" against {_seconds(outcome['comparator_seconds'])} for {_comparator_name(outcome['comparator'])}"
        if outcome["comparator"] else ""
    )
    rows = []
    for label, method, result in (
        ("Primary hypothesis", "nn_diff", outcome["primary"]),
        ("Secondary hypothesis", "nn_residual", outcome["secondary"]),
    ):
        row = _find(summary["primary"], method, "hybrid")
        rows.append({
            "label": label,
            "value": f"{'Supported' if result['supported'] else 'Not supported'}: {ROUTE_NAMES[method]} + fallback",
            "scope": (
                f"{_seconds(row['seconds_mean'])}{against}. {row['fallback_rate']:.1%} of requests "
                f"sent to simulation, p99 {row['price_p99_bp']:.2f} bp of strike and {row['delta_p99']:.5f} delta."
            ),
        })
    return rows


def _stress_p99(run, method: str, dataset: str) -> float:
    rows = [
        row["price_p99_bp"] for row in run["points"]["rows"]
        if row["method"] == method and row["dataset"] == dataset and row["scope"] == "fast_only"
    ]
    return sum(rows) / len(rows)


def _exploratory_rows(run, summary) -> list[dict]:
    """Observations outside the frozen rule. They are labeled that way on the page."""
    tolerance = summary["tolerance"]
    fast = [r for r in summary["primary"] if r["route"] == "fast_only" and passes(r, tolerance)]
    if not fast:
        return []
    best = min(fast, key=lambda row: row["price_p99_bp"])
    quickest = min(fast, key=lambda row: row["seconds_mean"])
    rows = [{
        "label": "Outside the frozen rule",
        "value": f"Most accurate fast route: {ROUTE_NAMES[best['method']]}",
        "scope": (
            f"Its worst of {best['requests']:,} requests missed by {best['price_max_bp']:.2f} bp and "
            f"{best['delta_max']:.4f} delta. {ROUTE_NAMES[quickest['method']]} reached "
            f"{quickest['price_max_bp']:.2f} bp and {quickest['delta_max']:.4f}."
        ),
    }]
    if best is not quickest:
        expiry = "stress_near_expiry"
        rows.append({
            "label": "Outside the frozen rule",
            "value": "The fastest route is the brittle one",
            "scope": (
                f"Below the fitted maturities, p99 price error was {_stress_p99(run, quickest['method'], expiry):,.0f} bp "
                f"for {ROUTE_NAMES[quickest['method']]} and {_stress_p99(run, best['method'], expiry):.2f} bp for "
                f"{ROUTE_NAMES[best['method']]}, which leans on a closed form."
            ),
        })
    return rows


def _check_rows(run, summary) -> list[dict]:
    tolerance = summary["tolerance"]
    gains = [
        row["efficiency_gain_vs_plain"] for row in run["variance_reduction"]
        if row["estimator"] == "conditional_control"
    ]
    rows = [{
        "label": "Simulation baseline",
        "value": f"{min(gains):,.0f}x to {max(gains):,.0f}x efficiency gain",
        "scope": "Against plain simulation, so the network had to beat a tuned engine.",
    }]
    panel = run["quantlib_panel"]
    if "skipped" not in panel:
        verb = "Agrees with" if panel["max_abs_z"] <= QUANTLIB_AGREEMENT_Z else "Differs from"
        rows.append({
            "label": "Outside check",
            "value": f"{verb} QuantLib on {panel['contracts']} random contracts",
            "scope": f"Largest gap {panel['max_abs_z']:.2f} standard errors, on separate seeds.",
        })
    noise = run["label_noise"]
    rows.append({
        "label": "Reference noise",
        "value": f"{noise['price_gap_p99_bp']:.3f} bp and {noise['delta_gap_p99']:.5f} delta",
        "scope": (
            f"p99 gap after relabeling {noise['n']:,} sealed points with new seeds, "
            f"{noise['price_gap_p99_bp'] / tolerance['price_bp']:.0%} and "
            f"{noise['delta_gap_p99'] / tolerance['delta']:.0%} of the tolerances."
        ),
    })
    return rows


def _limits(summary, runs: int) -> list[str]:
    """The map's own scope comes first, then the assumptions behind every number."""
    neural = _find(summary["primary"], COMPARED["neural"], "hybrid")
    return [
        (
            f"The map shows the first of {runs} recorded runs. Across runs, Neural sent "
            f"{neural['fallback_min']:.0%} to {neural['fallback_max']:.0%} of book requests to simulation."
        ),
        (
            "Errors in the map are for the whole book, against the reference, with longs and "
            "shorts netted, in bp of gross notional. The gate is per request, in bp of strike, "
            "so the two are not on one scale."
        ),
        *ASSUMPTIONS,
    ]


def build_data(run_dir: Path) -> dict:
    run = load_run(run_dir)
    summary = write_report(run_dir)
    book = run["book_primary"]
    tolerance = summary["tolerance"]
    outcome = summary["verdict"]
    primary = outcome["primary"]
    if outcome["comparator"]:
        verdict = (
            f"The neural route with fallback {'beat' if primary['supported'] else 'did not beat'} "
            f"the fastest non-neural route that met the accuracy gate, which was "
            f"{_comparator_name(outcome['comparator'])}."
        )
    else:
        verdict = (
            "No non-neural route met the accuracy gate, so the rule has nothing to compare "
            "against and neither hypothesis is supported."
        )
    supported = primary["supported"] or outcome["secondary"]["supported"]
    answer = "A frozen test says yes." if supported else "A frozen test says no."
    runs = summary["primary"][0]["runs"]
    scenarios = _scenarios(book)
    return {
        "question": (
            "Can a small neural network with simulation fallback beat the fastest numerical "
            f"baseline on a whole portfolio job? {answer}"
        ),
        "verdict": verdict,
        "gate": (
            f"The gate is p99 price within {tolerance['price_bp']:g} bp of strike and p99 delta "
            f"within {tolerance['delta']:g}, on {book['requests']:,} requests per job."
        ),
        "scenarios": scenarios,
        "spots": sorted({row["spot"] for row in scenarios}, reverse=True),
        "vols": sorted({row["vol"] for row in scenarios}),
        "gap_steps": list(GAP_STEPS),
        "positions": book["contracts"],
        "gross": book["gross_notional"],
        "runs": runs,
        "results": _headline_rows(summary) + _hypothesis_rows(summary) + _exploratory_rows(run, summary),
        "checks": _check_rows(run, summary),
        "limits": _limits(summary, runs),
    }


def write_demo(run_dir: Path, target: Path) -> None:
    template = (files("asianrisk") / "demo_template.html").read_text()
    for marker, name in (
        ("__FONT_SERIF__", "GeistSerif-Regular.woff2"),
        ("__FONT_SANS__", "Geist-Regular.woff2"),
        ("__FONT_SANS_MEDIUM__", "Geist-Medium.woff2"),
    ):
        font = (files("asianrisk") / "fonts" / name).read_bytes()
        template = template.replace(marker, base64.b64encode(font).decode())
    data = json.dumps(build_data(Path(run_dir)), separators=(",", ":")).replace("</", "<\\/")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(template.replace("__DATA__", data))
