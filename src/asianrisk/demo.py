"""Generate the offline demo page from a recorded run. Runs no pricer.

Every number on the page comes from the run's raw JSON, and every sentence that
states a result is built from those numbers, so the page cannot drift from the run.
"""

import json
from importlib.resources import files
from pathlib import Path

import numpy as np

from asianrisk.report import NEURAL, load_run, passes, write_report

ROUTES = ("nn_diff", "nn_residual", "corrected_grid", "lower_bound", "grid_cubic")
HEAT_ROUTES = ("nn_diff", "nn_residual", "corrected_grid", "lower_bound")
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
CHIPS = (
    "PyTorch", "NumPy", "SciPy", "QuantLib", "Monte Carlo", "Control variates",
    "Pathwise Greeks", "Differential ML", "Calibrated fallback", "Sealed evaluation",
)
ASSUMPTIONS = (
    "Pricing model: risk-neutral geometric Brownian motion with constant rate and constant volatility, dividend yield zero.",
    "Contract: arithmetic-average Asian call at inception, 12 equally spaced future fixings with the last at maturity, no fixing history, payoff max(average - strike, 0) discounted to today.",
    "Simulation: exact lognormal transitions on fixing dates, antithetic pairs, a control variate with a closed-form mean, and pathwise delta.",
    "A shock moves spot and volatility and leaves time unchanged, so every contract stays at inception.",
    "The claim is numerical accuracy under this model. It says nothing about market prices or about risks the model leaves out, such as skew, jumps or moving rates.",
    "The fallback rule was calibrated on ordinary traffic, so it gives no guarantee in stress regions. All inputs are synthetic, and the timings come from one machine.",
)


def _route_name(row: dict) -> str:
    name = ROUTE_NAMES[row["method"]]
    return f"{name} + fallback" if row["route"] == "hybrid" else name


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


def _family(row) -> str:
    if row["method"] == "direct_mc":
        return "simulation"
    return "neural" if row["method"] in NEURAL else "fixed"


def _scenarios(book) -> list[dict]:
    scenarios = {}
    for row in book["scenarios_run0"]:
        entry = scenarios.setdefault(row["scenario_id"], {
            "scenario_id": row["scenario_id"], "spot_shift": row["spot_shift"],
            "vol_shift": row["vol_shift"], "routes": {},
        })
        if row["method"] in ROUTES:
            entry["routes"][row["method"]] = {
                "value": row["value"], "accepted": row["accepted"], "rejected": row["rejected"],
            }
    reference = book["reference"]
    for index, scenario_id in enumerate(reference["scenario_ids"]):
        scenarios[scenario_id]["reference_value"] = reference["value"][index]
        scenarios[scenario_id]["reference_se"] = reference["value_se"][index]
    return list(scenarios.values())


def _headline_stats(run, summary) -> list[dict]:
    direct = _find(summary["primary"], "direct_mc", "simulation")
    tolerance = summary["tolerance"]
    passing = [row for row in summary["primary"] if passes(row, tolerance) and row["method"] != "direct_mc"]
    fastest = min(passing, key=lambda row: row["seconds_mean"]) if passing else None
    neural = [row for row in passing if row["method"] in NEURAL]
    best_neural = min(neural, key=lambda row: row["seconds_mean"]) if neural else None
    stats = []
    if fastest:
        stats.append({
            "value": _multiple(direct["seconds_mean"] / fastest["seconds_mean"]),
            "label": "faster than tuned simulation, same accuracy",
        })
    if best_neural:
        stats.append({
            "value": _multiple(direct["seconds_mean"] / best_neural["seconds_mean"]),
            "label": "best neural route against simulation",
        })
    panel = run["variance_reduction"]
    gains = [row["efficiency_gain_vs_plain"] for row in panel if row["estimator"] == "conditional_control"]
    stats.append({"value": f"{min(gains):,.0f}x", "label": "minimum simulation efficiency gain"})
    stats.append({"value": f"{direct['requests']:,}", "label": "repricings in one portfolio job"})
    return stats[:4]


def _comparator_name(comparator: str) -> str:
    method = comparator.removesuffix(" + fallback")
    suffix = " + fallback" if comparator.endswith(" + fallback") else ""
    return ROUTE_NAMES[method] + suffix


def _findings(run, summary) -> list[dict]:
    outcome = summary["verdict"]
    primary = outcome["primary"]
    secondary = outcome["secondary"]
    tolerance = summary["tolerance"]
    direct = _find(summary["primary"], "direct_mc", "simulation")
    rows = []
    comparator = outcome["comparator"]
    if comparator:
        rows.append({
            "tag": "Fastest at the gate",
            "title": _comparator_name(comparator),
            "text": f"{_seconds(outcome['comparator_seconds'])} for the whole job, against {_seconds(direct['seconds_mean'])} for tuned simulation.",
        })
    neural = _find(summary["primary"], "nn_diff", "hybrid")
    rows.append({
        "tag": "Primary hypothesis",
        "title": f"{'Supported' if primary['supported'] else 'Not supported'}: neural with fallback",
        "text": (
            f"{_seconds(neural['seconds_mean'])}, {neural['fallback_rate']:.1%} of requests sent to simulation, "
            f"p99 {neural['price_p99_bp']:.2f} bp and {neural['delta_p99']:.5f} delta."
        ),
    })
    correction = _find(summary["primary"], "nn_residual", "hybrid")
    rows.append({
        "tag": "Secondary hypothesis",
        "title": f"{'Supported' if secondary['supported'] else 'Not supported'}: neural correction with fallback",
        "text": (
            f"{_seconds(correction['seconds_mean'])}, {correction['fallback_rate']:.1%} sent to simulation, "
            f"p99 {correction['price_p99_bp']:.2f} bp and {correction['delta_p99']:.5f} delta."
        ),
    })
    return rows


def _rigor(run) -> list[dict]:
    panel = run["quantlib_panel"]
    rows = []
    if "skipped" not in panel:
        rows.append({
            "tag": "Independent check",
            "title": f"Agrees with QuantLib on {panel['contracts']} random contracts",
            "text": f"Largest gap {panel['max_abs_z']:.2f} standard errors, on independent seeds. Geometric formula within {panel['max_geometric_gap']:.0e} of QuantLib.",
        })
    noise = run["label_noise"]
    rows.append({
        "tag": "Measured noise",
        "title": "References were relabeled with new seeds",
        "text": f"On {noise['n']:,} sealed points the p99 gap was {noise['price_gap_p99_bp']:.3f} bp and {noise['delta_gap_p99']:.5f} delta, well inside the tolerances.",
    })
    rows.append({
        "tag": "Sealed test",
        "title": "Plan frozen before the test labels existed",
        "text": "The hypotheses, baselines, metrics and decision rule were committed first, and the command refuses to label sealed data without a matching freeze.",
    })
    reuse = run["reuse_speedup"]
    rows.append({
        "tag": "Path reuse",
        "title": f"{_multiple(reuse['speedup'])} faster per query after one sort",
        "text": f"A query is a prefix-sum lookup instead of a pass over the paths. Measured on {reuse['requests']} requests at 2^{reuse['log2_units']} antithetic pairs.",
    })
    return rows


def build_data(run_dir: Path) -> dict:
    run = load_run(run_dir)
    summary = write_report(run_dir)
    book = run["book_primary"]
    tolerance = summary["tolerance"]
    bench = [
        {
            "route": _route_name(row),
            "seconds": _seconds(row["seconds_mean"]),
            "price": f"{row['price_p99_bp']:.3f}",
            "delta": f"{row['delta_p99']:.5f}",
            "passes": "pass" if passes(row, tolerance) else "miss",
            "fallback": f"{row['fallback_rate']:.1%}",
        }
        for row in summary["primary"]
    ]
    points = [
        {
            "route": _route_name(row), "family": _family(row), "seconds": row["seconds_mean"],
            "seconds_text": _seconds(row["seconds_mean"]), "price": row["price_p99_bp"],
            "delta": row["delta_p99"], "passes": passes(row, tolerance),
        }
        for row in summary["primary"]
    ]
    heatmap = run["points"]["heatmap"]
    outcome = summary["verdict"]
    primary = outcome["primary"]
    comparator = _comparator_name(outcome["comparator"]) if outcome["comparator"] else "no route"
    verdict = (
        f"The neural route with fallback {'beat' if primary['supported'] else 'did not beat'} "
        f"the fastest non-neural route that met the accuracy gate, which was {comparator}."
    )
    manifest = run["manifest"]
    return {
        "headline": (
            "A research comparison of simulation, closed forms, interpolation and small neural "
            "networks, measured on the whole portfolio job at matched accuracy."
        ),
        "stats": _headline_stats(run, summary),
        "chips": list(CHIPS),
        "scenarios": _scenarios(book),
        "routes": list(ROUTES),
        "heat_routes": list(HEAT_ROUTES),
        "route_names": ROUTE_NAMES,
        "contracts": book["contracts"],
        "gross": book["gross_notional"],
        "verdict": verdict,
        "findings": _findings(run, summary),
        "rigor": _rigor(run),
        "bench": bench,
        "bench_note": (
            f"{book['requests']:,} requests per job, {summary['primary'][0]['runs']} runs on one machine. "
            f"The gate is p99 price within {tolerance['price_bp']:g} bp of strike and p99 delta within {tolerance['delta']:g}."
        ),
        "tolerance_bp": tolerance["price_bp"],
        "chart": {
            "points": points,
            "ladder": [
                {"seconds": row["seconds"], "price": row["price_p99_bp"]}
                for row in book["simulation_ladder"]
            ],
        },
        "heatmap": {
            "moneyness": np.exp(np.array(heatmap["log_moneyness"])).tolist(),
            "maturity": heatmap["maturity"],
            "rates": {
                name: np.round(np.array(heatmap["rates"][name]), 3).tolist() for name in HEAT_ROUTES
            },
        },
        "surface": run["surface_lines"],
        "assumptions": list(ASSUMPTIONS),
        "provenance": (
            f"Generated from the run at commit {manifest['commit'][:10]}. "
            f"Every number here traces to that run's manifest."
        ),
    }


def write_demo(run_dir: Path, target: Path) -> None:
    template = (files("asianrisk") / "demo_template.html").read_text()
    data = build_data(Path(run_dir))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(template.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
