"""Generate the offline demo page from a recorded run. Runs no pricer."""

import json
from importlib.resources import files
from pathlib import Path

from asianrisk.report import label, load_run, passes, write_report

ROUTES = ("nn_diff", "nn_residual", "corrected_grid", "lower_bound", "grid_cubic")
ASSUMPTIONS = (
    "Pricing model: risk-neutral geometric Brownian motion with constant rate and constant volatility, dividend yield zero.",
    "Contract: arithmetic-average Asian call at inception, 12 equally spaced future fixings with the last at maturity, no fixing history, cash payoff max(average - strike, 0) discounted to today.",
    "Simulation: exact lognormal transitions on fixing dates, antithetic pairs, a conditional control variate with a closed-form mean, and pathwise delta.",
    "A shock moves spot and volatility and leaves time unchanged, so every contract stays at inception.",
    "The claim is numerical accuracy under this model. It is not a claim about market prices, about any real instrument, or about risk that the model leaves out, such as skew, jumps or changing rates.",
    "The fallback rule was calibrated on ordinary traffic, so it gives no guarantee in stress regions. All inputs are synthetic.",
)


def build_data(run_dir: Path) -> dict:
    run = load_run(run_dir)
    summary = write_report(run_dir)
    book = run["book_primary"]
    tolerance = summary["tolerance"]
    reference = book["reference"]
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
    for index, scenario_id in enumerate(reference["scenario_ids"]):
        scenarios[scenario_id]["reference_value"] = reference["value"][index]
        scenarios[scenario_id]["reference_se"] = reference["value_se"][index]

    bench = [
        {
            "route": label(row),
            "seconds": f"{row['seconds_mean']:.3g} s",
            "price": f"{row['price_p99_bp']:.3f}",
            "delta": f"{row['delta_p99']:.5f}",
            "passes": "yes" if passes(row, tolerance) else "no",
            "fallback": f"{row['fallback_rate']:.1%}",
        }
        for row in summary["primary"]
    ]
    outcome = summary["verdict"]
    primary = outcome["primary"]
    comparator = outcome["comparator"] or "no non-neural route"
    verdict = (
        f"Result: the neural route with fallback "
        f"{'beat' if primary['supported'] else 'did not beat'} the best non-neural route that met "
        f"the accuracy gate ({comparator}) on this book."
    )
    manifest = run["manifest"]
    return {
        "scenarios": list(scenarios.values()),
        "routes": list(ROUTES),
        "contracts": book["contracts"],
        "gross": book.get("gross_notional", 1.0),
        "verdict": verdict,
        "bench": bench,
        "bench_note": (
            f"{book['requests']} requests, {primary and summary['primary'][0]['runs']} runs on one machine, "
            f"gate: p99 price within {tolerance['price_bp']:g} bp of strike and p99 delta within {tolerance['delta']:g}."
        ),
        "assumptions": list(ASSUMPTIONS),
        "provenance": f"Generated from run at commit {manifest['commit'][:10]} on {manifest['machine']['platform']}.",
    }


def write_demo(run_dir: Path, target: Path) -> None:
    template = (files("asianrisk") / "demo_template.html").read_text()
    data = build_data(Path(run_dir))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(template.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
