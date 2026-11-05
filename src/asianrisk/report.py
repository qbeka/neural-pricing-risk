"""Build the benchmark report, plots and verdict from raw JSON only.

Nothing here runs a pricer, so a report can be regenerated from a committed run
on any machine. The verdict applies the frozen decision rule in code.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reference palette, first three categorical slots, one per method family. Validated
# all-pairs for color vision deficiency. Marker shape carries identity as well.
FAMILY_COLORS = {"simulation": "#2a78d6", "non-neural fast": "#eb6834", "neural": "#1baf7a"}
FAMILY_MARKERS = {"simulation": "s", "non-neural fast": "o", "neural": "^"}
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
NEURAL = ("nn_price", "nn_diff", "nn_residual")


def load_run(run_dir: Path) -> dict:
    raw = Path(run_dir) / "raw"
    run = {path.stem: json.loads(path.read_text()) for path in raw.glob("*.json")}
    run["manifest"] = json.loads((Path(run_dir) / "manifest.json").read_text())
    return run


def label(row: dict) -> str:
    if row["route"] == "hybrid":
        return f"{row['method']} + fallback"
    return row["method"]


def aggregate_book(book: dict) -> list[dict]:
    """Mean and spread across runs for each route. Denominator is `runs`."""
    groups = defaultdict(list)
    for row in book["runs"]:
        groups[row["method"], row["route"]].append(row)
    summary = []
    for (method, route), rows in groups.items():
        seconds = [row["seconds"] for row in rows]
        summary.append({
            "method": method,
            "route": route,
            "runs": len(rows),
            "requests": book["requests"],
            "seconds_mean": float(np.mean(seconds)),
            "seconds_sd": float(np.std(seconds, ddof=1)) if len(rows) > 1 else 0.0,
            "cpu_seconds_mean": float(np.mean([row["cpu_seconds"] for row in rows])),
            "price_p99_bp": float(np.median([row["price_p99_bp"] for row in rows])),
            "price_p99_sd": float(np.std([row["price_p99_bp"] for row in rows])),
            "price_median_bp": float(np.median([row["price_median_bp"] for row in rows])),
            "delta_p99": float(np.median([row["delta_p99"] for row in rows])),
            "delta_p99_sd": float(np.std([row["delta_p99"] for row in rows])),
            "delta_median": float(np.median([row["delta_median"] for row in rows])),
            # Worst single request. The gate reads p99, so a fast route can hide 1% here.
            "price_max_bp": float(np.max([row["price_max_bp"] for row in rows])),
            "delta_max": float(np.max([row["delta_max"] for row in rows])),
            "book_error_max_bp": float(np.median(
                [row["book_value_error_max_bp_of_gross"] for row in rows])),
            "fallback_rate": float(np.mean([row["fallback_rate"] for row in rows])),
            "simulated_requests": float(np.mean([row["simulated_requests"] for row in rows])),
        })
    return sorted(summary, key=lambda row: row["seconds_mean"])


def passes(row: dict, tolerance: dict) -> bool:
    return row["price_p99_bp"] <= tolerance["price_bp"] and row["delta_p99"] <= tolerance["delta"]


def upfront_seconds(method: str, route: str, costs: dict) -> dict:
    """CPU and wall seconds to prepare a route. Calibration only counts for hybrids."""
    if method == "direct_mc":
        return {"cpu": 0.0, "wall": 0.0}
    cost = costs[method]
    cpu = cost.get("label_cpu", 0.0) + cost.get("train_cpu_upper", 0.0)
    wall = cost.get("label_wall", 0.0) + cost.get("train_wall", 0.0)
    if route == "hybrid":
        cpu += cost["calibration_cpu"]
        wall += cost["calibration_wall"]
    return {"cpu": cpu, "wall": wall}


def verdict(summary: list[dict], tolerance: dict) -> dict:
    """Apply the frozen rule. Returns the comparator and each hypothesis outcome."""
    passing = [row for row in summary if passes(row, tolerance)]
    non_neural = [row for row in passing if row["method"] not in NEURAL]
    comparator = min(non_neural, key=lambda row: row["seconds_mean"]) if non_neural else None

    def outcome(method: str) -> dict:
        route = next(r for r in summary if r["method"] == method and r["route"] == "hybrid")
        result = {
            "route": label(route),
            "passes_gate": passes(route, tolerance),
            "seconds_mean": route["seconds_mean"],
        }
        if comparator is None:
            # With nothing to beat, the rule cannot support a hypothesis.
            return {**result, "supported": False, "stretch": False, "speedup": None}
        margin = route["seconds_sd"] + comparator["seconds_sd"]
        result["speedup"] = comparator["seconds_mean"] / route["seconds_mean"]
        result["supported"] = (
            result["passes_gate"]
            and route["seconds_mean"] < comparator["seconds_mean"] - margin
        )
        result["stretch"] = result["supported"] and result["speedup"] >= 10.0
        return result

    return {
        "comparator": label(comparator) if comparator else None,
        "comparator_seconds": comparator["seconds_mean"] if comparator else None,
        "primary": outcome("nn_diff"),
        "secondary": outcome("nn_residual"),
    }


def break_even(row: dict, reference: dict, costs: dict) -> dict:
    """Jobs needed for a route's upfront cost to pay back against a reference route."""
    savings = reference["seconds_mean"] - row["seconds_mean"]
    upfront = upfront_seconds(row["method"], row["route"], costs)
    reference_upfront = upfront_seconds(reference["method"], reference["route"], costs)
    extra = upfront["wall"] - reference_upfront["wall"]
    if extra <= 0:
        jobs = 0.0 if savings > 0 else None
    else:
        jobs = extra / savings if savings > 0 else None
    return {"savings_per_job": savings, "extra_upfront_wall": extra, "jobs_to_recover": jobs}


def _family(row: dict) -> str:
    if row["method"] == "direct_mc":
        return "simulation"
    return "neural" if row["method"] in NEURAL else "non-neural fast"


def plot_accuracy_vs_compute(summary, ladder, tolerance, path: Path) -> None:
    """Wall time against p99 price error, one point per route plus the simulation ladder."""
    figure, axis = plt.subplots(figsize=(9.5, 5.8), dpi=160)
    seen = set()
    axis.plot(
        [row["seconds"] for row in ladder], [row["price_p99_bp"] for row in ladder],
        color=FAMILY_COLORS["simulation"], linewidth=2, zorder=2,
    )
    for row in (ladder[0], ladder[-1]):
        axis.annotate(
            f"2^{row['log2_units']} pairs", (row["seconds"], row["price_p99_bp"]),
            xytext=(0, 8), textcoords="offset points", fontsize=7.5, color=TEXT_SECONDARY, ha="center",
        )
    for row in summary:
        family = _family(row)
        axis.scatter(
            row["seconds_mean"], row["price_p99_bp"], s=64, color=FAMILY_COLORS[family],
            marker=FAMILY_MARKERS[family], edgecolor="white", linewidth=1.5, zorder=3,
            label=family if family not in seen else None,
        )
        seen.add(family)
    labelled = sorted(summary, key=lambda row: (row["seconds_mean"], row["price_p99_bp"]))
    for index, row in enumerate(labelled):
        passing = passes(row, tolerance)
        if not passing and row["route"] == "hybrid":
            continue
        axis.annotate(
            label(row), (row["seconds_mean"], row["price_p99_bp"]), xytext=(7, 6 if index % 2 else -12),
            textcoords="offset points", fontsize=7.5, color=TEXT_SECONDARY,
        )
    axis.axhline(tolerance["price_bp"], color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    axis.text(
        axis.get_xlim()[0], tolerance["price_bp"], f"  tolerance {tolerance['price_bp']:g} bp",
        va="bottom", fontsize=8, color=TEXT_SECONDARY,
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Portfolio wall time, seconds (log scale)", color=TEXT_PRIMARY)
    axis.set_ylabel("99th percentile |price error| / strike, bp (log scale)", color=TEXT_PRIMARY)
    axis.set_title("Accuracy against compute, primary book", loc="left", color=TEXT_PRIMARY, fontsize=12)
    axis.grid(True, color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    axis.legend(frameon=False, fontsize=8, loc="upper right")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def plot_fallback_heatmaps(heatmap, names, path: Path) -> None:
    """Fallback rate by moneyness and maturity, one panel per route."""
    figure, axes = plt.subplots(1, len(names), figsize=(4.1 * len(names), 4.0), dpi=160, sharey=True)
    axes = np.atleast_1d(axes)
    moneyness = np.exp(np.array(heatmap["log_moneyness"]))
    maturity = np.array(heatmap["maturity"])
    for axis, name in zip(axes, names):
        rates = np.array(heatmap["rates"][name]).T
        image = axis.pcolormesh(moneyness, maturity, rates, cmap="Blues", vmin=0.0, vmax=1.0, shading="nearest")
        axis.set_title(name, loc="left", fontsize=10, color=TEXT_PRIMARY)
        axis.set_xlabel("Moneyness S / K", color=TEXT_PRIMARY, fontsize=9)
        axis.tick_params(colors=TEXT_SECONDARY, labelsize=8)
        for side in axis.spines.values():
            side.set_visible(False)
    axes[0].set_ylabel("Maturity, years", color=TEXT_PRIMARY, fontsize=9)
    bar = figure.colorbar(image, ax=axes, shrink=0.85, pad=0.02)
    bar.set_label("Share of vols sent to simulation", fontsize=8, color=TEXT_SECONDARY)
    bar.ax.tick_params(labelsize=8, colors=TEXT_SECONDARY)
    bar.outline.set_visible(False)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _format_seconds(value: float) -> str:
    if value >= 1:
        return f"{value:.2f}"
    if value >= 1e-3:
        return f"{value * 1e3:.2f} ms"
    return f"{value * 1e6:.1f} us"


def book_table(summary, tolerance) -> list[str]:
    lines = [
        "| Route | Wall time (s, mean +/- sd) | CPU (s) | Requests per second | p99 price (bp) | Max price (bp) | p99 delta | Max delta | Passes gate | Sent to simulation |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {label(row)} | {row['seconds_mean']:.4g} +/- {row['seconds_sd']:.2g} "
            f"| {row['cpu_seconds_mean']:.3g} | {row['requests'] / row['seconds_mean']:,.0f} "
            f"| {row['price_p99_bp']:.3f} | {row['price_max_bp']:.2f} "
            f"| {row['delta_p99']:.5f} | {row['delta_max']:.4f} "
            f"| {'yes' if passes(row, tolerance) else 'no'} | {row['fallback_rate']:.1%} |"
        )
    return lines


def point_table(rows, dataset) -> list[str]:
    groups = defaultdict(list)
    for row in rows:
        if row["dataset"] == dataset:
            groups[row["method"], row["scope"]].append(row)
    lines = [
        "| Route | Scope | n | Median price (bp) | p99 price (bp) | Median delta | p99 delta | Sent to simulation |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (method, scope), members in sorted(groups.items()):
        lines.append(
            f"| {method} | {scope} | {members[0]['n']} "
            f"| {np.mean([m['price_median_bp'] for m in members]):.3f} "
            f"| {np.mean([m['price_p99_bp'] for m in members]):.3f} "
            f"| {np.mean([m['delta_median'] for m in members]):.5f} "
            f"| {np.mean([m['delta_p99'] for m in members]):.5f} "
            f"| {np.mean([m['fallback_rate'] for m in members]):.1%} |"
        )
    return lines


def write_report(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    run = load_run(run_dir)
    freeze = run["manifest"]["freeze"]
    tolerance = freeze["tolerance"]
    primary, bespoke = run["book_primary"], run["book_bespoke"]
    primary_summary = aggregate_book(primary)
    bespoke_summary = aggregate_book(bespoke)
    outcome = verdict(primary_summary, tolerance)
    costs = run["upfront"]

    direct = next(row for row in primary_summary if row["method"] == "direct_mc")
    comparator = next(
        (row for row in primary_summary if label(row) == outcome["comparator"]), None
    )
    recovery = []
    for row in primary_summary:
        if row["method"] == "direct_mc" or not passes(row, tolerance):
            continue
        entry = {"route": label(row), "upfront": upfront_seconds(row["method"], row["route"], costs)}
        entry["against_direct"] = break_even(row, direct, costs)
        if comparator is not None and row is not comparator:
            entry["against_comparator"] = break_even(row, comparator, costs)
        recovery.append(entry)

    plots = run_dir / "plots"
    plots.mkdir(exist_ok=True)
    plot_accuracy_vs_compute(
        primary_summary, primary["simulation_ladder"], tolerance, plots / "accuracy_vs_compute.png"
    )
    plot_fallback_heatmaps(
        run["points"]["heatmap"], ["corrected_grid", "nn_diff", "nn_residual"],
        plots / "fallback_heatmap.png",
    )
    summary = {
        "verdict": outcome, "primary": primary_summary, "bespoke": bespoke_summary,
        "recovery": recovery, "tolerance": tolerance,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (run_dir / "tables.md").write_text("\n".join(_tables(run, summary)) + "\n")
    return summary


def _tables(run, summary) -> list[str]:
    tolerance = summary["tolerance"]
    primary = run["book_primary"]
    bespoke = run["book_bespoke"]
    lines = ["# Generated tables", "", "Built by `python -m asianrisk report` from `raw/`. Do not edit by hand.", ""]
    for title, book, rows in (
        ("Primary book", primary, summary["primary"]),
        ("Bespoke book", bespoke, summary["bespoke"]),
    ):
        lines += [
            f"## {title}", "",
            f"Denominator: {book['requests']} requests ({book['contracts']} contracts by {book['scenarios']} "
            f"scenarios), {book['distinct_vol_maturity_pairs']} distinct vol and maturity pairs, "
            f"{rows[0]['runs']} runs. Errors are medians across runs.", "",
        ]
        lines += book_table(rows, tolerance) + [""]
    lines += ["## Upfront cost and recovery", "",
              "| Route | Upfront wall (s) | Upfront CPU (s, upper) | Saves per job vs direct (s) | Jobs to recover vs direct | Saves per job vs comparator (s) | Jobs to recover vs comparator |",
              "|---|---:|---:|---:|---:|---:|---:|"]

    def jobs(value):
        return "never" if value is None else f"{value:.1f}"

    for entry in summary["recovery"]:
        direct = entry["against_direct"]
        other = entry.get("against_comparator")
        lines.append(
            f"| {entry['route']} | {entry['upfront']['wall']:.0f} | {entry['upfront']['cpu']:.0f} "
            f"| {direct['savings_per_job']:.3f} | {jobs(direct['jobs_to_recover'])} "
            f"| {other['savings_per_job']:.4f} | {jobs(other['jobs_to_recover'])} |"
            if other else
            f"| {entry['route']} | {entry['upfront']['wall']:.0f} | {entry['upfront']['cpu']:.0f} "
            f"| {direct['savings_per_job']:.3f} | {jobs(direct['jobs_to_recover'])} | comparator | comparator |"
        )
    lines += ["", "## Simulation ladder, primary book, one run", "",
              "| Antithetic pairs | Wall time (s) | p99 price (bp) | p99 delta |", "|---:|---:|---:|---:|"]
    for row in primary["simulation_ladder"]:
        lines.append(
            f"| 2^{row['log2_units']} | {row['seconds']:.3f} | {row['price_p99_bp']:.3f} | {row['delta_p99']:.5f} |"
        )
    lines += ["", "## Single-request latency", "",
              "Median over 300 requests, one at a time, routing decision included. direct_mc uses 30 requests with paths already built.", "",
              "| Route | Latency |", "|---|---:|"]
    for name, value in sorted(primary["latency"].items(), key=lambda item: item[1]):
        lines.append(f"| {name} | {_format_seconds(value)} |")
    for dataset, title in (
        ("test", "Sealed ordinary test"),
        ("stress_near_expiry", "Stress: near expiry"),
        ("stress_kink", "Stress: payoff kink"),
    ):
        lines += ["", f"## {title}", "", "`fast_only` is the method alone on every point. `accepted` is the same method on the points it kept. `hybrid_all` adds simulation on the rest.", ""]
        lines += point_table(run["points"]["rows"], dataset)
    lines += ["", "## Differential comparison, sealed test", "",
              "| Train points | Paths per point | Arm | Seeds | p99 price (bp) | p99 delta |",
              "|---:|---:|---|---:|---:|---:|"]
    groups = defaultdict(list)
    for row in run["differential"]:
        groups[row["n_points"], row["arm"]].append(row)
    for (size, arm), members in sorted(groups.items()):
        lines.append(
            f"| {size} | {2 * members[0]['units_per_point']} | {arm} | {len(members)} "
            f"| {np.mean([m['price_p99_bp'] for m in members]):.2f} "
            f"| {np.mean([m['delta_p99'] for m in members]):.5f} |"
        )
    lines += ["", "## Accelerator sensitivity, direct simulation", "",
              "| Device | Precision | Wall (s) | Max price change vs 64-bit CPU (bp) | Max delta change |",
              "|---|---|---:|---:|---:|"]
    for row in run["accelerator"]["rows"]:
        lines.append(
            f"| {row['device']} | {row['dtype']} | {row['seconds']:.3f} "
            f"| {row['max_price_change_bp']:.4f} | {row['max_delta_change']:.5f} |"
        )
    noise = run["label_noise"]
    lines += ["", "## Reference noise", "",
              f"Sealed test labels relabeled with new seeds on {noise['n']} points: "
              f"price gap median {noise['price_gap_median_bp']:.4f} bp, p99 {noise['price_gap_p99_bp']:.4f} bp, "
              f"max {noise['price_gap_max_bp']:.4f} bp; delta gap median {noise['delta_gap_median']:.6f}, "
              f"p99 {noise['delta_gap_p99']:.6f}, max {noise['delta_gap_max']:.6f}.", ""]
    reference = primary["reference"]
    lines += [f"Book reference: {reference['replicates']} independent replicates of 2^{reference['log2_units_each']} antithetic pairs. "
              f"Price standard error median {reference['price_se_median_bp']:.4f} bp, max {reference['price_se_max_bp']:.4f} bp. "
              f"Delta standard error median {reference['delta_se_median']:.6f}, max {reference['delta_se_max']:.6f}."]
    lines += ["", "## Failure specimens", ""]
    for specimen in run["points"]["specimens"]:
        inputs = specimen["inputs"]
        reference_row = specimen["reference"]
        lines += [
            f"### {specimen['dataset']}", "",
            f"Worst answer of `{specimen['primary_hybrid']}` on this set"
            f" ({'among accepted points' if specimen['any_accepted'] else 'nothing was accepted, so among all points'}). "
            f"ln(S/K) = {inputs['log_moneyness']:.5f}, vol = {inputs['vol']:.5f}, T = {inputs['maturity']:.5f}. "
            f"Error is {specimen['score_in_tolerances']:.2f} tolerances.", "",
            f"Reference: price/K = {reference_row['price_over_strike']:.7f} +/- {reference_row['price_se']:.7f}, "
            f"delta = {reference_row['delta']:.6f} +/- {reference_row['delta_se']:.6f}.", "",
            "| Method | price/K | Error (bp) | delta | Error | Accepted |", "|---|---:|---:|---:|---:|---|",
        ]
        for method, answer in specimen["answers"].items():
            lines.append(
                f"| {method} | {answer['price_over_strike']:.7f} "
                f"| {(answer['price_over_strike'] - reference_row['price_over_strike']) / 1e-4:+.3f} "
                f"| {answer['delta']:.6f} | {answer['delta'] - reference_row['delta']:+.6f} "
                f"| {answer.get('accepted', 'n/a')} |"
            )
        lines.append("")
    return lines
