"""Sealed evaluation. Run only after the freeze.

Raw measurements go to JSON under the run directory. Plots and tables are built
from those files by `report.py`, so nothing here formats results.

Unit of replication: one run pairs a network training seed with a simulation
seed. Closed forms and grids have no training seed, so across runs they vary only
through timing noise and through the simulation answers on rejected requests.
"""

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from asianrisk.domain import load_domain
from asianrisk.fastmc import PathEngine
from asianrisk.labels import generate_labels, load_dataset
from asianrisk.methods import load_network
from asianrisk.metrics import BASIS_POINT, error_summary
from asianrisk.pipeline import book_reference, build_rules, fast_methods
from asianrisk.portfolio import all_requests, load_book, shock_grid

SEALED_SETS = ("test", "stress_near_expiry", "stress_kink")
TIMING_REPEATS = 5


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _engine(config, log2_units, seed, device="cpu", dtype=torch.float64):
    return PathEngine(
        config["rate"], config["n_fixings"], 2**log2_units, seed, device=device, dtype=dtype
    )


def _routed(surface, rule, points, simulate):
    """Fast answers where accepted, simulation elsewhere. Returns arrays and the mask."""
    price, delta = surface(points)
    accepted = rule.accept(points)
    rejected = ~accepted
    if rejected.any():
        result = simulate(points[rejected])
        price, delta = price.copy(), delta.copy()
        price[rejected], delta[rejected] = result["price"], result["delta"]
    return price, delta, accepted


def evaluate_points(config, out, log2_units, seeds) -> dict:
    """Accepted-case and all-case error, by dataset, method and run."""
    domain = load_domain(config)
    rows, specimens, heatmap = [], [], None
    for run, seed in enumerate(seeds):
        surfaces = fast_methods(config, out, seed)
        rules = build_rules(config, out, seed, surfaces)
        engine = _engine(config, log2_units, seed=1000 + run)
        for dataset in SEALED_SETS:
            labels = load_dataset(out / "data" / f"{dataset}.npz")
            points = labels["points"]
            reference = (labels["price"], labels["delta"])
            simulated = engine.price(points)
            rows.append({
                "dataset": dataset, "run": run, "method": "direct_mc", "scope": "all",
                "fallback_rate": 1.0, **error_summary(simulated["price"], simulated["delta"], *reference),
            })
            for name, surface in surfaces.items():
                price, delta = surface(points)
                accepted = rule_mask = rules[name].accept(points)
                base = {"dataset": dataset, "run": run, "method": name}
                rate = float(1.0 - rule_mask.mean())
                rows.append({**base, "scope": "fast_only", "fallback_rate": 0.0,
                             **error_summary(price, delta, *reference)})
                if accepted.any():
                    rows.append({**base, "scope": "accepted", "fallback_rate": rate,
                                 **error_summary(price[accepted], delta[accepted],
                                                 *(r[accepted] for r in reference))})
                hybrid_price = np.where(accepted, price, simulated["price"])
                hybrid_delta = np.where(accepted, delta, simulated["delta"])
                rows.append({**base, "scope": "hybrid_all", "fallback_rate": rate,
                             **error_summary(hybrid_price, hybrid_delta, *reference)})
            if run == 0:
                specimens.append(_failure_specimen(dataset, labels, surfaces, rules, simulated, config))
        if run == 0:
            heatmap = _fallback_heatmap(domain, config, rules)
    return {"rows": rows, "specimens": specimens, "heatmap": heatmap}


def _failure_specimen(dataset, labels, surfaces, rules, simulated, config) -> dict:
    """The worst accepted answer of the primary hybrid, with every method's answer."""
    points = labels["points"]
    name = "nn_diff"
    price, delta = surfaces[name](points)
    accepted = rules[name].accept(points)
    score = np.maximum(
        np.abs(price - labels["price"]) / (config["tolerance"]["price_bp"] * BASIS_POINT),
        np.abs(delta - labels["delta"]) / config["tolerance"]["delta"],
    )
    score_for_choice = np.where(accepted, score, -1.0) if accepted.any() else score
    worst = int(np.argmax(score_for_choice))
    answers = {}
    for method, surface in surfaces.items():
        value, slope = surface(points[[worst]])
        answers[method] = {
            "price_over_strike": float(value[0]), "delta": float(slope[0]),
            "accepted": bool(rules[method].accept(points[[worst]])[0]),
        }
    answers["direct_mc"] = {
        "price_over_strike": float(simulated["price"][worst]),
        "delta": float(simulated["delta"][worst]),
    }
    log_moneyness, vol, maturity = points[worst]
    return {
        "dataset": dataset,
        "primary_hybrid": name,
        "any_accepted": bool(accepted.any()),
        "inputs": {"log_moneyness": float(log_moneyness), "vol": float(vol), "maturity": float(maturity)},
        "reference": {
            "price_over_strike": float(labels["price"][worst]), "delta": float(labels["delta"][worst]),
            "price_se": float(labels["price_se"][worst]), "delta_se": float(labels["delta_se"][worst]),
        },
        "score_in_tolerances": float(score[worst]),
        "answers": answers,
    }


def _fallback_heatmap(domain, config, rules, cells=24) -> dict:
    """Fallback rate on a regular moneyness by maturity lattice, averaged over vol."""
    moneyness = np.linspace(*domain.log_moneyness, cells)
    root_maturity = np.linspace(*np.sqrt(domain.maturity), cells)
    vols = np.linspace(*domain.vol, 25)
    grids = np.meshgrid(moneyness, vols, root_maturity**2, indexing="ij")
    points = np.column_stack([axis.ravel() for axis in grids])
    result = {"log_moneyness": moneyness.tolist(), "maturity": (root_maturity**2).tolist(), "rates": {}}
    for name, rule in rules.items():
        rejected = ~rule.accept(points).reshape(cells, len(vols), cells)
        result["rates"][name] = rejected.mean(axis=1).tolist()
    return result


def _timed(work, repeats=TIMING_REPEATS):
    """Run `work` several times and return its last result and the median seconds."""
    seconds = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = work()
        seconds.append(time.perf_counter() - started)
    return result, float(np.median(seconds))


def _book_summary(price, delta, strikes, quantities, scenario_count, reference):
    """Error against the reference at contract level and at book level per scenario."""
    target_price = reference["price_replicates"].mean(axis=0)
    target_delta = reference["delta_replicates"].mean(axis=0)
    summary = error_summary(price, delta, target_price, target_delta)
    value = (strikes * quantities * price).reshape(scenario_count, -1).sum(axis=1)
    target_value = (strikes * quantities * target_price).reshape(scenario_count, -1).sum(axis=1)
    gross = (strikes * np.abs(quantities)).reshape(scenario_count, -1).sum(axis=1)
    relative_error = np.abs(value - target_value) / gross / BASIS_POINT
    return {
        **summary,
        "book_value_error_median_bp_of_gross": float(np.median(relative_error)),
        "book_value_error_max_bp_of_gross": float(np.max(relative_error)),
    }, value


def benchmark_book(config, out, books_dir, book_name, log2_units, seeds) -> dict:
    """Full portfolio repricing, timed end to end for every route."""
    torch.set_num_threads(config["simulation"]["torch_threads"])
    book_path = Path(books_dir) / f"book_{book_name}.json"
    book = load_book(book_path)
    scenarios = shock_grid()
    points, strikes, quantities = all_requests(book, scenarios)
    n_scenarios = len(scenarios)

    started = time.perf_counter()
    reference = book_reference(config, points, seed_base=9_000_000 + sum(map(ord, book_name)))
    reference_seconds = time.perf_counter() - started
    target_price = reference["price_replicates"].mean(axis=0)
    target_delta = reference["delta_replicates"].mean(axis=0)
    replicate_values = (strikes * quantities * reference["price_replicates"]).reshape(
        len(reference["price_replicates"]), n_scenarios, -1
    ).sum(axis=2)
    reference_value = replicate_values.mean(axis=0)
    reference_value_se = replicate_values.std(axis=0, ddof=1) / np.sqrt(len(replicate_values))
    replicates = reference["price_replicates"].shape[0]

    rows, scenario_rows = [], []
    for run, seed in enumerate(seeds):
        surfaces = fast_methods(config, out, seed)
        rules = build_rules(config, out, seed, surfaces)
        simulation_seed = 2000 + run

        def direct():
            return _engine(config, log2_units, simulation_seed).price(points)

        simulated, direct_seconds = _timed(direct, repeats=1)
        summary, value = _book_summary(
            simulated["price"], simulated["delta"], strikes, quantities, n_scenarios, reference
        )
        rows.append({
            "run": run, "method": "direct_mc", "route": "simulation", "seconds": direct_seconds,
            "fallback_rate": 1.0, "simulated_requests": len(points), **summary,
        })
        for name, surface in surfaces.items():
            answer, seconds = _timed(lambda: surface(points))
            summary, value = _book_summary(*answer, strikes, quantities, n_scenarios, reference)
            rows.append({
                "run": run, "method": name, "route": "fast_only", "seconds": seconds,
                "fallback_rate": 0.0, "simulated_requests": 0, **summary,
            })
            rule = rules[name]

            def simulate(rejected_points):
                return _engine(config, log2_units, simulation_seed).price(rejected_points)

            (price, delta, accepted), seconds = _timed(
                lambda: _routed(surface, rule, points, simulate), repeats=1
            )
            summary, value = _book_summary(price, delta, strikes, quantities, n_scenarios, reference)
            rows.append({
                "run": run, "method": name, "route": "hybrid", "seconds": seconds,
                "fallback_rate": float(1.0 - accepted.mean()),
                "simulated_requests": int((~accepted).sum()), **summary,
            })
            if run == 0:
                by_scenario = accepted.reshape(n_scenarios, -1)
                for index, scenario in enumerate(scenarios):
                    scenario_rows.append({
                        "scenario_id": scenario.scenario_id, "spot_shift": scenario.spot_shift,
                        "vol_shift": scenario.vol_shift, "method": name,
                        "value": float(value.reshape(-1)[index]),
                        "accepted": int(by_scenario[index].sum()),
                        "rejected": int((~by_scenario[index]).sum()),
                    })
    latency = _single_request_latency(config, out, seeds[0], points, log2_units)
    return {
        "book": book_name,
        "book_sha256": sha256(book_path),
        "requests": len(points),
        "contracts": len(book.contracts),
        "gross_notional": float(np.sum(np.abs([c.quantity * c.strike for c in book.contracts]))),
        "scenarios": n_scenarios,
        "distinct_vol_maturity_pairs": len(np.unique(points[:, 1:], axis=0)),
        "reference": {
            "seconds": reference_seconds,
            "replicates": replicates,
            "log2_units_each": config["simulation"]["reference_log2_units"],
            "price_se_median_bp": float(np.median(
                reference["price_replicates"].std(axis=0, ddof=1) / np.sqrt(replicates)) / BASIS_POINT),
            "price_se_max_bp": float(np.max(
                reference["price_replicates"].std(axis=0, ddof=1) / np.sqrt(replicates)) / BASIS_POINT),
            "delta_se_median": float(np.median(
                reference["delta_replicates"].std(axis=0, ddof=1) / np.sqrt(replicates))),
            "delta_se_max": float(np.max(
                reference["delta_replicates"].std(axis=0, ddof=1) / np.sqrt(replicates))),
            "scenario_ids": [scenario.scenario_id for scenario in scenarios],
            "value": reference_value.tolist(),
            "value_se": reference_value_se.tolist(),
        },
        "runs": rows,
        "scenarios_run0": scenario_rows,
        "latency": latency,
        "target_price_sample": target_price[:3].tolist(),
        "target_delta_sample": target_delta[:3].tolist(),
    }


def _single_request_latency(config, out, seed, points, log2_units, samples=300) -> dict:
    """Median seconds for one request at a time, with structures already loaded."""
    rng = np.random.default_rng(55)
    chosen = points[rng.choice(len(points), samples, replace=False)]
    surfaces = fast_methods(config, out, seed)
    rules = build_rules(config, out, seed, surfaces)
    result = {}
    for name, surface in surfaces.items():
        seconds = []
        for row in chosen:
            query = row[None, :]
            started = time.perf_counter()
            surface(query)
            rules[name].accept(query)
            seconds.append(time.perf_counter() - started)
        result[name] = float(np.median(seconds))
    started = time.perf_counter()
    engine = _engine(config, log2_units, 3000)
    build_seconds = time.perf_counter() - started
    seconds = []
    for row in chosen[:30]:
        started = time.perf_counter()
        engine.price(row[None, :])
        seconds.append(time.perf_counter() - started)
    result["direct_mc"] = float(np.median(seconds))
    result["direct_mc_path_build"] = build_seconds
    return result


def accelerator_sensitivity(config, books_dir, log2_units) -> dict:
    """Direct simulation in 32-bit on CPU and MPS, transfers included in wall time."""
    book = load_book(Path(books_dir) / "book_primary.json")
    points, _, _ = all_requests(book, shock_grid())
    rows = []
    devices = [("cpu", torch.float64), ("cpu", torch.float32)]
    if torch.backends.mps.is_available():
        devices.append(("mps", torch.float32))
    baseline = None
    for device, dtype in devices:
        def work():
            return _engine(config, log2_units, 4000, device, dtype).price(points)

        result, seconds = _timed(work, repeats=3)
        if baseline is None:
            baseline = result
        rows.append({
            "device": device, "dtype": str(dtype), "seconds": seconds,
            "max_price_change_bp": float(np.max(np.abs(result["price"] - baseline["price"])) / BASIS_POINT),
            "max_delta_change": float(np.max(np.abs(result["delta"] - baseline["delta"]))),
        })
    return {"rows": rows}


def label_noise_check(config, out, size=2000) -> dict:
    """Relabel part of the test set with new seeds and measure the disagreement."""
    labels = load_dataset(out / "data" / "test.npz")
    points = labels["points"][:size]
    spec = config["datasets"]["test"]
    repeat, _ = generate_labels(
        points, config["rate"], config["n_fixings"], 2 ** spec["log2_units"], spec["seed"] + 100
    )
    price_gap = np.abs(repeat["price"] - labels["price"][:size]) / BASIS_POINT
    delta_gap = np.abs(repeat["delta"] - labels["delta"][:size])
    return {
        "n": len(points),
        "price_gap_median_bp": float(np.median(price_gap)),
        "price_gap_p99_bp": float(np.quantile(price_gap, 0.99)),
        "price_gap_max_bp": float(np.max(price_gap)),
        "delta_gap_median": float(np.median(delta_gap)),
        "delta_gap_p99": float(np.quantile(delta_gap, 0.99)),
        "delta_gap_max": float(np.max(delta_gap)),
    }


def upfront_costs(config, out) -> dict:
    """CPU and wall seconds to build each route, from the saved timing records."""
    def sidecar(name):
        return json.loads((out / "data" / f"{name}.json").read_text())

    train = sidecar("train")
    calibration = sidecar("calibration")
    costs = {"lower_bound": {}, "moment_matched": {}}
    for name in config["grids"]:
        record = json.loads((out / "models" / f"{name}.json").read_text())
        costs[name] = {"label_cpu": record["cpu_seconds"], "label_wall": record["wall_seconds"]}
    threads = config["simulation"]["torch_threads"]
    for name, spec in config["networks"].items():
        fits = [
            json.loads((out / "models" / f"{name}_seed{seed}.json").read_text())["record"]
            for seed in config["network_seeds"]
        ]
        train_wall = float(np.mean([fit["wall_seconds"] for fit in fits]))
        share = spec["n_points"] / train["n_points"]
        costs[name] = {
            "label_cpu": train["cpu_seconds"] * share,
            "label_wall": train["wall_seconds"] * share,
            "train_wall": train_wall,
            "train_cpu_upper": train_wall * threads,
            "parameters": fits[0]["parameters"],
        }
    for cost in costs.values():
        cost["calibration_cpu"] = calibration["cpu_seconds"]
        cost["calibration_wall"] = calibration["wall_seconds"]
    return costs


def evaluate_differential(config, out) -> list[dict]:
    """Sealed error of the price-only and delta-supervised arms at each budget."""
    domain = load_domain(config)
    section = config["differential"]
    test = load_dataset(out / "data" / "test.npz")
    train_spec = config["datasets"]["train"]
    rows = []
    for size in section["n_points"]:
        for arm in ("price_only", "price_delta"):
            for seed in section["seeds"]:
                path = out / "models" / "differential" / f"{arm}_{size}_seed{seed}.pt"
                surface = load_network(path, domain, config["rate"], config["n_fixings"])
                price, delta = surface(test["points"])
                rows.append({
                    "arm": arm, "seed": seed, "n_points": size,
                    "units_per_point": 2 ** train_spec["log2_units"],
                    **error_summary(price, delta, test["price"], test["delta"]),
                })
    return rows


def run_sealed_evaluation(config, config_path, out, books_dir, run_dir, freeze) -> None:
    """Every sealed measurement, written as raw JSON with a manifest."""
    raw = Path(run_dir) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    log2_units = freeze["simulation_log2_units"]
    seeds = config["network_seeds"][: config["simulation"]["runs"]]
    started = time.perf_counter()

    def write(name, payload):
        (raw / f"{name}.json").write_text(json.dumps(payload, indent=1) + "\n")
        print(f"{time.perf_counter() - started:8.1f}s wrote {name}", flush=True)

    write("label_noise", label_noise_check(config, out))
    write("upfront", upfront_costs(config, out))
    write("differential", evaluate_differential(config, out))
    write("points", evaluate_points(config, out, log2_units, seeds))
    for book_name in ("primary", "bespoke"):
        write(f"book_{book_name}", benchmark_book(config, out, books_dir, book_name, log2_units, seeds))
    write("accelerator", accelerator_sensitivity(config, books_dir, log2_units))
    manifest = {
        "commit": git_commit(),
        "command": f"python -m asianrisk evaluate --config {config_path} --run {run_dir}",
        "config_sha256": sha256(config_path),
        "freeze": freeze,
        "books": {
            name: sha256(Path(books_dir) / f"book_{name}.json")
            for name in config["books"]
        },
        "inputs": {
            path.name: sha256(path) for path in sorted((out / "data").glob("*.npz"))
        },
        "models": {
            path.name: sha256(path) for path in sorted((out / "models").glob("*.pt"))
        },
        "seeds": {"network": seeds, "simulation_log2_units": log2_units},
        "machine": machine_record(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (Path(run_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def machine_record() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
    }


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
