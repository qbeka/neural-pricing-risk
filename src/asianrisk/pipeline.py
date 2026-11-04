"""The experiment stages: fit, calibrate, then evaluate after the freeze.

Stages read and write under one artifact directory, so the smoke run and the
full run are the same code on different configs.
"""

import json
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from asianrisk.domain import load_domain, sample_dataset
from asianrisk.fallback import Tolerance, calibrate_rule
from asianrisk.grid import node_points
from asianrisk.labels import generate_labels, load_dataset, save_dataset
from asianrisk.methods import (
    _base,
    load_grid,
    load_network,
    lower_bound_surface,
    moment_matched_surface,
    save_network,
)
from asianrisk.metrics import error_summary
from asianrisk.network import TrainConfig, train_surface
from asianrisk.portfolio import all_requests, generate_book, load_book, save_book, shock_grid
from asianrisk.fastmc import PathEngine

DEVELOPMENT_SETS = ("train", "validation", "calibration")


def write_books(config: dict, directory: Path) -> None:
    """Generate each book once. An existing file is never replaced."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, spec in config["books"].items():
        path = directory / f"book_{name}.json"
        if path.exists():
            continue
        book = generate_book(
            name, spec["seed"], config["rate"], config["n_fixings"],
            spec["n_contracts"], spec["n_underlyings"], spec["bespoke_maturities"],
        )
        save_book(book, path)


def generate_datasets(config: dict, out: Path, names=DEVELOPMENT_SETS) -> None:
    domain = load_domain(config)
    for name in names:
        path = out / "data" / f"{name}.npz"
        if path.exists():
            continue
        spec = config["datasets"][name]
        points = sample_dataset(name, domain, config)
        values, timing = generate_labels(
            points, config["rate"], config["n_fixings"], 2 ** spec["log2_units"], spec["seed"]
        )
        save_dataset(path, points, values, {"name": name, **spec, **timing})


def fit_grids(config: dict, out: Path) -> None:
    """Label the lattice nodes of each grid. The grid itself is built on load."""
    domain = load_domain(config)
    for index, (name, spec) in enumerate(config["grids"].items()):
        path = out / "models" / f"{name}.npz"
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        nodes = node_points(domain, tuple(spec["shape"]))
        labels, timing = generate_labels(
            nodes, config["rate"], config["n_fixings"], 2 ** spec["log2_units"], 5000 + index
        )
        np.savez_compressed(path, price=labels["price"], delta=labels["delta"])
        record = {"spec": spec, "nodes": len(nodes), **timing}
        path.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")


def fit_networks(config: dict, out: Path) -> None:
    """Train every network on every seed, each on a prefix of the train set."""
    domain = load_domain(config)
    rate, n_fixings = config["rate"], config["n_fixings"]
    train = load_dataset(out / "data" / "train.npz")
    torch.set_num_threads(config["simulation"]["torch_threads"])
    known = {field.name for field in fields(TrainConfig)}
    for name, spec in config["networks"].items():
        for seed in config["network_seeds"]:
            path = out / "models" / f"{name}_seed{seed}.pt"
            if path.exists():
                continue
            size = spec["n_points"]
            settings = {key: value for key, value in spec.items() if key in known}
            surface, record = train_surface(
                domain, train["points"][:size], train["price"][:size], train["delta"][:size],
                TrainConfig(seed=seed, **settings), _base(spec["base"], rate, n_fixings),
            )
            save_network(path, surface, spec, record)
            print(json.dumps({"name": name, "seed": seed, "seconds": record["wall_seconds"]}), flush=True)


def fit_differential(config: dict, out: Path) -> None:
    """Price-only and delta-supervised fits on equal labeling budgets.

    Both arms use the same points and the same simulated paths, because the
    pathwise delta comes from those paths. The training step count is fixed, so
    a smaller set is not also handed fewer gradient steps.
    """
    domain = load_domain(config)
    section = config["differential"]
    train = load_dataset(out / "data" / "train.npz")
    torch.set_num_threads(config["simulation"]["torch_threads"])
    for size in section["n_points"]:
        steps_per_epoch = max(size // section["batch_size"], 1)
        epochs = max(section["steps"] // steps_per_epoch, 1)
        for arm, delta_weight in (("price_only", 0.0), ("price_delta", 1.0)):
            for seed in section["seeds"]:
                path = out / "models" / "differential" / f"{arm}_{size}_seed{seed}.pt"
                if path.exists():
                    continue
                spec = {
                    "base": None, "width": section["width"], "depth": section["depth"],
                    "arm": arm, "n_points": size,
                }
                settings = TrainConfig(
                    width=section["width"], depth=section["depth"], epochs=epochs,
                    batch_size=section["batch_size"], delta_weight=delta_weight, seed=seed,
                )
                surface, record = train_surface(
                    domain, train["points"][:size], train["price"][:size], train["delta"][:size],
                    settings,
                )
                save_network(path, surface, spec, record)


def fast_methods(config: dict, out: Path, seed: int) -> dict:
    """Surfaces by name. Networks use one training seed, the rest have no seed."""
    domain = load_domain(config)
    rate, n_fixings = config["rate"], config["n_fixings"]
    surfaces = {
        "lower_bound": lower_bound_surface(rate, n_fixings),
        "moment_matched": moment_matched_surface(rate, n_fixings),
    }
    for name in config["grids"]:
        surfaces[name] = load_grid(out / "models" / f"{name}.npz", domain, rate, n_fixings)
    for name in config["networks"]:
        path = out / "models" / f"{name}_seed{seed}.pt"
        surfaces[name] = load_network(path, domain, rate, n_fixings)
    return surfaces


def build_rules(config: dict, out: Path, seed: int, surfaces: dict) -> dict:
    """One fallback rule per fast method, from the calibration set only."""
    domain = load_domain(config)
    calibration = load_dataset(out / "data" / "calibration.npz")
    tolerance = Tolerance(**config["tolerance"])
    rules = {}
    for name, surface in surfaces.items():
        price, delta = surface(calibration["points"])
        rules[name] = calibrate_rule(
            domain, config["rate"], calibration["points"], price, delta,
            calibration["price"], calibration["delta"], tolerance,
            config["fallback"]["coverage"], config["fallback"]["bins"],
        )
    return rules


def book_reference(config: dict, points, seed_base: int) -> dict:
    """High-precision answers from independent path sets.

    Replicates are independent, so their spread gives a standard error for any
    aggregate, including a book value. The seeds never meet a benchmark seed.
    """
    section = config["simulation"]
    torch.set_num_threads(section["torch_threads"])
    prices, deltas = [], []
    for replicate in range(section["reference_replicates"]):
        engine = PathEngine(
            config["rate"], config["n_fixings"], 2 ** section["reference_log2_units"],
            seed=seed_base + replicate,
        )
        result = engine.price(points)
        prices.append(result["price"])
        deltas.append(result["delta"])
    return {"price_replicates": np.array(prices), "delta_replicates": np.array(deltas)}


def select_simulation_units(config: dict, out: Path, directory: Path) -> dict:
    """Smallest ladder step meeting tolerance on the development book.

    Uses only the development book, never a benchmark book, and picks the units
    used by direct simulation and by every fallback.
    """
    book = load_book(directory / "book_development.json")
    points, _, _ = all_requests(book, shock_grid())
    reference = book_reference(config, points, seed_base=7_000_000)
    target_price = reference["price_replicates"].mean(axis=0)
    target_delta = reference["delta_replicates"].mean(axis=0)
    section = config["simulation"]
    tolerance = config["tolerance"]
    ladder = []
    chosen = None
    for log2_units in section["ladder_log2_units"]:
        started = time.perf_counter()
        engine = PathEngine(config["rate"], config["n_fixings"], 2**log2_units, seed=11)
        result = engine.price(points)
        seconds = time.perf_counter() - started
        summary = error_summary(result["price"], result["delta"], target_price, target_delta)
        passes = summary["price_p99_bp"] <= tolerance["price_bp"] and (
            summary["delta_p99"] <= tolerance["delta"]
        )
        ladder.append({"log2_units": log2_units, "seconds": seconds, "passes": passes, **summary})
        if passes and chosen is None:
            chosen = log2_units
    if chosen is None:
        chosen = section["ladder_log2_units"][-1]
    return {"chosen_log2_units": chosen, "ladder": ladder}
