"""Development only: closed forms and grids against the validation set.

Run from the repository root:  uv run python experiments/dev_grids.py
"""

import json
import time
from functools import partial
from pathlib import Path

import numpy as np

from asianrisk.analytic import conditional_lower_bound_call, moment_matched_call
from asianrisk.domain import load_domain
from asianrisk.grid import GridSurface, node_points
from asianrisk.labels import generate_labels, load_dataset
from asianrisk.metrics import error_summary

CONFIG = json.loads(Path("configs/experiment.json").read_text())
DOMAIN = load_domain(CONFIG)
RATE, N = CONFIG["rate"], CONFIG["n_fixings"]
SHAPES = [(17, 9, 9), (33, 17, 17), (65, 33, 33)]
NODE_UNITS = 2**15


def closed_form(formula, points):
    return formula(np.exp(points[:, 0]), 1.0, points[:, 1], RATE, points[:, 2], N)


def main():
    validation = load_dataset(Path("artifacts/data/validation.npz"))
    points = validation["points"]
    print("median and max label SE, bp:",
          np.median(validation["price_se"]) * 1e4, validation["price_se"].max() * 1e4,
          "delta:", np.median(validation["delta_se"]), validation["delta_se"].max())
    rows = []

    def record(name, surface, **extra):
        started = time.perf_counter()
        price, delta = surface(points)
        seconds = time.perf_counter() - started
        summary = error_summary(price, delta, validation["price"], validation["delta"])
        rows.append({"method": name, "seconds_per_query": seconds / len(points), **extra, **summary})
        print(json.dumps(rows[-1]))

    lower_bound = partial(closed_form, conditional_lower_bound_call)
    record("lower_bound", lower_bound)
    record("moment_matched", partial(closed_form, moment_matched_call))
    for shape in SHAPES:
        nodes = node_points(DOMAIN, shape)
        labels, timing = generate_labels(nodes, RATE, N, NODE_UNITS, seed=100 + shape[0])
        for order in (1, 3):
            for base_name, base in (("none", None), ("lower_bound", lower_bound)):
                surface = GridSurface.from_labels(
                    DOMAIN, shape, labels["price"], labels["delta"], order, base
                )
                record(f"grid_{order}_{base_name}", surface, shape=shape, label_cpu=timing["cpu_seconds"])
    Path("results/dev/grids.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
