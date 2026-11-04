"""Development only: network shape and supervision on the validation set.

Run from the repository root:  uv run python experiments/dev_network.py
"""

import json
from functools import partial
from pathlib import Path

import numpy as np
import torch

from asianrisk.analytic import conditional_lower_bound_call
from asianrisk.domain import load_domain
from asianrisk.labels import load_dataset
from asianrisk.metrics import error_summary
from asianrisk.network import TrainConfig, train_surface

CONFIG = json.loads(Path("configs/experiment.json").read_text())
DOMAIN = load_domain(CONFIG)
RATE, N = CONFIG["rate"], CONFIG["n_fixings"]


def lower_bound(points):
    return conditional_lower_bound_call(
        np.exp(points[:, 0]), 1.0, points[:, 1], RATE, points[:, 2], N
    )


def main():
    torch.set_num_threads(6)
    train = load_dataset(Path("artifacts/data/train.npz"))
    validation = load_dataset(Path("artifacts/data/validation.npz"))
    all_points = len(train["points"])
    trials = [
        ("price_only", None, all_points, TrainConfig(delta_weight=0.0, epochs=200)),
        ("diff", None, all_points, TrainConfig(delta_weight=1.0, epochs=200)),
        ("diff_e600", None, all_points, TrainConfig(delta_weight=1.0, epochs=600)),
        ("residual_30k", lower_bound, 30000, TrainConfig(epochs=800, batch_size=1024)),
        ("residual_6k", lower_bound, 6000, TrainConfig(epochs=2000, batch_size=512)),
        ("residual_1500", lower_bound, 1500, TrainConfig(epochs=4000, batch_size=256)),
    ]
    rows = []
    for name, base, size, config in trials:
        surface, record = train_surface(
            DOMAIN, train["points"][:size], train["price"][:size], train["delta"][:size],
            config, base,
        )
        price, delta = surface(validation["points"])
        summary = error_summary(price, delta, validation["price"], validation["delta"])
        rows.append({"name": name, **record, **summary})
        print(json.dumps(rows[-1]), flush=True)
    Path("results/dev/network.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
