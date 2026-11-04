"""Development only: how far can a raw price network go on the validation set?

Run from the repository root:
  uv run python experiments/dev_network_raw.py 64 3
"""

import json
import sys
from pathlib import Path

import torch

from asianrisk.domain import load_domain
from asianrisk.labels import load_dataset
from asianrisk.metrics import error_summary
from asianrisk.network import TrainConfig, train_surface

CONFIG = json.loads(Path("configs/experiment.json").read_text())
DOMAIN = load_domain(CONFIG)


def main():
    width, depth = int(sys.argv[1]), int(sys.argv[2])
    torch.set_num_threads(3)
    train = load_dataset(Path("artifacts/data/train.npz"))
    validation = load_dataset(Path("artifacts/data/validation.npz"))
    size = 30000
    config = TrainConfig(width=width, depth=depth, epochs=2000, batch_size=1024)
    surface, record = train_surface(
        DOMAIN, train["points"][:size], train["price"][:size], train["delta"][:size], config
    )
    price, delta = surface(validation["points"])
    row = {**record, **error_summary(price, delta, validation["price"], validation["delta"])}
    Path(f"results/dev/network_raw_w{width}_d{depth}.json").write_text(json.dumps(row, indent=2) + "\n")


if __name__ == "__main__":
    main()
