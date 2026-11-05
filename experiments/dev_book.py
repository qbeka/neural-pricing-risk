"""Dry run of the portfolio benchmark on the development book, one seed.

It uses no sealed data. It exists to catch a crash or an absurd number before
the freeze, and its output stays out of every result.
"""

import json
from pathlib import Path

import numpy as np

from asianrisk.evaluate import benchmark_book
from asianrisk.report import aggregate_book

config = json.loads(Path("configs/experiment.json").read_text())
ladder = json.loads(Path("artifacts/simulation_ladder.json").read_text())
book = benchmark_book(
    config, Path("artifacts"), Path("configs"), "development", ladder["chosen_log2_units"], [0]
)
rows = aggregate_book(book)
Path("results/dev/book_development.json").write_text(json.dumps(rows, indent=1) + "\n")
for row in rows:
    print(
        f"{row['method']:16s} {row['route']:10s} {row['seconds_mean']:9.4f}s cpu {row['cpu_seconds_mean']:8.3f} "
        f"p99 {row['price_p99_bp']:7.3f} bp max {row['price_max_bp']:7.2f} delta {row['delta_p99']:.5f} "
        f"fallback {row['fallback_rate']:6.1%}"
    )
print(json.dumps({k: v for k, v in book["latency"].items()}, indent=1))
print("load", book["load_average"], "mean", np.mean(book["load_average"]["before"]))
