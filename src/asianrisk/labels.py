"""Generate reference labels for parameter points, in parallel and reproducibly.

Each point gets its own generator from (seed, index), so results do not depend
on the worker count and labels at different points have independent errors.
Reusing one path set across points would be cheaper, but it would correlate the
label noise and a model could learn that noise as if it were the surface.
"""

import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from asianrisk.contract import Contract, Market
from asianrisk.reference import SimulationConfig, price_reference

LABEL_FIELDS = ("price", "delta", "price_se", "delta_se", "seconds")
_CHUNK = 256


def _label_chunk(job):
    points, start, rate, n_fixings, n_units, seed = job
    rows = np.empty((len(points), len(LABEL_FIELDS)))
    for offset, (log_moneyness, vol, maturity) in enumerate(points):
        # Strike 1, so price is the price divided by strike.
        contract = Contract("label", "X", 1.0, float(maturity), n_fixings)
        market = Market(float(np.exp(log_moneyness)), float(vol), rate)
        config = SimulationConfig(
            n_paths=n_units, control="conditional", seed=(seed, start + offset)
        )
        result = price_reference(contract, market, config)
        rows[offset] = (
            result.price, result.delta, result.price_se, result.delta_se, result.runtime_seconds
        )
    return rows


def generate_labels(points, rate, n_fixings, n_units, seed, workers=None):
    """Labels for an (n, 3) array of points, with wall and summed CPU seconds."""
    workers = workers or os.cpu_count()
    jobs = [
        (points[i : i + _CHUNK], i, rate, n_fixings, n_units, seed)
        for i in range(0, len(points), _CHUNK)
    ]
    started = time.perf_counter()
    if workers == 1 or len(jobs) == 1:
        chunks = [_label_chunk(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            chunks = list(pool.map(_label_chunk, jobs))
    rows = np.concatenate(chunks) if chunks else np.empty((0, len(LABEL_FIELDS)))
    labels = {field: rows[:, i] for i, field in enumerate(LABEL_FIELDS)}
    timing = {
        "wall_seconds": time.perf_counter() - started,
        "cpu_seconds": float(labels["seconds"].sum()),
        "workers": workers,
    }
    return labels, timing


def save_dataset(path: Path, points, labels, metadata: dict) -> str:
    """Write the arrays and a JSON sidecar. Returns the SHA-256 of the arrays."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, points=points, **labels)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = {**metadata, "sha256": digest, "n_points": len(points)}
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    return digest


def load_dataset(path: Path):
    with np.load(path) as arrays:
        return {name: arrays[name] for name in arrays.files}
