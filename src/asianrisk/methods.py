"""Every pricing route the benchmark compares, built from saved fit artifacts.

A surface is a callable: surface(points) -> (price / strike, delta).
"""

import json
from functools import partial
from pathlib import Path

import numpy as np
import torch

from asianrisk.analytic import conditional_lower_bound_call, moment_matched_call
from asianrisk.domain import Domain
from asianrisk.grid import GridSurface
from asianrisk.network import NeuralSurface, PriceNet


def closed_form(formula, rate, n_fixings, points):
    return formula(np.exp(points[:, 0]), 1.0, points[:, 1], rate, points[:, 2], n_fixings)


def lower_bound_surface(rate, n_fixings):
    return partial(closed_form, conditional_lower_bound_call, rate, n_fixings)


def moment_matched_surface(rate, n_fixings):
    return partial(closed_form, moment_matched_call, rate, n_fixings)


def _base(name, rate, n_fixings):
    return lower_bound_surface(rate, n_fixings) if name == "lower_bound" else None


def save_network(path: Path, surface: NeuralSurface, spec: dict, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(surface.net.state_dict(), path)
    path.with_suffix(".json").write_text(json.dumps({"spec": spec, "record": record}, indent=2) + "\n")


def load_network(path: Path, domain: Domain, rate, n_fixings) -> NeuralSurface:
    saved = json.loads(path.with_suffix(".json").read_text())
    spec = saved["spec"]
    state = torch.load(path, weights_only=True)
    net = PriceNet(spec["width"], spec["depth"]).to(torch.float64)
    net.load_state_dict(state)
    return NeuralSurface(domain, net.eval(), _base(spec["base"], rate, n_fixings))


def load_grid(path: Path, domain: Domain, rate, n_fixings) -> GridSurface:
    with np.load(path) as saved:
        price, delta = saved["price"], saved["delta"]
    spec = json.loads(path.with_suffix(".json").read_text())["spec"]
    base = _base(spec["base"], rate, n_fixings)
    return GridSurface.from_labels(domain, tuple(spec["shape"]), price, delta, spec["order"], base)
