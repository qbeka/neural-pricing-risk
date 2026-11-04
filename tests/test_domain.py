"""The split is by region, deterministic, and blind to labels."""

import json
from pathlib import Path

import numpy as np

from asianrisk.domain import (
    SPLITS,
    block_splits,
    load_domain,
    sample_dataset,
    split_of,
)
from asianrisk.labels import generate_labels

CONFIG = json.loads((Path(__file__).parent.parent / "configs" / "smoke.json").read_text())
DOMAIN = load_domain(CONFIG)
RATE = CONFIG["rate"]


def test_unit_mapping_round_trips():
    points = np.array([[0.1, 0.3, 1.2], [-0.2, 0.5, 0.2]])
    assert np.allclose(DOMAIN.from_unit(DOMAIN.to_unit(points)), points)


def test_every_block_has_exactly_one_split_in_the_requested_proportion():
    assignment = block_splits(CONFIG)
    assert len(assignment) == 16**3
    assert np.bincount(assignment).tolist() == [2458, 614, 614, 410]


def test_splits_are_deterministic_and_stay_inside_their_blocks():
    for index, name in enumerate(SPLITS):
        first = sample_dataset(name, DOMAIN, CONFIG)
        second = sample_dataset(name, DOMAIN, CONFIG)
        assert np.array_equal(first, second)
        assert len(first) > 0
        assert np.all(split_of(first, DOMAIN, CONFIG) == index)


def test_no_block_appears_in_two_splits():
    seen = {}
    for name in SPLITS:
        points = sample_dataset(name, DOMAIN, CONFIG)
        unit = np.clip((DOMAIN.to_unit(points) * 16).astype(int), 0, 15)
        seen[name] = {tuple(cell) for cell in unit}
    names = list(seen)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            assert not seen[left] & seen[right]


def test_reserved_stress_regions_are_absent_from_ordinary_splits():
    for name in SPLITS:
        points = sample_dataset(name, DOMAIN, CONFIG)
        assert not DOMAIN.in_kink_region(points, RATE).any()
        assert points[:, 2].min() >= DOMAIN.maturity[0]


def test_stress_sets_are_where_they_claim_to_be():
    near_expiry = sample_dataset("stress_near_expiry", DOMAIN, CONFIG)
    assert np.all(near_expiry[:, 2] < DOMAIN.maturity[0])
    assert not DOMAIN.supported(near_expiry, RATE).any()
    kink = sample_dataset("stress_kink", DOMAIN, CONFIG)
    assert len(kink) == CONFIG["datasets"]["stress_kink"]["max_points"]
    assert DOMAIN.in_kink_region(kink, RATE).all()


def test_labels_do_not_depend_on_the_worker_count():
    points = sample_dataset("validation", DOMAIN, CONFIG)[:300]
    serial, _ = generate_labels(points, RATE, 12, 2**8, seed=9, workers=1)
    parallel, _ = generate_labels(points, RATE, 12, 2**8, seed=9, workers=3)
    assert np.array_equal(serial["price"], parallel["price"])
    assert np.array_equal(serial["delta"], parallel["delta"])
