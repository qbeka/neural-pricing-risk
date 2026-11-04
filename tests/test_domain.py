"""The split is by region, deterministic, and blind to labels."""

import hashlib
import json
import subprocess
import sys
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


def _labels_command(tmp_path, freeze, config_path):
    return subprocess.run(
        [sys.executable, "-m", "asianrisk", "labels", "--config", str(config_path),
         "--out", str(tmp_path), "--freeze", str(freeze), "--sets", "test"],
        capture_output=True, text=True,
    )


def test_sealed_labels_need_a_valid_freeze_for_this_exact_config(tmp_path):
    config_path = Path(__file__).parent.parent / "configs" / "smoke.json"
    freeze = tmp_path / "freeze.json"
    assert _labels_command(tmp_path, freeze, config_path).returncode != 0  # missing
    freeze.write_text("")
    assert _labels_command(tmp_path, freeze, config_path).returncode != 0  # empty
    freeze.write_text(json.dumps({"config_sha256": "0" * 64}))
    assert _labels_command(tmp_path, freeze, config_path).returncode != 0  # other config
    assert not (tmp_path / "test.npz").exists()
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    freeze.write_text(json.dumps({"config_sha256": digest}))
    assert _labels_command(tmp_path, freeze, config_path).returncode == 0
    assert (tmp_path / "test.npz").exists()


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
