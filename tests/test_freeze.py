"""A freeze has to hold: each way of changing a pinned input stops the sealed run."""

import json

import pytest

from asianrisk.cli import pinned_files, require_pins
from asianrisk.evaluate import run_sealed_evaluation, sha256
from asianrisk.pipeline import expected_models


@pytest.fixture
def workspace(tmp_path):
    out = tmp_path / "artifacts"
    books = tmp_path / "configs"
    (out / "models" / "differential").mkdir(parents=True)
    (out / "data").mkdir()
    books.mkdir()
    (out / "simulation_ladder.json").write_text("{}")
    (out / "models" / "grid_cubic.npz").write_bytes(b"grid")
    (out / "models" / "differential" / "arm.pt").write_bytes(b"network")
    (out / "data" / "train.npz").write_bytes(b"labels")
    (books / "book_primary.json").write_text("{}")
    record = {
        "commit": "unused",
        "books": {"primary": sha256(books / "book_primary.json")},
        "pinned": pinned_files(out),
    }
    return out, books, record


def check(record, out, books):
    require_pins(record, out, books, allow_dirty=True)


def test_an_untouched_workspace_passes(workspace):
    out, books, record = workspace
    check(record, out, books)


@pytest.mark.parametrize(
    "path", ["models/grid_cubic.npz", "models/differential/arm.pt", "data/train.npz", "simulation_ladder.json"]
)
def test_any_changed_input_is_refused(workspace, path):
    out, books, record = workspace
    (out / path).write_bytes(b"refit")
    with pytest.raises(SystemExit, match="changed after the freeze"):
        check(record, out, books)


def test_a_changed_book_is_refused(workspace):
    out, books, record = workspace
    (books / "book_primary.json").write_text('{"changed": true}')
    with pytest.raises(SystemExit, match="book primary changed"):
        check(record, out, books)


def test_a_new_model_is_refused(workspace):
    out, books, record = workspace
    (out / "models" / "nn_late.pt").write_bytes(b"late")
    with pytest.raises(SystemExit, match="appeared after the freeze"):
        check(record, out, books)


def test_sealed_labels_do_not_count_as_a_change(workspace):
    out, books, record = workspace
    (out / "data" / "test.npz").write_bytes(b"sealed")
    (out / "data" / "test.json").write_text("{}")
    check(record, out, books)


def test_a_recorded_run_is_never_replaced(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(SystemExit, match="never replaced"):
        run_sealed_evaluation({}, None, None, None, run_dir, {})


def test_expected_models_cover_every_fit():
    config = json.load(open("configs/experiment.json"))
    names = expected_models(config)
    seeds, section = len(config["network_seeds"]), config["differential"]
    differential = len(section["n_points"]) * 2 * len(section["seeds"])
    assert len(names) == len(config["grids"]) + len(config["networks"]) * seeds + differential
    assert "differential/price_delta_24000_seed2.pt" in names
