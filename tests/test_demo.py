"""The page is built from the recorded run, so these check the run reached the page intact."""

import json
import re
import shutil
from pathlib import Path

import pytest

from asianrisk.demo import write_demo

RUN = Path(__file__).parent.parent / "results" / "run"


@pytest.fixture(scope="module")
def page(tmp_path_factory) -> str:
    # Building the page rewrites the report, so work on a copy of the run.
    root = tmp_path_factory.mktemp("demo")
    shutil.copytree(RUN, root / "run")
    target = root / "site" / "index.html"
    write_demo(root / "run", target)
    return target.read_text()


@pytest.fixture(scope="module")
def data(page) -> dict:
    return json.loads(re.search(r"const DATA = (.*);\n", page).group(1))


def test_page_is_one_self_contained_file(page):
    assert not re.search(r"__[A-Z_]+__", page)
    assert page.count("data:font/woff2;base64,") == 3
    assert not re.search(r"""(src|href)=["']/""", page)
    assert "url('/" not in page
    assert "<script src" not in page and "<link rel=\"stylesheet\"" not in page


def test_map_holds_every_recorded_shock(data):
    assert len(data["spots"]) * len(data["vols"]) == len(data["scenarios"]) == 42
    assert {(row["spot"], row["vol"]) for row in data["scenarios"]} == {
        (spot, vol) for spot in data["spots"] for vol in data["vols"]
    }
    assert (0, 0) in {(row["spot"], row["vol"]) for row in data["scenarios"]}


def test_every_position_takes_one_route(data):
    for row in data["scenarios"]:
        for route in (row["neural"], row["grid"]):
            assert route["fast"] + route["simulated"] == data["positions"]


def test_values_match_the_run(data):
    book = json.loads((RUN / "raw" / "book_primary.json").read_text())
    recorded = {
        (row["spot_shift"], row["vol_shift"], row["method"]): row for row in book["scenarios_run0"]
    }
    for row in data["scenarios"]:
        for key, method in (("neural", "nn_diff"), ("grid", "grid_cubic")):
            source = recorded[(row["spot"], row["vol"], method)]
            assert row[key]["value"] == source["value"]
            assert row[key]["simulated"] == source["rejected"]


def test_the_multiple_names_its_owner_and_baseline(data):
    first = data["results"][0]
    assert first["value"] == "Cubic grid, 311x faster than tuned simulation"
    assert "no fallback" in first["scope"]
    assert data["results"][1]["label"] == "Cubic grid in the map"


def test_failed_hypotheses_stay_stated(data):
    values = [row["value"] for row in data["results"]]
    assert "Not supported: Neural + fallback" in values
    assert "Not supported: Neural correction + fallback" in values


def test_copy_rules(page):
    assert "Move the market and compare the price</p>" in page
    assert not re.search("[–—]", page.split("<script>")[0].split("</style>")[-1])
    assert not re.search("[–—]", page.split("<script>")[-1])
    lowered = page.lower()
    for banned in ("independent review", "generated from the run", "/users/", "preregistered"):
        assert banned not in lowered


def test_no_transition_touches_a_cell(page):
    """Selecting must jump, so nothing may tween between two recorded cases."""
    style = page.split("<style>")[1].split("</style>")[0]
    for rule in re.findall(r"(\.cell[^{]*|\.cross[^{]*|\.share[^{]*)\{([^}]*)\}", style):
        assert "transition" not in rule[1]
        assert "animation" not in rule[1]
