"""The committed results files match the schema the app relies on."""

import json

from config.catalog import load_catalog
from config.results import RESULTS_DIR, validate_results


def test_committed_results_are_valid():
    assert validate_results() == []


def test_every_region_has_every_variable_with_a_sentence():
    index = json.loads((RESULTS_DIR / "index.json").read_text(encoding="utf-8"))
    codes = list(load_catalog().variables)
    for region, variables in index["regions"].items():
        assert variables == codes
        for code in variables:
            s = json.loads((RESULTS_DIR / region / code / "summary.json").read_text(encoding="utf-8"))
            assert s["trend"]["sentence"]
            assert s["trend"]["verdict"] in ("SIGNIFICANT", "NOT SIGNIFICANT")


def test_extremes_variables_have_a_yearly_maximum_trend():
    for region in ("florida", "phoenix"):
        for code in load_catalog().extreme_codes():
            s = json.loads((RESULTS_DIR / region / code / "summary.json").read_text(encoding="utf-8"))
            mt = s["extremes"]["max_trend"]
            assert mt["sentence"].startswith(load_catalog()[code].yearly_max_name)
            assert mt["verdict"] in ("SIGNIFICANT", "NOT SIGNIFICANT")


def test_heat_index_extremes_hide_the_unreliable_level():
    for region in ("florida", "phoenix"):
        s = json.loads((RESULTS_DIR / region / "HEAT_INDEX" / "summary.json").read_text(encoding="utf-8"))
        assert s["extremes"]["show_level"] is False
        assert s["extremes"]["level_old"] is None and s["extremes"]["note"]
