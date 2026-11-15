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


def test_every_file_uses_the_same_squares_as_the_region_mask():
    """One land mask per region: every map, count and analysis has the same squares."""
    import pandas as pd

    index = json.loads((RESULTS_DIR / "index.json").read_text(encoding="utf-8"))
    for region, variables in index["regions"].items():
        n = json.loads((RESULTS_DIR / region / "region.json").read_text(encoding="utf-8"))["n_land_cells"]
        squares = None
        for code in variables:
            d = RESULTS_DIR / region / code
            cells = pd.read_parquet(d / "cells.parquet")
            land = set(zip(cells.loc[cells["land"], "lat"], cells.loc[cells["land"], "lon"]))
            squares = squares or land
            assert land == squares, f"{region} {code}: different squares from the other variables"
            assert len(land) == n, f"{region} {code}: {len(land)} land squares, region mask has {n}"
            assert cells.loc[cells["land"], "slope_per_decade"].notna().sum() == n
            assert cells.loc[~cells["land"], "slope_per_decade"].isna().all()
            assert cells.loc[cells["land"], "pod_mode_1"].notna().sum() == n
            s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
            assert s["cells"]["n_land"] == n
            if "extremes" in s:
                assert s["extremes"]["max_trend_cells"]["n_land"] == n
                assert cells.loc[cells["land"], "max_trend_slope"].notna().sum() == n


def test_heat_index_extremes_hide_the_unreliable_level():
    for region in ("florida", "phoenix"):
        s = json.loads((RESULTS_DIR / region / "HEAT_INDEX" / "summary.json").read_text(encoding="utf-8"))
        assert s["extremes"]["show_level"] is False
        assert s["extremes"]["level_old"] is None and s["extremes"]["note"]
