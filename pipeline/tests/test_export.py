"""Tests for the export helpers and the results schema checks."""

import json

import numpy as np
import pandas as pd
import xarray as xr

from config.results import load_schema, validate_results
from pipeline.heatpod import export, trends


def test_clean_json_turns_nan_and_infinity_into_null():
    out = export.clean_json({"a": np.nan, "b": [np.inf, np.float32(1.5)], "c": np.bool_(True)})
    assert out == {"a": None, "b": [None, 1.5], "c": True}
    json.dumps(out)  # valid JSON


def test_monthly_series_band_contains_the_trend_line():
    time = pd.date_range("1981-01-01", periods=540, freq="MS")
    y = 0.03 * np.arange(540) / 12 + np.random.default_rng(0).normal(0, 0.3, 540)
    anom = xr.DataArray(y, dims="time", coords={"time": time})
    df = export.monthly_series(anom, trends.trend(anom))
    assert list(df.columns) == ["time", "anomaly", "trend", "trend_low", "trend_high"]
    assert (df["trend_low"] <= df["trend"] + 1e-12).all()
    assert (df["trend"] <= df["trend_high"] + 1e-12).all()
    # The line rises by the slope per decade over each decade.
    rise = df["trend"].iloc[120] - df["trend"].iloc[0]
    assert rise == np.float64(rise) and abs(rise - trends.trend(anom).slope_per_decade) < 1e-9


def tiny_results(root):
    """A minimal valid results folder with one region and one variable."""
    (root / "r" / "V").mkdir(parents=True)
    (root / "index.json").write_text(json.dumps(
        {"schema_version": load_schema()["version"], "period": {}, "regions": {"r": ["V"]}}))
    (root / "r" / "region.json").write_text(json.dumps(
        {"key": "r", "name": "R", "climate": "", "box": {}, "n_land_cells": 1, "variables": ["V"]}))
    (root / "r" / "outline.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": []}))
    summary = {"region": "r", "variable": "V", "name": "V", "unit": "C", "period": {},
               "trend": {k: 0 for k in ["slope_per_decade", "ci_low", "ci_high",
                                        "sen_slope_per_decade", "p_value", "significant",
                                        "verdict", "sentence"]},
               "cells": {"n_land": 1, "rising": 0, "falling": 0, "opposite": False},
               "pod": {"modes": []}}
    (root / "r" / "V" / "summary.json").write_text(json.dumps(summary))
    t = pd.date_range("2000-01-01", periods=2, freq="MS")
    d = root / "r" / "V"
    pd.DataFrame({"time": t, "anomaly": [0.0, 1.0], "trend": 0.0, "trend_low": 0.0,
                  "trend_high": 0.0}).to_parquet(d / "series_monthly.parquet")
    pd.DataFrame({"year": [2000], "mean": [1.0], "max": [2.0], "anomaly": [0.0]}).to_parquet(
        d / "series_annual.parquet")
    pd.DataFrame({"lat": [0.0], "lon": [0.0], "land": [True], "slope_per_decade": [0.1],
                  "p_value": [0.5], "q_value": [0.5], "significant": [False],
                  "pod_mode_1": [1.0], "pod_mode_2": [0.0], "pod_mode_3": [0.0]}).to_parquet(
        d / "cells.parquet")
    pd.DataFrame({"time": t, "mode": [1, 1], "value": [0.0, 1.0]}).to_parquet(d / "pod_series.parquet")


def test_validator_accepts_a_valid_folder(tmp_path):
    tiny_results(tmp_path)
    assert validate_results(tmp_path) == []


def test_validator_reports_missing_keys_columns_and_files(tmp_path):
    tiny_results(tmp_path)
    d = tmp_path / "r" / "V"
    s = json.loads((d / "summary.json").read_text())
    del s["trend"]["sentence"]
    (d / "summary.json").write_text(json.dumps(s))
    pd.DataFrame({"year": [2000.5], "mean": [1.0], "max": [2.0]}).to_parquet(d / "series_annual.parquet")
    (d / "pod_series.parquet").unlink()
    errors = "\n".join(validate_results(tmp_path))
    assert "missing 'trend.sentence'" in errors
    assert "missing column 'anomaly'" in errors
    assert "column 'year' is float64, expected int" in errors
    assert "pod_series.parquet: missing file" in errors
