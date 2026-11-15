"""Export results for the app (Module 7).

Runs the trend, pattern and extremes engines on the prepared data and writes
small files to data/results/, in the shape described by config/results_schema.yaml:

  data/results/index.json
  data/results/<region>/region.json        name, box, land cell count
  data/results/<region>/outline.geojson    coastline / land outline for maps
  data/results/<region>/<CODE>/summary.json        numbers and plain sentences
  data/results/<region>/<CODE>/series_monthly.parquet
  data/results/<region>/<CODE>/series_annual.parquet
  data/results/<region>/<CODE>/cells.parquet       one row per grid cell (maps)
  data/results/<region>/<CODE>/pod_series.parquet

Run from the repository root (about 2 minutes, mostly the extremes bootstrap):
  python -m pipeline.heatpod.export
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from config.catalog import Region, Variable, load_catalog, load_regions
from config.results import RESULTS_DIR, load_schema, validate_results
from pipeline.heatpod import extremes, pod, trends

log = logging.getLogger("heatpod.export")

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_json(obj):
    """Make values JSON safe: NaN and infinity become null, numpy types become Python."""
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not math.isfinite(f) else round(f, 6)
    return obj


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(clean_json(data), indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def write_parquet(path: Path, df: pd.DataFrame) -> None:
    df.to_parquet(path, index=False)


# ---------------------------------------------------------------------------
# Region-level files
# ---------------------------------------------------------------------------

def outline_geojson(region: Region, tolerance: float = 0.02) -> dict:
    """Natural Earth 1:50m coastlines, state and country borders inside the box, as lines.

    Lines (not filled polygons) so the box edges are not drawn as fake coastline.
    State borders give inland regions (Phoenix) a map reference.
    """
    import regionmask
    from shapely.geometry import box, mapping

    ne = regionmask.defined_regions.natural_earth_v5_1_2
    b = region.box
    clip = box(b.lon_min, b.lat_min, b.lon_max, b.lat_max)
    features = []
    polygons = list(ne.land_50.polygons) + list(ne.us_states_50.polygons) + list(ne.countries_50.polygons)
    for poly in polygons:
        part = poly.boundary.intersection(clip)
        if part.is_empty:
            continue
        part = part.simplify(tolerance, preserve_topology=True)
        features.append({"type": "Feature", "properties": {},
                         "geometry": _round_coords(mapping(part))})
    return {"type": "FeatureCollection", "features": features}


def _round_coords(geom, digits: int = 3):
    def r(c):
        if isinstance(c, (list, tuple)) and c and isinstance(c[0], (int, float)):
            return [round(float(v), digits) for v in c]
        return [r(x) for x in c]
    return {"type": geom["type"], "coordinates": r(geom["coordinates"])}


# ---------------------------------------------------------------------------
# Variable-level files
# ---------------------------------------------------------------------------

def monthly_series(anom: xr.DataArray, t: trends.TrendResult) -> pd.DataFrame:
    """Regional anomaly with its trend line and the band of lines within the slope's 95% interval."""
    years = trends.decimal_years(anom["time"])
    y = anom.values.astype(float)
    t0, y0 = years.mean(), y.mean()            # least-squares lines pass through the means
    line = lambda slope: y0 + slope / 10 * (years - t0)  # noqa: E731
    low, high = line(t.ci_low), line(t.ci_high)
    return pd.DataFrame({
        "time": pd.to_datetime(anom["time"].values),
        "anomaly": y,
        "trend": line(t.slope_per_decade),
        "trend_low": np.minimum(low, high),
        "trend_high": np.maximum(low, high),
    })


def annual_series(ds: xr.Dataset) -> pd.DataFrame:
    anom = ds["region_monthly_anomaly"].groupby("time.year").mean()
    return pd.DataFrame({
        "year": ds["year"].values.astype("int64"),
        "mean": ds["region_annual_mean"].values.astype(float),
        "max": ds["region_annual_max"].values.astype(float),
        "anomaly": anom.sel(year=ds["year"]).values.astype(float),
    })


def cells_table(mask: xr.DataArray, cells: xr.Dataset, patterns: xr.DataArray,
                ext_cells: xr.Dataset | None) -> pd.DataFrame:
    lat2, lon2 = xr.broadcast(mask["lat"], mask["lon"])
    df = pd.DataFrame({
        "lat": lat2.transpose("lat", "lon").values.ravel().astype(float),
        "lon": lon2.transpose("lat", "lon").values.ravel().astype(float),
        "land": mask.values.ravel().astype(bool),
        "slope_per_decade": cells["slope_per_decade"].values.ravel(),
        "p_value": cells["p_value"].values.ravel(),
        "q_value": cells["q_value"].values.ravel(),
        "significant": cells["significant"].values.ravel().astype(bool),
    })
    for m in (1, 2, 3):
        df[f"pod_mode_{m}"] = patterns.sel(mode=m).transpose("lat", "lon").values.ravel() \
            if m in patterns["mode"].values else np.nan
    if ext_cells is not None:
        for k in ("level_old", "level_new", "change", "period_new"):
            df[f"extreme_{k}"] = ext_cells[k].values.ravel()
        df["max_trend_slope"] = ext_cells["max_trend_slope"].values.ravel()
        df["max_trend_significant"] = ext_cells["max_trend_significant"].values.ravel().astype(bool)
    return df


def pod_series_table(series: xr.DataArray) -> pd.DataFrame:
    df = series.to_dataframe(name="value").reset_index()[["time", "mode", "value"]]
    df["mode"] = df["mode"].astype("int64")
    return df


def export_variable(region: Region, var: Variable, start_year: int, out: Path) -> None:
    with xr.open_dataset(RAW_DIR / region.key / "prepared" / f"{var.code}.nc") as f:
        ds = f.load()
    mask = ds["land_mask"]
    anom_region = ds["region_monthly_anomaly"]

    t = trends.trend(anom_region)
    cells = trends.cell_trends(ds["monthly_anomaly"], mask)
    p = pod.pod(ds["monthly_anomaly"], mask)
    opp = pod.opposite_trends(cells)

    modes = []
    for m in p.series["mode"].values:
        mt = trends.trend(p.series.sel(mode=m))
        modes.append({
            "mode": int(m),
            "variance_fraction": float(p.variance_fraction[m - 1]),
            "separated": bool(p.separated[m - 1]),
            "series_slope_per_decade": mt.slope_per_decade,
            "series_p_value": mt.p_value,
            "series_verdict": trends.verdict_label(mt),
        })

    summary = {
        "region": region.key,
        "variable": var.code,
        "name": var.name,
        "unit": var.unit,
        "definition": var.definition,
        "why_it_matters": var.why_it_matters,
        "limitation": var.limitation,
        "period": {"start": start_year, "end": int(ds["year"].max())},
        "trend": {**t.to_dict(), "verdict": trends.verdict_label(t),
                  "sentence": trends.sentence(var, t, start_year),
                  "test": "Mann-Kendall (Hamed-Rao, 12 lags)", "data": "monthly anomalies"},
        "cells": {"n_land": int(mask.sum()), **opp},
        "pod": {"modes": modes, "n_eff": p.n_eff},
    }

    ext_cells = None
    if var.extremes:
        e = extremes.analyze(ds["region_annual_max"])
        ext_cells = extremes.cell_extremes(ds["annual_max"], mask)
        # Trend of each year's maximum: more power than comparing two 22-year periods.
        mt = trends.yearly_trend(ds["region_annual_max"])
        mt_cells = trends.cell_trends(ds["annual_max"], mask, fn=trends.yearly_trend)
        ext_cells = ext_cells.assign(max_trend_slope=mt_cells["slope_per_decade"],
                                     max_trend_significant=mt_cells["significant"])
        block = e.to_dict()
        if not var.extremes_show_level:
            for k in ("level_old", "level_new", "loc_old", "loc_new"):
                block[k] = None   # not reliable enough to show (see extremes_note)
        summary["extremes"] = {
            **block,
            "verdict": "SIGNIFICANT" if e.significant else "NOT SIGNIFICANT",
            "sentence": extremes.sentence(var, e),
            "show_level": var.extremes_show_level,
            # True when the fitted shape sits on its limit: the fit is constrained,
            # so the uncertainty range is probably too narrow. The app shows a caution.
            "shape_at_bound": bool(abs(e.shape) >= max(abs(b) for b in extremes.SHAPE_BOUNDS) - 1e-3),
            "max_trend": {**mt.to_dict(), "verdict": trends.verdict_label(mt),
                          "sentence": trends.sentence(var, mt, start_year, subject=var.yearly_max_name),
                          "subject": var.yearly_max_name,
                          "test": "Mann-Kendall (Hamed-Rao, 3 lags) and 95% range must agree",
                          "data": "yearly maximums"},
            "max_trend_cells": {"n_land": int(mask.sum()),
                                **pod.opposite_trends(mt_cells)},
            "note": var.extremes_note,
            "periods": [list(p_) for p_ in extremes.PERIODS],
            "return_period": extremes.RETURN_PERIOD,
        }
        if not var.extremes_show_level:
            ext_cells = ext_cells.assign(level_old=np.nan * ext_cells["level_old"],
                                         level_new=np.nan * ext_cells["level_new"])

    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", summary)
    write_parquet(out / "series_monthly.parquet", monthly_series(anom_region, t))
    write_parquet(out / "series_annual.parquet", annual_series(ds))
    write_parquet(out / "cells.parquet", cells_table(mask, cells, p.patterns, ext_cells))
    write_parquet(out / "pod_series.parquet", pod_series_table(p.series))


def export_all(results_dir: Path = RESULTS_DIR) -> list[str]:
    catalog, regions = load_catalog(), load_regions()
    # Start clean so no stale files survive (only generated files live here).
    for child in results_dir.glob("*"):
        if child.is_dir():
            shutil.rmtree(child)
        elif child.name != ".gitkeep":
            child.unlink()

    index = {"schema_version": load_schema()["version"],
             "period": {"start": catalog.start.year, "end": catalog.end.year},
             "regions": {}}
    for key, region in regions.items():
        rdir = results_dir / key
        rdir.mkdir(parents=True, exist_ok=True)
        codes = list(catalog.variables)
        for code in codes:
            log.info("%s %s", key, code)
            export_variable(region, catalog[code], catalog.start.year, rdir / code)
        with xr.open_dataset(RAW_DIR / key / "prepared" / f"{codes[0]}.nc") as f:
            n_land = int(f["land_mask"].sum())
        write_json(rdir / "region.json", {
            "key": key, "name": region.name,
            **_region_extra(key),
            "box": vars(region.box), "n_land_cells": n_land, "variables": codes,
        })
        write_json(rdir / "outline.geojson", outline_geojson(region))
        index["regions"][key] = codes
    write_json(results_dir / "index.json", index)
    return validate_results(results_dir)


def _region_extra(key: str) -> dict:
    """Display-only fields from regions.yaml: climate words and labelled places."""
    import yaml
    from config.catalog import REGIONS_FILE
    r = yaml.safe_load(REGIONS_FILE.read_text(encoding="utf-8"))["regions"][key]
    return {"climate": r.get("climate", ""), "places": r.get("places", [])}


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Export results for the app.").parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    errors = export_all()
    total = sum(p.stat().st_size for p in RESULTS_DIR.rglob("*") if p.is_file())
    if errors:
        for e in errors:
            log.error(e)
        return 1
    log.info("Results valid: %.2f MB in %s", total / 1e6, RESULTS_DIR.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
