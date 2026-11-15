"""Cleaning and preparation (Module 3).

For each region and variable (the 4 POWER codes plus HEAT_INDEX):
1. Land mask: keep only grid cells whose center is on land (Natural Earth 1:50m).
2. Heat index: NOAA formula from daily max temperature and daily mean humidity.
3. Anomalies: monthly means minus each calendar month's long-term average
   (1991 to 2020, the WMO standard normal), which removes the seasonal cycle.
4. Yearly averages and yearly maximums, per grid cell and for the whole region.
5. Area weighting by cos(latitude), because grid cells shrink toward the poles.

Input:  data/raw/<region>/<CODE>.nc       (from download.py)
Output: data/raw/<region>/prepared/<CODE>.nc

Run from the repository root:
  python -m pipeline.heatpod.clean                 # every region
  python -m pipeline.heatpod.clean --region florida
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import xarray as xr

from config.catalog import Region, load_catalog, load_regions

log = logging.getLogger("heatpod.clean")

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
BASELINE = (1991, 2020)  # WMO standard climate normal period
HEAT_INDEX = "HEAT_INDEX"


# ---------------------------------------------------------------------------
# Land mask and area weights
# ---------------------------------------------------------------------------

def land_mask(lat: xr.DataArray, lon: xr.DataArray, state: str | None = None) -> xr.DataArray:
    """True where the grid cell center is on land (Natural Earth 1:50m land).

    With `state`, also inside that US state (Natural Earth 1:50m admin-1), so a
    box around Florida does not include squares in Alabama or Georgia.
    """
    import regionmask  # imported here: heavy, and only needed for real data

    ne = regionmask.defined_regions.natural_earth_v5_1_2
    mask = ne.land_50.mask(lon, lat).notnull()
    if state:
        states = ne.us_states_50
        mask = mask & (states.mask(lon, lat) == states.map_keys(state))
        mask = mask & ~on_land_border(lat, lon, state)
    return mask.rename("land_mask")


def on_land_border(lat: xr.DataArray, lon: xr.DataArray, state: str, tol: float = 0.01) -> xr.DataArray:
    """True where a cell center sits on the state's border with another state or country.

    Florida's northern border runs along 31.0 N, exactly where a row of grid
    centers lies; those squares are half in Alabama, so they are left out.
    Coastal squares are kept (the coast is not a land border).
    """
    import regionmask
    from shapely.geometry import Point
    from shapely.ops import unary_union

    ne = regionmask.defined_regions.natural_earth_v5_1_2
    poly = ne.us_states_50[state].polygon
    others = [p for n, p in zip(ne.us_states_50.names, ne.us_states_50.polygons) if n != state]
    others += [p for n, p in zip(ne.countries_50.names, ne.countries_50.polygons)
               if n != "United States of America"]
    neighbours = unary_union([p for p in others if p.distance(poly) < 0.05])
    border = poly.boundary.intersection(neighbours.buffer(0.02))
    hits = [[Point(float(lo), float(la)).distance(border) < tol for lo in lon.values] for la in lat.values]
    return xr.DataArray(np.array(hits), dims=("lat", "lon"), coords={"lat": lat.values, "lon": lon.values})


def area_weights(lat: xr.DataArray) -> xr.DataArray:
    """Relative cell area: proportional to cos(latitude)."""
    return np.cos(np.deg2rad(lat)).rename("area_weight")


def regional_mean(da: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    """Area-weighted mean over the land cells."""
    w = area_weights(da["lat"]).broadcast_like(mask).where(mask)
    return da.where(mask).weighted(w.fillna(0)).mean(("lat", "lon"))


# ---------------------------------------------------------------------------
# Heat index (NOAA Weather Prediction Center)
# https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml
# ---------------------------------------------------------------------------

def heat_index_f(t_f, rh):
    """NOAA heat index in Fahrenheit from temperature (F) and relative humidity (%)."""
    t = np.asarray(t_f, dtype=float)
    rh = np.asarray(rh, dtype=float)
    # Step 1: simple formula, averaged with the temperature.
    simple = 0.5 * (t + 61.0 + (t - 68.0) * 1.2 + rh * 0.094)
    hi = (simple + t) / 2.0
    # Step 2: where that is 80 F or more, use the Rothfusz regression.
    full = (-42.379 + 2.04901523 * t + 10.14333127 * rh - 0.22475541 * t * rh
            - 0.00683783 * t**2 - 0.05481717 * rh**2 + 0.00122874 * t**2 * rh
            + 0.00085282 * t * rh**2 - 0.00000199 * t**2 * rh**2)
    # Adjustments for very dry and for very humid air.
    dry = (rh < 13) & (t >= 80) & (t <= 112)
    full = np.where(dry, full - ((13 - rh) / 4) * np.sqrt(np.clip(17 - np.abs(t - 95), 0, None) / 17), full)
    wet = (rh > 85) & (t >= 80) & (t <= 87)
    full = np.where(wet, full + ((rh - 85) / 10) * ((87 - t) / 5), full)
    return np.where(hi >= 80, full, hi)


def heat_index_c(t_c, rh):
    """NOAA heat index in Celsius from temperature (C) and relative humidity (%)."""
    t_f = np.asarray(t_c, dtype=float) * 9 / 5 + 32
    return (heat_index_f(t_f, rh) - 32) * 5 / 9


def heat_index_grid(t2m_max: xr.DataArray, rh2m: xr.DataArray) -> xr.DataArray:
    """Daily heat index grid. Approximation: daily max temperature with daily mean humidity."""
    hi = xr.apply_ufunc(heat_index_c, t2m_max, rh2m)
    return hi.astype(np.float32).rename(HEAT_INDEX)


# ---------------------------------------------------------------------------
# Anomalies and yearly summaries
# ---------------------------------------------------------------------------

def to_monthly(daily: xr.DataArray) -> xr.DataArray:
    """Daily values to monthly means. The time axis is the first day of each month."""
    return daily.resample(time="MS").mean()


def monthly_climatology(monthly: xr.DataArray, baseline=BASELINE) -> xr.DataArray:
    """Average of each calendar month (1 to 12) over the baseline years."""
    years = monthly["time"].dt.year
    base = monthly.sel(time=(years >= baseline[0]) & (years <= baseline[1]))
    return base.groupby("time.month").mean("time")


def monthly_anomalies(monthly: xr.DataArray, baseline=BASELINE) -> xr.DataArray:
    """Monthly values minus the climatology of their calendar month."""
    clim = monthly_climatology(monthly, baseline)
    return (monthly.groupby("time.month") - clim).drop_vars("month")


def yearly(daily: xr.DataArray, how: str) -> xr.DataArray:
    """Yearly mean or max, with an integer 'year' axis."""
    g = daily.groupby("time.year")
    return g.mean("time") if how == "mean" else g.max("time")


def prepare_variable(daily: xr.DataArray, mask: xr.DataArray, baseline=BASELINE) -> xr.Dataset:
    """All Module 3 products for one variable in one region."""
    daily = daily.where(mask)
    monthly = to_monthly(daily)
    clim = monthly_climatology(monthly, baseline)
    anom = monthly_anomalies(monthly, baseline)
    region_daily = regional_mean(daily, mask)

    ds = xr.Dataset({
        "land_mask": mask.drop_vars([c for c in mask.coords if c not in ("lat", "lon")]),
        "area_weight": area_weights(daily["lat"]),
        # Monthly arrays use the "time" axis (first of month); yearly arrays use "year".
        "monthly_mean": monthly,
        "climatology": clim,
        "monthly_anomaly": anom,
        "annual_mean": yearly(daily, "mean"),
        "annual_max": yearly(daily, "max"),
        "region_monthly_anomaly": regional_mean(anom, mask),
        "region_annual_mean": yearly(region_daily, "mean"),
        # Region-wide yearly maximum: the hottest day of the area-weighted regional series.
        "region_annual_max": yearly(region_daily, "max"),
    })
    ds.attrs.update(baseline=f"{baseline[0]}-{baseline[1]}",
                    weighting="cos(latitude), land cells only",
                    land_mask="Natural Earth 1:50m land, cell center")
    return ds


# ---------------------------------------------------------------------------
# Files and command line
# ---------------------------------------------------------------------------

def load_daily(region: Region, code: str, raw_dir: Path = RAW_DIR) -> xr.DataArray:
    with xr.open_dataset(raw_dir / region.key / f"{code}.nc") as ds:
        return ds[code].load()


def prepare_region(region: Region, raw_dir: Path = RAW_DIR) -> list[Path]:
    """Prepare every catalog variable for one region and save the results."""
    catalog = load_catalog()
    out_dir = raw_dir / region.key / "prepared"
    out_dir.mkdir(parents=True, exist_ok=True)

    first = load_daily(region, catalog.power_codes()[0], raw_dir)
    mask = land_mask(first["lat"], first["lon"], region.state) if region.land_mask \
        else xr.ones_like(first.isel(time=0), dtype=bool).drop_vars("time").rename("land_mask")
    log.info("%s: %d of %d cells are land%s", region.key, int(mask.sum()), mask.size,
             f" in {region.state}" if region.state else "")

    written = []
    for code, var in catalog.variables.items():
        if code == HEAT_INDEX:
            daily = heat_index_grid(*(load_daily(region, c, raw_dir) for c in var.inputs))
        else:
            daily = load_daily(region, code, raw_dir)
        ds = prepare_variable(daily, mask)
        ds.attrs.update(region=region.key, variable=code, unit=var.unit)
        out = out_dir / f"{code}.nc"
        ds.to_netcdf(out)
        written.append(out)
        log.info("%s %s -> %s", region.key, code, out.relative_to(ROOT))
    return written


def main(argv: list[str] | None = None) -> int:
    regions = load_regions()
    parser = argparse.ArgumentParser(description="Clean and prepare HeatPOD data.")
    parser.add_argument("--region", action="append", choices=sorted(regions))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for key in args.region or sorted(regions):
        prepare_region(regions[key])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
