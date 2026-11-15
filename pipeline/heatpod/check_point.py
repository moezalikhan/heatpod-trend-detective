"""Check one grid cell against NASA POWER's point service (Module 11).

The point service is what the official POWER Data Access Viewer uses. If our
downloaded and merged grid is right, the values for one cell must match exactly.

Run from the repository root (needs the raw data from download.py):
  python -m pipeline.heatpod.check_point --region phoenix --lat 33.5 --lon -112.5 --year 2020
"""

from __future__ import annotations

import argparse

import httpx
import pandas as pd
import xarray as xr

from config.catalog import load_catalog
from pipeline.heatpod.download import COMMUNITY, RAW_DIR, TIME_STANDARD

POINT_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compare one cell with the POWER point service.")
    p.add_argument("--region", default="phoenix")
    p.add_argument("--lat", type=float, default=33.5)
    p.add_argument("--lon", type=float, default=-112.5)
    p.add_argument("--year", type=int, default=2020)
    a = p.parse_args(argv)
    codes = load_catalog().power_codes()
    r = httpx.get(POINT_URL, timeout=120, params={
        "parameters": ",".join(codes), "community": COMMUNITY, "latitude": a.lat,
        "longitude": a.lon, "start": f"{a.year}0101", "end": f"{a.year}1231",
        "format": "json", "time-standard": TIME_STANDARD})
    r.raise_for_status()
    params = r.json()["properties"]["parameter"]
    worst = 0.0
    for code in codes:
        point = pd.Series(params[code])
        point.index = pd.to_datetime(point.index, format="%Y%m%d")
        with xr.open_dataset(RAW_DIR / a.region / f"{code}.nc") as ds:
            grid = ds[code].sel(lat=a.lat, lon=a.lon, method="nearest") \
                .sel(time=slice(f"{a.year}-01-01", f"{a.year}-12-31")).to_series()
        diff = float((point - grid).abs().max())
        worst = max(worst, diff)
        print(f"{code:12s} {len(grid)} days, largest difference {diff:.3f}")
    print("MATCH" if worst < 0.01 else "MISMATCH")
    return 0 if worst < 0.01 else 1


if __name__ == "__main__":
    raise SystemExit(main())
