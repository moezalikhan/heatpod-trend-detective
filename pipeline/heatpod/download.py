"""NASA POWER regional downloader (Module 2).

Facts checked against the live API on 2026-11-15:
- One parameter per regional request.
- At most 366 days per request ("Please do not request a time period greater
  than 366 days."), so we request one calendar year at a time.
- Tiles of at most 4.5 x 4.5 degrees (config/regions.yaml).
- About 30 requests per 60 seconds, so requests are spaced out.
- The grid is MERRA-2: 0.5 degrees latitude x 0.625 degrees longitude. Tiles that
  share an edge on a grid line return that line twice; merge_tiles() removes
  duplicates and checks the values agree.

Files:
  data/raw/<region>/<CODE>/tile<N>_<YEAR>.nc   one cached API response each
  data/raw/<region>/<CODE>.nc                  merged grid for the whole region

Run from the repository root:
  python -m pipeline.heatpod.download --region florida --variable T2M
  python -m pipeline.heatpod.download            # every region and variable
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

import httpx
import numpy as np
import pandas as pd
import xarray as xr

from config.catalog import Box, Region, load_catalog, load_regions

log = logging.getLogger("heatpod.download")

API_URL = "https://power.larc.nasa.gov/api/temporal/daily/regional"
COMMUNITY = "AG"            # agroclimatology; temperatures in C, rain in mm/day
TIME_STANDARD = "lst"       # local solar time, so "daily max" means the local day
MAX_DAYS_PER_REQUEST = 366
MIN_SECONDS_BETWEEN_REQUESTS = 2.5  # 24 per minute, safely under the 30 limit
MAX_RETRIES = 8                   # waits 2, 4, 8, 16, 32, 60, 60 s: rides out a short network drop
REQUEST_DEADLINE_SECONDS = 180.0  # a slow server that trickles bytes is cut off here
RETRY_STATUSES = {429, 500, 502, 503, 504}
FILL_VALUE = -999.0         # POWER's missing-data marker

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"  # NetCDF4 files are HDF5 files


class DownloadError(RuntimeError):
    """The API refused a request or kept failing after all retries."""


# ---------------------------------------------------------------------------
# Planning: which requests to make
# ---------------------------------------------------------------------------

def year_chunks(start: date, end: date) -> list[tuple[date, date]]:
    """Split a date range into calendar-year pieces (each at most 366 days)."""
    chunks = []
    for year in range(start.year, end.year + 1):
        a = max(start, date(year, 1, 1))
        b = min(end, date(year, 12, 31))
        chunks.append((a, b))
    return chunks


@dataclass(frozen=True)
class Request:
    region: str
    code: str
    tile_index: int
    tile: Box
    start: date
    end: date

    def params(self) -> dict[str, str]:
        return {
            "parameters": self.code,
            "community": COMMUNITY,
            "latitude-min": str(self.tile.lat_min),
            "latitude-max": str(self.tile.lat_max),
            "longitude-min": str(self.tile.lon_min),
            "longitude-max": str(self.tile.lon_max),
            "start": self.start.strftime("%Y%m%d"),
            "end": self.end.strftime("%Y%m%d"),
            "format": "netcdf",
            "time-standard": TIME_STANDARD,
        }

    def path(self, raw_dir: Path = RAW_DIR) -> Path:
        return raw_dir / self.region / self.code / f"tile{self.tile_index}_{self.start.year}.nc"


def plan_requests(region: Region, code: str, start: date, end: date) -> list[Request]:
    """Every (tile, year) request needed for one variable in one region."""
    return [
        Request(region.key, code, i, tile, a, b)
        for i, tile in enumerate(region.tiles, start=1)
        for a, b in year_chunks(start, end)
    ]


# ---------------------------------------------------------------------------
# Fetching: rate limit, retries, cache
# ---------------------------------------------------------------------------

class RateLimiter:
    """Keeps a minimum gap between the start of two requests."""

    def __init__(self, min_interval: float, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.min_interval = min_interval
        self.clock, self.sleep = clock, sleep
        self._last: float | None = None

    def wait(self) -> None:
        if self._last is not None:
            gap = self.clock() - self._last
            if gap < self.min_interval:
                self.sleep(self.min_interval - gap)
        self._last = self.clock()


def is_valid_netcdf(path: Path) -> bool:
    """True if the file exists and looks like a complete NetCDF4 file."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb") as f:
        if f.read(8) != HDF5_SIGNATURE:
            return False
    try:
        with xr.open_dataset(path) as ds:
            return len(ds.data_vars) > 0
    except Exception:
        return False


def get_with_deadline(client: httpx.Client, params: dict[str, str],
                      deadline: float = REQUEST_DEADLINE_SECONDS,
                      clock: Callable[[], float] = time.monotonic) -> tuple[int, bytes]:
    """GET the API with a limit on the total time, not only the gap between bytes.

    The httpx read timeout only fires when no bytes arrive at all, so a connection
    that keeps sending tiny pieces could hang forever (seen on 2026-11-15).
    """
    started = clock()
    chunks = []
    with client.stream("GET", API_URL, params=params) as resp:
        for chunk in resp.iter_bytes():
            chunks.append(chunk)
            if clock() - started > deadline:
                raise httpx.ReadTimeout(f"request took longer than {deadline:.0f}s")
        return resp.status_code, b"".join(chunks)


def fetch(req: Request, client: httpx.Client, limiter: RateLimiter,
          raw_dir: Path = RAW_DIR, sleep: Callable[[float], None] = time.sleep) -> bool:
    """Download one request into the cache. Returns False if it was already cached."""
    path = req.path(raw_dir)
    if is_valid_netcdf(path):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)

    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        limiter.wait()
        try:
            status, body = get_with_deadline(client, req.params())
        except httpx.TransportError as exc:      # timeouts, dropped connections
            last_error = repr(exc)
        else:
            if status == 200 and body.startswith(HDF5_SIGNATURE):
                # Write to a temporary file first, so a crash never leaves a half file.
                tmp = path.with_suffix(".part")
                tmp.write_bytes(body)
                tmp.replace(path)
                return True
            last_error = f"HTTP {status}: {body[:300].decode('utf-8', 'replace')}"
            if status not in RETRY_STATUSES and status != 200:
                # A bad request will not get better by retrying.
                raise DownloadError(f"{req.path(raw_dir).name}: {last_error}")
        backoff = min(60.0, 2.0 ** attempt)
        log.warning("attempt %d/%d failed (%s), retrying in %.0fs",
                    attempt, MAX_RETRIES, last_error, backoff)
        sleep(backoff)
    raise DownloadError(f"{path.name}: gave up after {MAX_RETRIES} attempts: {last_error}")


def download_variable(region: Region, code: str, start: date, end: date,
                      client: httpx.Client, limiter: RateLimiter,
                      raw_dir: Path = RAW_DIR) -> None:
    """Download every tile and year for one variable, skipping cached files."""
    requests = plan_requests(region, code, start, end)
    for n, req in enumerate(requests, start=1):
        fresh = fetch(req, client, limiter, raw_dir)
        log.info("%s %s tile %d year %d [%d/%d] %s", region.key, code, req.tile_index,
                 req.start.year, n, len(requests), "downloaded" if fresh else "cached")


# ---------------------------------------------------------------------------
# Merging: tiles and years into one grid
# ---------------------------------------------------------------------------

def merge_tiles(tiles: list[xr.DataArray]) -> xr.DataArray:
    """Combine tiles into one grid. Shared edge cells must have equal values."""
    merged = xr.merge(tiles, join="outer", compat="no_conflicts", combine_attrs="drop_conflicts")
    return merged[tiles[0].name].sortby("lat").sortby("lon")


def find_gaps(da: xr.DataArray, start: date, end: date) -> list[str]:
    """Describe any missing days or missing values. Empty list means no gaps."""
    problems = []
    expected = pd.date_range(start, end, freq="D")
    times = pd.DatetimeIndex(da["time"].values)
    missing_days = expected.difference(times)
    if len(missing_days):
        problems.append(f"{len(missing_days)} missing days, first {missing_days[0].date()}")
    if times.has_duplicates:
        problems.append("duplicate days")
    n_nan = int(da.isnull().sum())
    if n_nan:
        problems.append(f"{n_nan} missing values")
    return problems


def merge_variable(region: Region, code: str, start: date, end: date,
                   raw_dir: Path = RAW_DIR) -> xr.DataArray:
    """Merge all cached tile-year files for one variable into one grid."""
    tiles = []
    for i in range(1, len(region.tiles) + 1):
        years = [Request(region.key, code, i, region.tiles[i - 1], a, b).path(raw_dir)
                 for a, b in year_chunks(start, end)]
        missing = [p.name for p in years if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{region.key} {code}: not downloaded yet: {missing[:3]}")
        parts = []
        for p in years:
            with xr.open_dataset(p) as ds:
                parts.append(ds[code].load())
        tile = xr.concat(parts, dim="time")
        parts.clear()
        tiles.append(tile.where(tile > FILL_VALUE))  # FILL_VALUE becomes NaN

    da = merge_tiles(tiles)
    da = da.astype(np.float32)
    da.attrs.update(region=region.key, source="NASA POWER daily regional API",
                    community=COMMUNITY, time_standard=TIME_STANDARD)
    return da


def save_merged(da: xr.DataArray, region: Region, code: str, raw_dir: Path = RAW_DIR) -> Path:
    out = raw_dir / region.key / f"{code}.nc"
    tmp = out.with_suffix(".part")
    da.to_dataset(name=code).to_netcdf(tmp)
    tmp.replace(out)
    return out


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    catalog, regions = load_catalog(), load_regions()
    parser = argparse.ArgumentParser(description="Download NASA POWER data for HeatPOD.")
    parser.add_argument("--region", action="append", choices=sorted(regions),
                        help="region key (repeatable; default: all)")
    parser.add_argument("--variable", action="append", choices=catalog.power_codes(),
                        help="POWER code (repeatable; default: all)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # hide one line per request

    start, end = catalog.start, catalog.end
    limiter = RateLimiter(MIN_SECONDS_BETWEEN_REQUESTS)
    failed = []
    with httpx.Client(timeout=httpx.Timeout(60.0, connect=20.0)) as client:
        for key in args.region or sorted(regions):
            for code in args.variable or catalog.power_codes():
                region = regions[key]
                try:
                    download_variable(region, code, start, end, client, limiter)
                    da = merge_variable(region, code, start, end)
                except (DownloadError, FileNotFoundError) as exc:
                    log.error("FAILED %s %s: %s", key, code, exc)
                    failed.append(f"{key} {code}")
                    continue
                gaps = find_gaps(da, start, end)
                out = save_merged(da, region, code)
                log.info("MERGED %s %s -> %s %s, %s", key, code, out.relative_to(ROOT),
                         dict(da.sizes), "no gaps" if not gaps else "GAPS: " + "; ".join(gaps))
                if gaps:
                    failed.append(f"{key} {code} (gaps)")
    if failed:
        log.error("Finished with problems: %s", ", ".join(failed))
        return 1
    log.info("Finished: all requested variables downloaded and merged with no gaps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
