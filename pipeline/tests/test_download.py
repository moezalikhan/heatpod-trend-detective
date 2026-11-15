"""Tests for the downloader using fake data and a fake HTTP server (no network)."""

from datetime import date

import httpx
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from config.catalog import Box, Region
from pipeline.heatpod import download as dl

TILE = Box(24.5, 27.75, -87.6, -83.8)


def fake_grid(lats, lons, start="2024-01-01", days=3, value=None):
    """A small T2M grid like the ones NASA POWER returns."""
    times = pd.date_range(start, periods=days, freq="D")
    data = np.random.default_rng(0).normal(25, 3, (days, len(lats), len(lons)))
    if value is not None:
        data[:] = value
    return xr.DataArray(data.astype("float32"), name="T2M",
                        coords={"time": times, "lat": lats, "lon": lons},
                        dims=("time", "lat", "lon"))


def netcdf_bytes(tmp_path, da):
    p = tmp_path / "resp.nc"
    da.to_dataset().to_netcdf(p)
    return p.read_bytes()


class NoWait(dl.RateLimiter):
    def __init__(self):
        super().__init__(0)


# --- planning ---------------------------------------------------------------

def test_year_chunks_cover_1981_to_2025_without_overlap():
    chunks = dl.year_chunks(date(1981, 1, 1), date(2025, 12, 31))
    assert len(chunks) == 45
    assert all((b - a).days + 1 <= dl.MAX_DAYS_PER_REQUEST for a, b in chunks)
    for (_, b), (a2, _) in zip(chunks, chunks[1:]):
        assert (a2 - b).days == 1


def test_plan_has_one_request_per_tile_and_year():
    region = Region("florida", "Florida", TILE, (TILE, TILE))
    reqs = dl.plan_requests(region, "T2M", date(1981, 1, 1), date(1982, 12, 31))
    assert len(reqs) == 4
    p = reqs[0].params()
    assert p["parameters"] == "T2M" and p["start"] == "19810101" and p["end"] == "19811231"
    assert p["format"] == "netcdf"


def test_rate_limiter_spaces_requests():
    now = [0.0]
    slept = []
    lim = dl.RateLimiter(2.5, clock=lambda: now[0], sleep=lambda s: slept.append(s))
    lim.wait()
    now[0] = 1.0
    lim.wait()
    assert slept == [pytest.approx(1.5)]


# --- fetching ---------------------------------------------------------------

def make_request():
    return dl.Request("florida", "T2M", 1, TILE, date(2024, 1, 1), date(2024, 12, 31))


def test_fetch_retries_server_errors_then_saves(tmp_path):
    body = netcdf_bytes(tmp_path, fake_grid([24.5], [-87.5]))
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503) if len(calls) < 3 else httpx.Response(200, content=body)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    raw = tmp_path / "raw"
    assert dl.fetch(make_request(), client, NoWait(), raw, sleep=lambda s: None) is True
    assert len(calls) == 3
    assert dl.is_valid_netcdf(make_request().path(raw))


def test_fetch_uses_cache_without_calling_the_api(tmp_path):
    raw = tmp_path / "raw"
    path = make_request().path(raw)
    path.parent.mkdir(parents=True)
    fake_grid([24.5], [-87.5]).to_dataset().to_netcdf(path)

    def handler(request):
        raise AssertionError("the API must not be called for cached files")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert dl.fetch(make_request(), client, NoWait(), raw) is False


def test_fetch_does_not_retry_a_bad_request(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(422, json={"messages": ["Please do not request a time period greater than 366 days."]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(dl.DownloadError, match="366 days"):
        dl.fetch(make_request(), client, NoWait(), tmp_path, sleep=lambda s: None)
    assert len(calls) == 1


def test_fetch_rejects_json_error_with_status_200(tmp_path):
    client = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"header": "error"})))
    with pytest.raises(dl.DownloadError):
        dl.fetch(make_request(), client, NoWait(), tmp_path, sleep=lambda s: None)
    assert not make_request().path(tmp_path).exists()


# --- merging ----------------------------------------------------------------

def test_merge_tiles_with_shared_edge_gives_full_grid():
    # Like Punjab: both tiles include the grid line at lat 31.0.
    full = fake_grid([28.0, 28.5, 31.0, 31.5], [70.0, 70.625])
    south = full.sel(lat=[28.0, 28.5, 31.0])
    north = full.sel(lat=[31.0, 31.5])
    merged = dl.merge_tiles([north, south])
    assert list(merged.lat.values) == [28.0, 28.5, 31.0, 31.5]
    xr.testing.assert_allclose(merged, full)


def test_merge_tiles_refuses_conflicting_edges():
    a = fake_grid([28.0, 31.0], [70.0], value=20.0)
    b = fake_grid([31.0, 31.5], [70.0], value=21.0)
    with pytest.raises(xr.MergeError):
        dl.merge_tiles([a, b])


def test_find_gaps_reports_missing_days_and_values():
    da = fake_grid([24.5], [-87.5], days=5)
    assert dl.find_gaps(da, date(2024, 1, 1), date(2024, 1, 5)) == []
    holed = da.drop_sel(time=[pd.Timestamp("2024-01-03")])
    assert "1 missing days" in dl.find_gaps(holed, date(2024, 1, 1), date(2024, 1, 5))[0]
    da2 = da.copy()
    da2[0, 0, 0] = np.nan
    assert dl.find_gaps(da2, date(2024, 1, 1), date(2024, 1, 5)) == ["1 missing values"]


def test_merge_variable_joins_years_and_tiles(tmp_path):
    west, east = Box(28, 30, 70, 71), Box(28, 30, 71, 72)
    region = Region("demo", "Demo", Box(28, 30, 70, 72), (west, east))
    start, end = date(2023, 12, 30), date(2024, 1, 2)
    for i, lons in ((1, [70.0, 70.625]), (2, [71.25, 71.875])):
        for a, b in dl.year_chunks(start, end):
            req = dl.Request("demo", "T2M", i, region.tiles[i - 1], a, b)
            req.path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
            fake_grid([28.0, 28.5], lons, start=str(a), days=(b - a).days + 1) \
                .to_dataset().to_netcdf(req.path(tmp_path))
    da = dl.merge_variable(region, "T2M", start, end, tmp_path)
    assert dict(da.sizes) == {"time": 4, "lat": 2, "lon": 4}
    assert dl.find_gaps(da, start, end) == []
