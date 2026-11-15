"""Tests for cleaning and preparation, on fake data with known answers."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pipeline.heatpod import clean


# --- heat index -------------------------------------------------------------

# Values from the NWS heat index chart (temperature F, humidity %, heat index F).
NWS_CHART = [
    (80, 40, 80),
    (90, 50, 95),
    (90, 70, 106),
    (100, 40, 109),
    (96, 65, 121),
    (86, 90, 105),
]


@pytest.mark.parametrize("t, rh, expected", NWS_CHART)
def test_heat_index_matches_nws_chart(t, rh, expected):
    assert clean.heat_index_f(t, rh) == pytest.approx(expected, abs=1.0)


def test_heat_index_below_80f_is_close_to_air_temperature():
    assert clean.heat_index_f(70, 50) == pytest.approx(69.5, abs=0.1)


def test_heat_index_celsius_matches_fahrenheit():
    t_c = (100 - 32) * 5 / 9
    assert clean.heat_index_c(t_c, 40) == pytest.approx((109 - 32) * 5 / 9, abs=0.6)


# --- fake grid --------------------------------------------------------------

TREND_PER_YEAR = 0.03


def fake_daily(lats=(25.0, 30.0), lons=(-82.0, -81.375), seed=0):
    """20 C plus a strong seasonal cycle, a known trend and a little noise."""
    time = pd.date_range("1981-01-01", "2025-12-31", freq="D")
    years = (time - time[0]).days / 365.25
    season = 8 * np.sin(2 * np.pi * (time.day_of_year - 105) / 365.25)
    signal = 20 + season + TREND_PER_YEAR * years
    noise = np.random.default_rng(seed).normal(0, 0.5, (len(time), len(lats), len(lons)))
    data = signal.values[:, None, None] + noise
    return xr.DataArray(data, name="T2M", dims=("time", "lat", "lon"),
                        coords={"time": time, "lat": list(lats), "lon": list(lons)})


def all_land(da):
    return xr.ones_like(da.isel(time=0), dtype=bool).drop_vars("time")


# --- anomalies --------------------------------------------------------------

def test_anomalies_remove_the_seasonal_cycle():
    daily = fake_daily()
    monthly = clean.to_monthly(daily)
    anom = clean.monthly_anomalies(monthly)
    # Size of the seasonal wave: spread of the 12 calendar-month averages.
    wave_before = float(monthly.groupby("time.month").mean().max("month").min()
                        - monthly.groupby("time.month").mean().min("month").max())
    by_month = anom.groupby("time.month").mean()
    wave_after = float((by_month.max("month") - by_month.min("month")).max())
    assert wave_before > 10      # a strong summer and winter wave in the raw data
    assert wave_after < 0.3      # gone after removing the climatology


def test_anomalies_average_zero_over_the_baseline():
    anom = clean.monthly_anomalies(clean.to_monthly(fake_daily()))
    base = anom.sel(time=slice("1991-01-01", "2020-12-31"))
    np.testing.assert_allclose(base.groupby("time.month").mean(), 0, atol=1e-9)


def test_anomalies_keep_the_trend():
    anom = clean.monthly_anomalies(clean.to_monthly(fake_daily()))
    series = anom.mean(("lat", "lon"))
    years = (series["time"].dt.year + (series["time"].dt.month - 0.5) / 12).values
    slope = np.polyfit(years, series.values, 1)[0]
    assert slope == pytest.approx(TREND_PER_YEAR, abs=0.003)


# --- weighting and masking --------------------------------------------------

def test_regional_mean_weights_by_cos_latitude():
    da = xr.DataArray([[0.0], [1.0]], dims=("lat", "lon"), coords={"lat": [0.0, 60.0], "lon": [0.0]})
    mask = xr.ones_like(da, dtype=bool)
    # cos(0) = 1, cos(60) = 0.5, so the mean is 0.5 / 1.5.
    assert float(clean.regional_mean(da, mask)) == pytest.approx(1 / 3)


def test_regional_mean_ignores_sea_cells():
    da = xr.DataArray([[10.0, 999.0]], dims=("lat", "lon"), coords={"lat": [25.0], "lon": [0.0, 1.0]})
    mask = xr.DataArray([[True, False]], dims=("lat", "lon"), coords=da.coords)
    assert float(clean.regional_mean(da, mask)) == pytest.approx(10.0)


# --- yearly summaries and full product --------------------------------------

def test_yearly_max_picks_the_hottest_day():
    daily = fake_daily()
    daily.loc[dict(time="2010-07-15")] = 60.0
    ds = clean.prepare_variable(daily, all_land(daily))
    assert float(ds["annual_max"].sel(year=2010).min()) == 60.0
    assert float(ds["region_annual_max"].sel(year=2010)) == pytest.approx(60.0)


def test_prepare_variable_shapes_and_masking():
    daily = fake_daily()
    mask = all_land(daily)
    mask[0, 0] = False  # one sea cell
    ds = clean.prepare_variable(daily, mask)
    assert ds.sizes["time"] == 45 * 12
    assert ds.sizes["year"] == 45
    assert ds.sizes["month"] == 12
    assert bool(ds["monthly_anomaly"].isel(lat=0, lon=0).isnull().all())
    assert int(ds["region_monthly_anomaly"].isnull().sum()) == 0
