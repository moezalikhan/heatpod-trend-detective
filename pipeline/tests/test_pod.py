"""POD tests on fake fields built from known patterns (Module 5)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pipeline.heatpod import pod, trends

N = 540
TIME = pd.date_range("1981-01-01", periods=N, freq="MS")
LAT = np.array([28.0, 28.5, 29.0, 29.5, 30.0, 30.5])
LON = np.array([-84.0, -83.375, -82.75, -82.125, -81.5])


def field(series_list, pattern_list, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    data = sum(s[:, None, None] * p[None] for s, p in zip(series_list, pattern_list))
    data = data + rng.normal(0, noise, data.shape)
    return xr.DataArray(data, dims=("time", "lat", "lon"),
                        coords={"time": TIME, "lat": LAT, "lon": LON})


def land(drop=()):
    m = xr.DataArray(np.ones((len(LAT), len(LON)), bool), dims=("lat", "lon"),
                     coords={"lat": LAT, "lon": LON})
    for i, j in drop:
        m[i, j] = False
    return m


def uniform():
    return np.ones((len(LAT), len(LON)))


def dipole():
    # North half up, south half down: "opposite ways in different places".
    p = np.ones((len(LAT), len(LON)))
    p[: len(LAT) // 2] = -1
    return p


def standard(x):
    return (x - x.mean()) / x.std()


def test_known_patterns_and_shares_are_recovered():
    rng = np.random.default_rng(1)
    s1, s2 = standard(rng.normal(size=N)), standard(rng.normal(size=N))
    s2 = standard(s2 - s1 * (s1 @ s2) / (s1 @ s1))   # make the two series uncorrelated
    da = field([2.0 * s1, 1.0 * s2], [uniform(), dipole()])
    res = pod.pod(da, land())
    # Equal-area cells would give 4:1; cos(lat) changes it only slightly.
    assert res.variance_fraction[0] == pytest.approx(0.8, abs=0.02)
    assert res.variance_fraction[1] == pytest.approx(0.2, abs=0.02)
    # Mode 1 pattern is the uniform field of size 2, mode 2 the dipole of size 1.
    np.testing.assert_allclose(res.patterns.sel(mode=1), 2.0 * uniform(), atol=0.05)
    np.testing.assert_allclose(abs(res.patterns.sel(mode=2)), 1.0 * uniform(), atol=0.05)
    assert abs(np.corrcoef(res.series.sel(mode=1), s1)[0, 1]) > 0.999
    assert abs(np.corrcoef(res.series.sel(mode=2), s2)[0, 1]) > 0.999


def test_series_have_unit_std_and_patterns_reconstruct_the_field():
    rng = np.random.default_rng(2)
    da = field([rng.normal(size=N), 0.5 * rng.normal(size=N)], [uniform(), dipole()], noise=0.0)
    res = pod.pod(da, land(), n_modes=2)
    np.testing.assert_allclose(res.series.std("time"), 1.0, rtol=1e-6)
    recon = (res.series * res.patterns).sum("mode")
    np.testing.assert_allclose(recon.transpose("time", "lat", "lon"), da - da.mean("time"), atol=1e-8)


def test_sign_makes_mode_one_warm_when_region_warms():
    da = field([-np.linspace(-1, 1, N)], [uniform()])  # whole region cools...
    res = pod.pod(da, land(), n_modes=1)
    assert float(res.patterns.sel(mode=1).mean()) > 0  # ...pattern stays positive
    assert float(res.series.sel(mode=1)[-1]) < 0       # and the series carries the sign


def test_trend_in_mode_series_is_detected():
    rng = np.random.default_rng(3)
    trend = np.arange(N) / 120 * 0.5
    # The trending series must be the larger one, so it is mode 1.
    da = field([trend + rng.normal(0, 0.5, N), 0.3 * rng.normal(size=N)], [uniform(), dipole()])
    res = pod.pod(da, land())
    t1 = trends.trend(res.series.sel(mode=1))
    assert t1.significant and t1.slope_per_decade > 0


def test_sea_cells_are_left_out():
    rng = np.random.default_rng(4)
    da = field([rng.normal(size=N)], [uniform()])
    res = pod.pod(da, land(drop=[(0, 0), (5, 4)]), n_modes=1)
    assert np.isnan(float(res.patterns.sel(mode=1)[0, 0]))
    assert np.isnan(float(res.patterns.sel(mode=1)[5, 4]))
    assert int(res.patterns.sel(mode=1).notnull().sum()) == len(LAT) * len(LON) - 2


def test_north_rule_flags_modes_with_equal_variance():
    rng = np.random.default_rng(5)
    a, b = rng.normal(size=N), rng.normal(size=N)
    other = np.tile(np.array([1, -1, 1, -1, 1]), (len(LAT), 1)).astype(float)
    res = pod.pod(field([a, b], [dipole(), other], noise=0.05), land(), n_modes=2)
    assert not res.separated[0] and not res.separated[1]   # same size: cannot be told apart
    res2 = pod.pod(field([3 * a, b], [uniform(), dipole()], noise=0.05), land(), n_modes=2)
    assert res2.separated[0]


def test_opposite_trends_are_counted():
    ramp = np.arange(N) / 120
    rng = np.random.default_rng(6)
    da = field([ramp], [dipole()], noise=0.3, seed=6)
    cells = trends.cell_trends(da, land())
    opp = pod.opposite_trends(cells)
    assert opp["opposite"] and opp["rising"] == 15 and opp["falling"] == 15
