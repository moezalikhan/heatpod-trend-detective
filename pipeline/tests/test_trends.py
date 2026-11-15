"""Trend engine tests on fake data with known answers (Module 4 'done' criteria)."""

import numpy as np
import pandas as pd
import pymannkendall as mk
import pytest
import xarray as xr

from config.catalog import load_catalog
from pipeline.heatpod import trends

N_MONTHS = 540  # 1981 to 2025
TIME = pd.date_range("1981-01-01", periods=N_MONTHS, freq="MS")


def ar1(n, phi, sd, rng):
    """Red noise: each month partly remembers the month before."""
    e = rng.normal(0, sd, n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def series(values):
    return xr.DataArray(values, dims="time", coords={"time": TIME})


def with_trend(per_decade, noise):
    years = np.arange(N_MONTHS) / 12
    return series(per_decade * years / 10 + noise)


# --- a known trend comes back -----------------------------------------------

def test_known_trend_is_recovered_and_significant():
    rng = np.random.default_rng(1)
    r = trends.trend(with_trend(0.30, ar1(N_MONTHS, 0.5, 0.4, rng)))
    assert r.slope_per_decade == pytest.approx(0.30, abs=0.06)
    assert r.sen_slope_per_decade == pytest.approx(0.30, abs=0.06)
    assert r.significant and r.p_value < 0.01


def test_confidence_interval_covers_the_true_trend():
    # Over many noisy series the 95% interval should contain the true slope about 95% of the time.
    rng = np.random.default_rng(8)
    covered = 0
    for _ in range(200):
        r = trends.trend(with_trend(0.30, ar1(N_MONTHS, 0.5, 0.4, rng)))
        covered += r.ci_low <= 0.30 <= r.ci_high
    assert covered / 200 >= 0.90


def test_negative_trend_is_recovered():
    rng = np.random.default_rng(2)
    r = trends.trend(with_trend(-0.50, rng.normal(0, 0.3, N_MONTHS)))
    assert r.slope_per_decade == pytest.approx(-0.50, abs=0.05)
    assert r.significant


def test_perfect_line_gives_exact_slope():
    r = trends.trend(with_trend(0.25, np.zeros(N_MONTHS)))
    assert r.slope_per_decade == pytest.approx(0.25, rel=1e-9)
    assert r.sen_slope_per_decade == pytest.approx(0.25, rel=1e-6)


# --- random data is not called significant ----------------------------------

def false_positive_rate(phi, runs=300, seed=3):
    rng = np.random.default_rng(seed)
    hits = sum(trends.trend(series(ar1(N_MONTHS, phi, 1.0, rng))).significant for _ in range(runs))
    return hits / runs


def test_white_noise_is_rarely_significant():
    assert false_positive_rate(phi=0.0) <= 0.09   # 5% expected


def test_autocorrelated_noise_is_rarely_significant():
    # The corrected test should stay near 5%, even when months are strongly linked
    # (simulation with 1500 runs gave 6.9%; the basic test gives 32%).
    assert false_positive_rate(phi=0.6) <= 0.11


def test_basic_mann_kendall_is_fooled_by_autocorrelation():
    # Why we use the corrected version: the basic test fires far too often.
    rng = np.random.default_rng(3)
    hits = sum(mk.original_test(ar1(N_MONTHS, 0.6, 1.0, rng)).h for _ in range(300))
    assert hits / 300 > 0.15


def test_confidence_interval_covers_zero_for_red_noise():
    rng = np.random.default_rng(4)
    covered = 0
    for _ in range(200):
        r = trends.trend(series(ar1(N_MONTHS, 0.6, 1.0, rng)))
        covered += r.ci_low <= 0 <= r.ci_high
    assert covered / 200 >= 0.90   # about 95% expected


# --- false discovery rate ---------------------------------------------------

def test_fdr_qvalues_match_hand_calculation():
    q = trends.fdr_qvalues(np.array([0.01, 0.04, 0.03, 0.2, np.nan]))
    np.testing.assert_allclose(q[:4], [0.04, 0.16 / 3, 0.16 / 3, 0.2])
    assert np.isnan(q[4])


def grid(cells_with_trend, seed):
    """10 x 10 cells of red noise; the first `cells_with_trend` cells get a real trend."""
    rng = np.random.default_rng(seed)
    data = np.stack([ar1(N_MONTHS, 0.3, 1.0, rng) for _ in range(100)], axis=1)
    data[:, :cells_with_trend] += 0.8 * (np.arange(N_MONTHS) / 120)[:, None]
    da = xr.DataArray(data.reshape(N_MONTHS, 10, 10), dims=("time", "lat", "lon"),
                      coords={"time": TIME, "lat": np.arange(10.0), "lon": np.arange(10.0)})
    return da, xr.ones_like(da.isel(time=0), dtype=bool).drop_vars("time")


def test_fdr_keeps_pure_noise_grid_clean():
    da, mask = grid(0, seed=5)
    cells = trends.cell_trends(da, mask)
    assert int((cells["p_value"] < 0.05).sum()) >= 1   # raw p-values give chance hits
    assert int(cells["significant"].sum()) <= 1        # FDR removes them


def test_fdr_still_finds_real_trends():
    da, mask = grid(20, seed=6)
    sig = trends.cell_trends(da, mask)["significant"].values.ravel()
    assert sig[:20].sum() >= 18      # real trends found
    assert sig[20:].sum() <= 2       # few false alarms


def test_sea_cells_are_skipped():
    da, mask = grid(0, seed=7)
    mask[0, 0] = False
    cells = trends.cell_trends(da, mask)
    assert np.isnan(float(cells["p_value"][0, 0]))
    assert not bool(cells["significant"][0, 0])


# --- sentences --------------------------------------------------------------

def make_result(slope, p):
    return trends.TrendResult(slope, slope - 0.1, slope + 0.1, slope, p, p < 0.05, 540, 300.0)


def test_sentence_for_a_significant_rise():
    var = load_catalog()["HEAT_INDEX"]
    s = trends.sentence(var, make_result(0.41, 0.002), 1981)
    assert s == ("Heat index has risen by 0.41 °C per decade since 1981. "
                 "This trend is very unlikely to be chance (p = 0.002).")


def test_sentence_for_no_clear_trend_uses_plain_units():
    var = load_catalog()["RH2M"]
    s = trends.sentence(var, make_result(-0.12, 0.31), 1981)
    assert "percentage points" in s and "could easily be chance (p = 0.310)" in s
    assert trends.verdict_label(make_result(-0.12, 0.31)) == "NOT SIGNIFICANT"
