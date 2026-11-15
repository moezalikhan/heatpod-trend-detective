"""Extremes engine tests on simulated GEV data with known answers (Module 6)."""

import numpy as np
import pytest
import xarray as xr
from scipy import stats

from config.catalog import load_catalog
from pipeline.heatpod import extremes as ex


def gev(xi, loc, scale, n, seed):
    """Draw GEV values. SciPy's c is minus our xi."""
    return stats.genextreme.rvs(-xi, loc=loc, scale=scale, size=n, random_state=seed)


def yearly(old, new):
    years = np.arange(1981, 1981 + len(old) + len(new))
    return xr.DataArray(np.concatenate([old, new]), dims="year", coords={"year": years})


# --- the fit gives back the parameters used to create the data ---------------

@pytest.mark.parametrize("xi", [-0.25, 0.0, 0.15])
def test_fit_recovers_known_parameters(xi):
    old, new = gev(xi, 40.0, 1.5, 3000, 1), gev(xi, 41.0, 2.0, 3000, 2)
    fit = ex.fit_gev_pair(old, new)
    assert fit.shape == pytest.approx(xi, abs=0.04)
    assert fit.loc[0] == pytest.approx(40.0, abs=0.1)
    assert fit.loc[1] == pytest.approx(41.0, abs=0.1)
    assert fit.scale[0] == pytest.approx(1.5, abs=0.08)
    assert fit.scale[1] == pytest.approx(2.0, abs=0.1)


def test_return_level_matches_the_formula():
    # Gumbel (xi = 0): level = loc - scale * ln(-ln(1 - 1/T))
    fit = ex.GevPair((40.0, 40.0), (1.5, 1.5), 0.0)
    expected = 40.0 - 1.5 * np.log(-np.log(1 - 1 / 20))
    assert ex.return_level(fit, 0) == pytest.approx(expected, rel=1e-9)
    # Same distribution in both periods: the 20-year level stays a 20-year level.
    assert ex.new_return_period(fit) == pytest.approx(20.0, rel=1e-6)


def test_shift_makes_the_old_level_more_common():
    # Location up by one scale unit: the old 20-year level becomes much more frequent.
    fit = ex.GevPair((40.0, 41.5), (1.5, 1.5), -0.1)
    assert ex.new_return_period(fit) < 8
    assert ex.return_level(fit, 1) - ex.return_level(fit, 0) == pytest.approx(1.5, rel=1e-9)


def test_shape_stays_inside_bounds_for_tiny_samples():
    fit = ex.fit_gev_pair(gev(0.0, 40, 1.5, 22, 3), gev(0.0, 40, 1.5, 23, 4))
    assert ex.SHAPE_BOUNDS[0] <= fit.shape <= ex.SHAPE_BOUNDS[1]


# --- realistic sample sizes with bootstrap ranges ---------------------------

def test_clear_increase_is_found_with_45_years():
    old, new = gev(-0.2, 40.0, 1.0, 23, 5), gev(-0.2, 43.0, 1.0, 22, 6)
    r = ex.analyze(yearly(old, new), n_boot=300)
    assert r.significant and r.change_low > 0
    assert r.change == pytest.approx(3.0, abs=1.0)
    assert r.period_new < 5


def test_no_change_is_not_called_significant_most_of_the_time():
    hits = 0
    for seed in range(20):
        old, new = gev(-0.2, 40.0, 1.0, 23, 100 + seed), gev(-0.2, 40.0, 1.0, 22, 200 + seed)
        hits += ex.analyze(yearly(old, new), n_boot=200, seed=seed).significant
    # A 100-case simulation gave a 6% false alarm rate. At 6%, 5 or more hits
    # out of 20 happens less than 1% of the time.
    assert hits <= 4


def test_cell_extremes_skips_sea_cells():
    rng = np.random.default_rng(7)
    data = gev(-0.2, 40, 1.0, 45 * 4, 7).reshape(45, 2, 2)
    da = xr.DataArray(data, dims=("year", "lat", "lon"),
                      coords={"year": np.arange(1981, 2026), "lat": [0.0, 1.0], "lon": [0.0, 1.0]})
    mask = xr.DataArray([[True, False], [True, True]], dims=("lat", "lon"), coords=da.isel(year=0).coords)
    cells = ex.cell_extremes(da, mask.drop_vars("year"))
    assert np.isnan(float(cells["change"][0, 1]))
    assert int(cells["change"].notnull().sum()) == 3


# --- sentence ---------------------------------------------------------------

def test_sentence_reads_plainly():
    var = load_catalog()["HEAT_INDEX"]
    r = ex.ExtremesResult(-0.2, 40, 1, 41, 1, 44.0, 45.5, 1.5, 0.6, 2.4, 3.2, 1.8, 7.5, True, 1000)
    assert ex.sentence(var, r) == (
        "The heat index level that used to be reached about once every 20 years (44.0 °C) "
        "is now reached about once every 3 years (likely range 1.8 to 8 years).")
