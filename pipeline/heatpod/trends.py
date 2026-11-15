"""Trend engine (Module 4).

Works on monthly anomalies (seasonal cycle already removed in Module 3).

- Linear trend: least-squares slope per decade with a 95% confidence interval.
  Neighbouring months are correlated, so the interval uses an effective sample
  size n_eff = n (1 - r1) / (1 + r1), where r1 is the lag-1 autocorrelation of
  the residuals (Santer et al. 2000). Without this the interval is too narrow.
- Mann-Kendall test, Hamed and Rao (1998) version, which corrects for
  autocorrelation. Its p-value decides SIGNIFICANT / NOT SIGNIFICANT.
  Settings checked by simulation (1500 runs, 540 months, 5% level):
    basic test on red noise (lag-1 autocorrelation 0.6): 32% false alarms;
    pymannkendall default Hamed-Rao: 10 to 16%;
    our setting, 12 lags and never shrinking the variance: 6 to 8%.
- Sen's slope: a robust trend size that outliers do not distort.
- Per grid cell: the same test in every land cell, then the Benjamini-Hochberg
  false discovery rate correction, so chance "hits" among many cells are not
  reported as real.

Run from the repository root to print every regional trend:
  python -m pipeline.heatpod.trends
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pymannkendall as mk
import xarray as xr
from scipy import stats

from config.catalog import Variable, load_catalog, load_regions

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
ALPHA = 0.05            # significance level
MONTHS_PER_DECADE = 120
MK_LAGS = 12            # one year of monthly autocorrelation in the Hamed-Rao correction


@dataclass(frozen=True)
class TrendResult:
    slope_per_decade: float      # least-squares slope
    ci_low: float                # 95% confidence interval of that slope
    ci_high: float
    sen_slope_per_decade: float  # robust slope (Sen)
    p_value: float               # Mann-Kendall (Hamed-Rao), drives the verdict
    significant: bool
    n: int                       # number of months
    n_eff: float                 # effective number of independent months

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# One series
# ---------------------------------------------------------------------------

def decimal_years(time: xr.DataArray) -> np.ndarray:
    """Month-start dates to decimal years at mid-month (2000-01 -> 2000.04)."""
    return (time.dt.year + (time.dt.month - 0.5) / 12).values.astype(float)


def linear_trend(y: np.ndarray, t_years: np.ndarray) -> tuple[float, float, float, float]:
    """Slope per decade and its 95% interval, widened for autocorrelation.

    Returns (slope, ci_low, ci_high, n_eff).
    """
    fit = stats.linregress(t_years, y)
    resid = y - (fit.intercept + fit.slope * t_years)
    n = len(y)
    r1 = np.corrcoef(resid[:-1], resid[1:])[0, 1]
    r1 = max(0.0, r1)                       # negative r1 would shrink the interval; do not
    n_eff = n * (1 - r1) / (1 + r1)
    n_eff = min(float(n), max(n_eff, 3.0))  # keep at least 1 degree of freedom
    # Rescale the standard error from n - 2 to n_eff - 2 degrees of freedom.
    se = fit.stderr * np.sqrt((n - 2) / (n_eff - 2))
    half = stats.t.ppf(1 - ALPHA / 2, n_eff - 2) * se
    return fit.slope * 10, (fit.slope - half) * 10, (fit.slope + half) * 10, n_eff


def mann_kendall(y: np.ndarray) -> tuple[float, float]:
    """Autocorrelation-corrected Mann-Kendall test.

    Returns (p_value, Sen's slope per time step).
    The Hamed-Rao correction can shrink the variance when the sample
    autocorrelation is negative by chance, which makes the test too eager.
    We never let it go below the basic test's variance.
    """
    r = mk.hamed_rao_modification_test(y, alpha=ALPHA, lag=MK_LAGS)
    n = len(y)
    var_basic = n * (n - 1) * (2 * n + 5) / 18
    var_s = float(r.var_s)
    if not np.isfinite(var_s) or var_s < var_basic:   # NaN can come back for very strong trends
        var_s = var_basic
    z = (r.s - np.sign(r.s)) / np.sqrt(var_s)
    p = float(2 * stats.norm.sf(abs(z)))
    return p, float(r.slope)


def trend(series: xr.DataArray) -> TrendResult:
    """All trend numbers for one monthly anomaly series (dimension 'time')."""
    s = series.dropna("time")
    y = s.values.astype(float)
    slope, lo, hi, n_eff = linear_trend(y, decimal_years(s["time"]))
    p, sen = mann_kendall(y)
    return TrendResult(
        slope_per_decade=float(slope), ci_low=float(lo), ci_high=float(hi),
        sen_slope_per_decade=sen * MONTHS_PER_DECADE,
        p_value=p, significant=p < ALPHA, n=len(y), n_eff=float(n_eff),
    )


# ---------------------------------------------------------------------------
# Every grid cell, with false discovery rate control
# ---------------------------------------------------------------------------

def fdr_qvalues(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values). NaN stays NaN."""
    q = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if ok.any():
        q[ok] = stats.false_discovery_control(p[ok], method="bh")
    return q


def cell_trends(grid: xr.DataArray, mask: xr.DataArray) -> xr.Dataset:
    """Trend in every land cell. 'significant' uses the FDR-corrected q-value."""
    names = ("slope_per_decade", "ci_low", "ci_high", "sen_slope_per_decade", "p_value")
    out = {k: xr.full_like(mask, np.nan, dtype=float) for k in names}
    for i in range(mask.sizes["lat"]):
        for j in range(mask.sizes["lon"]):
            if not bool(mask[i, j]):
                continue
            r = trend(grid.isel(lat=i, lon=j))
            for k in names:
                out[k][i, j] = getattr(r, k)
    ds = xr.Dataset(out)
    q = fdr_qvalues(ds["p_value"].values.ravel()).reshape(ds["p_value"].shape)
    ds["q_value"] = (ds["p_value"].dims, q)
    ds["significant"] = (ds["q_value"] < ALPHA) & mask
    ds.attrs.update(test="Mann-Kendall (Hamed-Rao)", fdr="Benjamini-Hochberg", alpha=ALPHA)
    return ds


# ---------------------------------------------------------------------------
# Plain-language sentences
# ---------------------------------------------------------------------------

def format_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def unit_words(unit: str) -> str:
    """Units as they read in a sentence."""
    return {"%": "percentage points"}.get(unit, unit)


def verdict_label(result: TrendResult) -> str:
    return "SIGNIFICANT" if result.significant else "NOT SIGNIFICANT"


def sentence(var: Variable, result: TrendResult, start_year: int) -> str:
    """One or two plain sentences a high school student can read."""
    size = f"{abs(result.slope_per_decade):.2f} {unit_words(var.unit)} per decade"
    p = format_p(result.p_value)
    if not result.significant:
        return (f"{var.name} has changed by {result.slope_per_decade:+.2f} "
                f"{unit_words(var.unit)} per decade since {start_year}, but this could "
                f"easily be chance ({p}). The data do not show a clear trend.")
    direction = "risen" if result.slope_per_decade > 0 else "fallen"
    strength = "very unlikely" if result.p_value < 0.01 else "unlikely"
    return (f"{var.name} has {direction} by {size} since {start_year}. "
            f"This trend is {strength} to be chance ({p}).")


# ---------------------------------------------------------------------------
# Command line: print the regional trends for a quick look
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    catalog, regions = load_catalog(), load_regions()
    parser = argparse.ArgumentParser(description="Print regional trends.")
    parser.add_argument("--region", action="append", choices=sorted(regions))
    args = parser.parse_args(argv)
    for key in args.region or sorted(regions):
        for code, var in catalog.variables.items():
            with xr.open_dataset(RAW_DIR / key / "prepared" / f"{code}.nc") as ds:
                r = trend(ds["region_monthly_anomaly"].load())
                cells = cell_trends(ds["monthly_anomaly"].load(), ds["land_mask"].load())
            n_land = int(ds["land_mask"].sum())
            n_sig = int(cells["significant"].sum())
            print(f"[{key} {code}] {verdict_label(r)}  slope {r.slope_per_decade:+.3f} "
                  f"({r.ci_low:+.3f} to {r.ci_high:+.3f}), Sen {r.sen_slope_per_decade:+.3f}, "
                  f"{format_p(r.p_value)}, n_eff {r.n_eff:.0f}/{r.n}; "
                  f"cells significant after FDR: {n_sig}/{n_land}")
            print("   ", sentence(var, r, catalog.start.year))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
