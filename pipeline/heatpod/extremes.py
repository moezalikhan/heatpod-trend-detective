"""Extremes engine: extreme value statistics (Module 6).

Question: are the worst days getting worse?

1. Take each year's maximum (from Module 3), region-wide and per grid cell.
2. Split the years into two periods: 1981 to 2003 and 2004 to 2025.
3. Fit a GEV (Generalized Extreme Value) distribution to each period.
   With only about 22 maximums per period, a free shape parameter is very
   unstable, so both periods share one shape (location and scale differ) and the
   shape is kept within -0.5 to 0.5, the usual range for climate extremes
   (Martins and Stedinger 2000).
4. 20-year return level: the value exceeded on average once every 20 years.
5. Plain words: the old 20-year level now has a return period of N years.
6. Bootstrap: resample the years within each period, refit, and take the middle
   95% of the results as the uncertainty range. The change is SIGNIFICANT when
   that range excludes zero. Checked by simulation (100 cases, 23 + 22 years):
   6% false alarms with no real change; it finds about half of moderate real
   shifts (one scale unit), so it is on the cautious side.

Shape convention: xi > 0 heavy tail, xi < 0 bounded tail. SciPy's genextreme
uses c = -xi.

Run from the repository root:
  python -m pipeline.heatpod.extremes
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import optimize, stats

from config.catalog import Variable, load_catalog, load_regions

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PERIODS = ((1981, 2003), (2004, 2025))
RETURN_PERIOD = 20
SHAPE_BOUNDS = (-0.5, 0.5)
N_BOOT = 1000
CONFIDENCE = 0.95


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GevPair:
    """GEV fits for the two periods with a shared shape."""

    loc: tuple[float, float]
    scale: tuple[float, float]
    shape: float  # xi

    def dist(self, period: int):
        return stats.genextreme(c=-self.shape, loc=self.loc[period], scale=self.scale[period])


def _start(x: np.ndarray) -> tuple[float, float]:
    """Gumbel method-of-moments starting values (location, scale)."""
    scale = max(np.std(x) * np.sqrt(6) / np.pi, 1e-3)
    return float(np.mean(x) - 0.5772 * scale), float(scale)


def fit_gev_pair(old: np.ndarray, new: np.ndarray, start: GevPair | None = None) -> GevPair:
    """Maximum likelihood fit: own location and scale per period, one shared shape.

    `start` (a previous fit) gives one good starting point, which makes the
    bootstrap fast; without it a few starting shapes are tried.
    """
    def nll(theta):
        mu1, ls1, mu2, ls2, xi = theta
        ll = (stats.genextreme.logpdf(old, -xi, mu1, np.exp(ls1)).sum()
              + stats.genextreme.logpdf(new, -xi, mu2, np.exp(ls2)).sum())
        return -ll if np.isfinite(ll) else 1e10   # outside the support: reject

    if start is not None:
        starts = [[start.loc[0], np.log(start.scale[0]), start.loc[1],
                   np.log(start.scale[1]), start.shape]]
    else:
        (m1, s1), (m2, s2) = _start(old), _start(new)
        starts = [[m1, np.log(s1), m2, np.log(s2), xi0] for xi0 in (-0.1, 0.1, -0.3)]
    best = None
    for theta0 in starts:                          # keep the best of the starts
        res = optimize.minimize(
            nll, theta0, method="L-BFGS-B",
            bounds=[(None, None), (None, None), (None, None), (None, None), SHAPE_BOUNDS])
        if best is None or res.fun < best.fun:
            best = res
    mu1, ls1, mu2, ls2, xi = best.x
    return GevPair((mu1, mu2), (np.exp(ls1), np.exp(ls2)), float(xi))


def return_level(fit: GevPair, period: int, years: int = RETURN_PERIOD) -> float:
    """Value exceeded on average once every `years` years."""
    return float(fit.dist(period).ppf(1 - 1 / years))


def new_return_period(fit: GevPair, years: int = RETURN_PERIOD) -> float:
    """How often the old `years`-year level is reached in the new period (in years)."""
    p = float(fit.dist(1).sf(return_level(fit, 0, years)))
    return np.inf if p <= 0 else 1 / p


# ---------------------------------------------------------------------------
# Analysis with bootstrap uncertainty
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExtremesResult:
    shape: float
    loc_old: float
    scale_old: float
    loc_new: float
    scale_new: float
    level_old: float               # 20-year return level, 1981-2003
    level_new: float               # 20-year return level, 2004-2025
    change: float                  # level_new - level_old
    change_low: float              # 95% bootstrap range of the change
    change_high: float
    period_new: float              # old 20-year level now happens once every N years
    period_new_low: float
    period_new_high: float
    significant: bool              # range of the change excludes zero
    n_boot: int

    def to_dict(self) -> dict:
        return asdict(self)


def split_periods(annual_max: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    years = annual_max["year"]
    parts = []
    for a, b in PERIODS:
        x = annual_max.sel(year=(years >= a) & (years <= b)).values.astype(float)
        parts.append(x[np.isfinite(x)])
    return parts[0], parts[1]


def analyze(annual_max: xr.DataArray, n_boot: int = N_BOOT, seed: int = 0) -> ExtremesResult:
    """GEV comparison of the two periods for one yearly-maximum series."""
    old, new = split_periods(annual_max)
    fit = fit_gev_pair(old, new)
    lo_q, hi_q = (1 - CONFIDENCE) / 2, 1 - (1 - CONFIDENCE) / 2

    changes, periods = [], []
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        b = fit_gev_pair(rng.choice(old, len(old)), rng.choice(new, len(new)), start=fit)
        changes.append(return_level(b, 1) - return_level(b, 0))
        periods.append(new_return_period(b))
    has_boot = n_boot > 0
    c_lo, c_hi = np.quantile(changes, [lo_q, hi_q]) if has_boot else (np.nan, np.nan)
    # "Never reached again" comes back as infinity; cap it so the quantiles stay finite.
    periods = np.minimum(periods, 1e6)
    p_lo, p_hi = np.quantile(periods, [lo_q, hi_q]) if has_boot else (np.nan, np.nan)

    level_old, level_new = return_level(fit, 0), return_level(fit, 1)
    return ExtremesResult(
        shape=fit.shape, loc_old=fit.loc[0], scale_old=fit.scale[0],
        loc_new=fit.loc[1], scale_new=fit.scale[1],
        level_old=level_old, level_new=level_new, change=level_new - level_old,
        change_low=float(c_lo), change_high=float(c_hi),
        period_new=float(new_return_period(fit)),
        period_new_low=float(p_lo), period_new_high=float(p_hi),
        significant=bool(has_boot and (c_lo > 0 or c_hi < 0)), n_boot=n_boot,
    )


def cell_extremes(annual_max: xr.DataArray, mask: xr.DataArray) -> xr.Dataset:
    """Return levels per land cell (point estimates; no bootstrap, to stay fast)."""
    names = ("level_old", "level_new", "change", "period_new")
    out = {k: xr.full_like(mask, np.nan, dtype=float) for k in names}
    for i in range(mask.sizes["lat"]):
        for j in range(mask.sizes["lon"]):
            if bool(mask[i, j]):
                r = analyze(annual_max.isel(lat=i, lon=j), n_boot=0)
                for k in names:
                    out[k][i, j] = getattr(r, k)
    return xr.Dataset(out)


# ---------------------------------------------------------------------------
# Plain-language sentence
# ---------------------------------------------------------------------------

def format_years(n: float) -> str:
    if not np.isfinite(n) or n > 1000:
        return "more than 1000"
    return f"{n:.0f}" if n >= 2 else f"{n:.1f}"


def sentence(var: Variable, r: ExtremesResult) -> str:
    level = f"{r.level_old:.1f} {var.unit}"
    head = (f"The {var.name.lower()} level that used to be reached about once every "
            f"{RETURN_PERIOD} years ({level}) is now reached about once every "
            f"{format_years(r.period_new)} years")
    rng = (f" (likely range {format_years(r.period_new_low)} to "
           f"{format_years(r.period_new_high)} years)")
    if r.significant:
        return head + rng + "."
    return head + rng + ", but this change could be chance."


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    catalog, regions = load_catalog(), load_regions()
    parser = argparse.ArgumentParser(description="Print extremes results.")
    parser.add_argument("--region", action="append", choices=sorted(regions))
    args = parser.parse_args(argv)
    for key in args.region or sorted(regions):
        for code in catalog.extreme_codes():
            with xr.open_dataset(RAW_DIR / key / "prepared" / f"{code}.nc") as ds:
                region_max, cell_max = ds["region_annual_max"].load(), ds["annual_max"].load()
                mask = ds["land_mask"].load()
            r = analyze(region_max)
            cells = cell_extremes(cell_max, mask)
            u = catalog[code].unit
            print(f"[{key} {code}] 20-yr level {r.level_old:.2f} -> {r.level_new:.2f} {u} "
                  f"(change {r.change:+.2f}, 95% {r.change_low:+.2f} to {r.change_high:+.2f}) "
                  f"shape {r.shape:+.2f}; cells with a higher level: "
                  f"{int((cells['change'] > 0).sum())}/{int(mask.sum())}")
            print("   ", sentence(catalog[code], r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
