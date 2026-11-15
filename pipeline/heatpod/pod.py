"""Pattern engine: POD / EOF analysis (Module 5).

Proper Orthogonal Decomposition splits the anomaly field into a few fixed
spatial patterns ("modes"), each with its own time series:

    anomaly(time, cell) = sum over modes of  series_k(time) * pattern_k(cell)

Steps:
1. Matrix X: months x land cells, monthly anomalies, each column centered.
2. Area weighting: multiply each column by sqrt(cos(latitude)), so a cell's
   share of the variance matches its area. Patterns are un-weighted afterwards.
3. SVD: X_w = U S V^T. Mode k explains S_k^2 / sum(S^2) of the variation.
4. Scaling: each mode's time series has standard deviation 1, and its pattern is
   in data units (for example C): how much each cell moves when the mode is at +1.
5. Sign: the pattern's area-weighted mean is made positive, so a rising series
   means the region as a whole goes up.
6. Each mode's series gets the Module 4 trend test.
7. North et al. (1982) rule of thumb: a mode is "separated" if its eigenvalue
   error bar does not overlap the next one. Unseparated modes can mix.

Run from the repository root:
  python -m pipeline.heatpod.pod                     # every region, every variable
  python -m pipeline.heatpod.pod --region florida --variable T2M --map
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from config.catalog import load_catalog, load_regions
from pipeline.heatpod import trends

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
N_MODES = 3


@dataclass
class PodResult:
    patterns: xr.DataArray        # (mode, lat, lon), data units per 1 std of the series
    series: xr.DataArray          # (mode, time), standard deviation 1
    variance_fraction: np.ndarray  # share of the total variation, per mode (all modes)
    separated: np.ndarray         # North's rule, per kept mode
    n_eff: float                  # effective sample size used for North's rule


def anomaly_matrix(anom: xr.DataArray, mask: xr.DataArray):
    """Stack land cells into columns. Returns (X, weights, cell index, times)."""
    stacked = anom.where(mask).stack(cell=("lat", "lon"))
    land = mask.stack(cell=("lat", "lon")).values
    stacked = stacked.isel(cell=np.flatnonzero(land))
    X = stacked.transpose("time", "cell").values.astype(float)
    X = X - X.mean(axis=0)                                   # center each cell
    w = np.sqrt(np.cos(np.deg2rad(stacked["lat"].values)))   # sqrt(cos lat)
    return X, w, stacked["cell"], stacked["time"]


def lag1_autocorrelation(x: np.ndarray) -> float:
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def pod(anom: xr.DataArray, mask: xr.DataArray, n_modes: int = N_MODES) -> PodResult:
    """Run POD on a (time, lat, lon) anomaly grid over the land cells."""
    X, w, cells, times = anomaly_matrix(anom, mask)
    U, S, Vt = np.linalg.svd(X * w, full_matrices=False)
    lam = S**2
    frac = lam / lam.sum()
    k = min(n_modes, len(S))

    # Scale: series with std 1, pattern in data units (un-weighted).
    n = X.shape[0]
    series = U[:, :k] * np.sqrt(n)                    # std 1 (U columns have unit norm)
    patterns = (Vt[:k] * S[:k, None] / np.sqrt(n)) / w  # data units per 1 std

    # Sign: area-weighted mean of each pattern positive.
    area = w**2
    for m in range(k):
        if np.sum(patterns[m] * area) < 0:
            patterns[m] *= -1
            series[:, m] *= -1

    # North's rule of thumb, with an effective sample size for correlated months.
    r1 = max(0.0, lag1_autocorrelation(series[:, 0]))
    n_eff = n * (1 - r1) / (1 + r1)
    err = lam * np.sqrt(2.0 / n_eff)
    separated = np.array([
        (m == 0 or lam[m - 1] - err[m - 1] > lam[m] + err[m])
        and (m + 1 >= len(lam) or lam[m] - err[m] > lam[m + 1] + err[m + 1])
        for m in range(k)
    ])

    modes = np.arange(1, k + 1)
    pat = xr.DataArray(patterns, dims=("mode", "cell"), coords={"mode": modes, "cell": cells})
    pat = pat.unstack("cell").reindex_like(mask)      # sea cells back as NaN
    ser = xr.DataArray(series.T, dims=("mode", "time"), coords={"mode": modes, "time": times})
    return PodResult(pat, ser, frac, separated, float(n_eff))


def opposite_trends(cells: xr.Dataset) -> dict:
    """Counts of significant rising and falling cells (after FDR)."""
    sig = cells["significant"]
    up = int((sig & (cells["slope_per_decade"] > 0)).sum())
    down = int((sig & (cells["slope_per_decade"] < 0)).sum())
    return {"rising": up, "falling": down, "opposite": up > 0 and down > 0}


# ---------------------------------------------------------------------------
# Command line: quick look
# ---------------------------------------------------------------------------

def ascii_map(pattern: xr.DataArray) -> str:
    """Rough text map: + above the pattern's land mean, - below, . sea."""
    mean = float(pattern.mean())
    rows = []
    for la in pattern["lat"].values[::-1]:
        row = pattern.sel(lat=la).values
        rows.append(f"{la:5.1f} " + "".join(
            "." if np.isnan(v) else ("+" if v > mean else "-") for v in row))
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    catalog, regions = load_catalog(), load_regions()
    parser = argparse.ArgumentParser(description="POD pattern summary.")
    parser.add_argument("--region", action="append", choices=sorted(regions))
    parser.add_argument("--variable", action="append", choices=list(catalog.variables))
    parser.add_argument("--map", action="store_true", help="print rough text maps")
    args = parser.parse_args(argv)
    for key in args.region or sorted(regions):
        for code in args.variable or list(catalog.variables):
            with xr.open_dataset(RAW_DIR / key / "prepared" / f"{code}.nc") as ds:
                anom, mask = ds["monthly_anomaly"].load(), ds["land_mask"].load()
            res = pod(anom, mask)
            opp = opposite_trends(trends.cell_trends(anom, mask))
            print(f"[{key} {code}] opposite cell trends: {opp}")
            for m in res.series["mode"].values:
                t = trends.trend(res.series.sel(mode=m))
                p = res.patterns.sel(mode=m)
                print(f"   mode {m}: {100 * res.variance_fraction[m - 1]:5.1f}% "
                      f"{'separated' if res.separated[m - 1] else 'NOT separated'}; "
                      f"pattern {float(p.min()):+.2f} to {float(p.max()):+.2f} {catalog[code].unit}; "
                      f"series trend {t.slope_per_decade:+.2f} std/decade, "
                      f"{trends.verdict_label(t)} ({trends.format_p(t.p_value)})")
                if args.map:
                    print(ascii_map(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
