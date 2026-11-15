"""Every time chart uses the same x axis ticks: every 5 years, 1985 to 2025."""

from app import data_loader as dl
from app.components import charts

EXPECTED = list(range(1985, 2026, 5))


def _x_tick_years(chart) -> list[list[int]]:
    """Years of the x axis tick values in every layer that sets them."""
    spec = chart.to_dict()
    found = []
    for layer in spec.get("layer", [spec]):
        axis = layer.get("encoding", {}).get("x", {}).get("axis") or {}
        values = axis.get("values")
        if values:
            found.append([v["year"] if isinstance(v, dict) else v for v in values])
    return found


def test_all_time_charts_tick_every_five_years():
    region, code = "florida", "HEAT_INDEX"
    s = dl.summary(region, code)
    annual = dl.table(region, code, "series_annual")
    pod = dl.table(region, code, "pod_series")
    built = {
        "stripes": charts.stripes(annual, "°C"),
        "time_series": charts.time_series(dl.table(region, code, "series_monthly"), "°C", s["trend"]),
        "pod_series": charts.pod_series(pod[pod["mode"] == 1]),
        "yearly_maxima": charts.yearly_maxima(annual, s["extremes"], "°C", show_level=False),
    }
    for name, chart in built.items():
        ticks = _x_tick_years(chart)
        assert ticks, f"{name}: no x tick values set"
        assert all(t == EXPECTED for t in ticks), f"{name}: {ticks}"
