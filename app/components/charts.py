"""Altair chart builders with the project palette (docs/PLAN.md Section 8).

All charts are flat: thin lines, no gradients, no shadows. Color is used only
for data. Show every chart with st.altair_chart(chart, theme=None, width="stretch").
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

# Interface colors (paper and ink).
INK = "#1D2430"
SLATE = "#5B6472"
RULE = "#D9D3C7"
PAPER = "#F6F3EC"
GREY = "#BDBAB3"          # not significant / no data

# Data colors (ColorBrewer, colorblind safe).
RDBU = ["#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "#F7F7F7",
        "#FDDBC7", "#F4A582", "#D6604D", "#B2182B"]          # cooler/less -> warmer/more
YLORRD = ["#FFFFCC", "#FED976", "#FD8D3C", "#E31A1C", "#800026"]
UP, DOWN = "#B2182B", "#2166AC"                             # rising, falling
PERIOD_OLD, PERIOD_NEW = "#4393C3", "#D6604D"               # 1981-2003, 2004-2025

SANS = "plex-sans, 'IBM Plex Sans', sans-serif"
MONO = "plex-mono, 'IBM Plex Mono', monospace"


def style(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
    """Case File look: paper background, recessive axes, project fonts."""
    return (
        chart.configure(background="transparent", font=SANS)
        .configure_view(stroke=None)
        .configure_axis(
            domainColor=RULE, gridColor=RULE, gridOpacity=0.6, tickColor=RULE,
            labelColor=SLATE, titleColor=SLATE, labelFont=MONO, labelFontSize=11,
            titleFont=SANS, titleFontSize=12, titleFontWeight="normal",
        )
        .configure_legend(labelColor=SLATE, titleColor=SLATE, labelFont=MONO,
                          titleFont=SANS, titleFontWeight="normal", orient="bottom")
        .configure_title(font=SANS, color=INK, fontSize=13, fontWeight="normal", anchor="start")
    )


def nice_limit(values) -> float:
    """Largest absolute value, rounded UP to one significant figure (0.43 -> 0.5, 1.6 -> 2)."""
    v = np.abs(np.asarray(values, dtype=float))
    m = float(np.nanmax(v)) if v.size and np.isfinite(v).any() else 1.0
    if m <= 0:
        return 1.0
    f = 10 ** np.floor(np.log10(m))
    return float(np.ceil(round(m / f, 6)) * f)


def diverging(values, title: str = "") -> alt.Scale:
    """Blue (less) to red (more) with white fixed at zero, symmetric around zero."""
    n = nice_limit(values)
    return alt.Scale(domain=[-n, n], domainMid=0, range=RDBU, clamp=True)


def three_tick_legend(values, title: str) -> alt.Legend:
    """Legend with only -n, 0, +n, rounded, so labels never overlap."""
    n = nice_limit(values)
    decimals = max(0, int(-np.floor(np.log10(n))))
    fmt = f"+.{decimals}f"
    return alt.Legend(title=title, titleLimit=400, gradientLength=220, values=[-n, 0, n],
                      labelExpr=f"datum.value == 0 ? '0' : format(datum.value, '{fmt}')",
                      labelOverlap=False)


def stripes(annual: pd.DataFrame, unit: str) -> alt.Chart:
    """Warming stripes: one flat bar per year, colored by the yearly anomaly."""
    df = annual.assign(label=annual["anomaly"].map(lambda v: f"{v:+.2f} {unit}"))
    chart = alt.Chart(df).mark_rect().encode(
        x=alt.X("year:O", title=None, axis=alt.Axis(
            values=[1981, 1990, 2000, 2010, 2020, 2025], labelAngle=0, ticks=False,
            domain=False, labelFont=MONO)),
        color=alt.Color("anomaly:Q", scale=diverging(df["anomaly"], unit), legend=None),
        tooltip=[alt.Tooltip("year:O", title="Year"), alt.Tooltip("label:N", title="Difference from normal")],
    ).properties(height=44, width="container")
    return style(chart)


# Time axes: one tick every 5 years (1985, 1990, ... 2025).
YEAR_TICKS = [alt.DateTime(year=y, month=1, date=1) for y in range(1985, 2026, 5)]


def _time_x() -> alt.X:
    return alt.X("time:T", title=None, axis=alt.Axis(values=YEAR_TICKS, format="%Y", labelAngle=0))


def time_series(monthly: pd.DataFrame, unit: str, verdict: dict) -> alt.LayerChart:
    """Monthly anomalies with the trend line and its 95% band, plus a hover crosshair."""
    color = GREY if not verdict["significant"] else (UP if verdict["slope_per_decade"] > 0 else DOWN)
    base = alt.Chart(monthly).encode(x=_time_x())
    y_title = f"Difference from normal ({unit})"

    band = base.mark_area(opacity=0.18, color=color).encode(
        y=alt.Y("trend_low:Q", title=y_title), y2="trend_high:Q")
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=RULE, strokeWidth=1).encode(y="y:Q")
    months = base.mark_line(color=SLATE, strokeWidth=0.8, opacity=0.55).encode(y="anomaly:Q")
    trend = base.mark_line(color=color, strokeWidth=2.5,
                           strokeDash=[] if verdict["significant"] else [6, 4]).encode(y="trend:Q")

    hover = alt.selection_point(nearest=True, on="pointerover", fields=["time"], empty=False)
    catcher = base.mark_point(opacity=0, size=60).encode(y="anomaly:Q").add_params(hover)
    rule = base.mark_rule(color=SLATE, strokeWidth=1).encode(
        opacity=alt.when(hover).then(alt.value(0.7)).otherwise(alt.value(0)),
        tooltip=[alt.Tooltip("time:T", title="Month", format="%B %Y"),
                 alt.Tooltip("anomaly:Q", title=f"Difference from normal ({unit})", format="+.2f"),
                 alt.Tooltip("trend:Q", title=f"Trend line ({unit})", format="+.2f")])
    dot = base.mark_point(filled=True, color=INK, size=40).encode(
        y="anomaly:Q", opacity=alt.when(hover).then(alt.value(1)).otherwise(alt.value(0)))

    chart = alt.layer(band, zero, months, trend, catcher, rule, dot).properties(
        height=320, width="container", padding={"top": 10, "left": 5, "right": 5, "bottom": 5})
    return style(chart)


# ---------------------------------------------------------------------------
# Maps: grid cells drawn as flat squares, with the land outline on top
# ---------------------------------------------------------------------------

def _cell_features(cells: pd.DataFrame, value: str, extra: list[str]) -> dict:
    lats, lons = np.sort(cells["lat"].unique()), np.sort(cells["lon"].unique())
    dlat = (np.diff(lats).min() if len(lats) > 1 else 0.5) / 2
    dlon = (np.diff(lons).min() if len(lons) > 1 else 0.625) / 2
    features = []
    for row in cells[cells["land"]].itertuples(index=False):
        r = row._asdict()
        la, lo = r["lat"], r["lon"]
        ring = [[lo - dlon, la - dlat], [lo - dlon, la + dlat], [lo + dlon, la + dlat],
                [lo + dlon, la - dlat], [lo - dlon, la - dlat]]   # clockwise for Vega
        props = {"value": _num(r[value]), "lat": la, "lon": lo}
        props.update({k: _num(r[k]) if not isinstance(r[k], (bool, np.bool_)) else bool(r[k]) for k in extra})
        features.append({"type": "Feature", "properties": props,
                         "geometry": {"type": "Polygon", "coordinates": [ring]}})
    return {"type": "FeatureCollection", "features": features}


def _num(v):
    v = float(v)
    return None if not np.isfinite(v) else v


def _geo(fc: dict) -> alt.InlineData:
    return alt.InlineData(values=fc, format=alt.DataFormat(property="features", type="json"))


def _outline_layers(outline: dict) -> list:
    """State and country borders in thin grey, the coastline in ink on top."""
    border = alt.Chart(_geo(outline)).transform_filter("datum.properties.kind == 'border'") \
        .mark_geoshape(fill=None, stroke=SLATE, strokeWidth=0.6, strokeOpacity=0.7)
    coast = alt.Chart(_geo(outline)).transform_filter("datum.properties.kind == 'coast'") \
        .mark_geoshape(fill=None, stroke=INK, strokeWidth=1)
    return [border, coast]


def cell_map(cells: pd.DataFrame, outline: dict, value: str, legend_title: str,
             tooltip_title: str, significant_col: str | None = None, fmt: str = "+.2f") -> alt.LayerChart:
    """Grid cells colored on a diverging scale; optional grey for not significant."""
    extra = [significant_col] if significant_col else []
    fc = _cell_features(cells, value, extra)
    vals = [f["properties"]["value"] for f in fc["features"] if f["properties"]["value"] is not None]
    color = alt.Color("properties.value:Q", scale=diverging(vals),
                      legend=three_tick_legend(vals, legend_title))
    if significant_col:
        color = alt.when(f"datum.properties.{significant_col}").then(color).otherwise(alt.value(GREY))
    tips = [alt.Tooltip("properties.value:Q", title=tooltip_title, format=fmt),
            alt.Tooltip("properties.lat:Q", title="Latitude", format=".2f"),
            alt.Tooltip("properties.lon:Q", title="Longitude", format=".3f")]
    if significant_col:
        tips.append(alt.Tooltip(f"properties.{significant_col}:N", title="Significant after correction"))

    squares = alt.Chart(_geo(fc)).mark_geoshape(stroke=PAPER, strokeWidth=1).encode(
        color=color, tooltip=tips)
    # Side padding keeps the outer legend labels (for example -0.4) inside the chart.
    chart = alt.layer(squares, *_outline_layers(outline)).project(type="mercator").properties(
        height=420, width="container", padding={"left": 24, "right": 24, "top": 4, "bottom": 4})
    return style(chart)


def region_map(cells: pd.DataFrame, outline: dict, places: list[dict] | None = None) -> alt.LayerChart:
    """The land cells used in the analysis, as plain squares under the outline, with a few cities."""
    fc = _cell_features(cells, "lat", [])
    squares = alt.Chart(_geo(fc)).mark_geoshape(fill=RULE, stroke=PAPER, strokeWidth=1).encode(
        tooltip=[alt.Tooltip("properties.lat:Q", title="Latitude", format=".2f"),
                 alt.Tooltip("properties.lon:Q", title="Longitude", format=".3f")])
    layers = [squares, *_outline_layers(outline)]
    if places:
        p = pd.DataFrame(places)
        layers.append(alt.Chart(p).mark_point(filled=True, color=INK, size=30, shape="square")
                      .encode(longitude="lon:Q", latitude="lat:Q", tooltip=["name:N"]))
        layers.append(alt.Chart(p).mark_text(align="left", dx=6, dy=-6, font=MONO, fontSize=11, color=INK)
                      .encode(longitude="lon:Q", latitude="lat:Q", text="name:N"))
    return style(alt.layer(*layers).project(type="mercator").properties(height=380, width="container"))


def pod_series(series: pd.DataFrame) -> alt.LayerChart:
    """One mode's time series: thin monthly line and a 12-month running mean."""
    df = series.sort_values("time").assign(
        smooth=lambda d: d["value"].rolling(12, center=True, min_periods=6).mean())
    base = alt.Chart(df).encode(x=_time_x())
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=RULE).encode(y="y:Q")
    raw = base.mark_line(color=SLATE, strokeWidth=0.8, opacity=0.5).encode(
        y=alt.Y("value:Q", title="Pattern strength (standard deviations)"))
    smooth = base.mark_line(color=INK, strokeWidth=2).encode(
        y="smooth:Q",
        tooltip=[alt.Tooltip("time:T", title="Month", format="%B %Y"),
                 alt.Tooltip("smooth:Q", title="12-month average", format="+.2f")])
    return style(alt.layer(zero, raw, smooth).properties(
        height=220, width="container", padding={"top": 10, "left": 5, "right": 5, "bottom": 5}))


# ---------------------------------------------------------------------------
# Extremes
# ---------------------------------------------------------------------------

def yearly_maxima(annual: pd.DataFrame, ext: dict, unit: str, show_level: bool) -> alt.LayerChart:
    """Each year's maximum, colored by period, with its trend line.

    The 20-year levels are drawn only when reliable (not for heat index).
    """
    (a0, a1), (b0, b1) = ext["periods"]
    df = annual.copy()
    df["period"] = np.where(df["year"] <= a1, f"{a0}-{a1}", f"{b0}-{b1}")
    if show_level:
        df["y"] = df["max"]
        y_title = f"Highest value of the year ({unit})"
    else:
        # Differences from the first period's average maximum (absolute values not reliable).
        df["y"] = df["max"] - df.loc[df["year"] <= a1, "max"].mean()
        y_title = f"Yearly maximum vs {a0}-{a1} average ({unit})"
    period_scale = alt.Scale(domain=[f"{a0}-{a1}", f"{b0}-{b1}"], range=[PERIOD_OLD, PERIOD_NEW])
    color = alt.Color("period:N", scale=period_scale, legend=alt.Legend(title=None))
    x = alt.X("year:Q", title=None, scale=alt.Scale(domain=[1980, 2026], nice=False),
              axis=alt.Axis(values=list(range(1985, 2026, 5)), format="d", labelAngle=0))
    dots = alt.Chart(df).mark_point(filled=True, size=55, opacity=1).encode(
        x=x, y=alt.Y("y:Q", title=y_title, scale=alt.Scale(zero=False)), color=color,
        tooltip=[alt.Tooltip("year:Q", title="Year", format="d"),
                 alt.Tooltip("y:Q", title=y_title, format=".2f")])
    layers = [dots]
    mt = ext.get("max_trend")
    if mt:
        # Least-squares line through the means, with the slope from the yearly trend test.
        line = df[["year"]].copy()
        line["fit"] = df["y"].mean() + mt["slope_per_decade"] / 10 * (df["year"] - df["year"].mean())
        color_line = GREY if not mt["significant"] else (UP if mt["slope_per_decade"] > 0 else DOWN)
        layers.append(alt.Chart(line).mark_line(
            color=color_line, strokeWidth=2.5,
            strokeDash=[] if mt["significant"] else [6, 4]).encode(x="year:Q", y="fit:Q"))
    if show_level and ext.get("level_old") is not None:
        levels = pd.DataFrame({
            "y": [ext["level_old"], ext["level_new"]],
            "label": [f"20-year level {a0}-{a1}: {ext['level_old']:.1f} {unit}",
                      f"20-year level {b0}-{b1}: {ext['level_new']:.1f} {unit}"],
            "period": [f"{a0}-{a1}", f"{b0}-{b1}"]})
        layers.append(alt.Chart(levels).mark_rule(strokeDash=[5, 3], strokeWidth=1.5).encode(
            y="y:Q", color=alt.Color("period:N", scale=period_scale, legend=None),
            tooltip=[alt.Tooltip("label:N", title="Level")]))
    return style(alt.layer(*layers).properties(height=300, width="container"))


def return_period(ext: dict) -> alt.LayerChart:
    """How often the old 20-year level happens now, with its likely range (log scale)."""
    rp = ext["return_period"]
    cap = lambda v: float(min(max(v, 1.0), 1000.0)) if v is not None else 1000.0  # noqa: E731
    df = pd.DataFrame({"row": ["now"], "n": [cap(ext["period_new"])],
                       "lo": [cap(ext["period_new_low"])], "hi": [cap(ext["period_new_high"])]})
    x_scale = alt.Scale(type="log", domain=[1, 1000])
    x_axis = alt.Axis(values=[1, 2, 5, 10, 20, 50, 100, 1000], format="d")
    x_title = "Reached about once every ... years (log scale)"
    x = alt.X("n:Q", title=x_title, scale=x_scale, axis=x_axis)
    ref = alt.Chart(pd.DataFrame({"n": [rp]})).mark_rule(color=SLATE, strokeDash=[4, 3]).encode(x=x)
    ref_label = alt.Chart(pd.DataFrame({"n": [rp], "t": [f"before: every {rp} years"]})).mark_text(
        align="left", dx=4, dy=-16, color=SLATE, font=MONO, fontSize=11).encode(x="n:Q", text="t:N")
    rng = alt.Chart(df).mark_rule(color=INK, strokeWidth=2).encode(
        x=alt.X("lo:Q", scale=x_scale, title=x_title, axis=x_axis), x2="hi:Q")
    dot = alt.Chart(df).mark_point(filled=True, color=INK, size=120, shape="square").encode(
        x=x, tooltip=[alt.Tooltip("n:Q", title="Now: once every (years)", format=".1f"),
                      alt.Tooltip("lo:Q", title="Likely range from", format=".1f"),
                      alt.Tooltip("hi:Q", title="to", format=".1f")])
    return style(alt.layer(ref, ref_label, rng, dot).properties(height=90, width="container"))
