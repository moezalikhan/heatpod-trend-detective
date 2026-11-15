"""Step 3. Is it changing? Time series, trend line and the verdict."""

import streamlit as st

from app import data_loader as dl
from app.components import charts, verdict


def render(case) -> None:
    t = case.summary["trend"]
    unit = case.var.unit
    st.header("3. Is it changing?", anchor=False)
    st.markdown(
        f"Each thin line point is one month's difference from normal for the whole region "
        f"(land squares, weighted by area). The thick line is the trend; the band shows its 95% range.")
    monthly = dl.table(case.region, case.code, "series_monthly")
    st.altair_chart(charts.time_series(monthly, unit, t), theme=None, width="stretch")

    p = "p < 0.001" if t["p_value"] < 0.001 else f"p = {t['p_value']:.3f}"
    verdict.show(
        t["verdict"],
        f"{t['slope_per_decade']:+.2f} {unit} per decade",
        [f"Mann-Kendall (corrected for linked months): {p}",
         f"95% range: {t['ci_low']:+.2f} to {t['ci_high']:+.2f} {unit} per decade",
         f"Sen's slope (robust check): {t['sen_slope_per_decade']:+.2f} {unit} per decade"],
        t["sentence"],
    )
