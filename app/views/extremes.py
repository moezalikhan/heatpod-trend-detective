"""Step 5. Are the worst days getting worse? Extreme value comparison."""

import streamlit as st

from app import data_loader as dl
from app.components import charts, verdict


def _years(v) -> str:
    if v is None or v > 1000:
        return "more than 1000"
    return f"{v:.0f}" if v >= 2 else f"{v:.1f}"


def render(case) -> None:
    st.header("5. Are the worst days getting worse?", anchor=False)
    ext = case.summary.get("extremes")
    if ext is None:
        names = ", ".join(v.name.lower() for v in dl.catalog().variables.values() if v.extremes)
        st.markdown(f"Extremes are analysed for: {names}. Pick one of these in the sidebar.")
        return

    unit = case.var.unit
    (a0, a1), (b0, b1) = ext["periods"]
    st.markdown(
        f"We take the most extreme day of every year, fit an extreme value curve to "
        f"{a0}-{a1} and to {b0}-{b1}, and ask how often the old 20-year level is reached now.")
    annual = dl.table(case.region, case.code, "series_annual")
    st.altair_chart(charts.yearly_maxima(annual, ext, unit, ext["show_level"]), theme=None, width="stretch")
    st.altair_chart(charts.return_period(ext), theme=None, width="stretch")

    caution = None
    if ext["shape_at_bound"]:
        caution = ("Caution: the fitted curve hit its shape limit, so the likely range is "
                   "probably too narrow.")
    if not ext["show_level"]:
        caution = " ".join(x for x in (ext["note"], caution) if x)
    details = [f"Likely range: once every {_years(ext['period_new_low'])} to "
               f"{_years(ext['period_new_high'])} years",
               f"Change of the 20-year level: {ext['change']:+.1f} {unit} "
               f"(range {ext['change_low']:+.1f} to {ext['change_high']:+.1f})"]
    verdict.show(ext["verdict"], f"once every {_years(ext['period_new'])} years (was 20)",
                 details, ext["sentence"], caution)

    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    st.markdown("**Change of the 20-year level in every square** (no ranges per square).")
    cells = dl.table(case.region, case.code, "cells")
    st.altair_chart(charts.cell_map(cells, dl.outline(case.region), "extreme_change",
                                    f"{unit} change", f"Change of 20-year level ({unit})"),
                    theme=None, width="stretch")
