"""Step 5. Are the worst days getting worse? Yearly maximum trend and extreme value comparison."""

import streamlit as st

from app import data_loader as dl
from app.components import charts, verdict


def _years(v) -> str:
    if v is None or v > 1000:
        return "more than 1000"
    return f"{v:.0f}" if v >= 2 else f"{v:.1f}"


def _p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _switch(code: str) -> None:
    # Runs before the rerun, so the sidebar picker and the URL follow the new variable.
    st.session_state.variable = code


def render(case) -> None:
    st.header("5. Are the worst days getting worse?", anchor=False)
    ext = case.summary.get("extremes")
    if ext is None:
        st.markdown(f"Extremes need a variable with a clear \"worst day\". For "
                    f"{case.var.name.lower()} we look at the hottest days through these two:")
        cat = dl.catalog()
        b1, b2, _ = st.columns([1, 1.25, 0.75])
        b1.button(f"Switch to {cat['HEAT_INDEX'].name.lower()}", on_click=_switch,
                  args=("HEAT_INDEX",), type="primary", width="stretch", key="to_hi")
        b2.button("Switch to daily max temperature", on_click=_switch,
                  args=("T2M_MAX",), width="stretch", key="to_tmax")
        return

    unit = case.var.unit
    (a0, a1), (b0, b1) = ext["periods"]
    annual = dl.table(case.region, case.code, "series_annual")
    cells = dl.table(case.region, case.code, "cells")
    mt, mc = ext["max_trend"], ext["max_trend_cells"]

    # --- 1. Trend of each year's most extreme day (45 values) ---------------------
    st.subheader("Is the most extreme day of the year rising?", anchor=False)
    st.markdown(f"Each dot is the most extreme day of one year, {a0} to {b1}. "
                "The line is the trend of these 45 yearly maximums.")
    st.altair_chart(charts.yearly_maxima(annual, ext, unit, ext["show_level"]), theme=None, width="stretch")
    verdict.show(
        mt["verdict"],
        f"{mt['slope_per_decade']:+.2f} {unit} per decade",
        [f"Mann-Kendall (corrected for linked years): {_p(mt['p_value'])}",
         f"95% range: {mt['ci_low']:+.2f} to {mt['ci_high']:+.2f} {unit} per decade",
         f"Sen's slope (robust check): {mt['sen_slope_per_decade']:+.2f} {unit} per decade",
         f"Squares with a significant rise: {mc['rising']} of {mc['n_land']}"
         + (f", fall: {mc['falling']}" if mc["falling"] else "")],
        mt["sentence"],
    )
    st.altair_chart(charts.cell_map(cells, dl.outline(case.region), "max_trend_slope",
                                    f"{unit} per decade", f"Trend of yearly maximum ({unit} per decade)",
                                    significant_col="max_trend_significant"),
                    theme=None, width="stretch")
    st.caption("Trend of the yearly maximum in every square. Grey: not significant after "
               "correcting for testing many squares at once.")

    # --- 2. How often is the old 20-year level reached now? (GEV) -------------------
    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    st.subheader("How often is the old 20-year level reached now?", anchor=False)
    st.markdown(f"We fit an extreme value curve to {a0}-{a1} and to {b0}-{b1} separately. "
                "With only about 22 years in each period, this comparison has wide ranges.")
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
