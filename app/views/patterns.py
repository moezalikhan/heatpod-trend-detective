"""Step 4. Is it changing the same way everywhere? Trend map and POD patterns."""

import streamlit as st

from app import data_loader as dl
from app.components import charts


def render(case) -> None:
    unit = case.var.unit
    s = case.summary
    cells = dl.table(case.region, case.code, "cells")
    outline = dl.outline(case.region)
    c = s["cells"]

    st.header("4. Is it changing the same way everywhere?", anchor=False)
    st.markdown("**Trend in every square.** Red: rising. Blue: falling. Grey: not significant "
                "after correcting for testing many squares at once.")
    st.altair_chart(charts.cell_map(cells, outline, "slope_per_decade", f"{unit} per decade",
                                    f"Trend ({unit} per decade)", significant_col="significant"),
                    theme=None, width="stretch")
    if c["opposite"]:
        finding = (f"Opposite directions: {c['rising']} squares rise and {c['falling']} squares "
                   f"fall significantly.")
    elif c["rising"] or c["falling"]:
        direction = "rising" if c["rising"] else "falling"
        n = c["rising"] or c["falling"]
        finding = (f"Same direction everywhere: {n} of {c['n_land']} squares are significantly "
                   f"{direction}, none go the other way. The size of the change still differs "
                   f"from place to place.")
    else:
        finding = f"No square shows a significant trend ({c['n_land']} squares tested)."
    st.markdown(f'<p class="finding">{finding}</p>', unsafe_allow_html=True)

    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    st.markdown("**Patterns (POD).** The map's ups and downs, split into a few fixed patterns.")
    modes = s["pod"]["modes"]
    labels = {m["mode"]: f"Pattern {m['mode']} ({100 * m['variance_fraction']:.0f}%)" for m in modes}
    choice = st.radio("Pattern", list(labels), format_func=labels.get, horizontal=True,
                      key="pod_mode", label_visibility="collapsed")
    m = next(x for x in modes if x["mode"] == choice)
    st.altair_chart(charts.cell_map(cells, outline, f"pod_mode_{choice}",
                                    f"{unit} when the pattern is at +1",
                                    f"Movement ({unit})"),
                    theme=None, width="stretch")
    series = dl.table(case.region, case.code, "pod_series")
    st.altair_chart(charts.pod_series(series[series["mode"] == choice]), theme=None, width="stretch")
    p = "p < 0.001" if m["series_p_value"] < 0.001 else f"p = {m['series_p_value']:.3f}"
    sep = "" if m["separated"] else " This pattern is close in size to its neighbour, so the two may be mixed."
    st.caption(f"Pattern {choice} explains {100 * m['variance_fraction']:.1f}% of the "
               f"variation. Its strength over time: {m['series_verdict']} ({p}).{sep}")
