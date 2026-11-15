"""Step 2. Where are we looking?"""

import streamlit as st

from app import data_loader as dl
from app.components import charts


def render(case) -> None:
    info = case.region_info
    b = info["box"]
    st.header("2. Where are we looking?", anchor=False)
    st.markdown(f'<div class="case-label">Region</div>', unsafe_allow_html=True)
    st.subheader(f"{info['name']} ({info['climate']})", anchor=False)
    st.markdown(
        f"The study box runs from latitude {b['lat_min']} to {b['lat_max']} and longitude "
        f"{b['lon_min']} to {b['lon_max']}. NASA's data come as squares of about 55 km. "
        f"We keep the **{info['n_land_cells']} squares on land**; sea squares are removed.")
    cells = dl.table(case.region, case.code, "cells")
    st.altair_chart(charts.region_map(cells, dl.outline(case.region), info.get("places")),
                    theme=None, width="stretch")
    st.caption("Grey squares: land grid cells used in every calculation. Lines: coastline, state "
               "and country borders. Squares mark a few cities for orientation.")
