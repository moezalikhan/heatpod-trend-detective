"""Warming stripes band: one flat colored bar per year (blue cooler, red warmer)."""

import streamlit as st

from app import data_loader as dl
from app.components import charts


def show(region: str, code: str, unit: str) -> None:
    annual = dl.table(region, code, "series_annual")
    st.altair_chart(charts.stripes(annual, unit), theme=None, width="stretch")
