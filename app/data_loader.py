"""Load results files for the app. Everything is read once and cached.

The app never calls the NASA API; it only reads data/results/ (and the configs).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from config.catalog import Catalog, load_catalog

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
GLOSSARY = Path(__file__).parent / "content" / "glossary.yaml"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def catalog() -> Catalog:
    return load_catalog()


@st.cache_data(show_spinner=False)
def index() -> dict:
    return _json(RESULTS / "index.json")


@st.cache_data(show_spinner=False)
def region(key: str) -> dict:
    return _json(RESULTS / key / "region.json")


@st.cache_data(show_spinner=False)
def outline(key: str) -> dict:
    return _json(RESULTS / key / "outline.geojson")


@st.cache_data(show_spinner=False)
def summary(key: str, code: str) -> dict:
    return _json(RESULTS / key / code / "summary.json")


@st.cache_data(show_spinner=False)
def table(key: str, code: str, name: str) -> pd.DataFrame:
    """One of: series_monthly, series_annual, cells, pod_series."""
    return pd.read_parquet(RESULTS / key / code / f"{name}.parquet")


@st.cache_data(show_spinner=False)
def glossary() -> dict:
    return yaml.safe_load(GLOSSARY.read_text(encoding="utf-8"))
