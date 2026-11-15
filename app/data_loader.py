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


# Every cache key includes the file's modification time and size. Streamlit Cloud
# can pull a new commit without restarting, and a cache keyed only by
# (region, variable) then mixed old and new results files on one page.

def _stamp(path: Path) -> tuple[int, int]:
    s = path.stat()
    return s.st_mtime_ns, s.st_size


@st.cache_data(show_spinner=False)
def _json_cached(path: str, stamp: tuple[int, int]) -> dict:
    return _json(Path(path))


@st.cache_data(show_spinner=False)
def _parquet_cached(path: str, stamp: tuple[int, int]) -> pd.DataFrame:
    return pd.read_parquet(path)


def _load_json(path: Path) -> dict:
    return _json_cached(str(path), _stamp(path))


def index() -> dict:
    return _load_json(RESULTS / "index.json")


def region(key: str) -> dict:
    return _load_json(RESULTS / key / "region.json")


def outline(key: str) -> dict:
    return _load_json(RESULTS / key / "outline.geojson")


def summary(key: str, code: str) -> dict:
    return _load_json(RESULTS / key / code / "summary.json")


def table(key: str, code: str, name: str) -> pd.DataFrame:
    """One of: series_monthly, series_annual, cells, pod_series."""
    path = RESULTS / key / code / f"{name}.parquet"
    return _parquet_cached(str(path), _stamp(path))


@st.cache_data(show_spinner=False)
def _glossary_cached(stamp: tuple[int, int]) -> dict:
    return yaml.safe_load(GLOSSARY.read_text(encoding="utf-8"))


def glossary() -> dict:
    return _glossary_cached(_stamp(GLOSSARY))
