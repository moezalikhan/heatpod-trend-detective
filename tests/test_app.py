"""Streamlit AppTest smoke tests for the app."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def test_placeholder_page_runs_without_exceptions():
    at = AppTest.from_file(str(APP)).run(timeout=30)
    assert not at.exception
    assert at.title[0].value == "HeatPOD Trend Detective"
