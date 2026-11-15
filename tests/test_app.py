"""Streamlit AppTest checks: every step loads for every region and variable."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from config.catalog import load_catalog, load_regions

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")
STEPS = ["what", "where", "evidence", "patterns", "extremes"]


def run_case(region: str, variable: str, step: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["region"] = region
    at.session_state["variable"] = variable
    at.session_state["step"] = step
    return at.run()


def test_first_page_runs_with_title():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "HeatPOD Trend Detective"


@pytest.mark.parametrize("region", sorted(load_regions()))
@pytest.mark.parametrize("variable", list(load_catalog().variables))
@pytest.mark.parametrize("step", STEPS)
def test_every_step_loads(region, variable, step):
    at = run_case(region, variable, step)
    assert not at.exception, at.exception


def test_evidence_shows_a_verdict():
    at = run_case("florida", "T2M", "evidence")
    html = " ".join(m.value for m in at.markdown)
    assert "SIGNIFICANT" in html and "per decade" in html


def test_extremes_hides_heat_index_level_and_shows_note():
    at = run_case("florida", "HEAT_INDEX", "extremes")
    html = " ".join(m.value for m in at.markdown)
    assert "run several degrees too high" in html


@pytest.mark.parametrize("target", ["HEAT_INDEX", "T2M_MAX"])
def test_extremes_dead_end_offers_a_switch(target):
    at = run_case("florida", "T2M", "extremes")
    key = {"HEAT_INDEX": "to_hi", "T2M_MAX": "to_tmax"}[target]
    at.button(key=key).click().run()
    assert not at.exception
    assert at.session_state["variable"] == target
    assert at.query_params["variable"] == [target] or at.query_params["variable"] == target
    assert any("per decade" in m.value for m in at.markdown)


def test_demo_ends_on_a_variable_with_extremes():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.button[0].click().run()
    at.session_state["variable"] = "T2M"          # visitor changes the variable mid-demo
    at.session_state["step"] = "patterns"
    at.run()
    next(b for b in at.button if b.label == "Next step").click().run()
    assert at.session_state["step"] == "extremes"
    assert load_catalog()[at.session_state["variable"]].extremes


def test_patterns_mode_switch():
    at = run_case("phoenix", "T2M", "patterns")
    at.radio(key="pod_mode").set_value(2).run()
    assert not at.exception


def test_demo_walks_through_all_steps():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.button[0].click().run()
    assert at.session_state["region"] == "florida" and at.session_state["variable"] == "HEAT_INDEX"
    for expected in STEPS[1:]:
        next_button = next(b for b in at.button if b.label == "Next step")
        next_button.click().run()
        assert not at.exception
        assert at.session_state["step"] == expected
