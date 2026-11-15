"""Tests for the shared variable catalog (pipeline side)."""

from datetime import date

import pytest

from config.catalog import load_catalog


def test_catalog_has_four_power_variables_and_heat_index():
    cat = load_catalog()
    assert cat.power_codes() == ["T2M", "T2M_MAX", "RH2M", "PRECTOTCORR"]
    assert cat["HEAT_INDEX"].is_calculated
    assert cat["HEAT_INDEX"].inputs == ("T2M_MAX", "RH2M")


def test_extremes_variables():
    assert load_catalog().extreme_codes() == ["T2M_MAX", "PRECTOTCORR", "HEAT_INDEX"]


def test_every_variable_has_plain_text_fields():
    for v in load_catalog().variables.values():
        assert v.name and v.unit and v.definition and v.why_it_matters


def test_period_is_1981_to_2025():
    cat = load_catalog()
    assert cat.start == date(1981, 1, 1)
    assert cat.end == date(2025, 12, 31)


def test_bad_file_is_rejected(tmp_path):
    bad = tmp_path / "variables.yaml"
    bad.write_text(
        "period: {start: 1981-01-01, end: 2025-12-31}\n"
        "variables:\n"
        "  HI: {name: x, unit: C, source: calculated, definition: d,"
        " why_it_matters: w, inputs: [NOPE]}\n"
    )
    with pytest.raises(ValueError, match="unknown inputs"):
        load_catalog(bad)
