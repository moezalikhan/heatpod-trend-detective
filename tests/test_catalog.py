"""The app reads the same catalog as the pipeline, using only app dependencies."""

from config.catalog import load_catalog


def test_app_can_load_catalog():
    cat = load_catalog()
    assert len(cat.variables) == 5
    assert cat["T2M"].unit == "°C"
