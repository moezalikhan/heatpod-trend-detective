"""Tests for the region config: tiles fit the API limit and cover each box."""

import pytest

from config.catalog import load_regions


def test_both_regions_load_with_tiles_inside_the_limit():
    regions = load_regions()
    assert set(regions) == {"florida", "phoenix"}
    assert len(regions["florida"].tiles) == 4
    assert len(regions["phoenix"].tiles) == 1
    for r in regions.values():
        assert all(t.height <= 4.5 and t.width <= 4.5 for t in r.tiles)


def test_oversized_tile_is_rejected(tmp_path):
    bad = tmp_path / "regions.yaml"
    bad.write_text(
        "max_tile_degrees: 4.5\n"
        "regions:\n"
        "  x:\n"
        "    name: X\n"
        "    box: {lat_min: 0, lat_max: 5, lon_min: 0, lon_max: 5}\n"
        "    tiles: [{lat_min: 0, lat_max: 5, lon_min: 0, lon_max: 5}]\n"
    )
    with pytest.raises(ValueError, match="larger than"):
        load_regions(bad)


def test_tiles_that_leave_a_hole_are_rejected(tmp_path):
    bad = tmp_path / "regions.yaml"
    bad.write_text(
        "max_tile_degrees: 4.5\n"
        "regions:\n"
        "  x:\n"
        "    name: X\n"
        "    box: {lat_min: 0, lat_max: 4, lon_min: 0, lon_max: 4}\n"
        "    tiles: [{lat_min: 0, lat_max: 2, lon_min: 0, lon_max: 4}]\n"
    )
    with pytest.raises(ValueError, match="do not cover"):
        load_regions(bad)
