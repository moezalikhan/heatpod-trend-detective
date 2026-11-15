"""Variable catalog loader, shared by the pipeline and the app.

Reads config/variables.yaml and checks it. Uses only PyYAML and the standard
library, so the light app install can import it too.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

VARIABLES_FILE = Path(__file__).with_name("variables.yaml")

REQUIRED_FIELDS = ("name", "unit", "source", "definition", "why_it_matters")
SOURCES = ("power", "calculated")


@dataclass(frozen=True)
class Variable:
    """One variable: a NASA POWER code or a calculated one like heat index."""

    code: str
    name: str
    unit: str
    source: str
    definition: str
    why_it_matters: str
    inputs: tuple[str, ...] = ()
    method: str | None = None
    limitation: str | None = None
    extremes: bool = False          # fit extreme value statistics (Module 6)
    extremes_show_level: bool = True  # False: show only how often, not the level
    extremes_note: str | None = None
    yearly_max_name: str | None = None   # e.g. "The hottest day of the year"

    @property
    def is_calculated(self) -> bool:
        return self.source == "calculated"


@dataclass(frozen=True)
class Catalog:
    """All variables plus the study period."""

    start: date
    end: date
    variables: dict[str, Variable]

    def power_codes(self) -> list[str]:
        """Codes the downloader requests from NASA POWER."""
        return [v.code for v in self.variables.values() if v.source == "power"]

    def extreme_codes(self) -> list[str]:
        """Codes whose yearly maximum gets extreme value analysis."""
        return [v.code for v in self.variables.values() if v.extremes]

    def __getitem__(self, code: str) -> Variable:
        return self.variables[code]


def load_catalog(path: str | Path = VARIABLES_FILE) -> Catalog:
    """Load and check the variable catalog. Raises ValueError on a bad file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    period = raw.get("period") or {}
    start, end = period.get("start"), period.get("end")
    if not isinstance(start, date) or not isinstance(end, date) or start >= end:
        raise ValueError("period needs a start date before the end date")

    entries = raw.get("variables") or {}
    if not entries:
        raise ValueError("no variables defined")

    variables: dict[str, Variable] = {}
    for code, fields in entries.items():
        missing = [f for f in REQUIRED_FIELDS if not fields.get(f)]
        if missing:
            raise ValueError(f"{code}: missing {', '.join(missing)}")
        if fields["source"] not in SOURCES:
            raise ValueError(f"{code}: source must be one of {SOURCES}")
        variables[code] = Variable(code=code, **{**fields, "inputs": tuple(fields.get("inputs", ()))})

    # Calculated variables must be built only from downloaded ones.
    power = {c for c, v in variables.items() if v.source == "power"}
    for v in variables.values():
        if v.is_calculated:
            if not v.inputs:
                raise ValueError(f"{v.code}: calculated variable needs inputs")
            unknown = set(v.inputs) - power
            if unknown:
                raise ValueError(f"{v.code}: unknown inputs {sorted(unknown)}")

    return Catalog(start=start, end=end, variables=variables)


# ---------------------------------------------------------------------------
# Regions (config/regions.yaml)
# ---------------------------------------------------------------------------

REGIONS_FILE = Path(__file__).with_name("regions.yaml")


@dataclass(frozen=True)
class Box:
    """A latitude/longitude box in degrees."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    @property
    def height(self) -> float:
        return self.lat_max - self.lat_min

    @property
    def width(self) -> float:
        return self.lon_max - self.lon_min


@dataclass(frozen=True)
class Region:
    """A study region: its full box and the tiles used for NASA POWER requests."""

    key: str
    name: str
    box: Box
    tiles: tuple[Box, ...]
    land_mask: bool = True
    state: str | None = None        # clip the grid to this US state (Natural Earth admin-1)


def load_regions(path: str | Path = REGIONS_FILE) -> dict[str, Region]:
    """Load regions and check every tile fits the NASA POWER size limit."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    max_deg = float(raw["max_tile_degrees"])

    regions: dict[str, Region] = {}
    for key, r in raw["regions"].items():
        box = Box(**r["box"])
        tiles = tuple(Box(**t) for t in r["tiles"])
        for t in tiles:
            if t.height > max_deg or t.width > max_deg:
                raise ValueError(f"{key}: tile {t} is larger than {max_deg} degrees")
            inside = (box.lat_min <= t.lat_min < t.lat_max <= box.lat_max
                      and box.lon_min <= t.lon_min < t.lon_max <= box.lon_max)
            if not inside:
                raise ValueError(f"{key}: tile {t} is outside the region box")
        # Tiles must cover the whole box (they do not overlap, so areas add up).
        if abs(sum(t.height * t.width for t in tiles) - box.height * box.width) > 1e-6:
            raise ValueError(f"{key}: tiles do not cover the region box exactly")
        regions[key] = Region(key=key, name=r["name"], box=box, tiles=tiles,
                              land_mask=bool(r.get("land_mask", True)),
                              state=r.get("state"))
    return regions
