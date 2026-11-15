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
