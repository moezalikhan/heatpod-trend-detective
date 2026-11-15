"""Results schema checks, shared by the pipeline (after export) and the app tests.

Uses only app dependencies (PyYAML, pandas, pyarrow).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

SCHEMA_FILE = Path(__file__).with_name("results_schema.yaml")
RESULTS_DIR = Path(__file__).resolve().parents[1] / "data" / "results"

# pandas dtype "kind" letters for each schema type.
KINDS = {"datetime": "M", "float": "f", "int": "iu", "bool": "b"}


def load_schema(path: Path = SCHEMA_FILE) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _has(d: dict, dotted: str) -> bool:
    for part in dotted.split("."):
        if not isinstance(d, dict) or part not in d:
            return False
        d = d[part]
    return True


def _check_json(path: Path, spec: dict, errors: list[str]) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        errors.append(f"{path}: cannot read JSON ({exc})")
        return None
    for key in spec.get("required", []):
        if not _has(data, key):
            errors.append(f"{path}: missing '{key}'")
    for block, keys in (spec.get("optional_blocks") or {}).items():
        if block in data:
            for key in keys:
                if not _has(data[block], key):
                    errors.append(f"{path}: missing '{block}.{key}'")
    return data


def _check_parquet(path: Path, spec: dict, errors: list[str]) -> None:
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - report any read problem
        errors.append(f"{path}: cannot read Parquet ({exc})")
        return
    if df.empty:
        errors.append(f"{path}: no rows")
    expected = dict(spec.get("columns", {}))
    for col, typ in expected.items():
        if col not in df.columns:
            errors.append(f"{path}: missing column '{col}'")
        elif df[col].dtype.kind not in KINDS[typ]:
            errors.append(f"{path}: column '{col}' is {df[col].dtype}, expected {typ}")
    optional = spec.get("optional_columns") or {}
    for col, typ in optional.items():
        if col in df.columns and df[col].dtype.kind not in KINDS[typ]:
            errors.append(f"{path}: column '{col}' is {df[col].dtype}, expected {typ}")
    extra = set(df.columns) - set(expected) - set(optional)
    if extra:
        errors.append(f"{path}: unexpected columns {sorted(extra)}")


def _check_file(path: Path, spec: dict, errors: list[str]) -> dict | None:
    if not path.exists():
        errors.append(f"{path}: missing file")
        return None
    if spec["type"] == "json":
        return _check_json(path, spec, errors)
    if spec["type"] == "geojson":
        data = _check_json(path, {"required": ["type", "features"]}, errors)
        if data is not None and data.get("type") != "FeatureCollection":
            errors.append(f"{path}: not a FeatureCollection")
        return data
    _check_parquet(path, spec, errors)
    return None


def validate_results(root: Path = RESULTS_DIR, schema: dict | None = None) -> list[str]:
    """Check data/results against the schema. Returns a list of problems (empty = valid)."""
    schema = schema or load_schema()
    errors: list[str] = []
    index = _check_file(root / "index.json", schema["root_files"]["index.json"], errors)
    if index is None:
        return errors
    if index.get("schema_version") != schema["version"]:
        errors.append(f"index.json: schema_version {index.get('schema_version')} "
                      f"!= {schema['version']}")

    for region, variables in index["regions"].items():
        rdir = root / region
        for name, spec in schema["region_files"].items():
            _check_file(rdir / name, spec, errors)
        for code in variables:
            for name, spec in schema["variable_files"].items():
                _check_file(rdir / code / name, spec, errors)

    total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    if total > schema["max_total_mb"] * 1024 * 1024:
        errors.append(f"results are {total / 1e6:.1f} MB, limit {schema['max_total_mb']} MB")
    return errors
