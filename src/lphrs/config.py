"""Project paths and cached YAML config loading."""

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


@lru_cache(maxsize=None)
def load_yaml(name: str) -> dict:
    """Load config/<name>.yaml (name without extension)."""
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def metric_specs() -> dict:
    return load_yaml("metrics")["metrics"]


def loinc_to_metric() -> dict[str, str]:
    """Map every LOINC code (primary and aliases) to its metric key."""
    out = {}
    for key, spec in metric_specs().items():
        for code in [spec.get("loinc")] + list(spec.get("loinc_aliases", [])):
            if code:
                out[str(code)] = key
    return out
