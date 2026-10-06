"""NormalizeHealthData: unit conversion, UTC timestamps, range validation, de-duplication.

No Z-scoring here; scaling happens inside models only (build plan 4.1).
Every function is pure: it takes a DataFrame and returns a new one.
"""

from __future__ import annotations

import pandas as pd

MEASUREMENT_COLUMNS = ["patient_id", "timestamp", "metric", "value", "unit", "code", "category"]


def to_utc(values: pd.Series) -> pd.Series:
    """Parse date or datetime strings into timezone-aware UTC timestamps (NaT stays NaT)."""
    return pd.to_datetime(values, utc=True, format="ISO8601", errors="coerce")


def map_codes(obs: pd.DataFrame, code_map: dict[str, str]) -> pd.DataFrame:
    """Keep observations whose LOINC code is known and attach the metric key."""
    out = obs[obs["code"].isin(code_map)].copy()
    out["metric"] = out["code"].map(code_map)
    return out


def convert_units(df: pd.DataFrame, specs: dict) -> tuple[pd.DataFrame, int]:
    """Convert each value into its metric's canonical unit.

    A row whose unit is neither canonical nor listed in alt_units is dropped.
    Returns the converted frame and the number of rows dropped.
    """
    factors = []
    for metric, unit in zip(df["metric"], df["unit"]):
        spec = specs[metric]
        if unit == spec["unit"]:
            factors.append(1.0)
        else:
            factors.append(spec.get("alt_units", {}).get(unit))
    factor = pd.Series(factors, index=df.index, dtype="float64")
    out = df[factor.notna()].copy()
    out["value"] = out["value"] * factor[factor.notna()]
    out["unit"] = out["metric"].map({m: s["unit"] for m, s in specs.items()})
    return out, int(factor.isna().sum())


def validate_ranges(df: pd.DataFrame, specs: dict) -> tuple[pd.DataFrame, int]:
    """Drop non-numeric values and values outside each metric's valid_range."""
    lo = df["metric"].map({m: s["valid_range"][0] for m, s in specs.items()})
    hi = df["metric"].map({m: s["valid_range"][1] for m, s in specs.items()})
    keep = df["value"].notna() & (df["value"] >= lo) & (df["value"] <= hi)
    return df[keep].copy(), int((~keep).sum())


def validate_bp_pairs(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop systolic/diastolic readings taken together where diastolic >= systolic."""
    bp = df[df["metric"].isin(["systolic_bp", "diastolic_bp"])]
    wide = bp.pivot_table(index=["patient_id", "timestamp"], columns="metric",
                          values="value", aggfunc="mean")
    if not {"systolic_bp", "diastolic_bp"} <= set(wide.columns):
        return df, 0
    bad = wide[wide["diastolic_bp"] >= wide["systolic_bp"]].index
    if len(bad) == 0:
        return df, 0
    key = pd.MultiIndex.from_frame(df[["patient_id", "timestamp"]])
    drop = key.isin(bad) & df["metric"].isin(["systolic_bp", "diastolic_bp"]).to_numpy()
    return df[~drop].copy(), int(drop.sum())


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """One value per (patient, timestamp, metric).

    Exact repeats collapse to one row; conflicting values at the same instant
    (e.g. blood and serum glucose from the same panel) are averaged.
    """
    keys = ["patient_id", "timestamp", "metric"]
    before = len(df)
    out = (df.groupby(keys, as_index=False, sort=False)
             .agg(value=("value", "mean"), unit=("unit", "first"),
                  code=("code", "first"), category=("category", "first")))
    return out, before - len(out)


def normalize_health_data(obs: pd.DataFrame, specs: dict, code_map: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    """Full observation normalisation. `obs` has raw columns
    patient_id, timestamp (string), code, value (string), unit, category.

    Returns the clean long frame (MEASUREMENT_COLUMNS) and a report of rows dropped per step.
    """
    report = {"input_rows": len(obs)}
    df = map_codes(obs, code_map)
    report["kept_codes"] = len(df)
    df["timestamp"] = to_utc(df["timestamp"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    bad_ts = df["timestamp"].isna()
    df = df[~bad_ts]
    report["dropped_bad_timestamp"] = int(bad_ts.sum())
    df, report["dropped_unknown_unit"] = convert_units(df, specs)
    df, report["dropped_out_of_range"] = validate_ranges(df, specs)
    df, report["dropped_bp_pair"] = validate_bp_pairs(df)
    df, report["merged_duplicates"] = deduplicate(df)
    df = df.sort_values(["patient_id", "timestamp", "metric"], kind="stable").reset_index(drop=True)
    report["output_rows"] = len(df)
    return df[MEASUREMENT_COLUMNS], report


def normalize_events(df: pd.DataFrame) -> pd.DataFrame:
    """Condition / medication / allergy rows: UTC start and stop, exact duplicates removed."""
    out = df.copy()
    out["timestamp"] = to_utc(out["timestamp"])
    out["stop"] = to_utc(out["stop"])
    out = out[out["timestamp"].notna()]
    return out.drop_duplicates().reset_index(drop=True)
