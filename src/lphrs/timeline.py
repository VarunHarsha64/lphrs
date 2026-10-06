"""BuildTimeline: one row per event, sorted per patient (build plan 4.2).

event_type is one of: observation, lifestyle, condition, medication, allergy.
Measurements (observation, lifestyle) carry `metric`, `value`, `unit`.
Clinical events carry `code`, `description` and a `stop` (NaT = ongoing).
"""

from __future__ import annotations

import pandas as pd

TIMELINE_COLUMNS = ["patient_id", "timestamp", "stop", "event_type", "metric", "value", "unit",
                    "code", "description", "source", "context", "confidence"]
SORT_KEYS = ["patient_id", "timestamp", "event_type", "metric", "code"]

CONFIDENCE = {"synthea": 1.0, "synthetic": 0.8}

_SEMANTIC_TAG = r"\(([^()]*)\)\s*$"  # "Prediabetes (finding)" -> "finding"


def _frame(df: pd.DataFrame, **cols) -> pd.DataFrame:
    """Project `df` onto TIMELINE_COLUMNS; keyword args give constants or Series."""
    out = pd.DataFrame(index=df.index)
    for c in TIMELINE_COLUMNS:
        if c in cols:
            out[c] = cols[c]
        elif c in df.columns:
            out[c] = df[c]
        else:
            out[c] = None
    return out


def observation_events(meas: pd.DataFrame) -> pd.DataFrame:
    return _frame(meas, stop=pd.NaT, event_type="observation", source="synthea",
                  context=meas["category"], confidence=CONFIDENCE["synthea"])


def lifestyle_events(daily: pd.DataFrame) -> pd.DataFrame:
    """`daily` has patient_id, timestamp, metric, value, unit (synthetic wearable data)."""
    return _frame(daily, stop=pd.NaT, event_type="lifestyle", code=None, source="synthetic",
                  context="wearable", confidence=CONFIDENCE["synthetic"])


def condition_events(cond: pd.DataFrame) -> pd.DataFrame:
    tag = cond["description"].str.extract(_SEMANTIC_TAG, expand=False).fillna("unspecified")
    return _frame(cond, event_type="condition", metric=None, value=float("nan"), unit=None,
                  source="synthea", context=tag, confidence=CONFIDENCE["synthea"])


def medication_events(med: pd.DataFrame) -> pd.DataFrame:
    return _frame(med, event_type="medication", metric=None, value=float("nan"), unit=None,
                  source="synthea", context=med.get("reason"), confidence=CONFIDENCE["synthea"])


def allergy_events(alg: pd.DataFrame) -> pd.DataFrame:
    ctx = alg["allergy_category"].fillna("unknown") + ":" + alg["allergy_type"].fillna("unknown")
    return _frame(alg, event_type="allergy", metric=None, value=float("nan"), unit=None,
                  source="synthea", context=ctx, confidence=CONFIDENCE["synthea"])


def build_timeline(measurements: pd.DataFrame, conditions: pd.DataFrame, medications: pd.DataFrame,
                   allergies: pd.DataFrame, lifestyle: pd.DataFrame | None = None) -> pd.DataFrame:
    """Stack all event sources into one frame sorted by patient, then time."""
    parts = [observation_events(measurements), condition_events(conditions),
             medication_events(medications), allergy_events(allergies)]
    if lifestyle is not None and len(lifestyle):
        parts.append(lifestyle_events(lifestyle))
    tl = pd.concat([p for p in parts if len(p)], ignore_index=True)
    for c in ("timestamp", "stop"):  # an all-NaT column arrives tz-naive; force UTC everywhere
        tl[c] = pd.to_datetime(tl[c], utc=True)
    tl["value"] = tl["value"].astype("float64")
    tl["confidence"] = tl["confidence"].astype("float64")
    for c in ("patient_id", "event_type", "metric", "unit", "code", "description", "source", "context"):
        tl[c] = tl[c].astype("string")
    tl = tl.sort_values(SORT_KEYS, kind="stable", na_position="first").reset_index(drop=True)
    return tl[TIMELINE_COLUMNS]


def is_sorted(tl: pd.DataFrame) -> bool:
    """True when rows are ordered by patient and, within each patient, by time."""
    same_patient = tl["patient_id"].eq(tl["patient_id"].shift())
    patient_order = (tl["patient_id"].shift().fillna("") <= tl["patient_id"]).all()
    time_order = (~same_patient | (tl["timestamp"] >= tl["timestamp"].shift())).all()
    return bool(patient_order and time_order)
