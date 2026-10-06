import pandas as pd
import pytest

from lphrs import normalize
from lphrs.config import loinc_to_metric, metric_specs

SPECS = metric_specs()
CODES = loinc_to_metric()


def raw(rows):
    return pd.DataFrame(rows, columns=["patient_id", "timestamp", "code", "value", "unit", "category"])


def test_unit_conversion_to_canonical():
    df = pd.DataFrame({
        "metric": ["glucose", "body_weight", "ldl", "systolic_bp", "sleep_hours"],
        "value": [5.5, 220.0, 4.0, 120.0, 420.0],
        "unit": ["mmol/L", "[lb_av]", "mmol/L", "mm[Hg]", "min"],
    })
    out, dropped = normalize.convert_units(df, SPECS)
    assert dropped == 0
    assert out["value"].tolist() == pytest.approx([99.088, 99.79032, 154.68, 120.0, 7.0], rel=1e-4)
    assert out["unit"].tolist() == ["mg/dL", "kg", "mg/dL", "mmHg", "h"]


def test_unknown_unit_is_dropped():
    df = pd.DataFrame({"metric": ["glucose", "glucose"], "value": [100.0, 5.0], "unit": ["mg/dL", "g/L"]})
    out, dropped = normalize.convert_units(df, SPECS)
    assert dropped == 1 and out["value"].tolist() == [100.0]


def test_range_validation_drops_impossible_values():
    df = pd.DataFrame({
        "metric": ["systolic_bp", "systolic_bp", "hba1c", "ldl", "bmi"],
        "value": [300.0, 120.0, 25.0, -5.0, float("nan")],
    })
    out, dropped = normalize.validate_ranges(df, SPECS)
    assert dropped == 4
    assert out["value"].tolist() == [120.0]


def test_range_bounds_are_inclusive():
    lo, hi = SPECS["heart_rate"]["valid_range"]
    df = pd.DataFrame({"metric": ["heart_rate"] * 4, "value": [lo, hi, lo - 0.1, hi + 0.1]})
    out, dropped = normalize.validate_ranges(df, SPECS)
    assert out["value"].tolist() == [lo, hi] and dropped == 2


def test_timestamps_become_utc():
    ts = normalize.to_utc(pd.Series(["2020-01-02", "2020-01-02T03:04:05Z", None]))
    assert str(ts.dt.tz) == "UTC"
    assert ts.iloc[0] == pd.Timestamp("2020-01-02T00:00:00Z")
    assert ts.iloc[1] == pd.Timestamp("2020-01-02T03:04:05Z")
    assert pd.isna(ts.iloc[2])


def test_full_normalisation_pipeline():
    t1, t2 = "2021-05-01T10:00:00Z", "2020-01-01T10:00:00Z"
    obs = raw([
        ["p1", t1, "8480-6", "130", "mm[Hg]", "vital-signs"],
        ["p1", t1, "8462-4", "85", "mm[Hg]", "vital-signs"],
        ["p1", t1, "8480-6", "130", "mm[Hg]", "vital-signs"],     # exact duplicate
        ["p1", t1, "2339-0", "100", "mg/dL", "laboratory"],
        ["p1", t1, "2345-7", "104", "mg/dL", "laboratory"],       # alias code, same instant -> averaged
        ["p1", t2, "8480-6", "80", "mm[Hg]", "vital-signs"],
        ["p1", t2, "8462-4", "95", "mm[Hg]", "vital-signs"],      # diastolic >= systolic -> both dropped
        ["p2", t2, "4548-4", "abc", "%", "laboratory"],           # non-numeric -> dropped
        ["p2", t2, "9999-9", "1", "x", "laboratory"],             # unknown code -> ignored
    ])
    out, rep = normalize.normalize_health_data(obs, SPECS, CODES)
    assert rep["kept_codes"] == 8
    assert rep["dropped_out_of_range"] == 1
    assert rep["dropped_bp_pair"] == 2
    assert rep["merged_duplicates"] == 2
    got = {(r.metric, r.value) for r in out.itertuples()}
    assert got == {("systolic_bp", 130.0), ("diastolic_bp", 85.0), ("glucose", 102.0)}
    # sorted by patient, timestamp, metric: diastolic_bp, glucose, systolic_bp
    assert out["metric"].tolist() == ["diastolic_bp", "glucose", "systolic_bp"]
    assert out["unit"].tolist() == ["mmHg", "mg/dL", "mmHg"]
    assert list(out.columns) == normalize.MEASUREMENT_COLUMNS


def test_bp_stored_as_two_metrics():
    obs = raw([["p1", "2021-01-01T00:00:00Z", "8480-6", "120", "mm[Hg]", "vital-signs"],
               ["p1", "2021-01-01T00:00:00Z", "8462-4", "80", "mm[Hg]", "vital-signs"]])
    out, _ = normalize.normalize_health_data(obs, SPECS, CODES)
    assert sorted(out["metric"]) == ["diastolic_bp", "systolic_bp"]
