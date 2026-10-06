import pandas as pd

from lphrs import timeline

UTC = "UTC"


def ts(s):
    return pd.Timestamp(s, tz=UTC)


def make_inputs():
    meas = pd.DataFrame({
        "patient_id": ["b", "a", "a"],
        "timestamp": [ts("2020-01-01"), ts("2022-01-01"), ts("2019-01-01")],
        "metric": ["bmi", "systolic_bp", "systolic_bp"],
        "value": [31.0, 135.0, 120.0], "unit": ["kg/m2", "mmHg", "mmHg"],
        "code": ["39156-5", "8480-6", "8480-6"], "category": ["vital-signs"] * 3,
    })
    cond = pd.DataFrame({
        "patient_id": ["a", "b"], "timestamp": [ts("2020-06-01"), ts("2010-01-01")],
        "stop": [pd.NaT, ts("2010-02-01")], "code": ["59621000", "444814009"],
        "description": ["Essential hypertension (disorder)", "Viral sinusitis (disorder)"],
    })
    meds = pd.DataFrame({
        "patient_id": ["a"], "timestamp": [ts("2020-07-01")], "stop": [pd.NaT],
        "code": ["314076"], "description": ["lisinopril 10 MG Oral Tablet"], "reason": ["Essential hypertension"],
    })
    alg = pd.DataFrame({
        "patient_id": ["b"], "timestamp": [ts("2005-01-01")], "stop": [pd.NaT], "code": ["91935009"],
        "description": ["Peanut (substance)"], "allergy_type": ["allergy"], "allergy_category": ["food"],
    })
    daily = pd.DataFrame({
        "patient_id": ["a", "a"], "timestamp": [ts("2021-03-02"), ts("2021-03-01")],
        "metric": ["steps", "steps"], "value": [6000.0, 5000.0], "unit": ["steps/day"] * 2,
    })
    return meas, cond, meds, alg, daily


def test_timeline_columns_and_types():
    tl = timeline.build_timeline(*make_inputs())
    assert list(tl.columns) == timeline.TIMELINE_COLUMNS
    assert set(tl["event_type"]) == {"observation", "condition", "medication", "allergy", "lifestyle"}
    assert str(tl["timestamp"].dt.tz) == UTC
    assert len(tl) == 9


def test_timeline_sorted_per_patient():
    tl = timeline.build_timeline(*make_inputs())
    assert timeline.is_sorted(tl)
    a = tl[tl["patient_id"] == "a"]["timestamp"].tolist()
    assert a == sorted(a)
    assert tl["patient_id"].tolist() == sorted(tl["patient_id"].tolist())


def test_is_sorted_detects_disorder():
    tl = timeline.build_timeline(*make_inputs())
    shuffled = tl.iloc[::-1].reset_index(drop=True)
    assert not timeline.is_sorted(shuffled)


def test_clinical_events_keep_stop_and_context():
    tl = timeline.build_timeline(*make_inputs())
    htn = tl[tl["description"] == "Essential hypertension (disorder)"].iloc[0]
    assert pd.isna(htn["stop"]) and htn["context"] == "disorder"
    sinus = tl[tl["description"] == "Viral sinusitis (disorder)"].iloc[0]
    assert sinus["stop"] == ts("2010-02-01")
    peanut = tl[tl["event_type"] == "allergy"].iloc[0]
    assert peanut["context"] == "food:allergy"


def test_sources_and_confidence():
    tl = timeline.build_timeline(*make_inputs())
    life = tl[tl["event_type"] == "lifestyle"]
    assert (life["source"] == "synthetic").all() and (life["confidence"] < 1).all()
    assert (tl[tl["event_type"] != "lifestyle"]["source"] == "synthea").all()
