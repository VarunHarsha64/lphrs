import pandas as pd
import pytest

from lphrs import profile as P
from lphrs import temporal as T
from lphrs.timeline import build_timeline

REF = pd.Timestamp("2026-10-01T00:00:00Z")


def ago(years):
    return REF - pd.Timedelta(days=years * T.DAYS_PER_YEAR)


@pytest.fixture
def tl():
    meas = pd.DataFrame({
        "patient_id": "p", "metric": "systolic_bp", "unit": "mmHg", "code": "8480-6", "category": "vital-signs",
        "timestamp": [ago(3), ago(2), ago(1), ago(0.01), REF + pd.Timedelta(days=30)],
        "value": [120.0, 124.0, 130.0, 136.0, 180.0],              # last one is after REF -> ignored
    })
    cond = pd.DataFrame({
        "patient_id": "p", "timestamp": [ago(6), ago(3)], "stop": [pd.NaT, pd.NaT], "code": ["1", "2"],
        "description": ["Essential hypertension (disorder)", "Full-time employment (finding)"],
    })
    meds = pd.DataFrame({
        "patient_id": "p", "timestamp": [ago(2), ago(1)], "stop": [pd.NaT, pd.NaT], "code": ["m", "m"],
        "description": ["lisinopril 10 MG Oral Tablet"] * 2, "reason": ["Essential hypertension (disorder)"] * 2,
    })
    alg = pd.DataFrame({
        "patient_id": "p", "timestamp": [ago(15)], "stop": [pd.NaT], "code": ["a"],
        "description": ["Peanut (substance)"], "allergy_type": ["allergy"], "allergy_category": ["food"],
    })
    days = pd.date_range(ago(1.5).normalize(), REF.normalize(), freq="D", tz="UTC")
    daily = pd.DataFrame({"patient_id": "p", "metric": "steps", "unit": "steps/day", "timestamp": days,
                          "value": 9000.0 - 1500.0 * (days - days[0]).days / 365.25})   # ends ~6750, above 5000
    return build_timeline(meas, cond, meds, alg, daily)


def test_build_health_profile(tl):
    vol = pd.DataFrame({"lambda": {"systolic_bp": 0.002, "steps": 0.004}})
    prof = P.build_health_profile("p", tl, REF, vol, user_goals=["lower_bp"])
    bp = prof["metrics"]["systolic_bp"]
    assert bp["current"] == 136.0                                  # future reading excluded
    assert bp["slope_per_year"] == pytest.approx(5.4, abs=0.1)
    assert bp["urgency"] > 0 and bp["trend"] == "rising"
    steps = prof["metrics"]["steps"]
    assert steps["trend"] == "falling" and steps["direction"] == "worsening"
    assert steps["slope_per_year"] == pytest.approx(-1500, rel=0.05)
    assert steps["urgency_status"] == "approaching"
    assert [c["description"] for c in prof["active_conditions"]] == ["Essential hypertension (disorder)"]
    assert len(prof["active_medications"]) == 1                    # repeated prescription collapsed
    assert prof["allergies"][0]["w_recency"] >= 1.0 - 1e-12
    assert prof["cvd_risk"] is None


def test_profile_as_of_earlier_date_hides_future(tl):
    vol = pd.DataFrame({"lambda": {"systolic_bp": 0.002}})
    early = ago(2.5)                                               # before the medication started
    prof = P.build_health_profile("p", tl, early, vol)
    assert prof["metrics"]["systolic_bp"]["current"] == 120.0
    assert prof["active_medications"] == []


def test_build_all_profiles_matches_single(tl):
    vol = pd.DataFrame({"lambda": {"systolic_bp": 0.002, "steps": 0.004}})
    metrics, events = P.build_all_profiles(tl, REF, vol, {"p": ["lower_bp"]})
    single = P.build_health_profile("p", tl, REF, vol, user_goals=["lower_bp"])
    row = metrics.set_index("metric").loc["systolic_bp"]
    assert row["urgency"] == pytest.approx(single["metrics"]["systolic_bp"]["urgency"])
    assert set(events["event_class"]) == {"chronic_condition", "social_finding", "medication", "allergy"}


def test_build_user_context():
    ctx = pd.DataFrame([{"patient_id": "p", "goals": ["lower_bp"], "diet_style": "vegetarian",
                         "liked_exercise": ["walking", "yoga"], "minutes_per_day": 30, "injury": "knee",
                         "stress": 5, "smoke": 0, "alco": 0, "active": 1}])
    pats = pd.DataFrame([{"patient_id": "p", "age": 72.0, "gender": "F"}])
    uc = P.build_user_context("p", ctx, pats)
    assert uc["constraints"] == ["injury:knee"] and uc["minutes_per_day"] == 30
    assert {"walking", "yoga", "vegetarian"} <= set(uc["preference_tags"])
    assert set(uc["intensity_caps"].values()) == {4}               # 4 + active - age>=70
    assert len(uc["intensity_caps"]) == 6
