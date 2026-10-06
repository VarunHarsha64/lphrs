import math

import numpy as np
import pandas as pd
import pytest

from lphrs import profile as P
from lphrs import temporal as T

CFG = P.temporal_cfg()
REF = pd.Timestamp("2026-10-01T00:00:00Z")


def years_ago(y):
    return REF - pd.Timedelta(days=y * T.DAYS_PER_YEAR)


# ---------------------------------------------------------------- acceptance: BP example

def test_bp_example_slope_and_finite_time_to_threshold():
    """120, 124, 130, 136 over 4 years -> slope ~5.4 mmHg/yr, finite time to 140."""
    fit = T.fit_trend([-3, -2, -1, 0], [120, 124, 130, 136])
    assert fit.slope == pytest.approx(5.4, abs=0.01)
    urgency, ttt, status = T.time_to_threshold_urgency(136, fit.slope, fit.r2, fit.n, 140, "up")
    assert status == "approaching"
    assert math.isfinite(ttt) and ttt == pytest.approx(4 / 5.4, rel=1e-6)
    expected = fit.r2 * min(1, 4 / 5) / (1 + ttt)
    assert urgency == pytest.approx(expected)
    assert 0 < urgency < 1


def test_bp_example_through_metric_profile():
    m = pd.DataFrame({"timestamp": [years_ago(3), years_ago(2), years_ago(1), REF],
                      "value": [120.0, 124.0, 130.0, 136.0]})
    p = P.metric_profile(m, "systolic_bp", REF, lam=0.002, user_goals={"lower_bp"})
    assert p["slope_per_year"] == pytest.approx(5.4, abs=0.02)
    assert p["trend"] == "rising" and p["direction"] == "worsening"
    assert math.isfinite(p["time_to_threshold_years"]) and p["urgency"] > 0
    assert p["current"] == 136 and p["goal_relevance"] == CFG["goal_relevance"]["match"]
    assert p["history_score"] == 0.0          # never past 140


def test_short_span_gives_no_trend():
    """Readings bunched within a month (hospital stay) must not produce a yearly trend."""
    days = [40, 30, 20, 10, 0]
    m = pd.DataFrame({"timestamp": [REF - pd.Timedelta(days=d) for d in days],
                      "value": [70.0, 75.0, 80.0, 85.0, 90.0]})
    p = P.metric_profile(m, "glucose", REF, lam=0.002, user_goals=set())
    assert p["slope_per_year"] > 100                     # raw fit is extreme ...
    assert p["urgency"] == 0.0 and p["urgency_status"] == "insufficient_data"
    assert p["trend"] == "insufficient"


# ---------------------------------------------------------------- urgency branches

def test_past_threshold_is_full_urgency():
    assert T.time_to_threshold_urgency(150, -3, 0.9, 5, 140, "up")[:2] == (1.0, 0.0)
    assert T.time_to_threshold_urgency(35, 2, 0.9, 5, 40, "down")[0] == 1.0  # HDL below 40


def test_moving_away_is_zero():
    u, ttt, status = T.time_to_threshold_urgency(120, -2.0, 0.9, 5, 140, "up")
    assert (u, status) == (0.0, "moving_away") and ttt == float("inf")
    assert T.time_to_threshold_urgency(55, 1.0, 0.9, 5, 40, "down")[0] == 0.0


def test_down_direction_approaching():
    u, ttt, status = T.time_to_threshold_urgency(50, -5.0, 1.0, 5, 40, "down")
    assert status == "approaching" and ttt == pytest.approx(2.0) and u == pytest.approx(1 / 3)


def test_insufficient_points_gives_no_urgency_unless_past():
    assert T.time_to_threshold_urgency(130, 5, 1.0, 2, 140, "up")[2] == "insufficient_data"
    assert T.time_to_threshold_urgency(145, 5, 1.0, 1, 140, "up")[0] == 1.0


def test_confidence_scales_with_points_and_fit():
    weak = T.time_to_threshold_urgency(130, 5, 0.5, 3, 140, "up")[0]
    strong = T.time_to_threshold_urgency(130, 5, 1.0, 5, 140, "up")[0]
    assert strong == pytest.approx(1 / 3) and weak == pytest.approx(0.5 * 0.6 / 3)


def test_no_threshold_metric():
    assert T.time_to_threshold_urgency(70, 1, 1, 5, None, "up")[2] == "no_threshold"


# ---------------------------------------------------------------- decay

def test_old_allergy_keeps_its_floor():
    """Acceptance: an allergy recorded 20 years ago keeps weight >= its floor."""
    ev = pd.DataFrame({
        "patient_id": ["p"] * 3, "event_type": ["allergy", "condition", "condition"],
        "timestamp": [years_ago(20), years_ago(12), years_ago(8)],
        "stop": [pd.NaT, pd.NaT, years_ago(7.9)],
        "description": ["Peanut (substance)", "Essential hypertension (disorder)", "Viral sinusitis (disorder)"],
        "context": ["food:allergy", "disorder", "disorder"],
    })
    w = P.event_weights(ev, REF, user_goals={"lower_bp"}).set_index("description")
    allergy_floor = T.event_floor("allergy", CFG)
    assert w.loc["Peanut (substance)", "event_class"] == "allergy"
    assert w.loc["Peanut (substance)", "w_recency"] >= allergy_floor > 0
    assert w.loc["Peanut (substance)", "w_plain"] < 0.001        # plain decay would forget it
    htn = w.loc["Essential hypertension (disorder)"]
    assert htn["event_class"] == "chronic_condition" and htn["w_recency"] >= T.event_floor("chronic_condition", CFG)
    assert htn["goal_relevance"] == CFG["goal_relevance"]["match"]
    sinus = w.loc["Viral sinusitis (disorder)"]
    assert sinus["event_class"] == "acute_condition" and sinus["w_recency"] < 0.01


def test_recency_weight_floor_and_monotonic():
    d = np.array([0, 100, 1000, 10000])
    w = T.recency_weight(d, 0.002)
    assert w[0] == 1.0 and np.all(np.diff(w) < 0)
    assert np.all(T.recency_weight(d, 0.002, 0.5) >= 0.5)
    assert T.recency_weight([-5], 0.002)[0] == 1.0          # future -> treated as now


def test_volatile_metrics_forget_faster_and_rate_is_clipped():
    lam = T.decay_rate(np.array([0.1, 1.0, 2.0, 10.0]), CFG)
    base = CFG["lambda_base"]
    assert lam.tolist() == pytest.approx([0.25 * base, base, 2 * base, 4 * base])


def test_cohort_volatility_is_unit_free():
    rng = np.random.default_rng(0)
    rows = []
    for pid in range(30):
        for i in range(6):
            rows.append(("p%d" % pid, REF - pd.Timedelta(days=100 * i), "a", 100 + rng.normal(0, 10)))
            rows.append(("p%d" % pid, REF - pd.Timedelta(days=100 * i), "b", 100 + rng.normal(0, 2)))
    meas = pd.DataFrame(rows, columns=["patient_id", "timestamp", "metric", "value"])
    v1 = T.cohort_volatility(meas, CFG)
    scaled = meas.copy()
    scaled.loc[scaled["metric"] == "a", "value"] *= 1000      # same metric in different units
    v2 = T.cohort_volatility(scaled, CFG)
    assert v1.loc["a", "rel_change_sd"] > v1.loc["b", "rel_change_sd"]
    assert v1.loc["a", "lambda"] > v1.loc["b", "lambda"]
    assert v2.loc["a", "rel_change_sd"] == pytest.approx(v1.loc["a", "rel_change_sd"], rel=1e-9)


# ---------------------------------------------------------------- classification and labels

@pytest.mark.parametrize("desc,resolved,expected", [
    ("Essential hypertension (disorder)", False, "chronic_condition"),
    ("Prediabetes (finding)", False, "chronic_condition"),
    ("Full-time employment (finding)", False, "social_finding"),
    ("Stress (finding)", False, "social_finding"),
    ("Medication review due (situation)", False, "social_finding"),
    ("History of myocardial infarction (situation)", False, "resolved_major"),
    ("Acute non-ST segment elevation myocardial infarction (disorder)", True, "resolved_major"),
    ("Fracture of ankle (disorder)", True, "resolved_major"),
    ("Viral sinusitis (disorder)", True, "acute_condition"),
])
def test_condition_class(desc, resolved, expected):
    assert T.condition_class(desc, resolved, CFG) == expected


def test_trend_labels():
    assert T.trend_label(5.4, 0.99, 4, 136) == "rising"
    assert T.trend_label(-5.4, 0.99, 4, 136) == "falling"
    assert T.trend_label(0.5, 0.99, 4, 136) == "stable"       # < 2 %/yr
    assert T.trend_label(5.4, 0.01, 4, 136) == "stable"       # no fit
    assert T.trend_label(5.4, 0.99, 2, 136) == "insufficient"
    assert T.direction_vs_risk("falling", "down") == "worsening"
    assert T.direction_vs_risk("rising", "down") == "improving"


def test_weekly_resample_never_passes_last_day():
    days = pd.date_range("2026-09-01", "2026-09-30", freq="D", tz="UTC")
    daily = pd.DataFrame({"patient_id": "p", "metric": "steps", "timestamp": days, "value": 1000.0})
    w = T.resample_lifestyle(daily)
    assert w["timestamp"].max() == days.max()
    assert len(w) == 5 and (w["value"] == 1000.0).all()
