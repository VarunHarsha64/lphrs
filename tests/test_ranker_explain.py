import math

import pandas as pd
import pytest

from lphrs import explain as X
from lphrs import ranker as RK


def cand(id, domain="fitness", tags=(), minutes=10, intensity=2, effort=0.2, targets=(), goals=()):
    return {"id": id, "domain": domain, "title": f"item {id}", "tags": list(tags), "minutes": minutes,
            "intensity": intensity, "effort": effort, "targets": list(targets), "goals": list(goals),
            "contraindications": [], "source": "template"}


def metric(current, threshold, bad="up", urgency=0.0, status="moving_away", slope=0.0, ttt=float("nan"),
           history=0.0, unit="mmHg"):
    return {"current": current, "threshold": threshold, "bad_direction": bad, "urgency": urgency,
            "urgency_status": status, "slope_per_year": slope, "time_to_threshold_years": ttt,
            "history_score": history, "unit": unit}


@pytest.fixture
def profile():
    return {
        "metrics": {
            "systolic_bp": metric(136, 140, urgency=0.4, status="approaching", slope=5.4, ttt=0.74),
            "bmi": metric(31.0, 30, urgency=1.0, status="past_threshold", unit="kg/m2", history=0.8),
            "sleep_hours": metric(7.2, 6, bad="down", unit="h"),
        },
        "active_conditions": [{"description": "Sleep apnea (disorder)", "w_recency": 0.8}],
        "cvd_risk": 0.6,
    }


@pytest.fixture
def uc():
    return {"goals": ["lower_bp", "better_sleep"], "preference_tags": ["walking", "yoga"],
            "minutes_per_day": 40, "intensity_caps": {"fitness": 3, "sleep": 3, "nutrition": 3}}


def test_terms_are_bounded_and_defined(profile, uc):
    cands = pd.DataFrame([
        cand("bp_walk", tags=["walking"], targets=["systolic_bp"], goals=["lower_bp"]),
        cand("weight", domain="nutrition", targets=["bmi"], goals=["weight_loss"]),
        cand("sleep", domain="sleep", targets=["sleep_hours"], goals=["better_sleep"], minutes=5),
        cand("hard", intensity=5, minutes=35, tags=["hiit"]),
        cand("toolong", minutes=60),
    ])
    t = RK.compute_terms(cands, profile, uc).set_index("id")
    assert t[RK.TERMS].apply(lambda s: s.between(0, 1).all()).all()
    # R: CVD risk lifts a BP item; past-threshold BMI gives 1; sleep apnea makes a sleep item relevant
    assert t.loc["bp_walk", "R"] == pytest.approx(0.6) and t.loc["bp_walk", "R_driver"] == "cvd_risk"
    assert t.loc["weight", "R"] == 1.0 and t.loc["weight", "U"] == 1.0
    assert t.loc["sleep", "R"] == pytest.approx(0.8) and t.loc["sleep", "R_driver"].startswith("condition:")
    assert t.loc["bp_walk", "U"] == pytest.approx(0.4) and t.loc["bp_walk", "U_driver"] == "systolic_bp"
    assert t.loc["bp_walk", "G"] == 0.5 and t.loc["weight", "G"] == 0.0
    assert t.loc["bp_walk", "P"] == 0.5
    assert t.loc["hard", "F"] == pytest.approx(0.5 * 0.0)     # 35 > 20 min half-budget -> 0.25; intensity 5 vs cap 3 -> 0
    assert t.loc["toolong", "F"] == 0.0
    assert (t["H"] == 0.5).all()


def test_feasibility_pieces():
    assert RK.time_fit(10, 40) == 1.0 and RK.time_fit(30, 40) == pytest.approx(0.5) and RK.time_fit(41, 40) == 0
    assert RK.intensity_fit(3, 3) == 1.0 and RK.intensity_fit(4, 3) == 0.5 and RK.intensity_fit(5, 3) == 0.0


def test_adherence_smoothing():
    assert RK.adherence(None, "x", "fitness") == 0.5
    h = {"by_id": {"x": (4, 4)}, "by_domain": {"fitness": (0, 10)}}
    assert RK.adherence(h, "x", "fitness") == pytest.approx((4 + 0 + 1) / (4 + 5 + 2))
    assert RK.adherence({"by_id": {"x": (10, 10)}}, "x", "sleep") > 0.9


def test_score_uses_weights_and_subtracts_cost():
    t = pd.DataFrame([{**cand("a"), **dict.fromkeys(RK.TERMS, 1.0)}])
    s = RK.score(t)
    w = RK.default_weights()
    assert s.loc[0, "score"] == pytest.approx(sum(w.values()) - 2 * w["C"])
    assert s.loc[0, "contrib_C"] == pytest.approx(-w["C"])


def scored_pool():
    rows = [cand("w1", tags=["walking", "evening_walk"], minutes=10), cand("w2", tags=["walking"], minutes=10),
            cand("bed", domain="sleep", tags=["earlier_bedtime"], minutes=5),
            cand("tea", domain="lifestyle", tags=["late_caffeine"], minutes=5),
            cand("f3", tags=["cycling"], minutes=30), cand("n1", domain="nutrition", minutes=2),
            cand("n2", domain="nutrition", minutes=2), cand("n3", domain="nutrition", minutes=2)]
    df = pd.DataFrame(rows)
    df["score"] = [0.9, 0.85, 0.5, 0.52, 0.8, 0.6, 0.59, 0.58]
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def test_greedy_respects_k_budget_and_domain_cap():
    sel = RK.select_set(scored_pool(), budget=40, k=5, alpha=0.3, max_per_domain=2)
    assert len(sel) == 5
    assert sel["minutes"].sum() <= 40
    assert sel["domain"].value_counts().max() <= 2


def test_synergy_changes_the_pick():
    table = RK.load_synergy()
    assert RK.synergy(["evening_walk"], ["earlier_bedtime"], table) == pytest.approx(0.5)
    assert RK.synergy(["late_caffeine"], ["earlier_bedtime"], table) == pytest.approx(-0.6)
    assert RK.synergy(["walking"], ["walking"], table) == pytest.approx(-0.4)
    pool = scored_pool()
    sel = RK.select_set(pool, budget=100, k=3, alpha=0.3, max_per_domain=2)
    # w1 first; w2 is penalised as a redundant walk; bedtime (+0.5 with evening walk) beats late tea (-0.6)
    assert sel["id"].tolist()[0] == "w1"
    assert "w2" not in sel["id"].tolist()
    top = RK.select_top_k(pool, 3)
    assert top["id"].tolist() == ["w1", "w2", "f3"]


def test_lookahead_fills_list_on_small_budget():
    # plain greedy would take w1 + w2 (20 min) and then could not fit 3 more in 26
    sel = RK.select_set(scored_pool(), budget=26, k=5, alpha=0.3, max_per_domain=2)
    assert len(sel) == 5 and sel["minutes"].sum() <= 26
    assert sel["domain"].value_counts().max() <= 2


def test_infeasible_budget_degrades_gracefully():
    # with at most 2 nutrition items, no 5-item list fits in 16 minutes; return the best shorter list
    sel = RK.select_set(scored_pool(), budget=16, k=5, alpha=0.3, max_per_domain=2)
    assert len(sel) == 4 and sel["minutes"].sum() <= 16


def test_stops_when_gain_not_positive():
    pool = scored_pool()
    pool["score"] = -0.1
    assert RK.select_set(pool, budget=100).empty


def test_list_conflicts():
    sel = pd.DataFrame({"tags": [["late_caffeine"], ["earlier_bedtime"], ["evening_walk"]]})
    assert RK.list_conflicts(sel) == 1


# ---------------------------------------------------------------- explanations

def test_urgency_sentence_matches_plan_example(profile):
    s = X.reason_urgency(profile["metrics"]["systolic_bp"], "systolic_bp")
    assert s == ("Your systolic blood pressure has risen about 5.4 mmHg per year and is projected "
                 "to reach 140 in about 9 months.")
    m = dict(profile["metrics"]["systolic_bp"], time_to_threshold_years=2.04)
    assert "in about 2 years" in X.reason_urgency(m, "systolic_bp")


def test_explain_picks_top_two_positive_contributors(profile, uc):
    cands = pd.DataFrame([cand("bp", tags=["walking"], targets=["systolic_bp"], goals=["lower_bp"])])
    row = RK.score(RK.compute_terms(cands, profile, uc)).iloc[0].to_dict()
    reasons = X.explain(row, profile, uc)
    assert len(reasons) == 2
    assert reasons[0].startswith("Your estimated cardiovascular risk is 60%")
    assert any("risen about 5.4" in r or "lower blood pressure" in r for r in reasons)
    assert all(not r.startswith("Similar suggestions") for r in reasons)   # no history yet


def test_condition_reason_names_only_related_goal(profile, uc):
    cands = pd.DataFrame([cand("s", domain="sleep", targets=["sleep_hours"], goals=["better_sleep", "lower_bp"])])
    row = RK.score(RK.compute_terms(cands, profile, uc)).iloc[0].to_dict()
    r = X.reason_relevance(row, profile)
    assert r == "Your record lists sleep apnea; this supports better sleep."


def test_no_trend_sentence_for_negligible_urgency(profile):
    row = {"U_driver": "sleep_hours", "R_driver": None, "goals": [], "tags": [], "minutes": 1, "H": 0.5}
    m = dict(profile["metrics"]["sleep_hours"], urgency=0.0001, urgency_status="approaching",
             slope_per_year=-0.001, time_to_threshold_years=1473.0)
    prof = dict(profile, metrics={**profile["metrics"], "sleep_hours": m})
    assert X.reason("U", row, prof, {"goals": [], "preference_tags": [], "minutes_per_day": 15}) is None


def test_formatting_helpers():
    assert X.fmt_years(0.74) == "9 months" and X.fmt_years(1.0) == "1 year" and X.fmt_years(5.6) == "6 years"
    assert X.fmt_threshold("bmi", 30.0) == "30" and X.fmt_threshold("steps", 5000) == "5,000"
    assert X.label("bmi") == "BMI" and X.label("systolic_bp") == "systolic blood pressure"
    assert not math.isnan(len(X.DISCLAIMER))
