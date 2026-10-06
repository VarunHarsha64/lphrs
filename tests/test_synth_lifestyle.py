import numpy as np
import pandas as pd
import pytest

from lphrs import synth_lifestyle as synth


@pytest.fixture(scope="module")
def sleep_df():
    rng = np.random.default_rng(0)
    n = 200
    age = rng.integers(27, 60, n)
    male = rng.random(n) < 0.5
    bmi_cat = rng.choice(["Normal", "Overweight", "Obese"], n, p=[0.5, 0.4, 0.1])
    return pd.DataFrame({
        "gender": np.where(male, "M", "F"), "age": age, "bmi_category": bmi_cat,
        "sleep_hours": 7.5 - 0.5 * (bmi_cat == "Obese") + rng.normal(0, 0.5, n),
        "steps": 7000 - 30 * (age - 43) + rng.normal(0, 800, n),
        "stress": np.clip(5 + rng.normal(0, 1.5, n), 1, 10),
    })


@pytest.fixture(scope="module")
def cohort():
    n = 40
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "patient_id": [f"p{i:02d}" for i in range(n)],
        "gender": rng.choice(["M", "F"], n), "age": rng.uniform(18, 90, n),
        "bmi": rng.uniform(18, 40, n), "sbp": rng.uniform(100, 160, n), "hba1c": rng.uniform(4, 8, n),
        "has_hypertension": rng.random(n) < 0.3, "has_glycemic": rng.random(n) < 0.3,
        "has_obesity": rng.random(n) < 0.3, "knee_condition": rng.random(n) < 0.1,
        "back_condition": rng.random(n) < 0.1, "smoker_condition": False, "alcohol_condition": False,
    })


END = pd.Timestamp("2026-10-05T12:00:00Z")


def test_fit_recovers_known_effects(sleep_df):
    m = synth.fit_lifestyle_model(sleep_df)
    coef = dict(zip(m["features"], m["targets"]["steps"]["coef"]))
    assert coef["age_centered"] == pytest.approx(-30, abs=15)
    sleep = dict(zip(m["features"], m["targets"]["sleep_hours"]["coef"]))
    assert sleep["obese"] < -0.2


def test_daily_series_shape_ranges_and_dates(sleep_df, cohort):
    m = synth.fit_lifestyle_model(sleep_df)
    daily, params = synth.generate_daily(cohort, m, END, days=100)
    assert set(daily["metric"]) == {"sleep_hours", "steps"}
    assert daily["timestamp"].max() == END.normalize()
    assert daily["timestamp"].min() == END.normalize() - pd.Timedelta(days=99)
    sleep = daily[daily["metric"] == "sleep_hours"]["value"]
    assert sleep.between(2, 14).all()
    assert (daily[daily["metric"] == "steps"]["value"] >= 0).all()
    per = daily.groupby(["patient_id", "metric"]).size()
    assert per.max() <= 100 and per.min() > 70           # ~8% missing days
    assert len(params) == len(cohort)


def test_trending_patients_have_real_trends(sleep_df, cohort):
    m = synth.fit_lifestyle_model(sleep_df)
    daily, params = synth.generate_daily(cohort, m, END, days=730)
    steps = daily[daily["metric"] == "steps"]
    t = (steps["timestamp"] - steps["timestamp"].min()).dt.days / 365.25
    fitted = steps.assign(t=t).groupby("patient_id").apply(
        lambda g: np.polyfit(g["t"], g["value"], 1)[0], include_groups=False)
    p = params.set_index("patient_id")
    trending = p[p["steps_trending"]].index
    assert len(trending) > 0
    # recovered slope has the generated sign for clearly trending patients
    for pid in trending:
        assert np.sign(fitted[pid]) == np.sign(p.loc[pid, "steps_slope_per_year"])


def test_same_seed_same_output(sleep_df, cohort):
    m = synth.fit_lifestyle_model(sleep_df)
    a, _ = synth.generate_daily(cohort, m, END, days=30, seed=7)
    b, _ = synth.generate_daily(cohort, m, END, days=30, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_context_and_hidden_taste(sleep_df, cohort):
    m = synth.fit_lifestyle_model(sleep_df)
    _, params = synth.generate_daily(cohort, m, END, days=30)
    ctx = synth.generate_context(cohort, params, m)
    assert ctx["goals"].apply(len).between(1, 3).all()
    assert ctx["goals"].apply(lambda g: set(g) <= set(synth.GOALS)).all()
    assert ctx["minutes_per_day"].between(15, 90).all()
    assert set(ctx["injury"]) <= {"none", "knee", "back"}
    assert set(ctx["diet_style"]) <= set(synth.DIET_STYLES)
    hidden = synth.generate_hidden_taste(cohort, ctx)
    assert hidden["capacity"].between(1, 5).all()
    assert {f"a_{t}" for t in synth.SCORE_TERMS} <= set(hidden.columns)
    assert len(hidden) == len(cohort)


def test_goals_follow_conditions(sleep_df):
    n = 300
    base = dict(gender="F", age=50.0, bmi=22.0, sbp=115.0, hba1c=5.0, has_hypertension=False,
                has_glycemic=False, has_obesity=False, knee_condition=False, back_condition=False,
                smoker_condition=False, alcohol_condition=False)
    sick = pd.DataFrame([{**base, "patient_id": f"s{i}", "has_glycemic": True} for i in range(n)])
    well = pd.DataFrame([{**base, "patient_id": f"w{i}"} for i in range(n)])
    m = synth.fit_lifestyle_model(sleep_df)
    share = []
    for c in (sick, well):
        _, params = synth.generate_daily(c, m, END, days=10)
        ctx = synth.generate_context(c, params, m)
        share.append(ctx["goals"].apply(lambda g: "glucose_control" in g).mean())
    assert share[0] > share[1] + 0.2
