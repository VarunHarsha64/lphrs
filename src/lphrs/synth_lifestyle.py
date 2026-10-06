"""Synthetic sleep, steps, goals, preferences, constraints and hidden taste vectors (build plan 3.5).

No public dataset links lifestyle, goals and feedback to clinical records, so these layers are
simulated. Distributions of sleep, steps and stress are fitted on the Sleep Health and Lifestyle
dataset (cross-sectional, ages 27-59) by age, gender and BMI category; everything else (daily
noise, trends, goals, preferences) uses the hand-set assumptions documented below.

`cohort` frames passed in here have one row per patient with columns:
    patient_id, gender ("M"/"F"), age, bmi, sbp, hba1c,
    has_hypertension, has_glycemic, has_obesity, knee_condition, back_condition,
    smoker_condition, alcohol_condition
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42
GOALS = ["weight_loss", "lower_bp", "better_sleep", "more_activity", "glucose_control"]
DOMAINS = ["nutrition", "fitness", "sleep", "lifestyle", "preventive_care", "wellness"]
EXERCISE_TYPES = ["walking", "cycling", "swimming", "running", "strength", "yoga", "dance", "hiit", "tai_chi"]
DIET_STYLES = {"omnivore": 0.45, "mediterranean": 0.20, "vegetarian": 0.15, "low_carb": 0.20}
SCORE_TERMS = ["R", "G", "U", "P", "F", "H", "C"]

# Fitted data covers ages 27-59; outside that we clamp rather than extrapolate.
FIT_AGE_RANGE = (27, 59)
FIT_AGE_CENTER = 43

# --- daily-series assumptions (not in the source data, which is one row per person)
TREND_PROB = 0.30                 # share of patients with a real trend, per metric
TREND_DOWN_PROB = 0.60            # trends are more often in the bad direction
SLEEP_SLOPE_RANGE = (0.3, 0.8)    # h per year for trending patients
STEPS_SLOPE_RANGE = (1000, 2500)  # steps/day per year
SLEEP_DAILY_SD = 0.6              # h
STEPS_DAILY_CV = 0.25             # fraction of baseline
AR_PHI = 0.6                      # day-to-day autocorrelation of the noise
MISSING_DAY_PROB = 0.08           # days the wearable was not worn


# ---------------------------------------------------------------- distributions

def bmi_category(bmi: pd.Series) -> pd.Series:
    """Map BMI to the sleep dataset's categories. Missing BMI -> "Normal"."""
    cat = pd.Series("Normal", index=bmi.index)
    cat[bmi >= 25] = "Overweight"
    cat[bmi >= 30] = "Obese"
    return cat


def _design(age, gender, bmi_cat) -> np.ndarray:
    age = np.clip(np.asarray(age, dtype=float), *FIT_AGE_RANGE) - FIT_AGE_CENTER
    gender, bmi_cat = np.asarray(gender), np.asarray(bmi_cat)
    return np.column_stack([np.ones(len(age)), age, gender == "M",
                            bmi_cat == "Overweight", bmi_cat == "Obese"]).astype(float)


def fit_lifestyle_model(sleep_df: pd.DataFrame) -> dict:
    """OLS of sleep hours, steps and stress on age, gender and BMI category.

    The residual SD is the between-person spread used to draw each patient's baseline.
    """
    X = _design(sleep_df["age"], sleep_df["gender"], sleep_df["bmi_category"])
    model = {"features": ["intercept", "age_centered", "male", "overweight", "obese"],
             "n": len(sleep_df), "targets": {}}
    for target in ("sleep_hours", "steps", "stress"):
        y = sleep_df[target].to_numpy(float)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        sd = float(np.sqrt(resid @ resid / (len(y) - X.shape[1])))
        model["targets"][target] = {"coef": coef.tolist(), "resid_sd": sd}
    return model


def predict(model: dict, target: str, cohort: pd.DataFrame) -> np.ndarray:
    X = _design(cohort["age"], cohort["gender"], bmi_category(cohort["bmi"]))
    return X @ np.asarray(model["targets"][target]["coef"])


# ---------------------------------------------------------------- daily series

def _trend_slopes(n: int, rng: np.random.Generator, slope_range, small_sd) -> tuple[np.ndarray, np.ndarray]:
    trending = rng.random(n) < TREND_PROB
    sign = np.where(rng.random(n) < TREND_DOWN_PROB, -1.0, 1.0)
    big = sign * rng.uniform(*slope_range, size=n)
    small = rng.normal(0.0, small_sd, size=n)
    return np.where(trending, big, small), trending


def _ar1(n: int, days: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance AR(1) noise, shape (n, days)."""
    eps = rng.normal(size=(n, days))
    out = np.empty_like(eps)
    out[:, 0] = eps[:, 0]
    scale = np.sqrt(1 - AR_PHI**2)
    for t in range(1, days):
        out[:, t] = AR_PHI * out[:, t - 1] + scale * eps[:, t]
    return out


def generate_daily(cohort: pd.DataFrame, model: dict, end_date: pd.Timestamp,
                   days: int = 730, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily sleep hours and steps for the `days` days ending at `end_date` (inclusive).

    Returns (daily long frame: patient_id, timestamp, metric, value, unit;
             per-patient parameters: baselines, slopes, trend flags).
    """
    rng = np.random.default_rng(seed)
    n = len(cohort)
    dates = pd.date_range(end=end_date.normalize(), periods=days, freq="D", tz="UTC")
    t_years = (np.arange(days) - (days - 1) / 2) / 365.25          # centred: baseline = window mean
    weekend = np.asarray(dates.dayofweek >= 5)

    sl = model["targets"]["sleep_hours"]
    st = model["targets"]["steps"]
    sleep_base = np.clip(predict(model, "sleep_hours", cohort) + rng.normal(0, sl["resid_sd"], n), 4.5, 9.5)
    steps_base = np.clip(predict(model, "steps", cohort) + rng.normal(0, st["resid_sd"], n), 1500, 16000)
    sleep_slope, sleep_trending = _trend_slopes(n, rng, SLEEP_SLOPE_RANGE, 0.05)
    steps_slope, steps_trending = _trend_slopes(n, rng, STEPS_SLOPE_RANGE, 150)

    sleep = (sleep_base[:, None] + sleep_slope[:, None] * t_years[None, :]
             + 0.4 * weekend[None, :] + SLEEP_DAILY_SD * _ar1(n, days, rng))
    steps = ((steps_base[:, None] + steps_slope[:, None] * t_years[None, :]) * np.where(weekend, 0.85, 1.0)
             * (1 + STEPS_DAILY_CV * _ar1(n, days, rng)))
    sleep = np.clip(sleep, 2.0, 14.0).round(2)
    steps = np.clip(steps, 0, 40000).round()

    frames = []
    for metric, values, unit in (("sleep_hours", sleep, "h"), ("steps", steps, "steps/day")):
        observed = rng.random((n, days)) >= MISSING_DAY_PROB
        rows, cols = np.nonzero(observed)
        frames.append(pd.DataFrame({
            "patient_id": cohort["patient_id"].to_numpy()[rows],
            "timestamp": dates[cols],
            "metric": metric, "value": values[rows, cols], "unit": unit}))
    daily = pd.concat(frames, ignore_index=True)

    params = pd.DataFrame({
        "patient_id": cohort["patient_id"].to_numpy(),
        "sleep_base": sleep_base.round(2), "sleep_slope_per_year": sleep_slope.round(3),
        "sleep_trending": sleep_trending,
        "steps_base": steps_base.round(), "steps_slope_per_year": steps_slope.round(),
        "steps_trending": steps_trending})
    return daily, params


# ---------------------------------------------------------------- user context

def _goal_weights(row) -> np.ndarray:
    """Base weight 1 per goal, raised when the patient's record makes the goal relevant."""
    w = dict.fromkeys(GOALS, 1.0)
    if row.has_obesity or row.bmi >= 30:
        w["weight_loss"] += 4
    elif row.bmi >= 25:
        w["weight_loss"] += 2
    if row.has_hypertension or row.sbp >= 130:
        w["lower_bp"] += 4
    if row.has_glycemic or row.hba1c >= 5.7:
        w["glucose_control"] += 4
    if row.sleep_base < 6.5:
        w["better_sleep"] += 3
    if row.steps_base < 6000:
        w["more_activity"] += 3
    return np.array([w[g] for g in GOALS])


_YOUNG = {"walking": 1.5, "cycling": 1.5, "swimming": 1.0, "running": 2.0, "strength": 2.0,
          "yoga": 1.5, "dance": 1.5, "hiit": 2.0, "tai_chi": 0.3}
_OLD = {"walking": 3.0, "cycling": 1.5, "swimming": 2.0, "running": 0.5, "strength": 1.0,
        "yoga": 2.0, "dance": 1.0, "hiit": 0.3, "tai_chi": 2.0}


def _exercise_weights(age: float) -> np.ndarray:
    t = (np.clip(age, 25, 70) - 25) / 45
    return np.array([(1 - t) * _YOUNG[e] + t * _OLD[e] for e in EXERCISE_TYPES])


def generate_context(cohort: pd.DataFrame, params: pd.DataFrame, model: dict,
                     seed: int = SEED) -> pd.DataFrame:
    """Goals, preferences, minute budget, injuries and CVD lifestyle flags per patient.

    `params` is the second output of generate_daily (needs sleep_base, steps_base).
    """
    rng = np.random.default_rng(seed + 1)
    df = cohort.merge(params[["patient_id", "sleep_base", "steps_base"]], on="patient_id")
    stress_mu = predict(model, "stress", df)
    stress_sd = model["targets"]["stress"]["resid_sd"]
    styles, style_p = list(DIET_STYLES), list(DIET_STYLES.values())

    rows = []
    for i, r in enumerate(df.itertuples(index=False)):
        gw = _goal_weights(r)
        k = rng.choice([1, 2, 3], p=[0.30, 0.45, 0.25])
        goals = list(rng.choice(GOALS, size=k, replace=False, p=gw / gw.sum()))
        ew = _exercise_weights(r.age)
        n_ex = rng.integers(1, 4)
        liked = list(rng.choice(EXERCISE_TYPES, size=n_ex, replace=False, p=ew / ew.sum()))

        if r.knee_condition and rng.random() < 0.85:
            injury = "knee"
        elif r.back_condition and rng.random() < 0.85:
            injury = "back"
        else:
            injury = rng.choice(["none", "knee", "back"], p=[0.82, 0.08, 0.10])

        rows.append({
            "patient_id": r.patient_id,
            "goals": goals,
            "diet_style": rng.choice(styles, p=style_p),
            "liked_exercise": liked,
            "minutes_per_day": int(rng.integers(3, 19) * 5),          # 15..90
            "injury": injury,
            "stress": int(np.clip(round(stress_mu[i] + rng.normal(0, stress_sd)), 1, 10)),
            "smoke": int(bool(r.smoker_condition) or rng.random() < 0.10),
            "alco": int(bool(r.alcohol_condition) or rng.random() < 0.05),
            "active": int(r.steps_base >= 7000),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- hidden taste

# Mean sensitivity of a simulated user to each score term (R, G, U, P, F, H, C).
A_MEAN = np.array([0.6, 1.0, 0.4, 0.8, 1.0, 0.6, -0.8])
A_SD = 0.3


def generate_hidden_taste(cohort: pd.DataFrame, context: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Hidden per-user vectors for the feedback simulator (Phase 7). Never read by the ranker.

    a_*:        sensitivity to each score term
    bias:       base log-odds of accepting anything
    taste_<domain>, taste_<exercise>: true liking; exercise taste is correlated with,
                but not identical to, the stated preferences
    capacity:   highest intensity (1-5) the user can sustain; above it -> TOO_DIFFICULT
    """
    rng = np.random.default_rng(seed + 2)
    df = cohort[["patient_id", "age"]].merge(context[["patient_id", "liked_exercise", "active"]], on="patient_id")
    n = len(df)
    out = pd.DataFrame({"patient_id": df["patient_id"]})
    a = A_MEAN + rng.normal(0, A_SD, size=(n, len(SCORE_TERMS)))
    for j, term in enumerate(SCORE_TERMS):
        out[f"a_{term}"] = a[:, j].round(3)
    out["bias"] = rng.normal(-1.0, 0.3, n).round(3)
    for d in DOMAINS:
        out[f"taste_{d}"] = rng.normal(0, 1, n).round(3)
    liked = np.array([[e in lk for e in EXERCISE_TYPES] for lk in df["liked_exercise"]], dtype=float)
    ex = 0.8 * liked + rng.normal(0, 0.6, size=liked.shape)
    for j, e in enumerate(EXERCISE_TYPES):
        out[f"taste_ex_{e}"] = ex[:, j].round(3)
    cap = 4.5 - (df["age"].to_numpy() - 30) / 20 + 0.5 * df["active"].to_numpy() + rng.normal(0, 0.6, n)
    out["capacity"] = np.clip(np.round(cap), 1, 5).astype(int)
    return out
