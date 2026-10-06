"""BuildHealthProfile and BuildUserContext (build plan 4.4, 4.5).

A health profile is computed "as of" a reference date from timeline rows up to that date,
so the same code serves the live pipeline and the hidden-future evaluation (experiment 3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import temporal as T
from .config import load_yaml, metric_specs

CLINICAL_EVENTS = ("condition", "medication", "allergy")


def temporal_cfg() -> dict:
    return load_yaml("metrics")["temporal"]


# ---------------------------------------------------------------- measurements

def measurement_frame(tl: pd.DataFrame, freq: str | None = None) -> pd.DataFrame:
    """Observations as-is plus lifestyle resampled to weekly means.
    Columns: patient_id, timestamp, metric, value."""
    freq = freq or temporal_cfg()["lifestyle_resample"]
    cols = ["patient_id", "timestamp", "metric", "value"]
    obs = tl.loc[tl["event_type"] == "observation", cols]
    life = tl.loc[tl["event_type"] == "lifestyle", cols]
    parts = [obs] + ([T.resample_lifestyle(life, freq)[cols]] if len(life) else [])
    out = pd.concat(parts, ignore_index=True)
    out["metric"] = out["metric"].astype(str)
    return out.sort_values(["patient_id", "metric", "timestamp"], kind="stable").reset_index(drop=True)


def metric_profile(m: pd.DataFrame, metric: str, ref: pd.Timestamp, lam: float,
                   user_goals: set[str], cfg: dict | None = None) -> dict | None:
    """Profile of one metric for one patient. `m` holds that patient's rows for the metric."""
    cfg = cfg or temporal_cfg()
    spec = metric_specs()[metric]
    m = m[m["timestamp"] <= ref].sort_values("timestamp")
    if m.empty:
        return None
    current = float(m["value"].iloc[-1])
    window = m[m["timestamp"] >= ref - pd.Timedelta(days=cfg["trend_window_years"] * T.DAYS_PER_YEAR)]
    t = T.years_before(ref, window["timestamp"])
    fit = T.fit_trend(t, window["value"])
    span = float(np.ptp(t)) if len(t) else 0.0
    # too short a span counts as too few points for a trend
    n_trend = fit.n if span >= cfg["trend_min_span_years"] else 0
    theta, bad = spec["threshold"], spec["bad_direction"]
    urgency, ttt, status = T.time_to_threshold_urgency(
        current, fit.slope, fit.r2, n_trend, theta, bad,
        cfg["trend_min_points"], cfg["trend_full_confidence_points"])
    label = T.trend_label(fit.slope, fit.r2, n_trend, current, cfg["trend_min_points"])

    w = T.recency_weight(T.days_since(ref, m["timestamp"]), lam)
    if theta is None:
        history = float("nan")
    else:
        past = (m["value"] >= theta) if bad == "up" else (m["value"] <= theta)
        history = float((w * past.to_numpy()).sum() / w.sum()) if w.sum() > 0 else 0.0
    gr = cfg["goal_relevance"]
    return {
        "metric": metric, "current": current, "unit": spec["unit"],
        "last_updated": m["timestamp"].iloc[-1], "n_total": len(m),
        "trend": label, "direction": T.direction_vs_risk(label, bad),
        "slope_per_year": fit.slope, "r2": fit.r2, "n_window": fit.n, "span_years": span,
        "variance": float(window["value"].var(ddof=1)) if fit.n > 1 else float("nan"),
        "threshold": theta, "bad_direction": bad,
        "urgency": urgency, "time_to_threshold_years": ttt, "urgency_status": status,
        "history_score": history, "decay_lambda": float(lam),
        "goal_relevance": gr["match"] if set(spec["goals"]) & user_goals else gr["other"],
    }


# ---------------------------------------------------------------- clinical events

def _goal_match(text: str | None, user_goals: set[str], cfg: dict) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(any(k in t for k in kws) for g, kws in cfg["condition_goals"].items() if g in user_goals)


def event_weights(ev: pd.DataFrame, ref: pd.Timestamp, user_goals: set[str],
                  cfg: dict | None = None) -> pd.DataFrame:
    """Decay weights for one patient's condition / medication / allergy rows up to `ref`.

    delta_t runs from the start of an ongoing event, or from the stop of a resolved one.
    Active medications get delta_t = 0.
    """
    cfg = cfg or temporal_cfg()
    ev = ev[ev["timestamp"] <= ref].copy()
    if ev.empty:
        return ev.assign(event_class=[], active=[], floor=[], delta_days=[], w_recency=[],
                         w_plain=[], goal_relevance=[], weight=[])
    ev["event_class"] = T.classify_events(ev, ref, cfg)
    ev["active"] = ev["stop"].isna() | (ev["stop"] > ref)
    anchor = ev["timestamp"].where(ev["active"], ev["stop"])
    delta = T.days_since(ref, anchor)
    delta[(ev["event_type"] == "medication").to_numpy() & ev["active"].to_numpy()] = 0.0
    ev["floor"] = [T.event_floor(c, cfg) for c in ev["event_class"]]
    ev["delta_days"] = delta
    lam = cfg["lambda_base"]
    ev["w_recency"] = T.recency_weight(delta, lam, ev["floor"].to_numpy())
    ev["w_plain"] = T.plain_exponential_weight(delta, lam)
    gr = cfg["goal_relevance"]
    text = ev["description"].fillna("") + " " + ev["context"].fillna("")
    matched = np.array([_goal_match(t, user_goals, cfg) for t in text])
    ev["goal_relevance"] = np.where((ev["event_type"] == "allergy").to_numpy() | matched, gr["match"], gr["other"])
    ev["weight"] = ev["w_recency"] * ev["goal_relevance"]
    return ev


# ---------------------------------------------------------------- profiles

def build_health_profile(patient_id: str, tl: pd.DataFrame, ref: pd.Timestamp, volatility: pd.DataFrame,
                         user_goals=(), meas: pd.DataFrame | None = None,
                         cvd_risk: float | None = None) -> dict:
    """Health profile for one patient as of `ref`.

    tl: timeline rows (may contain other patients); meas: optional pre-built measurement_frame.
    volatility: output of temporal.cohort_volatility (needs a `lambda` column).
    cvd_risk: calibrated probability from risk_model (data/processed/cvd_risk.parquet).
    """
    cfg = temporal_cfg()
    goals = set(user_goals)
    rows = tl[tl["patient_id"] == patient_id]
    if meas is None:
        meas = measurement_frame(rows)
    else:
        meas = meas[meas["patient_id"] == patient_id]

    metrics = {}
    for metric, m in meas.groupby("metric"):
        lam = float(volatility["lambda"].get(metric, cfg["lambda_base"]))
        p = metric_profile(m, metric, ref, lam, goals, cfg)
        if p:
            metrics[metric] = p

    ev = event_weights(rows[rows["event_type"].isin(CLINICAL_EVENTS)], ref, goals, cfg)
    clinical = ev[ev["event_class"] != "social_finding"] if len(ev) else ev

    def unique_records(df, cols):
        if df.empty:
            return []
        df = df.sort_values("timestamp").drop_duplicates("description", keep="last")
        return df[cols].to_dict("records")

    ev_cols = ["description", "code", "timestamp", "stop", "event_class", "w_recency", "weight"]
    cond = clinical[clinical["event_type"] == "condition"] if len(clinical) else clinical
    return {
        "patient_id": patient_id,
        "ref_date": ref,
        "metrics": metrics,
        "active_conditions": unique_records(cond[cond["active"]] if len(cond) else cond, ev_cols),
        "past_major_events": unique_records(
            cond[(~cond["active"]) & (cond["event_class"] == "resolved_major")] if len(cond) else cond, ev_cols),
        "active_medications": unique_records(
            clinical[(clinical["event_type"] == "medication") & clinical["active"]] if len(clinical) else clinical,
            ev_cols),
        "allergies": unique_records(clinical[clinical["event_type"] == "allergy"] if len(clinical) else clinical,
                                    ev_cols + ["context"]),
        "events": ev,
        "cvd_risk": None if cvd_risk is None else float(cvd_risk),
    }


def build_all_profiles(tl: pd.DataFrame, ref: pd.Timestamp, volatility: pd.DataFrame,
                       goals_by_patient: dict[str, list[str]], patient_ids=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Profiles for many patients as two flat tables: (metric rows, event-weight rows)."""
    cfg = temporal_cfg()
    patient_ids = list(patient_ids) if patient_ids is not None else sorted(tl["patient_id"].unique())
    tl = tl[tl["patient_id"].isin(patient_ids)]
    meas = measurement_frame(tl)
    meas_groups = dict(tuple(meas.groupby("patient_id")))
    ev_all = tl[tl["event_type"].isin(CLINICAL_EVENTS)]
    ev_groups = dict(tuple(ev_all.groupby("patient_id")))
    lam = volatility["lambda"].to_dict()

    metric_rows, event_frames = [], []
    for pid in patient_ids:
        goals = set(goals_by_patient.get(pid, []))
        for metric, m in meas_groups.get(pid, meas.iloc[:0]).groupby("metric"):
            p = metric_profile(m, metric, ref, lam.get(metric, cfg["lambda_base"]), goals, cfg)
            if p:
                metric_rows.append({"patient_id": pid, **p})
        ev = event_weights(ev_groups.get(pid, ev_all.iloc[:0]), ref, goals, cfg)
        if len(ev):
            event_frames.append(ev)
    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    return pd.DataFrame(metric_rows), events


# ---------------------------------------------------------------- user context

def initial_intensity_cap(age: float, active: int) -> int:
    """Starting per-domain intensity cap (feedback adjusts it in Phase 7)."""
    cap = 4 + (1 if active else 0) - (1 if age >= 70 else 0)
    return int(np.clip(cap, 2, 5))


def build_user_context(patient_id: str, context: pd.DataFrame, patients: pd.DataFrame) -> dict:
    """Goals, preferences, constraints, daily minute budget and intensity caps for one patient."""
    c = context.loc[context["patient_id"] == patient_id].iloc[0]
    p = patients.loc[patients["patient_id"] == patient_id].iloc[0]
    prefs = load_yaml("interventions")["preferences"]
    liked = list(c["liked_exercise"])
    preference_tags = sorted(set(prefs["diet_styles"][c["diet_style"]]) | set(liked))
    cap = initial_intensity_cap(float(p["age"]), int(c["active"]))
    domains = sorted({i["domain"] for i in load_yaml("interventions")["interventions"]})
    return {
        "patient_id": patient_id,
        "age": float(p["age"]), "gender": p["gender"],
        "goals": list(c["goals"]),
        "diet_style": c["diet_style"],
        "liked_exercise": liked,
        "preference_tags": preference_tags,
        "minutes_per_day": int(c["minutes_per_day"]),
        "injury": c["injury"],
        "constraints": [] if c["injury"] == "none" else [f"injury:{c['injury']}"],
        "intensity_caps": dict.fromkeys(domains, cap),
        "smoke": int(c["smoke"]), "alco": int(c["alco"]), "active": int(c["active"]),
        "stress": int(c["stress"]),
    }
