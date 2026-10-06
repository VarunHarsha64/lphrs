"""Temporal engine: persistence-floored adaptive decay (5.1), trend detection and
time-to-threshold urgency (5.2).

    lambda_m  = lambda_base * clip(volatility_m, 0.25, 4.0)
    w_recency = max(floor_e, exp(-lambda_m * delta_t_days))
    floor_e   = severity_e * persistence_e
    W_e       = w_recency * goal_relevance_e

volatility_m is made unit-free before comparing metrics: the cohort SD of visit-to-visit
change divided by the cohort mean level of that metric, then divided by the median of that
ratio across metrics. (Raw SDs in mmHg, %, mg/dL ... are not comparable.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.25


# ---------------------------------------------------------------- event classes and floors

def condition_class(description: str, resolved: bool, cfg: dict) -> str:
    """Classify a Synthea condition into an event class from metrics.yaml temporal.event_types."""
    d = description.lower()
    if any(k in d for k in cfg["social_keywords"]):
        return "social_finding"
    if any(d.startswith(p) for p in cfg["history_prefixes"]):
        return "resolved_major"
    if not resolved:
        return "chronic_condition"
    if any(k in d for k in cfg["major_event_keywords"]):
        return "resolved_major"
    return "acute_condition"


def classify_events(events: pd.DataFrame, ref: pd.Timestamp, cfg: dict) -> pd.Series:
    """Event class for each timeline row of type condition / medication / allergy."""
    out = pd.Series(index=events.index, dtype="object")
    et = events["event_type"]
    out[et == "allergy"] = "allergy"
    out[et == "medication"] = "medication"
    cond = events[et == "condition"]
    resolved = cond["stop"].notna() & (cond["stop"] <= ref)
    out[cond.index] = [condition_class(d, r, cfg) for d, r in zip(cond["description"], resolved)]
    return out


def event_floor(event_class: str, cfg: dict) -> float:
    spec = cfg["event_types"][event_class]
    return float(spec["severity"] * spec["persistence"])


# ---------------------------------------------------------------- volatility and decay

def cohort_volatility(meas: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per-metric unit-free volatility from visit-to-visit changes across the cohort.

    `meas` has patient_id, timestamp, metric, value (lifestyle already resampled).
    Returns a frame indexed by metric with change_sd, mean_level, rel_change_sd,
    volatility (ratio to the median across metrics) and lambda (per day).
    """
    m = meas.sort_values(["patient_id", "metric", "timestamp"])
    diffs = m.groupby(["patient_id", "metric"])["value"].diff()
    stats = pd.DataFrame({
        "change_sd": diffs.groupby(m["metric"]).std(),
        "mean_level": m.groupby("metric")["value"].mean().abs(),
    })
    stats["rel_change_sd"] = stats["change_sd"] / stats["mean_level"]
    stats["volatility"] = stats["rel_change_sd"] / stats["rel_change_sd"].median()
    stats["lambda"] = decay_rate(stats["volatility"], cfg)
    return stats


def decay_rate(volatility, cfg: dict):
    lo, hi = cfg["volatility_clip"]
    return cfg["lambda_base"] * np.clip(volatility, lo, hi)


def recency_weight(delta_days, lam, floor=0.0):
    """max(floor, exp(-lambda * delta_t)); negative deltas (future) count as 0 days."""
    delta = np.clip(np.asarray(delta_days, dtype=float), 0, None)
    return np.maximum(floor, np.exp(-np.asarray(lam, dtype=float) * delta))


def plain_exponential_weight(delta_days, lam):
    """Baseline for experiment 4: no floor, no volatility scaling."""
    return recency_weight(delta_days, lam, 0.0)


# ---------------------------------------------------------------- trend and urgency

@dataclass
class TrendFit:
    slope: float        # units per year
    intercept: float
    r2: float
    n: int


def fit_trend(t_years, values) -> TrendFit:
    """Ordinary least squares of value on time (years)."""
    t = np.asarray(t_years, dtype=float)
    y = np.asarray(values, dtype=float)
    n = len(t)
    if n < 2 or np.ptp(t) == 0:
        return TrendFit(0.0, float(y.mean()) if n else float("nan"), 0.0, n)
    slope, intercept = np.polyfit(t, y, 1)
    resid = y - (slope * t + intercept)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else 0.0
    return TrendFit(float(slope), float(intercept), max(0.0, r2), n)


def is_past_threshold(x_now: float, theta: float, bad_direction: str) -> bool:
    return x_now >= theta if bad_direction == "up" else x_now <= theta


def time_to_threshold_urgency(x_now: float, slope: float, r2: float, n: int, theta: float | None,
                              bad_direction: str, min_points: int = 3,
                              full_conf_points: int = 5) -> tuple[float, float, str]:
    """Return (urgency in [0, 1], time-to-threshold in years or nan, status).

    status: no_threshold | past_threshold | insufficient_data | moving_away | approaching
    """
    if theta is None or np.isnan(x_now):
        return 0.0, float("nan"), "no_threshold"
    if is_past_threshold(x_now, theta, bad_direction):
        return 1.0, 0.0, "past_threshold"
    if n < min_points:
        return 0.0, float("nan"), "insufficient_data"
    toward = slope > 0 if bad_direction == "up" else slope < 0
    if not toward:
        return 0.0, float("inf"), "moving_away"
    ttt = (theta - x_now) / slope
    confidence = r2 * min(1.0, n / full_conf_points)
    return float(confidence / (1.0 + ttt)), float(ttt), "approaching"


def trend_label(slope: float, r2: float, n: int, level: float, min_points: int = 3,
                rel_per_year: float = 0.02, min_r2: float = 0.05) -> str:
    """rising / falling / stable / insufficient. A trend needs >= 2 % change per year of the
    current level and a minimal fit (r2 >= 0.05)."""
    if n < min_points:
        return "insufficient"
    if r2 < min_r2 or abs(slope) < rel_per_year * max(abs(level), 1e-9):
        return "stable"
    return "rising" if slope > 0 else "falling"


def direction_vs_risk(label: str, bad_direction: str) -> str:
    """worsening / improving / neutral relative to the metric's bad direction."""
    if label not in ("rising", "falling"):
        return "neutral"
    bad = (label == "rising") == (bad_direction == "up")
    return "worsening" if bad else "improving"


# ---------------------------------------------------------------- helpers

def years_before(ref: pd.Timestamp, timestamps: pd.Series) -> np.ndarray:
    """Time in years relative to `ref` (negative = in the past)."""
    return np.array((timestamps - ref).dt.total_seconds() / 86400 / DAYS_PER_YEAR, dtype=float)


def days_since(ref: pd.Timestamp, timestamps: pd.Series) -> np.ndarray:
    return np.array((ref - timestamps).dt.total_seconds() / 86400, dtype=float)


def resample_lifestyle(daily: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """Weekly mean of daily lifestyle values; timestamp = last observed day in that week
    (so a partial final week never lands after the reference date)."""
    d = daily[["patient_id", "metric", "timestamp", "value"]].copy()
    d["period"] = d["timestamp"].dt.tz_convert(None).dt.to_period(freq)
    out = d.groupby(["patient_id", "metric", "period"], as_index=False).agg(
        timestamp=("timestamp", "max"), value=("value", "mean"))
    return out.drop(columns="period")
