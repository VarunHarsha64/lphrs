"""Plain-language reasons for each recommendation (build plan 4.9).

Each item lists its top 2 positive score contributors. H (history) is only explained once
there is real feedback; with no history it sits at the neutral prior and says nothing useful.
"""

from __future__ import annotations

import re

from .config import load_yaml, metric_specs

GOAL_LABELS = {"weight_loss": "weight loss", "lower_bp": "lower blood pressure", "better_sleep": "better sleep",
               "more_activity": "more activity", "glucose_control": "glucose control"}
DISCLAIMER = "Wellness guidance only. Not medical advice."
MIN_CONTRIBUTION = 0.01     # smaller score contributions are not worth explaining
MIN_URGENCY = 0.02          # a trend sentence needs a real, near-term projection
MAX_PROJECTION_YEARS = 10


def fmt_value(metric: str, v: float, threshold: float | None = None) -> str:
    """Format a value; add decimals if rounding would make it look equal to its threshold."""
    s = _fmt(metric, v)
    if threshold is not None and v != threshold and float(s.replace(",", "")) == float(threshold):
        return f"{v:.2f}"
    return s


def _fmt(metric: str, v: float) -> str:
    if metric == "steps":
        return f"{v:,.0f}"
    if metric in ("hba1c", "sleep_hours", "bmi"):
        return f"{v:.1f}"
    return f"{v:.0f}"


def fmt_rate(metric: str, v: float) -> str:
    v = abs(v)
    if metric == "steps":
        return f"{round(v, -1):,.0f}"
    return f"{v:.1f}" if v < 10 else f"{v:.0f}"


def fmt_years(t: float) -> str:
    if t < 1:
        months = max(1, round(t * 12))
        return f"{months} month{'s' if months != 1 else ''}"
    y = round(t, 1) if t < 3 else round(t)
    return f"{y:g} year{'s' if y != 1 else ''}"


def _unit(metric: str) -> str:
    u = metric_specs()[metric]["unit"]
    return {"steps/day": "steps a day", "h": "hours", "/min": "beats/min"}.get(u, u)


LIFESTYLE_METRICS = {"steps", "sleep_hours"}
SHORT_LABELS = {"bmi": "BMI", "hba1c": "HbA1c", "ldl": "LDL cholesterol", "hdl": "HDL cholesterol"}


def label(metric: str) -> str:
    if metric in SHORT_LABELS:
        return SHORT_LABELS[metric]
    lbl = metric_specs()[metric]["label"]
    return lbl[0].lower() + lbl[1:]


def fmt_threshold(metric: str, v: float) -> str:
    return f"{v:g}" if metric != "steps" else f"{v:,.0f}"


def reason_urgency(m: dict, metric: str) -> str:
    verb = "risen" if m["slope_per_year"] > 0 else "fallen"
    return (f"Your {label(metric)} has {verb} about {fmt_rate(metric, m['slope_per_year'])} {_unit(metric)} "
            f"per year and is projected to reach {fmt_threshold(metric, m['threshold'])} in about "
            f"{fmt_years(m['time_to_threshold_years'])}.")


def reason_relevance(row: dict, profile: dict) -> str:
    drv = row["R_driver"]
    if drv.startswith("condition:"):
        desc = drv.split(":", 1)[1]
        cond = re.sub(r"\s*\([^()]*\)\s*$", "", desc)
        cg = load_yaml("metrics")["temporal"]["condition_goals"]
        served = [GOAL_LABELS[g] for g in row["goals"]
                  if any(k in desc.lower() for k in cg.get(g, []))]
        return f"Your record lists {cond.lower()}; this supports {' and '.join(served[:2])}."
    if drv == "cvd_risk":
        return (f"Your estimated cardiovascular risk is {profile['cvd_risk']:.0%}, and this supports "
                f"heart health.")
    m = profile["metrics"][drv]
    side = "above" if m["bad_direction"] == "up" else "below"
    if m["urgency_status"] == "past_threshold" and drv in LIFESTYLE_METRICS:
        if drv == "steps":
            return (f"Your daily steps recently average {fmt_value(drv, m['current'], m['threshold'])}, {side} the "
                    f"{fmt_threshold(drv, m['threshold'])} threshold.")
        return (f"Your sleep recently averages {fmt_value(drv, m['current'], m['threshold'])} hours a night, {side} the "
                f"{fmt_threshold(drv, m['threshold'])}-hour threshold.")
    if m["urgency_status"] == "past_threshold":
        return (f"Your latest {label(drv)} is {fmt_value(drv, m['current'], m['threshold'])} {_unit(drv)}, {side} the "
                f"{fmt_threshold(drv, m['threshold'])} threshold.")
    return f"Your {label(drv)} has often been {side} {fmt_threshold(drv, m['threshold'])} in your record."


def reason(term: str, row: dict, profile: dict, user_context: dict) -> str | None:
    if term == "U" and row["U_driver"]:
        m = profile["metrics"][row["U_driver"]]
        if (m["urgency_status"] == "approaching" and m["urgency"] >= MIN_URGENCY
                and m["time_to_threshold_years"] <= MAX_PROJECTION_YEARS):
            return reason_urgency(m, row["U_driver"])
    if term == "R" and row["R_driver"]:
        return reason_relevance(row, profile)
    if term == "G":
        served = [GOAL_LABELS[g] for g in user_context["goals"] if g in set(row["goals"])]
        if served:
            return f"Supports your goal{'s' if len(served) > 1 else ''}: {', '.join(served)}."
    if term == "P":
        shared = sorted(set(user_context["preference_tags"]) & set(row["tags"]))
        if shared:
            return f"Matches your preferences ({', '.join(s.replace('_', ' ') for s in shared)})."
    if term == "F":
        return f"Fits your day: {row['minutes']} of your {user_context['minutes_per_day']} available minutes."
    if term == "H":
        return "Similar suggestions have worked well for you before." if row["H"] > 0.5 else None
    return None


def explain(row: dict, profile: dict, user_context: dict, has_history: bool = False, top: int = 2) -> list[str]:
    """Up to `top` reasons, from the largest positive score contributions."""
    terms = ["R", "G", "U", "P", "F"] + (["H"] if has_history else [])
    ranked = sorted(terms, key=lambda t: row[f"contrib_{t}"], reverse=True)
    out = []
    for t in ranked:
        if row[f"contrib_{t}"] < MIN_CONTRIBUTION:
            break
        r = reason(t, row, profile, user_context)
        if r:
            out.append(r)
        if len(out) == top:
            break
    return out
