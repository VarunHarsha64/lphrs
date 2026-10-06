"""Experiments 1-7 and case-study helpers (build plan Section 7).

Every number reported comes from these functions run on the processed data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import profile as P
from . import ranker as RK
from . import safety as S
from . import temporal as T
from .config import metric_specs

THRESHOLD_METRICS = ["systolic_bp", "diastolic_bp", "bmi", "hba1c", "glucose", "total_cholesterol", "ldl", "hdl"]


# ---------------------------------------------------------------- per-patient bundle

@dataclass
class Bundle:
    patient_id: str
    profile: dict
    user_context: dict
    facts: S.SafetyFacts
    terms: pd.DataFrame          # compute_terms on ALL candidates
    kept: np.ndarray             # bool mask: survives the safety filter


def build_bundle(pid: str, data) -> Bundle:
    uc = P.build_user_context(pid, data.context, data.patients)
    prof = P.build_health_profile(pid, data.patient_timeline(pid), data.ref, data.volatility, uc["goals"],
                                  cvd_risk=data.cvd_risk.get(pid))
    facts = S.facts_from_profile(prof, uc)
    viol = S.find_violations(data.candidates, facts)
    kept = ~data.candidates["id"].isin(set(viol["candidate_id"])).to_numpy()
    terms = RK.compute_terms(data.candidates, prof, uc)
    return Bundle(pid, prof, uc, facts, terms, kept)


def recommend(b: Bundle, *, safety=True, urgency=True, selection="greedy", weights=None, table=None) -> pd.DataFrame:
    t = b.terms[b.kept] if safety else b.terms
    if not urgency:
        t = t.assign(U=0.0, U_driver=None)
    scored = RK.score(t, weights)
    if selection == "greedy":
        return RK.select_set(scored, budget=b.user_context["minutes_per_day"], table=table)
    return RK.select_top_k(scored)


# ---------------------------------------------------------------- experiment 1: safety

def exp1_safety(bundles, cands, table) -> dict:
    rows = []
    for b in bundles:
        for arm, safe in (("with_filter", True), ("without_filter", False)):
            recs = recommend(b, safety=safe, table=table)
            v = S.find_violations(cands[cands["id"].isin(set(recs["id"]))], b.facts)
            rows.append({"patient_id": b.patient_id, "arm": arm, "items": len(recs),
                         "violating_items": v["candidate_id"].nunique(),
                         "rules": sorted(v["rule_id"].unique())})
    df = pd.DataFrame(rows)
    summary = df.groupby("arm").agg(lists=("patient_id", "size"), items=("items", "sum"),
                                    violating_items=("violating_items", "sum"),
                                    lists_with_violation=("violating_items", lambda s: int((s > 0).sum())))
    summary["item_violation_rate"] = summary["violating_items"] / summary["items"]
    summary["list_violation_rate"] = summary["lists_with_violation"] / summary["lists"]
    rule_counts = (df[df["arm"] == "without_filter"].explode("rules")["rules"].dropna().value_counts())
    return {"summary": summary, "rules_without_filter": rule_counts, "rows": df}


# ---------------------------------------------------------------- experiment 2: snapshot vs longitudinal

def rising_but_normal(profile: dict) -> list[str]:
    """Metrics not yet past threshold whose trend moves toward it (urgency status 'approaching')."""
    return [m for m, p in profile["metrics"].items()
            if p["threshold"] is not None and p["urgency_status"] == "approaching" and p["direction"] == "worsening"]


def exp2_snapshot_vs_longitudinal(bundles, table) -> dict:
    rows = []
    for b in bundles:
        metrics = rising_but_normal(b.profile)
        if not metrics:
            continue
        recs = {arm: recommend(b, urgency=u, table=table) for arm, u in (("snapshot_U_off", False),
                                                                        ("longitudinal_U_on", True))}
        for m in metrics:
            row = {"patient_id": b.patient_id, "metric": m, "urgency": b.profile["metrics"][m]["urgency"]}
            for arm, r in recs.items():
                row[arm] = bool(any(m in t for t in r["targets"]))
            rows.append(row)
    df = pd.DataFrame(rows)
    overall = df[["snapshot_U_off", "longitudinal_U_on"]].mean()
    df["urgency_band"] = pd.cut(df["urgency"], [0, 0.05, 0.2, 1.0], include_lowest=True,
                                labels=["< 0.05", "0.05-0.2", ">= 0.2"])
    by_urgency = df.groupby("urgency_band", observed=False).agg(
        pairs=("patient_id", "size"), snapshot_U_off=("snapshot_U_off", "mean"),
        longitudinal_U_on=("longitudinal_U_on", "mean"))
    per_metric = df.groupby("metric").agg(pairs=("patient_id", "size"), snapshot_U_off=("snapshot_U_off", "mean"),
                                          longitudinal_U_on=("longitudinal_U_on", "mean"))
    return {"pairs": len(df), "patients": df["patient_id"].nunique(), "overall": overall,
            "per_metric": per_metric, "by_urgency": by_urgency, "rows": df}


# ---------------------------------------------------------------- experiment 3: early warning

def _urgency_series(t_years: np.ndarray, v: np.ndarray, theta, bad, cfg) -> np.ndarray:
    """Urgency at each reading using only data up to that reading (same rules as profile)."""
    out = np.zeros(len(t_years))
    for i in range(len(t_years)):
        now = t_years[i]
        w = (t_years <= now) & (t_years >= now - cfg["trend_window_years"])
        tw, vw = t_years[w], v[w]
        fit = T.fit_trend(tw - now, vw)
        n_trend = fit.n if np.ptp(tw) >= cfg["trend_min_span_years"] else 0
        out[i] = T.time_to_threshold_urgency(v[i], fit.slope, fit.r2, n_trend, theta, bad,
                                             cfg["trend_min_points"], cfg["trend_full_confidence_points"])[0]
    return out


def _streak_start(times: np.ndarray, flags: np.ndarray) -> float | None:
    """Start time of the run of flags that is still active at the last checkpoint."""
    if not len(flags) or not flags[-1]:
        return None
    i = len(flags) - 1
    while i > 0 and flags[i - 1]:
        i -= 1
    return times[i]


def exp3_early_warning(obs: pd.DataFrame, ref: pd.Timestamp, taus=(0.1, 0.2, 0.3),
                       bands=(0.02, 0.05, 0.10), hide_years: float = 2.0) -> dict:
    """Hide the last `hide_years`; flag metrics on earlier data; check crossings in the hidden part.

    Urgency flag:      urgency >= tau at the last pre-cut reading (lead measured from the start of
                       the flag streak that is still active then).
    Latest-value flag: latest value within a band (2 / 5 / 10 %) of the threshold on the bad side;
                       several bands so methods can be compared at similar alarm counts.
    Only (patient, metric) pairs NOT past threshold at the cut, with readings on both sides.
    """
    cfg = P.temporal_cfg()
    specs = metric_specs()
    cut = ref - pd.Timedelta(days=hide_years * T.DAYS_PER_YEAR)
    rows = []
    for (pid, metric), g in obs[obs["metric"].isin(THRESHOLD_METRICS)].groupby(["patient_id", "metric"]):
        g = g.sort_values("timestamp")
        pre, post = g[g["timestamp"] <= cut], g[g["timestamp"] > cut]
        if pre.empty or post.empty:
            continue
        theta, bad = specs[metric]["threshold"], specs[metric]["bad_direction"]
        if T.is_past_threshold(pre["value"].iloc[-1], theta, bad):
            continue
        past_post = post["value"] >= theta if bad == "up" else post["value"] <= theta
        crossed = bool(past_post.any())
        t_cross = post["timestamp"][past_post].iloc[0] if crossed else pd.NaT
        t_pre = T.years_before(cut, pre["timestamp"])
        v_pre = pre["value"].to_numpy(float)
        urg = _urgency_series(t_pre, v_pre, theta, bad, cfg)
        row = {"patient_id": pid, "metric": metric, "crossed": crossed,
               "months_cut_to_cross": ((t_cross - cut).days / 30.44) if crossed else np.nan}
        near = {f"latest_value_{b}": (v_pre >= theta * (1 - b)) if bad == "up" else (v_pre <= theta * (1 + b))
                for b in bands}
        for name, flags in [(f"urgency_{tau}", urg >= tau) for tau in taus] + list(near.items()):
            start = _streak_start(t_pre, flags)
            row[f"flag_{name}"] = start is not None
            # lead = time from flag start to crossing (start is in years relative to the cut, <= 0)
            row[f"lead_{name}"] = (row["months_cut_to_cross"] - start * 12) if (start is not None and crossed) else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    methods = [f"urgency_{tau}" for tau in taus] + [f"latest_value_{b}" for b in bands]
    crossers = df[df["crossed"]]
    summary = []
    for m in methods:
        flagged = df[f"flag_{m}"]
        lead = crossers[f"lead_{m}"]
        summary.append({
            "method": m, "flagged": int(flagged.sum()),
            "recall": float(crossers[f"flag_{m}"].mean()) if len(crossers) else np.nan,
            "precision": float(df.loc[flagged, "crossed"].mean()) if flagged.any() else np.nan,
            "median_lead_months_detected": float(lead.median()) if lead.notna().any() else np.nan,
            "iqr_lead_months_detected": (float(lead.quantile(0.25)), float(lead.quantile(0.75)))
            if lead.notna().any() else (np.nan, np.nan),
            "median_lead_months_all_crossers": float(lead.fillna(0).median()) if len(crossers) else np.nan,
        })
    per_metric = df.groupby("metric").agg(pairs=("crossed", "size"), crossed=("crossed", "sum"))
    return {"cut": cut, "pairs": len(df), "crossers": len(crossers), "summary": pd.DataFrame(summary),
            "per_metric": per_metric, "rows": df}


# ---------------------------------------------------------------- experiment 4: decay

def exp4_decay(events: pd.DataFrame, years: float = 5.0) -> dict:
    old = events[events["delta_days"] > years * T.DAYS_PER_YEAR]
    tab = old.groupby("event_class").agg(events=("w_recency", "size"), patients=("patient_id", "nunique"),
                                         floored_mean=("w_recency", "mean"), floored_min=("w_recency", "min"),
                                         plain_mean=("w_plain", "mean"), plain_max=("w_plain", "max"))
    return {"table": tab}


# ---------------------------------------------------------------- experiment 5: set selection

def exp5_set_selection(bundles, table) -> dict:
    rows = []
    for b in bundles:
        budget = b.user_context["minutes_per_day"]
        for arm, sel in (("independent_top_k", "topk"), ("greedy_set", "greedy")):
            r = recommend(b, selection=sel, table=table)
            rows.append({"patient_id": b.patient_id, "arm": arm, "items": len(r),
                         "conflicts": RK.list_conflicts(r, table), "domains": r["domain"].nunique(),
                         "minutes": int(r["minutes"].sum()), "within_budget": bool(r["minutes"].sum() <= budget),
                         "score_sum": float(r["score"].sum())})
    df = pd.DataFrame(rows)
    summary = df.groupby("arm").agg(lists=("patient_id", "size"), mean_conflicts=("conflicts", "mean"),
                                    lists_with_conflict=("conflicts", lambda s: float((s > 0).mean())),
                                    mean_domains=("domains", "mean"), within_budget=("within_budget", "mean"),
                                    mean_minutes=("minutes", "mean"), mean_score_sum=("score_sum", "mean"))
    return {"summary": summary, "rows": df}


# ---------------------------------------------------------------- experiment 6 extra: what the weights favour

def exp6_list_composition(bundles, learned: dict, table) -> pd.DataFrame:
    """Mean term values of final lists under default vs learned weights (no feedback history)."""
    rows = []
    for b in bundles:
        for arm, w in (("default", None), ("learned", learned)):
            r = recommend(b, weights=w, table=table)
            rows.append({"arm": arm, **{t: r[t].mean() for t in RK.TERMS}})
    return pd.DataFrame(rows).groupby("arm").mean()
