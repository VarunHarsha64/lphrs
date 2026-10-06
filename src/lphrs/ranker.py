"""Candidate scoring (build plan 5.3) and cross-domain set selection (5.4).

    Score = w1*R + w2*G + w3*U + w4*P + w5*F + w6*H - w7*C        (all terms in [0, 1])

Term definitions (the build plan names the terms; these are the concrete choices):
    R  health relevance: max of (a) target-metric "abnormality" (1 if past threshold, else the
       decay-weighted share of past readings beyond threshold), (b) the patient's CVD risk for
       candidates targeting a cardiovascular metric, (c) the decay weight of an active condition
       whose related goal (metrics.yaml condition_goals) the candidate serves.
    G  goal alignment: share of the user's goals the candidate serves.
    U  urgency: max time-to-threshold urgency over target metrics.
    P  preference: tag overlap with the user's preference tags (1 shared tag = 0.5, 2+ = 1).
    F  feasibility: time fit (full credit up to half the daily budget, falling to 0 at the
       budget) x intensity fit (penalty per level above the domain's intensity cap).
    H  historical adherence: smoothed acceptance of the same item and its domain (0.5 = no data).
    C  effort cost from the candidate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import load_yaml

TERMS = ["R", "G", "U", "P", "F", "H", "C"]
WEIGHT_KEYS = {"R": "relevance", "G": "goal", "U": "urgency", "P": "preference",
               "F": "feasibility", "H": "history", "C": "cost"}
CV_METRICS = {"systolic_bp", "diastolic_bp", "ldl", "total_cholesterol", "hdl"}
H_PRIOR = 2.0          # pseudo-observations at 0.5 for the adherence estimate
H_DOMAIN_WEIGHT = 0.5  # domain history counts half as much as the item's own history


def default_weights() -> dict[str, float]:
    w = load_yaml("weights")["weights"]
    return {t: float(w[k]) for t, k in WEIGHT_KEYS.items()}


def selection_params() -> dict:
    return load_yaml("weights")["set_selection"]


# ---------------------------------------------------------------- term computation

def _metric_signals(profile: dict) -> tuple[dict, dict]:
    """Per metric: abnormality in [0,1] and urgency in [0,1]."""
    abnormal, urgency = {}, {}
    for m, p in profile["metrics"].items():
        hist = p.get("history_score")
        hist = 0.0 if hist is None or np.isnan(hist) else float(hist)
        abnormal[m] = 1.0 if p["urgency_status"] == "past_threshold" else hist
        urgency[m] = float(p["urgency"])
    return abnormal, urgency


def condition_goal_weights(profile: dict) -> dict[str, tuple[float, str]]:
    """goal -> (strongest active-condition weight, that condition's description)."""
    cg = load_yaml("metrics")["temporal"]["condition_goals"]
    out: dict[str, tuple[float, str]] = {}
    for c in profile.get("active_conditions", []):
        d = c["description"].lower()
        for goal, kws in cg.items():
            if any(k in d for k in kws) and c["w_recency"] > out.get(goal, (0.0, ""))[0]:
                out[goal] = (float(c["w_recency"]), c["description"])
    return out


def adherence(history: dict | None, cand_id: str, domain: str) -> float:
    """Smoothed acceptance rate. history = {"by_id": {id: (accepted, shown)},
    "by_domain": {domain: (accepted, shown)}}; missing -> 0.5."""
    if not history:
        return 0.5
    a_id, n_id = history.get("by_id", {}).get(cand_id, (0, 0))
    a_d, n_d = history.get("by_domain", {}).get(domain, (0, 0))
    a = a_id + H_DOMAIN_WEIGHT * a_d + 0.5 * H_PRIOR
    n = n_id + H_DOMAIN_WEIGHT * n_d + H_PRIOR
    return float(a / n)


def time_fit(minutes: float, budget: float) -> float:
    if budget <= 0:
        return 0.0
    if minutes <= 0.5 * budget:
        return 1.0
    if minutes > budget:
        return 0.0
    return float(2 * (1 - minutes / budget))


def intensity_fit(intensity: int, cap: int) -> float:
    return 1.0 if intensity <= cap else float(max(0.0, 1 - 0.5 * (intensity - cap)))


def compute_terms(cands: pd.DataFrame, profile: dict, user_context: dict,
                  history: dict | None = None) -> pd.DataFrame:
    """Add R, G, U, P, F, H, C columns plus the metric that drove R and U (for explanations)."""
    abnormal, urgency = _metric_signals(profile)
    cond_goals = condition_goal_weights(profile)
    cvd = profile.get("cvd_risk") or 0.0
    goals = set(user_context["goals"])
    prefs = set(user_context["preference_tags"])
    budget = user_context["minutes_per_day"]
    caps = user_context["intensity_caps"]

    R, R_driver, U, U_driver, G, P, F, H = [], [], [], [], [], [], [], []
    for c in cands.itertuples(index=False):
        targets = [t for t in c.targets if t in profile["metrics"]]
        r_val, r_drv = 0.0, None
        for t in targets:
            if abnormal[t] > r_val:
                r_val, r_drv = abnormal[t], t
        if CV_METRICS & set(c.targets) and cvd > r_val:
            r_val, r_drv = cvd, "cvd_risk"
        for g in c.goals:
            w, desc = cond_goals.get(g, (0.0, ""))
            if w > r_val:
                r_val, r_drv = w, f"condition:{desc}"
        u_val, u_drv = 0.0, None
        for t in targets:
            if urgency[t] > u_val:
                u_val, u_drv = urgency[t], t
        R.append(r_val); R_driver.append(r_drv); U.append(u_val); U_driver.append(u_drv)
        G.append(len(goals & set(c.goals)) / len(goals) if goals else 0.0)
        P.append(min(1.0, len(prefs & set(c.tags)) / 2))
        F.append(time_fit(c.minutes, budget) * intensity_fit(c.intensity, caps.get(c.domain, 5)))
        H.append(adherence(history, c.id, c.domain))

    out = cands.copy()
    out["R"], out["G"], out["U"], out["P"], out["F"], out["H"] = R, G, U, P, F, H
    out["C"] = cands["effort"].astype(float)
    out["R_driver"], out["U_driver"] = R_driver, U_driver
    return out


def score(terms: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Add per-term contributions (w*term; C negative) and the total score."""
    w = weights or default_weights()
    out = terms.copy()
    for t in TERMS:
        sign = -1.0 if t == "C" else 1.0
        out[f"contrib_{t}"] = sign * w[t] * out[t]
    out["score"] = out[[f"contrib_{t}" for t in TERMS]].sum(axis=1)
    return out.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------- synergy and selection

def load_synergy() -> dict[frozenset, float]:
    """Tag pair -> value. A same-tag pair (redundancy) is keyed by a one-element frozenset."""
    return {frozenset((a, b)): float(v) for a, b, v in load_yaml("synergy")["pairs"]}


def synergy(tags_a, tags_b, table: dict[frozenset, float]) -> float:
    total = 0.0
    for a in tags_a:
        for b in tags_b:
            total += table.get(frozenset((a, b)), 0.0)
    return float(np.clip(total, -1.0, 1.0))


def select_set(scored: pd.DataFrame, budget: float, k: int | None = None, alpha: float | None = None,
               max_per_domain: int | None = None, table: dict | None = None) -> pd.DataFrame:
    """Greedy cross-domain selection with synergy, time budget and per-domain cap (5.4).

    Look-ahead on the budget: candidates are tried in order of gain, and the first one whose
    leftover budget can still be filled by the shortest allowed items (respecting the domain
    cap) is taken. Without this a few mid-length picks can exhaust a small budget and end the
    list early. If no candidate can complete the list, it aims for the longest list that fits.
    Returns the selected rows in pick order with `gain` and `synergy_bonus` columns.
    """
    p = selection_params()
    k = k if k is not None else p["k"]
    alpha = alpha if alpha is not None else p["alpha"]
    max_per_domain = max_per_domain if max_per_domain is not None else p["max_per_domain"]
    table = table if table is not None else load_synergy()

    df = scored.reset_index(drop=True)
    n = len(df)
    if n == 0:
        return pd.DataFrame()
    scores = df["score"].to_numpy(float)
    minutes = df["minutes"].to_numpy(float)
    dom_names, dom = np.unique(df["domain"].to_numpy(), return_inverse=True)
    M, S = _tag_synergy_matrices(df["tags"], table)
    by_minutes = np.argsort(minutes, kind="stable")

    available = np.ones(n, dtype=bool)
    bonus = np.zeros(n)                       # sum over selected items of synergy(c, s)
    counts = np.zeros(len(dom_names), dtype=int)
    picks = []
    while len(picks) < k and available.any():
        ok = available & (minutes <= budget) & (counts[dom] < max_per_domain)
        if not ok.any():
            break
        gains = np.where(ok, scores + alpha * bonus, -np.inf)
        order = np.argsort(-gains, kind="stable")
        if gains[order[0]] <= 0:
            break
        positive = order[gains[order] > 0]
        # aim for a full list; if impossible, for the longest list that still fits
        i = None
        for slots in range(k - len(picks) - 1, -1, -1):
            cost = _fill_costs(positive, by_minutes, available, minutes, dom, counts, max_per_domain, slots)
            fits = minutes[positive] + cost <= budget
            if fits.any():
                i = int(positive[np.argmax(fits)])
                break
        if i is None:
            break
        picks.append((i, float(gains[i]), float(alpha * bonus[i])))
        available[i] = False
        budget -= minutes[i]
        counts[dom[i]] += 1
        bonus += np.clip(M @ (S @ M[i]), -1.0, 1.0)

    out = df.iloc[[p[0] for p in picks]].reset_index(drop=True)
    out["gain"] = [p[1] for p in picks]
    out["synergy_bonus"] = [p[2] for p in picks]
    return out


def _tag_synergy_matrices(tags: pd.Series, table: dict[frozenset, float]) -> tuple[np.ndarray, np.ndarray]:
    """Item x tag indicator matrix M and symmetric tag x tag synergy matrix S, so that
    synergy(a, b) before clipping = M[a] @ S @ M[b] (same sum as `synergy`)."""
    vocab = {t: j for j, t in enumerate(sorted({t for ts in tags for t in ts}))}
    M = np.zeros((len(tags), len(vocab)))
    for i, ts in enumerate(tags):
        for t in ts:
            M[i, vocab[t]] = 1.0
    S = np.zeros((len(vocab), len(vocab)))
    for key, v in table.items():
        a, b = (tuple(key) * 2)[:2]          # one-element key = same-tag pair
        if a in vocab and b in vocab:
            S[vocab[a], vocab[b]] = S[vocab[b], vocab[a]] = v
    return M, S


def _fill(by_minutes, available, minutes, dom, counts, max_per_domain, slots, exclude=-1):
    """Minutes needed to fill `slots` with the shortest available items under the domain cap,
    and the items used. Shortest-first is optimal: a per-domain cap is a partition matroid."""
    if slots <= 0:
        return 0.0, ()
    c = counts.copy()
    total, used = 0.0, []
    for j in by_minutes:
        if j == exclude or not available[j] or c[dom[j]] >= max_per_domain:
            continue
        total += minutes[j]
        c[dom[j]] += 1
        used.append(j)
        if len(used) == slots:
            return total, tuple(used)
    return np.inf, tuple(used)


def _fill_costs(cands, by_minutes, available, minutes, dom, counts, max_per_domain, slots) -> np.ndarray:
    """For each candidate: minutes the other open slots need if it is picked.
    Computed once per domain; recomputed exactly only for candidates that are themselves in
    that domain's cheapest fill."""
    cost = np.empty(len(cands))
    per_dom = {}
    for k, j in enumerate(cands):
        d = dom[j]
        if d not in per_dom:
            c = counts.copy()
            c[d] += 1
            per_dom[d] = (c, *_fill(by_minutes, available, minutes, dom, c, max_per_domain, slots))
        c, total, used = per_dom[d]
        cost[k] = _fill(by_minutes, available, minutes, dom, c, max_per_domain, slots, exclude=j)[0] \
            if j in used else total
    return cost


def select_top_k(scored: pd.DataFrame, k: int | None = None) -> pd.DataFrame:
    """Baseline for experiment 5: the K highest scores, chosen independently."""
    k = k if k is not None else selection_params()["k"]
    return scored.head(k).reset_index(drop=True)


def list_conflicts(selected: pd.DataFrame, table: dict | None = None) -> int:
    """Number of pairs in a list with negative synergy."""
    table = table if table is not None else load_synergy()
    tags = selected["tags"].tolist()
    return sum(synergy(tags[i], tags[j], table) < 0 for i in range(len(tags)) for j in range(i + 1, len(tags)))
