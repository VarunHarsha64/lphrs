"""AdaptProfileFromFeedback, the simulated user, and the fast batch scorer used by the
feedback simulation (build plan 4.10 and 6.2).

Feedback rules (the design document was not available; these are this project's choices):
    ACCEPTED       item and its domain: +1 accepted, +1 shown; item tags: affinity +1
    COMPLETED      as ACCEPTED but counted twice (+2 / +2, affinity +2); two completions at the
                   domain's intensity cap raise that cap by 1 (max 5)
    REJECTED       +1 shown; item tags: affinity -1
    TOO_DIFFICULT  +1 shown; the domain's intensity cap drops to the item's intensity - 1 (min 1)
Tags whose affinity reaches +2 join the user's preference tags; at -2 they leave them.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import ranker as RK

OUTCOMES = ("ACCEPTED", "COMPLETED", "REJECTED", "TOO_DIFFICULT")
ACCEPT_OUTCOMES = {"ACCEPTED", "COMPLETED"}
AFFINITY_JOIN, AFFINITY_LEAVE = 2, -2
COMPLETIONS_TO_RAISE_CAP = 2
IGNORED_AFFINITY_TAGS = {"recipe"}


# ---------------------------------------------------------------- state and adaptation

def initial_state(user_context: dict) -> dict:
    """Mutable feedback state for one user, seeded from BuildUserContext."""
    return {"by_id": {}, "by_domain": {}, "caps": dict(user_context["intensity_caps"]),
            "affinity": {}, "completions_at_cap": {},
            "base_preference_tags": list(user_context["preference_tags"])}


def _bump(table: dict, key, accepted: float, shown: float) -> None:
    a, n = table.get(key, (0.0, 0.0))
    table[key] = (a + accepted, n + shown)


def adapt_profile_from_feedback(state: dict, item: dict, outcome: str) -> dict:
    """Return a new state updated with one piece of feedback.
    `item` needs id, domain, intensity, tags."""
    if outcome not in OUTCOMES:
        raise ValueError(f"unknown outcome {outcome}")
    s = copy.deepcopy(state)
    dom, cap = item["domain"], s["caps"].get(item["domain"], 5)
    acc, shown, aff = {"ACCEPTED": (1, 1, 1), "COMPLETED": (2, 2, 2),
                       "REJECTED": (0, 1, -1), "TOO_DIFFICULT": (0, 1, 0)}[outcome]
    _bump(s["by_id"], item["id"], acc, shown)
    _bump(s["by_domain"], dom, acc, shown)
    for t in item["tags"]:
        if t not in IGNORED_AFFINITY_TAGS and aff:
            s["affinity"][t] = s["affinity"].get(t, 0) + aff
    if outcome == "TOO_DIFFICULT":
        s["caps"][dom] = max(1, min(cap, int(item["intensity"]) - 1))
        s["completions_at_cap"][dom] = 0
    elif outcome == "COMPLETED" and int(item["intensity"]) >= cap:
        n = s["completions_at_cap"].get(dom, 0) + 1
        if n >= COMPLETIONS_TO_RAISE_CAP and cap < 5:
            s["caps"][dom] = cap + 1
            n = 0
        s["completions_at_cap"][dom] = n
    return s


def preference_tags(state: dict) -> list[str]:
    tags = set(state["base_preference_tags"])
    tags |= {t for t, v in state["affinity"].items() if v >= AFFINITY_JOIN}
    tags -= {t for t, v in state["affinity"].items() if v <= AFFINITY_LEAVE}
    return sorted(tags)


def apply_state(user_context: dict, state: dict) -> tuple[dict, dict]:
    """(user_context with adapted caps/preferences, history for the ranker's H term)."""
    uc = dict(user_context, intensity_caps=dict(state["caps"]), preference_tags=preference_tags(state))
    return uc, {"by_id": state["by_id"], "by_domain": state["by_domain"]}


# ---------------------------------------------------------------- simulated user

@dataclass
class SimulatedUser:
    """Hidden preferences of one patient (from hidden_taste.parquet). Never seen by the ranker."""
    a: np.ndarray            # sensitivity to R, G, U, P, F, H, C
    bias: float
    taste_domain: dict
    taste_exercise: dict
    capacity: int
    noise_sd: float = 0.5
    completion_prob: float = 0.7
    difficulty_weight: float = 0.8
    base_offset: float = 0.0   # simulator-level calibration of the overall acceptance rate

    @classmethod
    def from_row(cls, row: pd.Series) -> "SimulatedUser":
        return cls(a=np.array([row[f"a_{t}"] for t in RK.TERMS], dtype=float), bias=float(row["bias"]),
                   taste_domain={k[6:]: float(v) for k, v in row.items()
                                 if k.startswith("taste_") and not k.startswith("taste_ex_")},
                   taste_exercise={k[9:]: float(v) for k, v in row.items() if k.startswith("taste_ex_")},
                   capacity=int(row["capacity"]))

    def taste_match(self, domain: str, tags) -> float:
        ex = [self.taste_exercise[t] for t in tags if t in self.taste_exercise]
        return 0.5 * self.taste_domain.get(domain, 0.0) + (0.5 * float(np.mean(ex)) if ex else 0.0)

    def accept_probability(self, terms: np.ndarray, domain: str, tags, intensity: int, noise: float) -> float:
        difficulty = self.difficulty_weight * max(0, intensity - (self.capacity - 1))
        z = self.base_offset + self.bias + self.a @ terms + self.taste_match(domain, tags) - difficulty + noise
        return float(1 / (1 + np.exp(-z)))

    def respond(self, terms: np.ndarray, domain: str, tags, intensity: int, u: np.ndarray) -> str:
        """u: three uniforms/normal draws (noise, accept, complete) for common random numbers."""
        if intensity > self.capacity:
            return "TOO_DIFFICULT"
        p = self.accept_probability(terms, domain, tags, intensity, self.noise_sd * u[0])
        if u[1] >= p:
            return "REJECTED"
        return "COMPLETED" if u[2] < self.completion_prob else "ACCEPTED"


# ---------------------------------------------------------------- fast batch scoring

class CandidateMatrix:
    """Candidate table plus a dense item x tag matrix, shared by all simulated patients."""

    def __init__(self, cands: pd.DataFrame):
        self.cands = cands.reset_index(drop=True)
        self.index = {cid: i for i, cid in enumerate(self.cands["id"])}
        tags = sorted({t for ts in self.cands["tags"] for t in ts})
        self.tag_index = {t: j for j, t in enumerate(tags)}
        m = np.zeros((len(self.cands), len(tags)), dtype=np.float32)
        for i, ts in enumerate(self.cands["tags"]):
            for t in ts:
                m[i, self.tag_index[t]] = 1
        self.tag_matrix = m
        self.domains = sorted(self.cands["domain"].unique())
        self.domain_code = self.cands["domain"].map({d: k for k, d in enumerate(self.domains)}).to_numpy()
        self.minutes = self.cands["minutes"].to_numpy(float)
        self.intensity = self.cands["intensity"].to_numpy(int)

    def preference_overlap(self, rows: np.ndarray, tags) -> np.ndarray:
        cols = [self.tag_index[t] for t in tags if t in self.tag_index]
        if not cols:
            return np.zeros(len(rows))
        return self.tag_matrix[np.ix_(rows, cols)].sum(axis=1)


@dataclass
class PatientStatic:
    """Per-patient parts of the score that do not change with feedback."""
    patient_id: str
    rows: np.ndarray          # indices into CandidateMatrix of safety-kept candidates
    R: np.ndarray
    G: np.ndarray
    U: np.ndarray
    C: np.ndarray
    time_fit: np.ndarray
    budget: int


def build_static(patient_id: str, kept: pd.DataFrame, profile: dict, user_context: dict,
                 cm: CandidateMatrix) -> PatientStatic:
    t = RK.compute_terms(kept, profile, user_context)
    rows = kept["id"].map(cm.index).to_numpy()
    budget = user_context["minutes_per_day"]
    tf = np.array([RK.time_fit(m, budget) for m in cm.minutes[rows]])
    return PatientStatic(patient_id, rows, t["R"].to_numpy(float), t["G"].to_numpy(float),
                         t["U"].to_numpy(float), t["C"].to_numpy(float), tf, budget)


def dynamic_terms(ps: PatientStatic, state: dict, cm: CandidateMatrix) -> dict[str, np.ndarray]:
    """P, F, H for the current feedback state (same formulas as ranker.compute_terms)."""
    rows = ps.rows
    P = np.minimum(1.0, cm.preference_overlap(rows, preference_tags(state)) / 2)
    caps = np.array([state["caps"].get(d, 5) for d in cm.domains])[cm.domain_code[rows]]
    over = cm.intensity[rows] - caps
    F = ps.time_fit * np.where(over <= 0, 1.0, np.maximum(0.0, 1 - 0.5 * over))
    a_id = np.zeros(len(rows)); n_id = np.zeros(len(rows))
    pos = {r: k for k, r in enumerate(rows)}
    for cid, (a, n) in state["by_id"].items():
        k = pos.get(cm.index.get(cid, -1))
        if k is not None:
            a_id[k], n_id[k] = a, n
    dom_a = np.array([state["by_domain"].get(d, (0, 0))[0] for d in cm.domains], dtype=float)
    dom_n = np.array([state["by_domain"].get(d, (0, 0))[1] for d in cm.domains], dtype=float)
    dc = cm.domain_code[rows]
    H = (a_id + RK.H_DOMAIN_WEIGHT * dom_a[dc] + 0.5 * RK.H_PRIOR) / \
        (n_id + RK.H_DOMAIN_WEIGHT * dom_n[dc] + RK.H_PRIOR)
    return {"P": P, "F": F, "H": H}


def recommend_fast(ps: PatientStatic, state: dict, cm: CandidateMatrix, weights: dict,
                   shortlist: int | None = None, shortest: int = 30, table: dict | None = None) -> pd.DataFrame:
    """Score with the current state and run set selection; gives exactly the pipeline's list.
    `shortlist` (top-N by score plus the `shortest` quickest items) is an optional speed-up that
    can change results when scores tie, so the simulation does not use it."""
    dyn = dynamic_terms(ps, state, cm)
    terms = {"R": ps.R, "G": ps.G, "U": ps.U, "P": dyn["P"], "F": dyn["F"], "H": dyn["H"], "C": ps.C}
    score = sum(weights[t] * terms[t] for t in RK.TERMS if t != "C") - weights["C"] * terms["C"]
    if shortlist is None or len(score) <= shortlist + shortest:
        keep = np.arange(len(score))
    else:
        top = np.argpartition(-score, shortlist)[:shortlist]
        quick = np.argsort(cm.minutes[ps.rows], kind="stable")[:shortest]
        keep = np.union1d(top, quick)
    rows = ps.rows[keep]
    df = cm.cands.iloc[rows][["id", "domain", "title", "tags", "minutes", "intensity"]].reset_index(drop=True)
    for t in RK.TERMS:
        df[t] = terms[t][keep]
    df["score"] = score[keep]
    df = df.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    return RK.select_set(df, budget=ps.budget, table=table)
