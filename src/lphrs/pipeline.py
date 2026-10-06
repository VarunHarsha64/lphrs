"""PipelineMasterExecution: end-to-end recommendation run for one patient.

    profile -> user context -> candidates -> SafetyFilter -> scoring -> set selection -> explanations

Switches (use_safety, use_urgency, selection) exist for the evaluation experiments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd

from . import candidates as C
from . import explain as X
from . import profile as P
from . import ranker as RK
from . import safety as S
from .config import PROCESSED_DIR
from .risk_model import load_cvd_risk

DISCLAIMER = X.DISCLAIMER


@dataclass
class PipelineData:
    """Everything the pipeline reads from disk, loaded once and shared across runs."""
    ref: pd.Timestamp
    patients: pd.DataFrame
    context: pd.DataFrame
    volatility: pd.DataFrame
    cvd_risk: dict
    candidates: pd.DataFrame
    timeline: pd.DataFrame | None = None
    _groups: dict = field(default_factory=dict, repr=False)

    def patient_timeline(self, patient_id: str) -> pd.DataFrame:
        if self.timeline is not None:
            if not self._groups:
                self._groups = dict(tuple(self.timeline.groupby("patient_id")))
            return self._groups.get(patient_id, self.timeline.iloc[:0])
        return pd.read_parquet(PROCESSED_DIR / "timeline.parquet", filters=[("patient_id", "==", patient_id)])

    @property
    def cohort_ids(self) -> list[str]:
        return self.context["patient_id"].tolist()


def load_data(full_timeline: bool = False) -> PipelineData:
    """Load processed data. full_timeline=True keeps the whole timeline in memory (batch runs)."""
    meta = json.loads((PROCESSED_DIR / "build_meta.json").read_text())
    return PipelineData(
        ref=pd.Timestamp(meta["reference_date"]),
        patients=pd.read_parquet(PROCESSED_DIR / "patients.parquet"),
        context=pd.read_parquet(PROCESSED_DIR / "user_context.parquet"),
        volatility=pd.read_json(PROCESSED_DIR / "temporal_params.json", orient="index"),
        cvd_risk=load_cvd_risk(),
        candidates=C.generate_candidates(),
        timeline=pd.read_parquet(PROCESSED_DIR / "timeline.parquet") if full_timeline else None,
    )


def resolve_patient_id(data: PipelineData, prefix: str) -> str:
    hits = [p for p in data.cohort_ids if p.startswith(prefix)]
    if len(hits) != 1:
        raise ValueError(f"'{prefix}' matches {len(hits)} cohort patients; give more of the id")
    return hits[0]


def run_pipeline(patient_id: str, data: PipelineData, *, weights: dict | None = None,
                 history: dict | None = None, user_context: dict | None = None,
                 use_safety: bool = True, use_urgency: bool = True, selection: str = "greedy",
                 ref: pd.Timestamp | None = None, profile: dict | None = None) -> dict:
    """Run the full pipeline for one patient and return every intermediate result."""
    ref = ref if ref is not None else data.ref
    uc = user_context or P.build_user_context(patient_id, data.context, data.patients)
    if profile is None:
        profile = P.build_health_profile(patient_id, data.patient_timeline(patient_id), ref, data.volatility,
                                         uc["goals"], cvd_risk=data.cvd_risk.get(patient_id))

    cands = data.candidates
    facts = S.facts_from_profile(profile, uc)
    if use_safety:
        kept, drops = S.safety_filter(cands, facts)
    else:
        kept, drops = cands, S.find_violations(cands.iloc[:0], facts)

    terms = RK.compute_terms(kept, profile, uc, history)
    if not use_urgency:
        terms["U"] = 0.0
        terms["U_driver"] = None
    scored = RK.score(terms, weights)
    if selection == "greedy":
        recs = RK.select_set(scored, budget=uc["minutes_per_day"])
    else:
        recs = RK.select_top_k(scored)

    has_history = bool(history) and any(n for _, n in history.get("by_id", {}).values())
    items = []
    for row in recs.to_dict("records"):
        items.append({
            "id": row["id"], "title": row["title"], "domain": row["domain"], "minutes": int(row["minutes"]),
            "intensity": int(row["intensity"]), "score": float(row["score"]), "gain": row.get("gain"),
            "terms": {t: float(row[t]) for t in RK.TERMS},
            "contributions": {t: float(row[f"contrib_{t}"]) for t in RK.TERMS},
            "tags": list(row["tags"]),
            "reasons": X.explain(row, profile, uc, has_history),
            "disclaimer": DISCLAIMER,
        })
    return {"patient_id": patient_id, "ref_date": ref, "profile": profile, "user_context": uc,
            "safety_facts": facts, "n_candidates": len(cands), "n_after_safety": len(kept),
            "drops": drops, "scored": scored, "recommendations": items,
            "minutes_used": int(sum(i["minutes"] for i in items)), "disclaimer": DISCLAIMER}
