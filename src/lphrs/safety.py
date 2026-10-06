"""SafetyFilter: deterministic hard-constraint rules from safety_rules.yaml (build plan 4.7).

Every dropped candidate is logged with the rule(s) that fired; the log feeds explanations
and evaluation experiment 1. Simplified illustrations, not clinically reviewed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .candidates import match_keywords
from .config import load_yaml

DROP_LOG_COLUMNS = ["candidate_id", "title", "domain", "rule_id", "reason"]


@dataclass
class SafetyFacts:
    """What the rules need to know about one patient."""
    allergens: set[str] = field(default_factory=set)
    conditions: list[str] = field(default_factory=list)    # active condition descriptions
    medications: list[str] = field(default_factory=list)   # active medication descriptions
    injuries: set[str] = field(default_factory=set)        # survey: knee / back


def allergens_from_descriptions(descriptions, cfg: dict | None = None) -> set[str]:
    """Map Synthea allergy descriptions to canonical allergens (e.g. "Glycine max" -> soy)."""
    cfg = cfg or load_yaml("safety_rules")
    out = set()
    for name, spec in cfg["allergens"].items():
        if match_keywords(list(descriptions), spec["allergy_keywords"], spec.get("exclude_keywords", [])):
            out.add(name)
    return out


def facts_from_profile(profile: dict, user_context: dict) -> SafetyFacts:
    """Build SafetyFacts from BuildHealthProfile + BuildUserContext outputs."""
    allergies = [a["description"] for a in profile.get("allergies", [])
                 if pd.isna(a.get("stop")) or a["stop"] > profile["ref_date"]]
    injury = user_context.get("injury", "none")
    return SafetyFacts(
        allergens=allergens_from_descriptions(allergies),
        conditions=[c["description"] for c in profile.get("active_conditions", [])],
        medications=[m["description"] for m in profile.get("active_medications", [])],
        injuries=set() if injury in (None, "none") else {injury},
    )


# ---------------------------------------------------------------- rule logic

def rule_applies(rule: dict, facts: SafetyFacts) -> bool:
    """Does the rule's trigger match this patient?"""
    for t in rule["trigger"]["any_of"]:
        kind = t["type"]
        if kind == "allergy" and facts.allergens:
            return True
        if kind == "condition" and match_keywords(facts.conditions, t["keywords"]):
            return True
        if kind == "medication" and match_keywords(facts.medications, t["keywords"]):
            return True
        if kind == "injury" and facts.injuries & set(t["values"]):
            return True
    return False


def _in_scope(cands: pd.DataFrame, applies_to: list[str]) -> pd.Series:
    if "all" in applies_to:
        return pd.Series(True, index=cands.index)
    return cands["domain"].isin(applies_to) | (cands["source"].isin(applies_to))


def violates(cands: pd.DataFrame, rule: dict, facts: SafetyFacts) -> pd.Series:
    """Boolean mask of candidates the rule removes (assuming it applies to the patient)."""
    cond = rule["drop_when"]
    flags = [set(t) | set(c) for t, c in zip(cands["tags"], cands["contraindications"])]
    mask = pd.Series(False, index=cands.index)
    if cond.get("allergen_overlap"):
        bad = {f"allergen:{a}" for a in facts.allergens}
        mask |= pd.Series([bool(f & bad) for f in flags], index=cands.index)
    if "tags_any" in cond:
        bad = set(cond["tags_any"])
        mask |= pd.Series([bool(f & bad) for f in flags], index=cands.index)
    if "nutrient_above" in cond:
        n = cond["nutrient_above"]
        mask |= cands[n["field"]].fillna(-1) > n["value"]
    if "intensity_at_least" in cond:
        mask |= cands["intensity"] >= cond["intensity_at_least"]
    return mask & _in_scope(cands, rule["applies_to"])


def find_violations(cands: pd.DataFrame, facts: SafetyFacts, rules: list[dict] | None = None) -> pd.DataFrame:
    """One row per (candidate, rule) that fires. Used by the filter and by experiment 1."""
    rules = rules if rules is not None else load_yaml("safety_rules")["rules"]
    logs = []
    for rule in rules:
        if not rule_applies(rule, facts):
            continue
        hit = cands[violates(cands, rule, facts)]
        if len(hit):
            logs.append(pd.DataFrame({"candidate_id": hit["id"], "title": hit["title"], "domain": hit["domain"],
                                      "rule_id": rule["id"], "reason": rule["description"]}))
    return pd.concat(logs, ignore_index=True) if logs else pd.DataFrame(columns=DROP_LOG_COLUMNS)


def safety_filter(cands: pd.DataFrame, facts: SafetyFacts,
                  rules: list[dict] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (kept candidates, drop log)."""
    log = find_violations(cands, facts, rules)
    kept = cands[~cands["id"].isin(set(log["candidate_id"]))].reset_index(drop=True)
    return kept, log
