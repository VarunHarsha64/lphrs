"""GenerateCandidates from interventions.yaml and the Food.com recipe sample (build plan 4.6).

Every candidate has: id, domain, title, tags, intensity (1-5), minutes, effort, targets,
goals, contraindications, source ("template" / "recipe") and, for recipes, nutrient PDVs.
"""

from __future__ import annotations

import ast
import re

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, RAW_DIR, load_yaml

SEED = 42
NUTRITION_COLUMNS = ["calories", "fat_pdv", "sugar_pdv", "sodium_pdv", "protein_pdv", "sat_fat_pdv", "carbs_pdv"]
CANDIDATE_COLUMNS = ["id", "domain", "title", "tags", "intensity", "minutes", "effort", "targets", "goals",
                     "contraindications", "source"] + NUTRITION_COLUMNS
MAX_CALORIES = 1500           # per serving; larger values are whole-batch figures in the source data
RECIPE_CACHE = PROCESSED_DIR / "recipes.parquet"


# ---------------------------------------------------------------- keyword tagging

def _strip(text: str, excludes: list[str]) -> str:
    for ex in excludes:
        text = text.replace(ex, " ")
    return text


def match_keywords(texts: list[str], keywords: list[str], excludes: list[str] = ()) -> bool:
    """True if any text contains a keyword after removing excluded phrases
    (so "eggplant" is not egg, but "eggplant and egg" still is)."""
    for t in texts:
        t = _strip(str(t).lower(), excludes)
        if any(k in t for k in keywords):
            return True
    return False


def allergen_tags(ingredients: list[str], allergens: dict) -> list[str]:
    return [f"allergen:{name}" for name, spec in allergens.items()
            if match_keywords(ingredients, spec["ingredient_keywords"], spec.get("exclude_keywords", []))]


def interaction_tags(ingredients: list[str], interactions: dict) -> list[str]:
    return [tag for tag, kws in interactions.items() if match_keywords(ingredients, kws)]


def diet_tags(row: dict, ingredients: list[str], rules: dict, exceptions=()) -> list[str]:
    out = []
    for tag, rule in rules.items():
        if "field" in rule:
            v = row[rule["field"]]
            if ("max" in rule and v <= rule["max"]) or ("min" in rule and v >= rule["min"]):
                out.append(tag)
        elif "exclude_ingredient_keywords" in rule:
            if not match_keywords(ingredients, rule["exclude_ingredient_keywords"], exceptions):
                out.append(tag)
    return out


# ---------------------------------------------------------------- templates

def load_templates() -> pd.DataFrame:
    df = pd.DataFrame(load_yaml("interventions")["interventions"])
    df["source"] = "template"
    for c in NUTRITION_COLUMNS:
        df[c] = np.nan
    return df[CANDIDATE_COLUMNS]


# ---------------------------------------------------------------- recipes

def _title(name: str) -> str:
    name = re.sub(r"\s+", " ", str(name)).strip()
    return f"Cook: {name}"


def build_recipe_pool(raw: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Sample recipes, parse nutrition, derive allergen / interaction / diet tags."""
    rcfg = load_yaml("interventions")["recipes"]
    safety = load_yaml("safety_rules")
    df = raw[raw["name"].notna() & raw["minutes"].between(rcfg["min_minutes"], rcfg["max_minutes"])].copy()
    # wellness recipes only: drop alcohol and processed meat
    exceptions = rcfg.get("keyword_exceptions", [])
    allowed = [not match_keywords(ast.literal_eval(i), rcfg["exclude_ingredient_keywords"], exceptions)
               for i in df["ingredients"]]
    df = df[allowed]
    meal_tags = df["tags"].map(ast.literal_eval).map(set)
    inc, exc = set(rcfg["include_meal_tags"]), set(rcfg["exclude_meal_tags"])
    df = df[meal_tags.map(lambda t: bool(t & inc) and not (t & exc))]
    nut = pd.DataFrame(df["nutrition"].map(ast.literal_eval).tolist(), columns=NUTRITION_COLUMNS, index=df.index)
    df = df.join(nut)
    df = df[df["calories"].between(1, MAX_CALORIES)]
    df = df.sample(n=min(rcfg["sample_size"], len(df)), random_state=seed).sort_values("id")

    rows = []
    for r in df.itertuples(index=False):
        ingredients = ast.literal_eval(r.ingredients)
        rd = r._asdict()
        d_tags = diet_tags(rd, ingredients, rcfg["diet_tags"], exceptions)
        a_tags = allergen_tags(ingredients, safety["allergens"])
        i_tags = interaction_tags(ingredients, safety["interaction_tags"])
        targets = sorted({m for t in d_tags for m in rcfg["diet_tag_targets"].get(t, [])})
        goals = sorted({g for t in d_tags for g in rcfg["diet_tag_goals"].get(t, [])})
        rows.append({
            "id": f"R{r.id}", "domain": rcfg["domain"], "title": _title(r.name),
            "tags": ["recipe"] + d_tags + i_tags, "intensity": rcfg["intensity"], "minutes": int(r.minutes),
            "effort": round(min(1.0, r.minutes * rcfg["effort_per_minute"]), 3),
            "targets": targets, "goals": goals, "contraindications": a_tags + i_tags, "source": "recipe",
            "ingredients": ingredients,
            **{c: float(rd[c]) for c in NUTRITION_COLUMNS},
        })
    return pd.DataFrame(rows)


def load_recipe_pool(rebuild: bool = False) -> pd.DataFrame:
    """Cached recipe sample (data/processed/recipes.parquet); built from the raw CSV if missing."""
    if RECIPE_CACHE.exists() and not rebuild:
        return pd.read_parquet(RECIPE_CACHE)
    raw = pd.read_csv(RAW_DIR / "recipes" / "RAW_recipes.csv",
                      usecols=["name", "id", "minutes", "tags", "nutrition", "ingredients"])
    pool = build_recipe_pool(raw)
    RECIPE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(RECIPE_CACHE, index=False)
    return pool


# ---------------------------------------------------------------- candidates

def generate_candidates(profile: dict | None = None, user_context: dict | None = None,
                        recipes: pd.DataFrame | None = None) -> pd.DataFrame:
    """All candidates for a patient: every template plus the recipe pool.

    Selection by relevance happens in the ranker and safety in SafetyFilter, so the pool is
    the same for everyone; profile and user_context are accepted for interface stability.
    """
    recipes = load_recipe_pool() if recipes is None else recipes
    parts = [load_templates(), recipes.reindex(columns=CANDIDATE_COLUMNS)]
    out = pd.concat(parts, ignore_index=True)
    for c in ("tags", "targets", "goals", "contraindications"):
        out[c] = out[c].map(lambda v: list(v) if v is not None and not isinstance(v, float) else [])
    return out
