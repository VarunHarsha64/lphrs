import pandas as pd
import pytest

from lphrs import candidates as C
from lphrs import safety as S
from lphrs.config import load_yaml

SAFETY = load_yaml("safety_rules")
RULES = {r["id"]: r for r in SAFETY["rules"]}


def raw_recipes():
    rows = [
        # id, name, minutes, nutrition [cal, fat, sugar, sodium, protein, satfat, carbs], ingredients
        (1, "peanut noodles", 20, [400, 20, 10, 30, 35, 10, 15], ["peanut butter", "rice noodles", "soy sauce"]),
        (2, "eggplant parmesan", 45, [350, 25, 12, 20, 20, 30, 10], ["eggplant", "parmesan cheese", "flour"]),
        (3, "plain eggplant", 30, [120, 5, 5, 5, 4, 2, 4], ["eggplant", "olive oil", "garlic"]),
        (4, "kale salad", 10, [150, 10, 4, 6, 8, 5, 3], ["kale", "lemon juice", "olive oil"]),
        (5, "grapefruit bowl", 5, [100, 0, 40, 0, 2, 0, 8], ["grapefruit", "honey"]),
        (6, "salty soup", 30, [200, 5, 3, 80, 10, 5, 5], ["chicken broth", "salt", "carrots"]),
        (7, "slow roast", 300, [500, 30, 2, 20, 60, 40, 2], ["beef"]),                 # > 60 min
        (8, "shrimp tacos", 25, [450, 15, 6, 22, 40, 10, 12], ["shrimp", "corn tortilla", "lime"]),
        (9, "whole batch cake", 50, [9000, 300, 900, 100, 50, 400, 300], ["flour", "sugar", "butter"]),  # whole batch
        (10, "quick dip", 10, [100, 5, 2, 5, 3, 2, 2], ["yogurt", "dill"]),                               # dip tag
        (11, "vodka pasta", 30, [500, 20, 5, 20, 20, 10, 20], ["pasta", "vodka", "tomatoes"]),              # alcohol
        (12, "garnish", 2, [10, 0, 0, 0, 0, 0, 0], ["parsley"]),                                             # < 5 min
        (13, "toasty dogs", 10, [300, 20, 5, 20, 30, 20, 5], ["hot dogs", "buns"]),                           # processed meat
        (14, "ginger crumb cod", 20, [300, 10, 5, 10, 40, 5, 5],
         ["extra virgin olive oil", "fresh ginger", "breadcrumbs", "cod", "red wine vinegar"]),               # NOT alcohol
        (15, "graham parfait", 10, [250, 5, 15, 5, 10, 5, 10], ["graham crackers", "yogurt"]),                # NOT ham
    ]
    meal_tags = {10: ["appetizers", "dips"]}
    return pd.DataFrame([{"id": i, "name": n, "minutes": m, "nutrition": str(nu), "ingredients": str(ing),
                          "tags": str(meal_tags.get(i, ["main-dish"]))}
                         for i, n, m, nu, ing in rows])


@pytest.fixture(scope="module")
def pool():
    return C.build_recipe_pool(raw_recipes()).set_index("id", drop=False)


@pytest.fixture(scope="module")
def cands(pool):
    return C.generate_candidates(recipes=pool.reset_index(drop=True))


# ---------------------------------------------------------------- tagging

def test_recipe_filters(pool):
    assert "R7" not in pool.index           # longer than 60 minutes
    assert "R9" not in pool.index           # whole-batch nutrition
    assert {"R10", "R11", "R12", "R13"}.isdisjoint(pool.index)   # dip, alcohol, under 5 minutes, hot dogs
    assert {"R14", "R15"} <= set(pool.index)                     # ginger / virgin / crumbs / graham are fine
    assert "vegetarian" in pool.loc["R15", "tags"]               # graham is not ham
    assert len(pool) == 9


def test_allergen_tags_with_exclusions(pool):
    peanut = set(pool.loc["R1", "contraindications"])
    assert "allergen:peanut" in peanut and "allergen:soy" in peanut
    assert "allergen:milk" not in peanut                       # peanut butter is not milk
    assert "allergen:wheat" not in peanut                      # rice noodles excluded
    parm = set(pool.loc["R2", "contraindications"])
    assert {"allergen:milk", "allergen:wheat"} <= parm and "allergen:egg" not in parm
    assert pool.loc["R3", "contraindications"] == []           # eggplant is not egg
    shrimp = set(pool.loc["R8", "contraindications"])
    assert "allergen:shellfish" in shrimp and "allergen:wheat" not in shrimp  # corn tortilla


def test_interaction_and_diet_tags(pool):
    assert "high_vitamin_k" in pool.loc["R4", "tags"] and "high_vitamin_k" in pool.loc["R4", "contraindications"]
    assert "grapefruit" in pool.loc["R5", "contraindications"]
    assert {"low_sodium", "low_sugar", "vegetarian"} <= set(pool.loc["R4", "tags"])
    assert "vegetarian" not in pool.loc["R6", "tags"]          # chicken broth
    assert "high_protein" in pool.loc["R1", "tags"]
    assert {"systolic_bp", "glucose"} <= set(pool.loc["R4", "targets"])


def test_candidates_have_required_fields(cands):
    assert set(C.CANDIDATE_COLUMNS) <= set(cands.columns)
    assert cands["id"].is_unique
    assert (cands["source"] == "template").sum() == len(load_yaml("interventions")["interventions"])
    assert cands["intensity"].between(1, 5).all() and cands["effort"].between(0, 1).all()
    for c in ("tags", "targets", "goals", "contraindications"):
        assert cands[c].map(lambda v: isinstance(v, list)).all()


# ---------------------------------------------------------------- acceptance

def test_peanut_allergy_and_knee_injury(cands):
    """Acceptance (Phase 4): zero peanut recipes and zero high-impact exercises."""
    facts = S.SafetyFacts(allergens=S.allergens_from_descriptions(["Peanut (substance)"]),
                          injuries={"knee"})
    kept, log = S.safety_filter(cands, facts)
    assert facts.allergens == {"peanut"}
    peanut = kept["contraindications"].map(lambda c: "allergen:peanut" in c)
    assert not peanut.any()
    assert not kept["tags"].map(lambda t: "high_impact" in t).any()
    assert not kept["tags"].map(lambda t: "knee_load" in t).any()
    assert {"R1", "N14"} <= set(log[log["rule_id"] == "allergy_match"]["candidate_id"])
    assert {"F04", "F05", "F11"} <= set(log[log["rule_id"] == "joint_high_impact"]["candidate_id"])
    assert {"F08", "F14"} <= set(log[log["rule_id"] == "knee_load"]["candidate_id"])
    assert "F01" in set(kept["id"])                            # brisk walk is still allowed
    assert set(log.columns) == set(S.DROP_LOG_COLUMNS)


# ---------------------------------------------------------------- individual rules

def run(cands, **facts):
    kept, log = S.safety_filter(cands, S.SafetyFacts(**facts))
    return set(kept["id"]), log


def test_no_facts_drops_nothing(cands):
    kept, log = run(cands)
    assert len(kept) == len(cands) and log.empty


def test_hypertension_sodium_cap(cands):
    kept, log = run(cands, conditions=["Essential hypertension (disorder)"])
    assert "R6" not in kept and "R1" not in kept                # sodium 80 % and 30 % DV
    assert "R2" in kept and "N01" in kept                        # 20 % DV; templates unaffected
    assert set(log["rule_id"]) == {"hypertension_sodium_cap"}


def test_diabetes_sugar_cap(cands):
    kept, _ = run(cands, conditions=["Prediabetes (finding)"])
    assert "R5" not in kept and "R4" in kept


def test_cardiac_intensity_cap(cands):
    kept, log = run(cands, conditions=["Chronic congestive heart failure (disorder)"])
    dropped = set(log["candidate_id"])
    assert {"F04", "F05", "F10", "F11"} <= dropped and "F06" in kept   # F06 is intensity 3
    assert (cands.set_index("id").loc[list(dropped), "intensity"] >= 4).all()


def test_cardiac_rule_ignores_imaging_finding(cands):
    kept, log = run(cands, conditions=["Abnormal findings diagnostic imaging heart+coronary circulat (finding)"])
    assert log.empty


def test_warfarin_and_statin(cands):
    kept, log = run(cands, medications=["Warfarin Sodium 5 MG Oral Tablet", "Simvastatin 20 MG Oral Tablet"])
    assert {"R4", "N03"}.isdisjoint(kept)                       # vitamin K
    assert {"R5", "N13"}.isdisjoint(kept)                       # grapefruit
    assert set(log["rule_id"]) == {"warfarin_vitamin_k", "statin_grapefruit"}


def test_nystatin_is_not_a_statin(cands):
    kept, log = run(cands, medications=["Nystatin 100000 UNT/ML Oral Suspension"])
    assert log.empty


def test_back_injury_and_osteoarthritis(cands):
    kept, log = run(cands, injuries={"back"})
    assert "F10" not in kept and "F05" not in kept               # heavy spinal load, high impact
    assert "F08" in kept                                          # knee load only
    kept_oa, _ = run(cands, conditions=["Osteoarthritis of knee (disorder)"])
    assert "F05" not in kept_oa and "F14" not in kept_oa
    kept_hand, log_hand = run(cands, conditions=["Localized  primary osteoarthritis of the hand (disorder)"])
    assert log_hand.empty


def test_synthea_allergy_mapping():
    got = S.allergens_from_descriptions([
        "Shellfish (substance)", "Glycine max (substance)", "Cow's milk (substance)",
        "Eggs (edible) (substance)", "Tree pollen (substance)", "Allergic disposition (finding)"])
    assert got == {"shellfish", "soy", "milk", "egg"}            # shellfish is not fish; pollen is not nut


def test_every_drop_is_logged_with_rule(cands):
    facts = S.SafetyFacts(allergens={"milk", "peanut"}, conditions=["Essential hypertension (disorder)"],
                          injuries={"knee"})
    kept, log = S.safety_filter(cands, facts)
    assert len(kept) + log["candidate_id"].nunique() == len(cands)
    assert log["reason"].str.len().gt(0).all()
    assert len(log) >= log["candidate_id"].nunique()             # one row per rule that fired


@pytest.mark.parametrize("ingredient", ["basa fillets", "hake fillets", "orange roughy fillets", "fillet of sole",
                                        "mahi mahi", "grouper fillets", "sea bass", "squid", "seafood"])
def test_seafood_names_are_tagged(ingredient):
    tags = C.allergen_tags([ingredient], SAFETY["allergens"])
    assert {"allergen:fish", "allergen:shellfish"} & set(tags), ingredient


@pytest.mark.parametrize("ingredient", ["spike seasoning", "seafood seasoning", "peeled tomatoes", "balsamic vinegar"])
def test_seafood_false_positives_excluded(ingredient):
    tags = C.allergen_tags([ingredient], SAFETY["allergens"])
    assert not {"allergen:fish", "allergen:shellfish"} & set(tags), ingredient
