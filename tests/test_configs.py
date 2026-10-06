"""Consistency checks across the YAML configs."""

from pathlib import Path

import pytest
import yaml

CONFIG = Path(__file__).resolve().parents[1] / "config"


def load(name):
    with open(CONFIG / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def cfg():
    return {n: load(f"{n}.yaml") for n in ("metrics", "safety_rules", "interventions", "synergy", "weights")}


def test_weights_sum_to_one(cfg):
    assert sum(cfg["weights"]["weights"].values()) == pytest.approx(1.0)
    assert cfg["weights"]["seed"] == 42


def test_metrics_have_valid_ranges_and_directions(cfg):
    for key, m in cfg["metrics"]["metrics"].items():
        lo, hi = m["valid_range"]
        assert lo < hi, key
        assert m["bad_direction"] in ("up", "down"), key
        if m["threshold"] is not None:
            assert lo <= m["threshold"] <= hi, key
        assert set(m["goals"]) <= set(cfg["metrics"]["goals"]), key


def test_interventions_well_formed(cfg):
    items = cfg["interventions"]["interventions"]
    metrics = set(cfg["metrics"]["metrics"])
    goals = set(cfg["metrics"]["goals"])
    domains = {"nutrition", "fitness", "sleep", "lifestyle", "preventive_care", "wellness"}
    assert 60 <= len(items) <= 80
    assert len({i["id"] for i in items}) == len(items)
    assert {i["domain"] for i in items} == domains
    for i in items:
        assert i["domain"] in domains, i["id"]
        assert 1 <= i["intensity"] <= 5, i["id"]
        assert i["minutes"] >= 0, i["id"]
        assert 0 <= i["effort"] <= 1, i["id"]
        assert set(i["targets"]) <= metrics, i["id"]
        assert set(i["goals"]) <= goals, i["id"]


def test_contraindications_are_known_to_safety_rules(cfg):
    rules = cfg["safety_rules"]
    filtered_tags = set(rules["interaction_tags"])
    for r in rules["rules"]:
        filtered_tags |= set(r["drop_when"].get("tags_any", []))
    allergens = {f"allergen:{a}" for a in rules["allergens"]}
    for i in cfg["interventions"]["interventions"]:
        for c in i["contraindications"]:
            assert c in filtered_tags | allergens, (i["id"], c)
        # a contraindication tag should also appear in tags unless it is an allergen
        for c in i["contraindications"]:
            if not c.startswith("allergen:"):
                assert c in i["tags"], (i["id"], c)


def test_synergy_pairs_in_range_and_use_known_tags(cfg):
    tags = set()
    for i in cfg["interventions"]["interventions"]:
        tags |= set(i["tags"])
    tags |= set(cfg["interventions"]["recipes"]["diet_tags"]) | {"recipe"}   # tags added to every recipe
    tags |= set(cfg["safety_rules"]["interaction_tags"])
    seen = set()
    for a, b, v in cfg["synergy"]["pairs"]:
        assert -1 <= v <= 1
        assert a in tags and b in tags, (a, b)
        pair = frozenset((a, b))
        assert pair not in seen, (a, b)
        seen.add(pair)


def test_safety_rules_well_formed(cfg):
    ids = [r["id"] for r in cfg["safety_rules"]["rules"]]
    assert len(ids) == len(set(ids))
    trigger_types = {"allergy", "condition", "medication", "injury"}
    drop_kinds = {"allergen_overlap", "tags_any", "nutrient_above", "intensity_at_least"}
    for r in cfg["safety_rules"]["rules"]:
        assert r["description"]
        assert all(t["type"] in trigger_types for t in r["trigger"]["any_of"]), r["id"]
        assert set(r["drop_when"]) <= drop_kinds, r["id"]
    for kw in ("warfarin",):
        assert any(kw in t.get("keywords", []) for r in cfg["safety_rules"]["rules"] for t in r["trigger"]["any_of"])
