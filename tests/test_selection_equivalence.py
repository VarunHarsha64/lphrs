"""The vectorised select_set must pick exactly what the straightforward implementation picks."""

import numpy as np
import pandas as pd
import pytest

from lphrs import ranker as RK


def reference_select_set(scored, budget, k, alpha, max_per_domain, table):
    """The original (slow, obviously-correct) implementation."""
    def can_fill(by_minutes, exclude, slots, per_domain, budget):
        if slots <= 0:
            return budget >= 0
        used, total, counts = 0, 0.0, dict(per_domain)
        for idx, minutes, domain in zip(by_minutes.index, by_minutes["minutes"], by_minutes["domain"]):
            if idx == exclude or counts.get(domain, 0) >= max_per_domain:
                continue
            total += minutes
            if total > budget:
                return False
            counts[domain] = counts.get(domain, 0) + 1
            used += 1
            if used == slots:
                return True
        return False

    remaining = scored.reset_index(drop=True)
    selected, per_domain = [], {}
    while len(selected) < k and len(remaining):
        ok = (remaining["minutes"] <= budget) & remaining["domain"].map(lambda d: per_domain.get(d, 0) < max_per_domain)
        pool = remaining[ok]
        if pool.empty:
            break
        bonus = np.array([sum(RK.synergy(t, s["tags"], table) for s in selected) for t in pool["tags"]])
        gains = pool["score"].to_numpy() + alpha * bonus
        order = np.argsort(-gains, kind="stable")
        if gains[order[0]] <= 0:
            break
        by_minutes = remaining.sort_values("minutes", kind="stable")
        i = None
        for slots in range(k - len(selected) - 1, -1, -1):
            for j in order:
                d = pool.iloc[j]["domain"]
                if gains[j] > 0 and can_fill(by_minutes, pool.index[j], slots,
                                             {**per_domain, d: per_domain.get(d, 0) + 1},
                                             budget - pool.iloc[j]["minutes"]):
                    i = j
                    break
            if i is not None:
                break
        if i is None:
            break
        pick = pool.iloc[i].to_dict()
        pick["gain"] = float(gains[i])
        selected.append(pick)
        budget -= pick["minutes"]
        per_domain[pick["domain"]] = per_domain.get(pick["domain"], 0) + 1
        remaining = remaining.drop(index=pool.index[i])
    return [s["id"] for s in selected], [s["gain"] for s in selected]


TAGS = ["walking", "evening_walk", "earlier_bedtime", "late_caffeine", "sleep_hygiene", "recipe",
        "aerobic", "low_sodium", "dash", "hiit", "calorie_deficit", "strength", "other"]
DOMAINS = ["nutrition", "fitness", "sleep", "lifestyle", "wellness", "preventive_care"]


@pytest.mark.parametrize("seed", range(200))
def test_matches_reference(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 60))
    df = pd.DataFrame({
        "id": [f"c{i}" for i in range(n)],
        "domain": rng.choice(DOMAINS[: rng.integers(2, 7)], n),
        "minutes": rng.choice([1, 2, 5, 10, 15, 20, 30, 45], n),
        "tags": [list(rng.choice(TAGS, rng.integers(0, 4), replace=False)) for _ in range(n)],
        # coarse scores create ties, negative scores exercise the stop rule
        "score": np.round(rng.uniform(-0.2, 1.0, n), 1),
    }).sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    budget = int(rng.choice([5, 12, 20, 30, 60, 90]))
    table = RK.load_synergy()
    k, alpha, cap = 5, 0.3, int(rng.choice([1, 2]))
    ref_ids, ref_gains = reference_select_set(df, budget, k, alpha, cap, table)
    got = RK.select_set(df, budget, k=k, alpha=alpha, max_per_domain=cap, table=table)
    got_ids = got["id"].tolist() if len(got) else []
    assert got_ids == ref_ids
    if ref_ids:
        assert got["gain"].to_numpy() == pytest.approx(np.array(ref_gains))
