import numpy as np
import pandas as pd
import pytest

from lphrs import feedback as FB
from lphrs import ranker as RK
from lphrs import weight_learner as WL

UC = {"intensity_caps": {"fitness": 4, "sleep": 4}, "preference_tags": ["walking"]}
HARD = {"id": "F05", "domain": "fitness", "intensity": 5, "tags": ["hiit", "high_intensity"]}
EASY = {"id": "F01", "domain": "fitness", "intensity": 4, "tags": ["walking", "aerobic"]}


def test_accept_and_reject_update_history_and_affinity():
    s0 = FB.initial_state(UC)
    s1 = FB.adapt_profile_from_feedback(s0, EASY, "ACCEPTED")
    assert s1["by_id"]["F01"] == (1, 1) and s1["by_domain"]["fitness"] == (1, 1)
    assert s0["by_id"] == {}                                  # pure: input state untouched
    s2 = FB.adapt_profile_from_feedback(s1, EASY, "COMPLETED")
    assert s2["by_id"]["F01"] == (3, 3)
    assert "aerobic" in FB.preference_tags(s2)               # affinity 3 >= 2 joins preferences
    s3 = FB.adapt_profile_from_feedback(s0, EASY, "REJECTED")
    s3 = FB.adapt_profile_from_feedback(s3, EASY, "REJECTED")
    assert s3["by_id"]["F01"] == (0, 2)
    assert "walking" not in FB.preference_tags(s3)            # affinity -2 leaves preferences


def test_too_difficult_lowers_cap_and_completions_raise_it():
    s = FB.adapt_profile_from_feedback(FB.initial_state(UC), HARD, "TOO_DIFFICULT")
    assert s["caps"]["fitness"] == 4                          # min(cap, 5 - 1)
    s = FB.adapt_profile_from_feedback(s, EASY, "TOO_DIFFICULT")
    assert s["caps"]["fitness"] == 3
    at_cap = dict(EASY, intensity=3)
    s = FB.adapt_profile_from_feedback(s, at_cap, "COMPLETED")
    assert s["caps"]["fitness"] == 3
    s = FB.adapt_profile_from_feedback(s, at_cap, "COMPLETED")
    assert s["caps"]["fitness"] == 4                          # two completions at the cap
    with pytest.raises(ValueError):
        FB.adapt_profile_from_feedback(s, EASY, "MAYBE")


def test_apply_state_feeds_ranker():
    s = FB.adapt_profile_from_feedback(FB.initial_state(UC), EASY, "COMPLETED")
    uc, hist = FB.apply_state({"intensity_caps": UC["intensity_caps"], "preference_tags": ["walking"]}, s)
    assert RK.adherence(hist, "F01", "fitness") > 0.5
    assert "aerobic" in uc["preference_tags"]


def user(**kw):
    base = dict(a=np.array([0.6, 1.0, 0.4, 0.8, 1.0, 0.6, -0.8]), bias=-1.0,
                taste_domain={"fitness": 0.0}, taste_exercise={"walking": 0.0}, capacity=3)
    base.update(kw)
    return FB.SimulatedUser(**base)


def test_simulated_user_behaviour():
    u = user()
    good = np.array([1, 1, 1, 1, 1, 1, 0.0])
    bad = np.array([0, 0, 0, 0, 0, 0, 1.0])
    assert u.accept_probability(good, "fitness", [], 1, 0) > u.accept_probability(bad, "fitness", [], 1, 0)
    assert u.respond(good, "fitness", [], 4, np.array([0, 0, 0])) == "TOO_DIFFICULT"
    assert u.respond(good, "fitness", [], 1, np.array([0, 0.0, 0.1])) == "COMPLETED"
    assert u.respond(good, "fitness", [], 1, np.array([0, 0.0, 0.9])) == "ACCEPTED"
    assert u.respond(bad, "fitness", [], 1, np.array([0, 0.99, 0])) == "REJECTED"
    liked = user(taste_exercise={"walking": 2.0})
    assert liked.taste_match("fitness", ["walking"]) == pytest.approx(1.0)
    shifted = user(base_offset=-2.0)
    assert shifted.accept_probability(good, "fitness", [], 1, 0) < u.accept_probability(good, "fitness", [], 1, 0)


def test_weight_learner_recovers_signal():
    rng = np.random.default_rng(0)
    n = 4000
    X = rng.random((n, 7))
    true = np.array([0.0, 2.0, 0.0, 1.0, 1.0, 0.0, -1.5])
    y = rng.random(n) < 1 / (1 + np.exp(-(X @ true - 1)))
    log = pd.DataFrame(X, columns=RK.TERMS).assign(accepted=y.astype(int))
    w, coef = WL.learn_weights(log)
    assert sum(w.values()) == pytest.approx(1.0) and all(v >= 0 for v in w.values())
    assert w["G"] > w["P"] > w["R"] and w["C"] > 0.15
    assert coef["C"] < 0


def test_coefficients_to_weights_rules():
    w = WL.coefficients_to_weights({"R": -1, "G": 1, "U": 0, "P": 0, "F": 0, "H": 0, "C": -1})
    assert w == {"R": 0, "G": 0.5, "U": 0, "P": 0, "F": 0, "H": 0, "C": 0.5}
    w = WL.coefficients_to_weights({"R": 0, "G": 0, "U": 0, "P": 0, "F": 0, "H": 0, "C": 2})
    assert w == {t: 1 / 7 for t in RK.TERMS}                     # all clipped -> uniform fallback


def test_save_and_load_learned(tmp_path):
    w = {"R": 0.1, "G": 0.2, "U": 0.1, "P": 0.2, "F": 0.2, "H": 0.1, "C": 0.1}
    WL.save_learned_weights(w, {"rounds": 3}, tmp_path / "w.yaml")
    assert WL.load_learned_weights(tmp_path / "w.yaml") == pytest.approx(w)
