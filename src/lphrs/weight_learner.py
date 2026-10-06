"""Learns ranking weights w1..w7 from feedback logs (build plan 6.2, step 3).

Logistic regression of accepted (ACCEPTED/COMPLETED) on the seven score terms. Coefficients
become weights: negatives are clipped to 0 for R..H; cost enters the score with a minus sign,
so w7 = max(0, -coef_C). The weights are then normalised to sum to 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression

from .config import CONFIG_DIR
from .ranker import TERMS, WEIGHT_KEYS

SEED = 42
LEARNED_PATH = CONFIG_DIR / "weights_learned.yaml"


def fit_coefficients(log: pd.DataFrame) -> dict[str, float]:
    model = LogisticRegression(max_iter=2000, random_state=SEED)
    model.fit(log[TERMS].to_numpy(float), log["accepted"].to_numpy(int))
    out = dict(zip(TERMS, model.coef_[0].astype(float)))
    out["intercept"] = float(model.intercept_[0])
    return out


def coefficients_to_weights(coef: dict[str, float]) -> dict[str, float]:
    raw = {t: max(0.0, coef[t]) for t in TERMS if t != "C"}
    raw["C"] = max(0.0, -coef["C"])
    total = sum(raw.values())
    if total == 0:
        return {t: 1 / len(TERMS) for t in TERMS}
    return {t: raw[t] / total for t in TERMS}


def learn_weights(log: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    """(weights summing to 1, raw coefficients)."""
    coef = fit_coefficients(log)
    return coefficients_to_weights(coef), coef


def save_learned_weights(weights: dict[str, float], meta: dict, path=LEARNED_PATH) -> None:
    doc = {"note": "Learned on SIMULATED users (scripts/05_simulate_feedback.py); reflects the simulator's "
                   "assumptions, not real-world effectiveness.",
           "weights": {WEIGHT_KEYS[t]: round(float(w), 4) for t, w in weights.items()},
           **meta}
    path.write_text(yaml.safe_dump(doc, sort_keys=False))


def load_learned_weights(path=LEARNED_PATH) -> dict[str, float] | None:
    if not path.exists():
        return None
    w = yaml.safe_load(path.read_text())["weights"]
    return {t: float(w[k]) for t, k in WEIGHT_KEYS.items()}


def weights_as_array(w: dict[str, float]) -> np.ndarray:
    return np.array([w[t] for t in TERMS])
