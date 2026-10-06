import numpy as np
import pandas as pd
import pytest

from lphrs import risk_model as RM


def cvd_rows(n=400, seed=0):
    rng = np.random.default_rng(seed)
    ap_hi = rng.integers(100, 180, n)
    df = pd.DataFrame({
        "id": range(n), "age": rng.integers(30 * 365, 65 * 365, n), "gender": rng.integers(1, 3, n),
        "height": rng.integers(150, 190, n), "weight": rng.uniform(50, 110, n).round(1),
        "ap_hi": ap_hi, "ap_lo": ap_hi - rng.integers(30, 50, n),
        "cholesterol": rng.integers(1, 4, n), "gluc": rng.integers(1, 4, n),
        "smoke": rng.integers(0, 2, n), "alco": rng.integers(0, 2, n), "active": rng.integers(0, 2, n),
    })
    logit = 0.06 * (df["ap_hi"] - 140) + 0.5 * (df["cholesterol"] - 2)
    df["cardio"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


def test_clean_drops_impossible_bp_and_body_size():
    df = cvd_rows(10)
    df.loc[0, "ap_hi"] = 16020            # > 250
    df.loc[1, "ap_lo"] = -70              # < 40
    df.loc[2, ["ap_hi", "ap_lo"]] = [120, 130]  # lo >= hi
    df.loc[3, "height"] = 55              # < 120 cm
    df.loc[4, "weight"] = 10              # < 30 kg
    out, rep = RM.clean_cvd(df)
    assert rep["dropped_bp"] == 3 and rep["dropped_height_weight"] == 2 and len(out) == 5
    assert out["age_years"].between(29, 66).all()
    row = out.iloc[0]
    assert row["bmi"] == pytest.approx(row["weight"] / (row["height"] / 100) ** 2)


def test_split_is_70_15_15_stratified():
    df, _ = RM.clean_cvd(cvd_rows(1000))
    tr, va, te = RM.split(df)
    assert len(tr) + len(va) + len(te) == len(df)
    assert len(tr) / len(df) == pytest.approx(0.70, abs=0.01)
    assert len(va) == pytest.approx(len(te), abs=1)
    for part in (tr, va, te):
        assert part["cardio"].mean() == pytest.approx(df["cardio"].mean(), abs=0.03)


def test_levels_from_lab_values():
    v = pd.Series([150, 199.9, 200, 239, 240, np.nan])
    assert RM.level(v, RM.CHOLESTEROL_BINS).tolist() == [1, 1, 2, 2, 3, 1]


def test_synthea_feature_mapping():
    pm = pd.DataFrame([
        ("a", "body_weight", 80.0), ("a", "bmi", 24.69), ("a", "systolic_bp", 150.0), ("a", "diastolic_bp", 95.0),
        ("a", "total_cholesterol", 245.0), ("a", "glucose", 110.0),
        ("b", "body_weight", 60.0), ("b", "bmi", 22.0), ("b", "systolic_bp", 115.0), ("b", "diastolic_bp", 75.0),
    ], columns=["patient_id", "metric", "current"])
    pats = pd.DataFrame({"patient_id": ["a", "b"], "age": [55.0, 20.0], "gender": ["M", "F"]})
    ctx = pd.DataFrame({"patient_id": ["a", "b"], "smoke": [1, 0], "alco": [0, 0], "active": [0, 1]})
    X = RM.synthea_features(pm, pats, ctx).set_index("patient_id")
    assert X.loc["a", "height"] == pytest.approx(180.0, abs=0.1)
    assert X.loc["a", ["gender", "cholesterol", "gluc", "smoke"]].tolist() == [2, 3, 2, 1]
    assert X.loc["b", ["gender", "cholesterol", "gluc"]].tolist() == [1, 1, 1]
    assert X.loc["b", "cholesterol_imputed"] and X.loc["b", "gluc_imputed"]
    assert not X.loc["a", "age_outside_training"] and X.loc["b", "age_outside_training"]
    assert set(RM.FEATURES) <= set(X.columns)


def test_train_all_end_to_end_small():
    df, _ = RM.clean_cvd(cvd_rows(1500))
    small_grid = {"max_depth": [2, 3], "learning_rate": [0.1], "n_estimators": [50]}
    res = RM.train_all(df, grid=small_grid)
    assert len(res["xgb_grid"]) == 2
    assert res["chosen"] in ("logistic_regression", "xgboost")
    assert res["calibration_method"] in ("isotonic", "sigmoid")
    for m in res["test_metrics"].values():
        assert 0.5 < m["roc_auc"] <= 1 and 0 <= m["brier"] <= 0.25
    p = res["models"]["calibrated"].predict_proba(df[RM.FEATURES].head(5))[:, 1]
    assert ((p >= 0) & (p <= 1)).all()
