"""Cardiovascular risk classifier (build plan 6.1): cleaning, training, calibration, prediction,
and mapping Synthea patients onto the Kaggle CVD feature space."""

from __future__ import annotations

import itertools

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import MODELS_DIR, PROCESSED_DIR, RAW_DIR

SEED = 42
TARGET = "cardio"
FEATURES = ["age_years", "gender", "height", "weight", "bmi", "ap_hi", "ap_lo",
            "cholesterol", "gluc", "smoke", "alco", "active"]
MODEL_PATH = MODELS_DIR / "cvd_risk.joblib"
RISK_PATH = PROCESSED_DIR / "cvd_risk.parquet"

XGB_GRID = {"max_depth": [3, 4, 5, 6], "learning_rate": [0.05, 0.1], "n_estimators": [200, 400]}

# Synthea lab value -> dataset level (1 normal, 2 above normal, 3 well above normal)
CHOLESTEROL_BINS = (200, 240)   # total cholesterol mg/dL
GLUCOSE_BINS = (100, 126)       # glucose mg/dL


# ---------------------------------------------------------------- data

def load_cvd() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "cvd" / "cardio_train.csv", sep=";")


def clean_cvd(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop impossible BP (build plan), implausible height/weight (agreed addition); add age, BMI."""
    report = {"input_rows": len(df)}
    bp = df["ap_hi"].between(70, 250) & df["ap_lo"].between(40, 150) & (df["ap_lo"] < df["ap_hi"])
    report["dropped_bp"] = int((~bp).sum())
    df = df[bp]
    hw = df["height"].between(120, 220) & df["weight"].between(30, 200)
    report["dropped_height_weight"] = int((~hw).sum())
    df = df[hw].copy()
    df["age_years"] = df["age"] / 365.25
    df["bmi"] = df["weight"] / (df["height"] / 100) ** 2
    report["output_rows"] = len(df)
    report["positive_rate"] = float(df[TARGET].mean())
    return df.reset_index(drop=True), report


def split(df: pd.DataFrame, seed: int = SEED):
    """70 / 15 / 15 stratified split -> (train, val, test)."""
    train, rest = train_test_split(df, test_size=0.30, stratify=df[TARGET], random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest[TARGET], random_state=seed)
    return train, val, test


# ---------------------------------------------------------------- models

def make_logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))


def make_xgb(**params) -> XGBClassifier:
    return XGBClassifier(random_state=SEED, n_jobs=-1, eval_metric="logloss", tree_method="hist", **params)


def evaluate(y, p, threshold: float = 0.5) -> dict:
    pred = (p >= threshold).astype(int)
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "accuracy": float(accuracy_score(y, pred)), "f1": float(f1_score(y, pred)),
            "brier": float(brier_score_loss(y, p))}


def tune_xgb(train: pd.DataFrame, val: pd.DataFrame, grid: dict = XGB_GRID) -> tuple[dict, pd.DataFrame]:
    """Light grid search on the validation set, ranked by ROC-AUC."""
    rows = []
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        m = make_xgb(**params).fit(train[FEATURES], train[TARGET])
        rows.append({**params, "val_roc_auc": roc_auc_score(val[TARGET], m.predict_proba(val[FEATURES])[:, 1])})
    res = pd.DataFrame(rows).sort_values("val_roc_auc", ascending=False).reset_index(drop=True)
    best = {k: (int(res.loc[0, k]) if k != "learning_rate" else float(res.loc[0, k])) for k in keys}
    return best, res


def calibrate(model, X_val, y_val) -> tuple[CalibratedClassifierCV, str, dict]:
    """Fit isotonic and Platt (sigmoid) on the validation set; keep the lower validation Brier."""
    briers, fitted = {}, {}
    for method in ("isotonic", "sigmoid"):
        cal = CalibratedClassifierCV(FrozenEstimator(model), method=method).fit(X_val, y_val)
        fitted[method] = cal
        briers[method] = float(brier_score_loss(y_val, cal.predict_proba(X_val)[:, 1]))
    best = min(briers, key=briers.get)
    return fitted[best], best, briers


def train_all(df: pd.DataFrame, seed: int = SEED, grid: dict | None = None) -> dict:
    """Train LR and XGBoost, pick by validation ROC-AUC, calibrate, evaluate on test."""
    train, val, test = split(df, seed)
    Xtr, ytr = train[FEATURES], train[TARGET]
    Xva, yva = val[FEATURES], val[TARGET]
    Xte, yte = test[FEATURES], test[TARGET]

    lr = make_logreg().fit(Xtr, ytr)
    best_params, grid = tune_xgb(train, val, grid or XGB_GRID)
    xgb = make_xgb(**best_params).fit(Xtr, ytr)

    val_auc = {"logistic_regression": roc_auc_score(yva, lr.predict_proba(Xva)[:, 1]),
               "xgboost": roc_auc_score(yva, xgb.predict_proba(Xva)[:, 1])}
    chosen = max(val_auc, key=val_auc.get)
    base = xgb if chosen == "xgboost" else lr
    calibrated, method, cal_briers = calibrate(base, Xva, yva)

    preds = {"logistic_regression": lr.predict_proba(Xte)[:, 1],
             "xgboost": xgb.predict_proba(Xte)[:, 1],
             f"{chosen}_calibrated": calibrated.predict_proba(Xte)[:, 1]}
    return {
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "xgb_best_params": best_params, "xgb_grid": grid,
        "val_roc_auc": val_auc, "chosen": chosen, "calibration_method": method,
        "calibration_val_brier": cal_briers,
        "test_metrics": {k: evaluate(yte, p) for k, p in preds.items()},
        "test_predictions": preds, "y_test": yte.to_numpy(),
        "models": {"logistic_regression": lr, "xgboost": xgb, "calibrated": calibrated},
    }


def save_model(calibrated, chosen: str, method: str, path=MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": calibrated, "features": FEATURES, "chosen": chosen,
                 "calibration_method": method, "seed": SEED}, path)


def load_model(path=MODEL_PATH) -> dict:
    return joblib.load(path)


def predict_risk(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    return bundle["model"].predict_proba(X[bundle["features"]])[:, 1]


# ---------------------------------------------------------------- Synthea mapping

def level(values: pd.Series, bins: tuple[float, float]) -> pd.Series:
    """1 / 2 / 3 levels; missing -> 1."""
    out = pd.Series(1, index=values.index)
    out[values >= bins[0]] = 2
    out[values >= bins[1]] = 3
    return out


def synthea_features(profile_metrics: pd.DataFrame, patients: pd.DataFrame,
                     context: pd.DataFrame) -> pd.DataFrame:
    """Map each cohort patient's latest values into the CVD dataset's features.

    Height comes from latest weight and BMI (height_m = sqrt(weight / BMI)).
    Missing labs default to level 1 and are flagged; missing vitals stay NaN and are flagged.
    """
    cur = profile_metrics.pivot_table(index="patient_id", columns="metric", values="current", aggfunc="first")
    ids = context["patient_id"]
    cur = cur.reindex(ids)
    p = patients.set_index("patient_id").reindex(ids)
    c = context.set_index("patient_id").reindex(ids)
    get = lambda m: cur[m] if m in cur.columns else pd.Series(np.nan, index=cur.index)  # noqa: E731

    X = pd.DataFrame(index=ids)
    X["age_years"] = p["age"].to_numpy()
    X["gender"] = p["gender"].map({"F": 1, "M": 2}).to_numpy()
    X["weight"] = get("body_weight")
    X["bmi"] = get("bmi")
    X["height"] = 100 * np.sqrt(X["weight"] / X["bmi"])
    X["ap_hi"] = get("systolic_bp")
    X["ap_lo"] = get("diastolic_bp")
    X["cholesterol"] = level(get("total_cholesterol"), CHOLESTEROL_BINS)
    X["gluc"] = level(get("glucose"), GLUCOSE_BINS)
    for col in ("smoke", "alco", "active"):
        X[col] = c[col].to_numpy()
    X["cholesterol_imputed"] = get("total_cholesterol").isna()
    X["gluc_imputed"] = get("glucose").isna()
    X["vitals_missing"] = X[["weight", "bmi", "ap_hi", "ap_lo"]].isna().any(axis=1)
    X["age_outside_training"] = ~X["age_years"].between(29, 65)
    return X.reset_index().rename(columns={"index": "patient_id"})


def load_cvd_risk(path=RISK_PATH) -> dict[str, float]:
    """patient_id -> calibrated CVD risk (empty if not yet computed)."""
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    return dict(zip(df["patient_id"], df["cvd_risk"]))
