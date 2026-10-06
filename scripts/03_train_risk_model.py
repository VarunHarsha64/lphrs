"""Phase 5: train and calibrate the cardiovascular risk model, then score every cohort patient.

Writes:
    models/cvd_risk.joblib
    reports/risk_model_metrics.json
    reports/figures/cvd_calibration.png
    reports/figures/cvd_feature_importance.png
    data/processed/cvd_risk.parquet

    python scripts/03_train_risk_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lphrs import risk_model as RM  # noqa: E402
from lphrs.config import PROCESSED_DIR, REPORTS_DIR  # noqa: E402

FIG_DIR = REPORTS_DIR / "figures"
LABELS = {"logistic_regression": "Logistic regression", "xgboost": "XGBoost (uncalibrated)"}


def calibration_plot(res: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="Perfect calibration")
    for name, p in res["test_predictions"].items():
        frac, mean = calibration_curve(res["y_test"], p, n_bins=10, strategy="quantile")
        base = "XGBoost" if name.startswith("xgboost") else "Logistic regression"
        label = LABELS.get(name, f"{base} + {res['calibration_method']} calibration")
        brier = res["test_metrics"][name]["brier"]
        ax.plot(mean, frac, marker="o", ms=4, label=f"{label} (Brier {brier:.3f})")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed CVD rate")
    ax.set_title("CVD risk model calibration (test set)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def importance_plot(res: dict, path: Path) -> None:
    xgb = res["models"]["xgboost"]
    imp = pd.Series(xgb.get_booster().get_score(importance_type="gain")).reindex(RM.FEATURES).fillna(0)
    imp = (imp / imp.sum()).sort_values()
    lr = res["models"]["logistic_regression"][-1]
    coef = pd.Series(lr.coef_[0], index=RM.FEATURES).reindex(imp.index)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].barh(imp.index, imp.values, color="#4C72B0")
    axes[0].set_title("XGBoost feature importance (share of gain)")
    axes[1].barh(coef.index, coef.values, color=np.where(coef.values >= 0, "#C44E52", "#55A868"))
    axes[1].axvline(0, c="grey", lw=0.8)
    axes[1].set_title("Logistic regression coefficients (standardised)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    raw = RM.load_cvd()
    df, clean_report = RM.clean_cvd(raw)
    print(f"CVD data: {clean_report}")

    print("Training (logistic regression, XGBoost grid of "
          f"{np.prod([len(v) for v in RM.XGB_GRID.values()])} configs) ...")
    res = RM.train_all(df)
    RM.save_model(res["models"]["calibrated"], res["chosen"], res["calibration_method"])
    calibration_plot(res, FIG_DIR / "cvd_calibration.png")
    importance_plot(res, FIG_DIR / "cvd_feature_importance.png")

    metrics = {k: res[k] for k in ("split_sizes", "xgb_best_params", "val_roc_auc", "chosen",
                                   "calibration_method", "calibration_val_brier", "test_metrics")}
    metrics["cleaning"] = clean_report
    metrics["xgb_grid"] = res["xgb_grid"].to_dict("records")
    (REPORTS_DIR / "risk_model_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    print(f"\nXGBoost best params (validation): {res['xgb_best_params']}")
    print(f"Validation ROC-AUC: " + ", ".join(f"{k} {v:.4f}" for k, v in res["val_roc_auc"].items()))
    print(f"Chosen: {res['chosen']} + {res['calibration_method']} calibration "
          f"(validation Brier: {', '.join(f'{k} {v:.4f}' for k, v in res['calibration_val_brier'].items())})")
    print("\nTest set:")
    print(pd.DataFrame(res["test_metrics"]).T.round(4).to_string())

    # ---------------------------------------------------------------- score Synthea cohort
    ctx = pd.read_parquet(PROCESSED_DIR / "user_context.parquet")
    pats = pd.read_parquet(PROCESSED_DIR / "patients.parquet")
    pm = pd.read_parquet(PROCESSED_DIR / "profile_metrics.parquet")
    X = RM.synthea_features(pm, pats, ctx)
    bundle = RM.load_model()
    X["cvd_risk"] = RM.predict_risk(bundle, X)
    X.to_parquet(RM.RISK_PATH, index=False)

    print(f"\nCohort CVD risk: {X['cvd_risk'].notna().sum()} / {len(X)} patients scored")
    print(f"  risk quantiles: {X['cvd_risk'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(3).to_dict()}")
    print(f"  imputed cholesterol level: {X['cholesterol_imputed'].sum()} | imputed glucose level: "
          f"{X['gluc_imputed'].sum()} | missing vitals: {X['vitals_missing'].sum()} | "
          f"age outside 29-65: {X['age_outside_training'].sum()}")
    print(f"\nSaved model to {RM.MODEL_PATH}, metrics and figures to {REPORTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
