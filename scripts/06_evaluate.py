"""Phase 8: run all evaluation experiments and write reports/results.md (build plan Section 7).

Every number in the report is computed here from the processed data and earlier phase outputs
(reports/risk_model_metrics.json, reports/feedback_results.json).

    python scripts/06_evaluate.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lphrs import SEED  # noqa: E402
from lphrs import evaluate as E  # noqa: E402
from lphrs import explain as X  # noqa: E402
from lphrs import pipeline as PL  # noqa: E402
from lphrs import profile as P  # noqa: E402
from lphrs import ranker as RK  # noqa: E402
from lphrs import temporal as T  # noqa: E402
from lphrs.config import PROCESSED_DIR, RAW_DIR, REPORTS_DIR, metric_specs  # noqa: E402
from lphrs.weight_learner import load_learned_weights  # noqa: E402

FIG = REPORTS_DIR / "figures"
BLUE, ORANGE, GREY, RED, GREEN = "#4C72B0", "#DD8452", "#8C8C8C", "#C44E52", "#55A868"


def method_label(m: str) -> str:
    """urgency_0.2 -> U≥0.2 ; latest_value_0.02 -> latest ±2%"""
    if m.startswith("latest_value_"):
        return f"latest ±{float(m.rsplit('_', 1)[1]) * 100:g}%"
    return m.replace("urgency_", "U≥")


# ---------------------------------------------------------------- markdown helpers

def md_table(df: pd.DataFrame, fmt: dict | None = None, index: bool = True) -> str:
    fmt = fmt or {}
    d = df.reset_index() if index else df
    cols = list(d.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in d.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            f = fmt.get(c)
            if f and isinstance(v, (int, float, np.floating, np.integer)) and not pd.isna(v):
                cells.append(f.format(v))
            elif isinstance(v, (float, np.floating)):
                cells.append("–" if pd.isna(v) else f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pct(x: float, digits: int = 1) -> str:
    return "–" if pd.isna(x) else f"{100 * x:.{digits}f}%"


def compare(a: float, b: float, higher="higher", lower="lower", same="the same as") -> str:
    if np.isclose(a, b):
        return same
    return higher if a > b else lower


def exp2_reading(e2) -> str:
    o, bu = e2["overall"], e2["by_urgency"]
    hi, lo = bu.loc[">= 0.2"], bu.loc["< 0.05"]
    return (f"Reading: switching U on changes the overall share targeted from {pct(o['snapshot_U_off'])} to "
            f"{pct(o['longitudinal_U_on'])}. For pairs with urgency ≥ 0.2 ({int(hi['pairs'])} pairs) it goes from "
            f"{pct(hi['snapshot_U_off'])} to {pct(hi['longitudinal_U_on'])}; for urgency < 0.05 ({int(lo['pairs'])} pairs) "
            f"from {pct(lo['snapshot_U_off'])} to {pct(lo['longitudinal_U_on'])}. Most rising metrics have low urgency, "
            "and many are targeted anyway through relevance, goals or broad interventions. U also lifts items for "
            "metrics already past their threshold (urgency 1), which compete for the same five slots, so the effect "
            "on an individual rising metric can be negative.")


def exp3_reading(e3) -> str:
    s = e3["summary"].set_index("method")
    u = s.loc["urgency_0.2"]
    lv = s[s.index.str.startswith("latest_value")]
    near = lv.iloc[(lv["flagged"] - u["flagged"]).abs().argsort()].iloc[0]
    name = method_label(near.name)
    verdict = ("does not beat" if near["recall"] >= u["recall"] and near["precision"] >= u["precision"]
               else "trades off against")
    return (f"Reading: at a similar number of alarms (U≥0.2: {int(u['flagged'])}; {name}: {int(near['flagged'])}), the "
            f"latest-value flag has {compare(near['recall'], u['recall'])} recall ({pct(near['recall'])} vs "
            f"{pct(u['recall'])}) and {compare(near['precision'], u['precision'])} precision ({pct(near['precision'])} vs "
            f"{pct(u['precision'])}). For the crossings each method does detect, the median lead is "
            f"{u['median_lead_months_detected']:.0f} months (urgency) vs {near['median_lead_months_detected']:.0f} months "
            f"(latest value). Wider latest-value bands raise recall at the cost of many more alarms and lower precision; "
            f"their long lead times mostly reflect values that sit near the threshold for years. On this synthetic data "
            f"the trend-based urgency flag {verdict} a simple near-threshold flag at detecting crossings within 2 years.")


# ---------------------------------------------------------------- figures

def fig_exp1(e1, path):
    s = e1["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    arms = ["without_filter", "with_filter"]
    axes[0].bar(["Without filter", "With filter"], [s.loc[a, "list_violation_rate"] for a in arms], color=[RED, GREEN])
    axes[0].set(title="Lists (top-5) with at least one violation", ylabel="Share of patients")
    for i, a in enumerate(arms):
        axes[0].text(i, s.loc[a, "list_violation_rate"], pct(s.loc[a, "list_violation_rate"]), ha="center", va="bottom")
    rc = e1["rules_without_filter"].sort_values()
    axes[1].barh(rc.index, rc.values, color=RED)
    axes[1].set(title="Rules violated without the filter", xlabel="Lists affected")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_exp2(e2, path):
    pm, bu = e2["per_metric"], e2["by_urgency"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(pm))
    axes[0].bar(x - 0.2, pm["snapshot_U_off"], 0.4, label="Snapshot (U off)", color=GREY)
    axes[0].bar(x + 0.2, pm["longitudinal_U_on"], 0.4, label="Longitudinal (U on)", color=BLUE)
    axes[0].set_xticks(x, [f"{m}\n(n={n})" for m, n in zip(pm.index, pm["pairs"])], fontsize=7)
    axes[0].set(title="Rising-but-normal metric targeted, by metric", ylabel="Share of patient-metric pairs")
    axes[0].legend(fontsize=8)
    x = np.arange(len(bu))
    axes[1].bar(x - 0.2, bu["snapshot_U_off"], 0.4, label="Snapshot (U off)", color=GREY)
    axes[1].bar(x + 0.2, bu["longitudinal_U_on"], 0.4, label="Longitudinal (U on)", color=BLUE)
    axes[1].set_xticks(x, [f"urgency {b}\n(n={n})" for b, n in zip(bu.index, bu["pairs"])], fontsize=8)
    axes[1].set(title="... by the metric's urgency")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_exp3(e3, path):
    s = e3["summary"].set_index("method")
    rows = e3["rows"][e3["rows"]["crossed"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for m, r in s.iterrows():
        c = BLUE if m.startswith("urgency") else ORANGE
        axes[0].scatter(r["recall"], r["precision"], s=30 + r["flagged"] / 5, color=c)
        axes[0].annotate(method_label(m), (r["recall"], r["precision"]),
                         fontsize=8, xytext=(4, 4), textcoords="offset points")
    axes[0].set(title="Early warning: precision vs recall\n(point size = number of alarms)",
                xlabel="Recall (crossers flagged before crossing)", ylabel="Precision (flagged that crossed)")
    data = [rows[f"lead_{m}"].dropna() for m in s.index]
    axes[1].boxplot(data, tick_labels=[method_label(m) for m in s.index],
                    showfliers=False)
    axes[1].set(title="Lead time for detected crossers", ylabel="Months from first flag to crossing")
    axes[1].tick_params(axis="x", labelsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_exp4(e4, path):
    cfg = P.temporal_cfg()
    lam = cfg["lambda_base"]
    years = np.linspace(0, 15, 300)
    days = years * T.DAYS_PER_YEAR
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(years, T.plain_exponential_weight(days, lam), c=GREY, ls="--", label="Plain exponential (all events)")
    for cls, c in (("allergy", RED), ("chronic_condition", BLUE), ("resolved_major", ORANGE), ("acute_condition", GREEN)):
        axes[0].plot(years, T.recency_weight(days, lam, T.event_floor(cls, cfg)), c=c, label=f"Floored: {cls}")
    axes[0].axvline(5, c="k", lw=0.6, ls=":")
    axes[0].set(title="Recency weight by event age", xlabel="Years since event", ylabel="w_recency")
    axes[0].legend(fontsize=7)
    t = e4["table"].reindex(["allergy", "chronic_condition", "resolved_major", "acute_condition", "medication"]).dropna(how="all")
    x = np.arange(len(t))
    axes[1].bar(x - 0.2, t["plain_mean"], 0.4, color=GREY, label="Plain exponential")
    axes[1].bar(x + 0.2, t["floored_mean"], 0.4, color=BLUE, label="Persistence-floored")
    axes[1].set_xticks(x, [f"{i}\n(n={int(n)})" for i, n in zip(t.index, t["events"])], fontsize=7)
    axes[1].set(title="Mean weight of events older than 5 years (cohort)", ylabel="Mean weight")
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_exp5(e5, path):
    s = e5["summary"].reindex(["independent_top_k", "greedy_set"])
    labels = ["Independent top-K", "Greedy set"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, col, title in ((axes[0], "mean_conflicts", "Conflicting pairs per list"),
                           (axes[1], "mean_domains", "Distinct domains per list"),
                           (axes[2], "within_budget", "Lists within time budget")):
        ax.bar(labels, s[col], color=[GREY, BLUE])
        ax.set_title(title)
        for i, v in enumerate(s[col]):
            ax.text(i, v, pct(v) if col == "within_budget" else f"{v:.2f}", ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_case(case, b, data, path):
    specs = metric_specs()
    obs = data.patient_timeline(b.patient_id)
    meas = P.measurement_frame(obs)
    metrics = case["plot_metrics"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 3.6), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        d = meas[meas["metric"] == m].sort_values("timestamp")
        ax.plot(d["timestamp"].dt.tz_convert(None), d["value"], marker="o" if m not in ("steps", "sleep_hours") else None,
                ms=3, lw=1, c=BLUE, label="Recorded" if m not in ("steps", "sleep_hours") else "Weekly mean")
        theta = specs[m]["threshold"]
        if theta is not None:
            ax.axhline(theta, c=RED, ls="--", lw=1, label=f"Threshold {theta:g}")
        p = b.profile["metrics"].get(m)
        if p and p["urgency_status"] == "approaching" and np.isfinite(p["time_to_threshold_years"]):
            horizon = min(p["time_to_threshold_years"], 10)
            t0 = data.ref.tz_convert(None)
            t1 = t0 + pd.Timedelta(days=horizon * T.DAYS_PER_YEAR)
            ax.plot([t0, t1], [p["current"], p["current"] + p["slope_per_year"] * horizon], c=ORANGE, ls=":",
                    lw=2, label=f"Projection (crosses in ~{X.fmt_years(p['time_to_threshold_years'])})")
        ax.set_title(f"{X.label(m)} ({specs[m]['unit']})")
        ax.legend(fontsize=7)
    fig.suptitle(f"Case study: {case['title']} (patient {b.patient_id[:8]})", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ---------------------------------------------------------------- case studies

def pick_cases(bundles) -> list[tuple[dict, E.Bundle]]:
    used, cases = set(), []

    def best(cands, key):
        cands = [b for b in cands if b.patient_id not in used]
        return max(cands, key=key) if cands else None

    def urg(b, m):
        p = b.profile["metrics"].get(m)
        return p["urgency"] if p and p["urgency_status"] == "approaching" and p["direction"] == "worsening" else -1

    b = best([x for x in bundles if urg(x, "systolic_bp") > 0], lambda x: urg(x, "systolic_bp"))
    if b:
        used.add(b.patient_id)
        cases.append(({"title": "rising blood pressure (hypertensive trend)",
                       "plot_metrics": ["systolic_bp", "diastolic_bp"]}, b))
    def has(x, kw):
        return any(kw in c["description"].lower() for c in x.profile["active_conditions"])
    pre = [x for x in bundles if has(x, "prediabetes") and not has(x, "diabetes mellitus")]
    b = best([x for x in pre if max(urg(x, "hba1c"), urg(x, "glucose")) > 0],
             lambda x: max(urg(x, "hba1c"), urg(x, "glucose")))
    if b:
        used.add(b.patient_id)
        cases.append(({"title": "prediabetes (no diabetes) with a rising glycaemic trend",
                       "plot_metrics": ["hba1c", "glucose"]}, b))
    ai = [x for x in bundles if x.facts.allergens and x.facts.injuries]
    b = best(ai, lambda x: (len(x.facts.allergens), int((~x.kept).sum())))
    if b:
        used.add(b.patient_id)
        cases.append(({"title": "food allergy and an injury", "plot_metrics": ["bmi", "steps"]}, b))
    return cases


def case_markdown(i: int, case: dict, b: E.Bundle, data, table, fig_name: str) -> str:
    uc, prof = b.user_context, b.profile
    res = PL.run_pipeline(b.patient_id, data, user_context=uc, profile=prof)
    lines = [f"### Case {i}: {case['title']}", "",
             f"Patient `{b.patient_id[:8]}`: {uc['gender']}, {uc['age']:.0f} years. "
             f"Goals: {', '.join(X.GOAL_LABELS[g] for g in uc['goals'])}. Daily budget {uc['minutes_per_day']} min. "
             f"Survey injury: {uc['injury']}. Diet: {uc['diet_style']}. CVD risk (model): {prof['cvd_risk']:.0%}.", "",
             f"![{case['title']}](figures/{fig_name})", "", "**Profile (metrics with a threshold):**", ""]
    mrows = [{"metric": X.label(m), "current": X.fmt_value(m, p["current"]) + f" {p['unit']}",
              "threshold": X.fmt_threshold(m, p["threshold"]), "trend": p["trend"],
              "slope / year": f"{p['slope_per_year']:+.2f}", "urgency": f"{p['urgency']:.2f}",
              "status": p["urgency_status"].replace("_", " ")}
             for m, p in prof["metrics"].items() if p["threshold"] is not None]
    lines += [md_table(pd.DataFrame(mrows), index=False), ""]
    conds = [c["description"] for c in prof["active_conditions"]]
    lines.append(f"**Active conditions ({len(conds)}):** " + ("; ".join(conds[:10]) + (" ..." if len(conds) > 10 else "")
                                                             if conds else "none"))
    meds = sorted({m["description"] for m in prof["active_medications"]})
    lines.append(f"\n**Active medications:** {'; '.join(meds) if meds else 'none'}")
    lines.append(f"\n**Allergies:** {'; '.join(a['description'] for a in prof['allergies']) or 'none'} "
                 f"→ mapped allergens: {', '.join(sorted(b.facts.allergens)) or 'none'}")
    drops = res["drops"]
    lines += ["", f"**Safety filter:** {res['n_candidates']:,} candidates → {res['n_after_safety']:,} kept "
                  f"({drops['candidate_id'].nunique():,} removed).", ""]
    if len(drops):
        d = drops.groupby(["rule_id", "reason"])["candidate_id"].nunique().rename("candidates removed").reset_index()
        ex = drops.groupby("rule_id")["title"].first().rename("example")
        lines += [md_table(d.merge(ex, on="rule_id"), index=False), ""]
    lines += ["**Final recommendations (default weights):**", ""]
    for j, r in enumerate(res["recommendations"], 1):
        terms = " ".join(f"{t}={r['terms'][t]:.2f}" for t in RK.TERMS)
        lines.append(f"{j}. **{r['title']}** [{r['domain']}, {r['minutes']} min, score {r['score']:.3f}]  ")
        for reason in r["reasons"]:
            lines.append(f"   - {reason}")
        lines.append(f"   - _terms: {terms}_")
    lines += ["", f"_{X.DISCLAIMER}_", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- main

def main() -> int:
    t_start = time.time()
    FIG.mkdir(parents=True, exist_ok=True)
    data = PL.load_data(full_timeline=True)
    table = RK.load_synergy()
    ids = data.cohort_ids
    print(f"Building bundles for {len(ids)} patients ...", flush=True)
    bundles = [E.build_bundle(p, data) for p in ids]
    print(f"  done in {time.time() - t_start:.0f}s", flush=True)

    print("Experiment 1: safety", flush=True)
    e1 = E.exp1_safety(bundles, data.candidates, table)
    print("Experiment 2: snapshot vs longitudinal", flush=True)
    e2 = E.exp2_snapshot_vs_longitudinal(bundles, table)
    print("Experiment 3: early warning", flush=True)
    obs = data.timeline[(data.timeline["event_type"] == "observation") & data.timeline["patient_id"].isin(ids)]
    e3 = E.exp3_early_warning(obs, data.ref)
    print("Experiment 4: decay", flush=True)
    e4 = E.exp4_decay(pd.read_parquet(PROCESSED_DIR / "profile_events.parquet"))
    print("Experiment 5: set selection", flush=True)
    e5 = E.exp5_set_selection(bundles, table)
    print("Experiment 6: weight learning", flush=True)
    fb = json.loads((REPORTS_DIR / "feedback_results.json").read_text())
    learned = load_learned_weights()
    e6_comp = E.exp6_list_composition(bundles, learned, table)
    print("Experiment 7: risk model", flush=True)
    rm = json.loads((REPORTS_DIR / "risk_model_metrics.json").read_text())
    risk = pd.read_parquet(PROCESSED_DIR / "cvd_risk.parquet")

    fig_exp1(e1, FIG / "exp1_safety.png")
    fig_exp2(e2, FIG / "exp2_snapshot_vs_longitudinal.png")
    fig_exp3(e3, FIG / "exp3_early_warning.png")
    fig_exp4(e4, FIG / "exp4_decay.png")
    fig_exp5(e5, FIG / "exp5_set_selection.png")

    print("Case studies", flush=True)
    cases = pick_cases(bundles)
    case_md = []
    for i, (case, b) in enumerate(cases, 1):
        name = f"case{i}_{b.patient_id[:8]}.png"
        fig_case(case, b, data, FIG / name)
        case_md.append(case_markdown(i, case, b, data, table, name))

    # ------------------------------------------------------------ assemble results.md
    meta = json.loads((PROCESSED_DIR / "build_meta.json").read_text())
    pats = data.patients
    src = json.loads((RAW_DIR / "synthea" / "SOURCE.json").read_text())
    s1, s5 = e1["summary"], e5["summary"]
    s3 = e3["summary"].set_index("method")
    tm = rm["test_metrics"]
    fb_first, fb_last = fb["acceptance_rounds_1_5"], fb["acceptance_last_5_rounds"]
    old = e4["table"]

    md = []
    md += ["# LPHRS evaluation results", "",
           f"_Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by `scripts/06_evaluate.py` (seed {SEED}). "
           f"Every number below is computed by code in this repository._", "",
           f"> **{X.DISCLAIMER}** This is a research prototype on synthetic data.", "",
           "## Data", "",
           f"- **Clinical records:** Synthea, generated locally ({src.get('kind')}; Java {src.get('java_major')}; "
           f"`{' '.join(src.get('args', []))}`). {len(pats)} patients ({int(pats['alive'].sum())} alive, "
           f"{int((~pats['alive']).sum())} deceased). Reference date {meta['reference_date'][:10]}.",
           f"- **Evaluation cohort:** {len(ids)} living adults (18+). Children and deceased patients stay in the "
           "timeline and in cohort-level statistics only.",
           "- **Lifestyle, goals, preferences, injuries and feedback are synthetically generated.** No public dataset "
           "links these to clinical records. Sleep and steps are calibrated on the Sleep Health and Lifestyle dataset "
           "(374 people, ages 27-59); trends, noise, goals and preferences follow documented assumptions in "
           "`src/lphrs/synth_lifestyle.py`.",
           "- **Risk model training data:** Kaggle Cardiovascular Disease dataset; **recipes:** 5,000 sampled Food.com "
           "meals (5-60 min, no alcohol, meal-type tags).", "",
           "## Summary", "",
           md_table(pd.DataFrame([
               {"#": 1, "experiment": "Safety", "result":
                f"lists with a violation: {pct(s1.loc['without_filter', 'list_violation_rate'])} without filter → "
                f"{pct(s1.loc['with_filter', 'list_violation_rate'])} with filter"},
               {"#": 2, "experiment": "Snapshot vs longitudinal", "result":
                f"rising-but-normal metric targeted: {pct(e2['overall']['snapshot_U_off'])} (U off) vs "
                f"{pct(e2['overall']['longitudinal_U_on'])} (U on); {e2['pairs']} pairs"},
               {"#": 3, "experiment": "Early warning", "result":
                f"U≥0.2: recall {pct(s3.loc['urgency_0.2', 'recall'])}, precision {pct(s3.loc['urgency_0.2', 'precision'])}, "
                f"median lead {s3.loc['urgency_0.2', 'median_lead_months_detected']:.0f} months (detected); "
                f"latest ±5%: recall {pct(s3.loc['latest_value_0.05', 'recall'])}, precision "
                f"{pct(s3.loc['latest_value_0.05', 'precision'])}"},
               {"#": 4, "experiment": "Decay", "result":
                f"events > 5 y old: allergies keep {old.loc['allergy', 'floored_mean']:.2f} (plain "
                f"{old.loc['allergy', 'plain_mean']:.4f}); chronic conditions {old.loc['chronic_condition', 'floored_mean']:.2f} "
                f"(plain {old.loc['chronic_condition', 'plain_mean']:.4f})"},
               {"#": 5, "experiment": "Set selection", "result":
                f"conflicts/list {s5.loc['independent_top_k', 'mean_conflicts']:.2f} → {s5.loc['greedy_set', 'mean_conflicts']:.2f}; "
                f"domains {s5.loc['independent_top_k', 'mean_domains']:.2f} → {s5.loc['greedy_set', 'mean_domains']:.2f}; "
                f"within budget {pct(s5.loc['independent_top_k', 'within_budget'])} → {pct(s5.loc['greedy_set', 'within_budget'])}"},
               {"#": 6, "experiment": "Weight learning (simulated)", "result":
                f"acceptance, last 5 rounds: fixed {pct(fb_last['fixed'])} vs learned {pct(fb_last['learned'])} "
                f"(both {pct(fb_first['fixed'])} in rounds 1-5)"},
               {"#": 7, "experiment": "Risk model", "result":
                f"test ROC-AUC LR {tm['logistic_regression']['roc_auc']:.3f} / XGBoost {tm['xgboost']['roc_auc']:.3f}; "
                f"Brier {tm['logistic_regression']['brier']:.3f} / {tm['xgboost']['brier']:.3f}"},
           ]), index=False), ""]

    # 1
    md += ["## 1. Safety: pipeline with vs without SafetyFilter", "",
           "Same patients, same scoring and set selection; only the filter is switched. A violation is a final "
           "top-5 item that any safety rule would remove for that patient.", "",
           md_table(s1[["lists", "items", "violating_items", "item_violation_rate", "lists_with_violation",
                        "list_violation_rate"]], fmt={"item_violation_rate": "{:.2%}", "list_violation_rate": "{:.2%}"}), "",
           "Rules that would have been broken without the filter (lists affected):", "",
           md_table(e1["rules_without_filter"].rename("lists").to_frame()), "",
           f"With the filter the violation rate is {pct(s1.loc['with_filter', 'item_violation_rate'], 2)} "
           f"(target 0%).", "", "![Experiment 1](figures/exp1_safety.png)", ""]

    # 2
    bu = e2["by_urgency"]
    md += ["## 2. Snapshot vs longitudinal (urgency term off vs on)", "",
           "Population: patient-metric pairs whose metric is still on the safe side of its threshold but trending "
           "toward it (urgency status *approaching*, direction *worsening*). Outcome: does the final list contain an "
           "item targeting that metric? The snapshot arm sets U = 0; everything else is identical.", "",
           f"{e2['pairs']} pairs from {e2['patients']} patients. Overall share targeted: "
           f"**{pct(e2['overall']['snapshot_U_off'])} (U off) vs {pct(e2['overall']['longitudinal_U_on'])} (U on)**.", "",
           md_table(e2["per_metric"], fmt={"snapshot_U_off": "{:.1%}", "longitudinal_U_on": "{:.1%}"}), "",
           "By the metric's urgency value:", "",
           md_table(bu, fmt={"snapshot_U_off": "{:.1%}", "longitudinal_U_on": "{:.1%}"}), "",
           exp2_reading(e2), "",
           "![Experiment 2](figures/exp2_snapshot_vs_longitudinal.png)", ""]

    # 3
    s3d = e3["summary"].copy()
    s3d["iqr_lead_months_detected"] = s3d["iqr_lead_months_detected"].map(
        lambda t: "–" if pd.isna(t[0]) else f"{t[0]:.0f}–{t[1]:.0f}")
    md += ["## 3. Early warning: urgency vs latest-value flag", "",
           f"Each patient's most recent 2 years are hidden (cut = {e3['cut']:%Y-%m-%d}). Flags are computed only from "
           "readings up to the cut; outcomes are threshold crossings in the hidden part. Included: "
           f"{e3['pairs']:,} patient-metric pairs (observation metrics with a threshold) that were not past the "
           f"threshold at the cut and have readings on both sides; **{e3['crossers']} crossed** in the hidden 2 years.", "",
           "- *Urgency flag:* time-to-threshold urgency ≥ τ at the last pre-cut reading (τ = 0.1 / 0.2 / 0.3).",
           "- *Latest-value flag:* the latest reading is within 2 / 5 / 10 % of the threshold on the bad side.",
           "- *Lead time:* months from the start of the flag (the run of consecutive flagged readings still active at "
           "the cut) to the first hidden reading past the threshold. Readings are roughly yearly, so lead times are coarse.", "",
           md_table(s3d, fmt={"recall": "{:.1%}", "precision": "{:.1%}", "median_lead_months_detected": "{:.0f}",
                              "median_lead_months_all_crossers": "{:.0f}"}, index=False), "",
           "Crossings per metric:", "", md_table(e3["per_metric"]), "",
           exp3_reading(e3), "",
           "![Experiment 3](figures/exp3_early_warning.png)", ""]

    # 4
    md += ["## 4. Decay: plain exponential vs persistence-floored adaptive", "",
           "Weights of every condition / medication / allergy event older than 5 years in the cohort profiles "
           "(`w_plain` = exp(-λ·Δt); `w_recency` = max(floor, exp(-λ·Δt)), floor = severity × persistence).", "",
           md_table(old, fmt={"floored_mean": "{:.3f}", "floored_min": "{:.3f}", "plain_mean": "{:.4f}",
                              "plain_max": "{:.4f}"}), "",
           f"Allergies and chronic conditions older than 5 years keep weights of "
           f"{old.loc['allergy', 'floored_mean']:.2f} and {old.loc['chronic_condition', 'floored_mean']:.2f} with the floor, "
           f"versus {old.loc['allergy', 'plain_mean']:.4f} and {old.loc['chronic_condition', 'plain_mean']:.4f} with plain "
           "decay; acute conditions and old medications fade identically under both schemes.", "",
           "Metric decay rates from cohort volatility (half-life = ln 2 / λ):", "",
           md_table(data.volatility.assign(half_life_days=np.log(2) / data.volatility["lambda"])
                    [["rel_change_sd", "volatility", "lambda", "half_life_days"]].sort_values("volatility"),
                    fmt={"rel_change_sd": "{:.4f}", "volatility": "{:.2f}", "lambda": "{:.5f}", "half_life_days": "{:.0f}"}), "",
           "![Experiment 4](figures/exp4_decay.png)", ""]

    # 5
    md += ["## 5. Set selection: independent top-K vs greedy cross-domain set", "",
           "Same scores; top-K takes the 5 highest independently, the greedy set applies synergy (α = 0.3), the time "
           "budget (with look-ahead) and at most 2 items per domain.", "",
           md_table(s5, fmt={"mean_conflicts": "{:.2f}", "lists_with_conflict": "{:.1%}", "mean_domains": "{:.2f}",
                             "within_budget": "{:.1%}", "mean_minutes": "{:.1f}", "mean_score_sum": "{:.3f}"}), "",
           f"The greedy set has {compare(s5.loc['greedy_set', 'mean_conflicts'], s5.loc['independent_top_k', 'mean_conflicts'], 'more', 'fewer')} "
           f"conflicts per list, covers {compare(s5.loc['greedy_set', 'mean_domains'], s5.loc['independent_top_k', 'mean_domains'], 'more', 'fewer')} "
           f"domains and stays within budget in {pct(s5.loc['greedy_set', 'within_budget'])} of lists, at the cost of a "
           f"{compare(s5.loc['greedy_set', 'mean_score_sum'], s5.loc['independent_top_k', 'mean_score_sum'])} summed score "
           f"({s5.loc['greedy_set', 'mean_score_sum']:.3f} vs {s5.loc['independent_top_k', 'mean_score_sum']:.3f}). "
           "Conflicts counted include same-tag redundancy pairs (e.g. two recipes).", "",
           "![Experiment 5](figures/exp5_set_selection.png)", ""]

    # 6
    wh = pd.DataFrame(fb["weight_history"]).set_index("round")[RK.TERMS]
    md += ["## 6. Weight learning: fixed vs learned weights (simulated users)", "",
           f"{fb['patients']} simulated users × {fb['rounds']} rounds × 5 items, two arms with common random numbers. "
           "Both arms adapt each user's history, intensity caps and preferred tags from feedback; the learned arm also "
           "refits w1..w7 every 5 rounds by logistic regression on its own log. Simulator calibration: base offset "
           f"{fb['sim_base_offset']} so that about half of round-1 items are accepted.", "",
           md_table(pd.DataFrame({"rounds 1-5": fb_first, "last 5 rounds": fb_last}).T,
                    fmt={"fixed": "{:.1%}", "learned": "{:.1%}"}), "",
           "Weights after each refit:", "", md_table(wh, fmt={t: "{:.3f}" for t in RK.TERMS}), "",
           "What the learned weights favour: mean term values of the final lists for the 775 patients (no feedback "
           "history, so H is constant):", "",
           md_table(e6_comp, fmt={t: "{:.3f}" for t in RK.TERMS}), "",
           "**Caveats.** (1) This demonstrates the adaptation mechanism on simulated users, not real-world effectiveness; "
           "the learned weights reflect the simulator's assumptions. (2) H (historical adherence) is computed from the "
           "same feedback the learner trains on, so its weight is inflated by a feedback loop (users who accept more "
           "have higher H). (3) Maximising acceptance pushes down health relevance and urgency, so the learned weights "
           "are **not** used as the live setting; the hand-set defaults remain live and the learned weights are opt-in "
           "(`04_run_pipeline.py --weights learned`).", "",
           "![Learning curve](figures/feedback_learning_curve.png)", "", "![Learned weights](figures/learned_weights.png)", ""]

    # 7
    tmd = pd.DataFrame(tm).T
    md += ["## 7. Risk model: logistic regression vs XGBoost", "",
           f"Kaggle CVD data after cleaning: {rm['cleaning']['output_rows']:,} rows (dropped {rm['cleaning']['dropped_bp']:,} "
           f"for impossible blood pressure, {rm['cleaning']['dropped_height_weight']} for implausible height/weight). "
           f"Split 70/15/15 stratified: {rm['split_sizes']}. XGBoost tuned on validation "
           f"({len(rm['xgb_grid'])} configurations; best {rm['xgb_best_params']}); chosen model "
           f"{rm['chosen']} with {rm['calibration_method']} calibration.", "",
           "Test set:", "", md_table(tmd, fmt={c: "{:.4f}" for c in tmd.columns}), "",
           "![Calibration](figures/cvd_calibration.png)", "", "![Feature importance](figures/cvd_feature_importance.png)", "",
           f"Applied to the Synthea cohort, features come from the latest readings. Missing labs default to level 1 "
           f"(cholesterol imputed for {int(risk['cholesterol_imputed'].sum())}, glucose for {int(risk['gluc_imputed'].sum())} "
           f"of {len(risk)} patients) and {int(risk['age_outside_training'].sum())} patients are outside the training age "
           "range (29-65), where the model treats age as the nearest training age.", ""]

    md += ["## Case studies", ""] + case_md
    md += ["## Limitations", "",
           "- Clinical data is synthetic (Synthea); lifestyle, goals, preferences, injuries and feedback are simulated.",
           "- Thresholds and safety rules are simplified illustrations and have not been reviewed by a clinician.",
           "- Recipe allergens are found by ingredient keyword matching, which can miss unlisted names. An audit against "
           "Food.com's own fish/seafood tags during this evaluation found 38 untagged seafood recipes (e.g. basa, hake, "
           "orange roughy); the keyword lists were extended and all results regenerated, but the method remains "
           "imperfect and real use would need a curated ingredient-allergen database.",
           "- The linear trend projection assumes the recent trend continues; Synthea readings are roughly yearly.",
           "- Recipe diet tags only look at sodium, sugar, protein and meat content, not overall food quality. In an "
           "earlier run a hot-dog recipe ranked first for a prediabetic patient; processed meats and alcohol are now "
           "excluded from the recipe pool, but other low-quality foods could still score well.",
           "- Learned weights reflect the simulator's assumptions and a feedback loop in the history term.",
           "- The risk model is trained on a different population (Kaggle, ages 29-65) than it is applied to.",
           "- The system gives wellness guidance only and does not diagnose or prescribe.", ""]

    (REPORTS_DIR / "results.md").write_text("\n".join(md), encoding="utf-8")
    raw = {"exp1": e1["summary"].to_dict("index"), "exp2": {"overall": e2["overall"].to_dict(), "pairs": e2["pairs"]},
           "exp3": e3["summary"].to_dict("records"), "exp4": old.to_dict("index"), "exp5": s5.to_dict("index"),
           "exp6_list_terms": e6_comp.to_dict("index"), "cases": [b.patient_id for _, b in cases]}
    (REPORTS_DIR / "evaluation_results.json").write_text(json.dumps(raw, indent=2, default=str))
    print(f"\nWrote {REPORTS_DIR / 'results.md'} ({len(cases)} case studies) in {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
