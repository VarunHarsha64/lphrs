"""Phase 6: run the recommendation pipeline for one patient.

    python scripts/04_run_pipeline.py                     # list example patient ids
    python scripts/04_run_pipeline.py --patient 307966cc  # an id prefix is enough
    python scripts/04_run_pipeline.py --patient <id> --breakdown
    python scripts/04_run_pipeline.py --patient <id> --weights learned   # weights from Phase 7

Default weights (config/weights.yaml) are the live setting. The learned weights come from
SIMULATED feedback and nearly ignore health relevance and urgency, so they are opt-in.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from lphrs import pipeline as PL  # noqa: E402
from lphrs.explain import GOAL_LABELS, fmt_value, label  # noqa: E402
from lphrs.ranker import TERMS  # noqa: E402
from lphrs.weight_learner import load_learned_weights  # noqa: E402


def list_examples(data: PL.PipelineData) -> None:
    ctx = data.context.merge(data.patients[["patient_id", "age", "gender"]], on="patient_id")
    ctx["cvd"] = ctx["patient_id"].map(data.cvd_risk)
    print("Example cohort patients (use --patient <id prefix>):\n")
    picks = pd.concat([ctx.sort_values("cvd", ascending=False).head(3), ctx.sort_values("cvd").head(2)])
    for r in picks.itertuples():
        goals = ", ".join(GOAL_LABELS[g] for g in r.goals)
        print(f"  {r.patient_id[:8]}  {r.gender} {r.age:4.0f}y  CVD risk {r.cvd:.0%}  injury {r.injury:<5} goals: {goals}")
    print(f"\n{len(ctx)} patients in the cohort.")


def print_result(res: dict, breakdown: bool) -> None:
    uc, prof = res["user_context"], res["profile"]
    print("=" * 78)
    print(f"Patient {res['patient_id']}  ({uc['gender']}, {uc['age']:.0f} y)   as of {res['ref_date']:%Y-%m-%d}")
    print("=" * 78)
    print(f"Goals: {', '.join(GOAL_LABELS[g] for g in uc['goals'])} | budget {uc['minutes_per_day']} min/day | "
          f"injury: {uc['injury']} | diet: {uc['diet_style']} | likes: {', '.join(uc['liked_exercise'])}")
    print(f"CVD risk: {prof['cvd_risk']:.0%}" if prof["cvd_risk"] is not None else "CVD risk: n/a")
    flagged = [(m, p) for m, p in prof["metrics"].items() if p["urgency"] >= 0.01]
    if flagged:
        print("Metrics needing attention:")
        for m, p in sorted(flagged, key=lambda x: -x[1]["urgency"]):
            print(f"  {label(m):<26} {fmt_value(m, p['current']):>7} {p['unit']:<10} trend {p['trend']:<9} "
                  f"urgency {p['urgency']:.2f} ({p['urgency_status'].replace('_', ' ')})")
    conds = [c["description"] for c in prof["active_conditions"]]
    if conds:
        print("Active conditions: " + "; ".join(conds[:8]) + (" ..." if len(conds) > 8 else ""))
    if prof["allergies"]:
        print("Allergies: " + "; ".join(a["description"] for a in prof["allergies"]))

    drops = res["drops"]
    print(f"\nCandidates: {res['n_candidates']:,} -> {res['n_after_safety']:,} after the safety filter "
          f"({drops['candidate_id'].nunique():,} removed)")
    if len(drops):
        for rule, g in drops.groupby("rule_id"):
            print(f"  {rule:<26} {g['candidate_id'].nunique():>5}  ({g['reason'].iloc[0]})")

    print(f"\nRecommendations ({len(res['recommendations'])}, {res['minutes_used']} of "
          f"{uc['minutes_per_day']} min):\n")
    for i, r in enumerate(res["recommendations"], 1):
        print(f"{i}. [{r['domain']}] {r['title']}  ({r['minutes']} min, score {r['score']:.3f})")
        for reason in r["reasons"]:
            print(textwrap.fill(reason, width=78, initial_indent="     - ", subsequent_indent="       "))
        if breakdown:
            print("     " + "  ".join(f"{t}={r['terms'][t]:.2f}" for t in TERMS)
                  + f"  | synergy {r['gain'] - r['score']:+.3f}")
    print(f"\n{res['disclaimer']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patient", help="patient id or unique prefix")
    ap.add_argument("--breakdown", action="store_true", help="show score terms per recommendation")
    ap.add_argument("--weights", choices=["default", "learned"], default="default",
                    help="ranking weights (default: hand-set; learned: from the Phase 7 simulation)")
    args = ap.parse_args()

    data = PL.load_data()
    if not args.patient:
        list_examples(data)
        return 0
    pid = PL.resolve_patient_id(data, args.patient)
    weights = None
    if args.weights == "learned":
        weights = load_learned_weights()
        if weights is None:
            ap.error("config/weights_learned.yaml not found; run scripts/05_simulate_feedback.py first")
        print("Using LEARNED weights (simulated feedback): "
              + ", ".join(f"{t}={w:.2f}" for t, w in weights.items()) + "\n")
    print_result(PL.run_pipeline(pid, data, weights=weights), args.breakdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
