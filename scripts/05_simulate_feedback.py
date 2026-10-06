"""Phase 7: simulated feedback loop and weight learning (build plan 6.2).

Two arms run on the same simulated users with common random numbers:
    fixed    default hand-set weights for all rounds
    learned  starts from the defaults; every 5 rounds a logistic regression on this arm's own
             feedback log re-estimates w1..w7
Each arm keeps its own feedback state (history, intensity caps, tag affinity) per user.

This demonstrates the adaptation mechanism on SIMULATED users, not real-world effectiveness.

Writes:
    data/processed/feedback_log.parquet
    config/weights_learned.yaml
    reports/feedback_results.json
    reports/figures/feedback_learning_curve.png
    reports/figures/learned_weights.png

    python scripts/05_simulate_feedback.py [--rounds 30] [--patients N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lphrs import SEED  # noqa: E402
from lphrs import feedback as FB  # noqa: E402
from lphrs import pipeline as PL  # noqa: E402
from lphrs import profile as P  # noqa: E402
from lphrs import ranker as RK  # noqa: E402
from lphrs import safety as S  # noqa: E402
from lphrs import weight_learner as WL  # noqa: E402
from lphrs.config import PROCESSED_DIR, REPORTS_DIR  # noqa: E402

LEARN_EVERY = 5
# Simulator calibration: shifts every user's acceptance log-odds so that about half of round-1
# items are accepted under the default ranker (calibrated on 100 patients: offset 0 -> 82 %,
# -1.5 -> 57 %, -2.0 -> 47 %). Without it acceptance starts near a ceiling of ~85 %.
SIM_BASE_OFFSET = -1.75
K = RK.selection_params()["k"]


def build_statics(data: PL.PipelineData, cm: FB.CandidateMatrix, ids: list[str]):
    statics, contexts = {}, {}
    for pid in ids:
        uc = P.build_user_context(pid, data.context, data.patients)
        prof = P.build_health_profile(pid, data.patient_timeline(pid), data.ref, data.volatility, uc["goals"],
                                      cvd_risk=data.cvd_risk.get(pid))
        kept, _ = S.safety_filter(data.candidates, S.facts_from_profile(prof, uc))
        statics[pid] = FB.build_static(pid, kept, prof, uc, cm)
        contexts[pid] = uc
    return statics, contexts


def run(rounds: int, ids: list[str], data, cm, statics, contexts, users) -> tuple[pd.DataFrame, list[dict]]:
    table = RK.load_synergy()
    weights = {"fixed": RK.default_weights(), "learned": RK.default_weights()}
    states = {arm: {pid: FB.initial_state(contexts[pid]) for pid in ids} for arm in weights}
    weight_history = [{"round": 0, **weights["learned"]}]
    log = []
    for rnd in range(1, rounds + 1):
        t0 = time.time()
        for arm, w in weights.items():
            for idx, pid in enumerate(ids):
                state = states[arm][pid]
                recs = FB.recommend_fast(statics[pid], state, cm, w, table=table)
                # common random numbers: the same draws per (round, patient, list position) in both arms
                rng = np.random.default_rng([SEED, rnd, idx])
                noise, u_acc, u_done = rng.standard_normal(K), rng.random(K), rng.random(K)
                user = users[pid]
                for pos, rec in enumerate(recs.to_dict("records")):
                    terms = np.array([rec[t] for t in RK.TERMS])
                    outcome = user.respond(terms, rec["domain"], rec["tags"], int(rec["intensity"]),
                                           np.array([noise[pos], u_acc[pos], u_done[pos]]))
                    log.append({"round": rnd, "arm": arm, "patient_id": pid, "position": pos + 1,
                                "item_id": rec["id"], "domain": rec["domain"], "intensity": int(rec["intensity"]),
                                **{t: float(rec[t]) for t in RK.TERMS}, "score": float(rec["score"]),
                                "outcome": outcome, "accepted": int(outcome in FB.ACCEPT_OUTCOMES)})
                    state = FB.adapt_profile_from_feedback(state, rec, outcome)
                states[arm][pid] = state
        cur = pd.DataFrame([r for r in log[-2 * len(ids) * K:] if r["round"] == rnd])
        rates = cur.groupby("arm")["accepted"].mean()
        msg = f"round {rnd:2d}: acceptance fixed {rates['fixed']:.3f} | learned {rates['learned']:.3f}"
        if rnd % LEARN_EVERY == 0:
            lw = pd.DataFrame([r for r in log if r["arm"] == "learned"])
            new_w, coef = WL.learn_weights(lw)
            weight_history.append({"round": rnd, **new_w, **{f"coef_{k}": v for k, v in coef.items()}})
            if rnd < rounds:
                weights["learned"] = new_w
            msg += "  -> learned weights " + " ".join(f"{t}={new_w[t]:.2f}" for t in RK.TERMS)
        print(f"{msg}   ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(log), weight_history


def plot_learning_curve(log: pd.DataFrame, path: Path) -> pd.DataFrame:
    by = log.groupby(["round", "arm"]).agg(acceptance=("accepted", "mean"),
                                           too_difficult=("outcome", lambda o: (o == "TOO_DIFFICULT").mean()))
    by = by.reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for arm, style in (("fixed", dict(c="#4C72B0", label="Fixed default weights")),
                       ("learned", dict(c="#DD8452", label="Learned weights (refit every 5 rounds)"))):
        d = by[by["arm"] == arm]
        axes[0].plot(d["round"], d["acceptance"], marker="o", ms=3, **style)
        axes[1].plot(d["round"], d["too_difficult"], marker="o", ms=3, **style)
    for r in range(LEARN_EVERY, int(by["round"].max()), LEARN_EVERY):
        axes[0].axvline(r + 0.5, c="grey", lw=0.6, ls=":")
    axes[0].set(title="Simulated acceptance rate per round", xlabel="Round", ylabel="Share of items accepted")
    axes[1].set(title="Share of items reported TOO_DIFFICULT", xlabel="Round", ylabel="Share of items")
    axes[0].legend(fontsize=8)
    fig.suptitle("Feedback simulation (simulated users; not real-world effectiveness)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return by


def plot_weights(default: dict, learned: dict, path: Path) -> None:
    x = np.arange(len(RK.TERMS))
    names = [f"{t}\n{RK.WEIGHT_KEYS[t]}" for t in RK.TERMS]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, [default[t] for t in RK.TERMS], 0.4, label="Default (hand-set)", color="#4C72B0")
    ax.bar(x + 0.2, [learned[t] for t in RK.TERMS], 0.4, label="Learned (simulated feedback)", color="#DD8452")
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylabel("Weight (sums to 1)")
    ax.set_title("Ranking weights: default vs learned (C is subtracted)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--patients", type=int, default=None, help="limit to the first N cohort patients (testing)")
    args = ap.parse_args()

    t0 = time.time()
    data = PL.load_data(full_timeline=True)
    ids = data.cohort_ids[: args.patients] if args.patients else data.cohort_ids
    cm = FB.CandidateMatrix(data.candidates)
    hidden = pd.read_parquet(PROCESSED_DIR / "hidden_taste.parquet").set_index("patient_id")
    users = {pid: FB.SimulatedUser.from_row(hidden.loc[pid]) for pid in ids}
    for u in users.values():
        u.base_offset = SIM_BASE_OFFSET
    statics, contexts = build_statics(data, cm, ids)
    print(f"Prepared {len(ids)} patients in {time.time() - t0:.0f}s; running {args.rounds} rounds x 2 arms ...")

    log, weight_history = run(args.rounds, ids, data, cm, statics, contexts, users)
    log.to_parquet(PROCESSED_DIR / "feedback_log.parquet", index=False)

    fig_dir = REPORTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    curve = plot_learning_curve(log, fig_dir / "feedback_learning_curve.png")
    final = {t: weight_history[-1][t] for t in RK.TERMS}
    plot_weights(RK.default_weights(), final, fig_dir / "learned_weights.png")
    WL.save_learned_weights(final, {"rounds": args.rounds, "patients": len(ids), "learn_every": LEARN_EVERY,
                                    "sim_base_offset": SIM_BASE_OFFSET,
                                    "log_rows": int((log["arm"] == "learned").sum())})

    acc = curve.pivot(index="round", columns="arm", values="acceptance")
    last5 = acc.tail(5).mean()
    summary = {
        "simulated": True, "sim_base_offset": SIM_BASE_OFFSET, "rounds": args.rounds, "patients": len(ids), "items_per_list": K,
        "acceptance_by_round": acc.round(4).reset_index().to_dict("records"),
        "acceptance_rounds_1_5": acc.head(5).mean().round(4).to_dict(),
        "acceptance_last_5_rounds": last5.round(4).to_dict(),
        "outcomes": log.groupby(["arm", "outcome"]).size().unstack(fill_value=0).to_dict("index"),
        "weight_history": weight_history,
        "default_weights": RK.default_weights(), "learned_weights": final,
    }
    (REPORTS_DIR / "feedback_results.json").write_text(json.dumps(summary, indent=2, default=float))

    print(f"\nAcceptance rounds 1-5:  fixed {summary['acceptance_rounds_1_5']['fixed']:.3f} | "
          f"learned {summary['acceptance_rounds_1_5']['learned']:.3f}")
    print(f"Acceptance last 5 rounds: fixed {last5['fixed']:.3f} | learned {last5['learned']:.3f}")
    print("Outcomes:\n" + log.groupby(["arm", "outcome"]).size().unstack(fill_value=0).to_string())
    print("Final learned weights: " + ", ".join(f"{t}={final[t]:.3f}" for t in RK.TERMS))
    print(f"\nWrote config/weights_learned.yaml and figures in {fig_dir}  (total {time.time() - t0:.0f}s)")
    print("Note: simulated users; this shows the adaptation mechanism, not real-world effectiveness.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
