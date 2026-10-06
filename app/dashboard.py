"""Phase 9: Streamlit dashboard.

    streamlit run app/dashboard.py

Patient picker, metric timelines with thresholds and projected crossings, profile summary,
safety drops with reasons, recommendations with score breakdowns, and feedback buttons that
update the user's state (AdaptProfileFromFeedback) and re-rank live.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lphrs import explain as X  # noqa: E402
from lphrs import feedback as FB  # noqa: E402
from lphrs import pipeline as PL  # noqa: E402
from lphrs import profile as P  # noqa: E402
from lphrs import ranker as RK  # noqa: E402
from lphrs import temporal as T  # noqa: E402
from lphrs.config import metric_specs  # noqa: E402
from lphrs.weight_learner import load_learned_weights  # noqa: E402

OUTCOME_LABELS = {"ACCEPTED": "👍 Accept", "COMPLETED": "✅ Completed", "REJECTED": "👎 Reject",
                  "TOO_DIFFICULT": "😓 Too difficult"}
TERM_NAMES = {"R": "Health relevance", "G": "Goal alignment", "U": "Urgency", "P": "Preference",
              "F": "Feasibility", "H": "History", "C": "Effort cost"}

st.set_page_config(page_title="LPHRS", page_icon="🩺", layout="wide")


# ---------------------------------------------------------------- cached data

@st.cache_resource(show_spinner="Loading processed data ...")
def load():
    return PL.load_data(full_timeline=True)


@st.cache_resource(show_spinner=False)
def base_context(pid: str) -> dict:
    data = load()
    return P.build_user_context(pid, data.context, data.patients)


@st.cache_resource(show_spinner="Building health profile ...")
def health_profile(pid: str) -> dict:
    data = load()
    uc = base_context(pid)
    return P.build_health_profile(pid, data.patient_timeline(pid), data.ref, data.volatility, uc["goals"],
                                  cvd_risk=data.cvd_risk.get(pid))


@st.cache_resource(show_spinner=False)
def measurements(pid: str) -> pd.DataFrame:
    return P.measurement_frame(load().patient_timeline(pid))


@st.cache_data(show_spinner=False)
def patient_options() -> list[tuple[str, str]]:
    data = load()
    df = data.context[["patient_id"]].merge(data.patients[["patient_id", "age", "gender"]], on="patient_id")
    df["cvd"] = df["patient_id"].map(data.cvd_risk)
    df = df.sort_values("cvd", ascending=False)
    return [(r.patient_id, f"{r.patient_id[:8]} · {r.gender} {r.age:.0f}y · CVD {r.cvd:.0%}") for r in df.itertuples()]


# ---------------------------------------------------------------- session state

def feedback_state(pid: str) -> dict:
    fb = st.session_state.setdefault("fb", {})
    if pid not in fb:
        fb[pid] = FB.initial_state(base_context(pid))
    return fb[pid]


def feedback_log(pid: str) -> list[dict]:
    return st.session_state.setdefault("fb_log", {}).setdefault(pid, [])


def give_feedback(pid: str, rec: dict, outcome: str) -> None:
    st.session_state["fb"][pid] = FB.adapt_profile_from_feedback(feedback_state(pid), rec, outcome)
    feedback_log(pid).append({"time": datetime.now().strftime("%H:%M:%S"), "item": rec["title"],
                              "domain": rec["domain"], "intensity": rec["intensity"], "outcome": outcome})
    st.session_state.setdefault("prev_ids", {})[pid] = [r["id"] for r in st.session_state["last_recs"]]


def reset_feedback(pid: str) -> None:
    st.session_state.get("fb", {}).pop(pid, None)
    st.session_state.get("fb_log", {}).pop(pid, None)
    st.session_state.get("prev_ids", {}).pop(pid, None)


# ---------------------------------------------------------------- charts

def metric_chart(pid: str, metric: str, prof: dict, ref: pd.Timestamp) -> alt.LayerChart | None:
    spec = metric_specs()[metric]
    d = measurements(pid)
    d = d[d["metric"] == metric].copy()
    if d.empty:
        return None
    d["date"] = d["timestamp"].dt.tz_convert(None)
    lifestyle = metric in ("steps", "sleep_hours")
    base = alt.Chart(d).encode(x=alt.X("date:T", title=None),
                               y=alt.Y("value:Q", title=spec["unit"], scale=alt.Scale(zero=False)),
                               tooltip=[alt.Tooltip("date:T"), alt.Tooltip("value:Q", format=".1f")])
    layers = [base.mark_line(color="#4C72B0", point=not lifestyle, strokeWidth=1.5 if not lifestyle else 1)]
    p = prof["metrics"].get(metric)
    if spec["threshold"] is not None:
        layers.append(alt.Chart(pd.DataFrame({"y": [spec["threshold"]]})).mark_rule(color="#C44E52", strokeDash=[6, 4])
                      .encode(y="y:Q"))
    if p and p["urgency_status"] == "approaching" and np.isfinite(p["time_to_threshold_years"]) \
            and p["time_to_threshold_years"] <= 10:
        t0 = ref.tz_convert(None)
        ttt = p["time_to_threshold_years"]
        proj = pd.DataFrame({"date": [t0, t0 + pd.Timedelta(days=ttt * T.DAYS_PER_YEAR)],
                             "value": [p["current"], spec["threshold"]]})
        layers.append(alt.Chart(proj).mark_line(color="#DD8452", strokeDash=[2, 3], strokeWidth=2.5)
                      .encode(x="date:T", y="value:Q"))
        layers.append(alt.Chart(proj.tail(1)).mark_point(color="#DD8452", size=80, filled=True)
                      .encode(x="date:T", y="value:Q",
                              tooltip=[alt.Tooltip("date:T", title="projected crossing")]))
    return alt.layer(*layers).properties(height=220, title=X.label(metric))


def contribution_chart(rec: dict) -> alt.Chart:
    df = pd.DataFrame({"term": [TERM_NAMES[t] for t in RK.TERMS],
                       "contribution": [rec["contributions"][t] for t in RK.TERMS],
                       "value": [rec["terms"][t] for t in RK.TERMS]})
    return (alt.Chart(df).mark_bar()
            .encode(x=alt.X("contribution:Q", title="weight × term"), y=alt.Y("term:N", sort=None, title=None),
                    color=alt.condition("datum.contribution < 0", alt.value("#C44E52"), alt.value("#4C72B0")),
                    tooltip=["term", alt.Tooltip("value:Q", format=".2f", title="term value"),
                             alt.Tooltip("contribution:Q", format=".3f")])
            .properties(height=170))


# ---------------------------------------------------------------- page

def main() -> None:
    data = load()
    st.warning(f"**{X.DISCLAIMER}** Research prototype on synthetic data (Synthea + simulated lifestyle).", icon="⚠️")

    with st.sidebar:
        st.title("🩺 LPHRS")
        opts = patient_options()
        # options are the display labels (no format_func) so the widget also works under AppTest
        label_to_pid = {lab: p for p, lab in opts}
        pid = label_to_pid[st.selectbox("Patient (sorted by CVD risk)", list(label_to_pid), key="patient_label")]
        st.session_state["patient"] = pid
        weights_choice = st.radio("Ranking weights", ["Default (hand-set)", "Learned (simulated feedback)"],
                                  key="weights",
                                  help="Learned weights come from simulated users and nearly ignore health relevance "
                                       "and urgency, so the defaults are the live setting.")
        if st.button("Reset feedback for this patient", key="reset"):
            reset_feedback(pid)
        st.caption(f"Data as of {data.ref:%Y-%m-%d}. {len(opts)} adult patients.")

    weights = None
    if weights_choice.startswith("Learned"):
        weights = load_learned_weights()
        if weights is None:
            st.sidebar.error("config/weights_learned.yaml not found; using defaults.")

    prof = health_profile(pid)
    state = feedback_state(pid)
    uc, history = FB.apply_state(base_context(pid), state)
    res = PL.run_pipeline(pid, data, weights=weights, history=history, user_context=uc, profile=prof)
    recs = res["recommendations"]
    st.session_state["last_recs"] = recs

    # ------------------------------------------------------------ header
    st.header(f"Patient {pid[:8]}")
    c = st.columns(5)
    c[0].metric("Age / sex", f"{uc['age']:.0f} / {uc['gender']}")
    c[1].metric("CVD risk (model)", f"{prof['cvd_risk']:.0%}" if prof["cvd_risk"] is not None else "n/a")
    c[2].metric("Daily time budget", f"{uc['minutes_per_day']} min")
    c[3].metric("Injury (survey)", uc["injury"])
    c[4].metric("Feedback given", len(feedback_log(pid)))
    st.caption("Goals: " + ", ".join(X.GOAL_LABELS[g] for g in uc["goals"]) + f" · Diet: {uc['diet_style']} · "
               f"Likes: {', '.join(uc['liked_exercise'])}")

    tab_rec, tab_time, tab_prof, tab_safe = st.tabs(["Recommendations", "Metric timelines", "Profile", "Safety filter"])

    # ------------------------------------------------------------ recommendations
    with tab_rec:
        prev = st.session_state.get("prev_ids", {}).get(pid)
        if prev is not None:
            new = [r["title"] for r in recs if r["id"] not in prev]
            gone = len([i for i in prev if i not in {r["id"] for r in recs}])
            st.info(f"Re-ranked after your feedback: {len(new)} new item(s), {gone} removed."
                    + (f" New: {'; '.join(new)}" if new else ""), icon="🔄")
        st.subheader(f"Today's plan · {res['minutes_used']} of {uc['minutes_per_day']} min")
        for i, r in enumerate(recs, 1):
            with st.container(border=True):
                left, right = st.columns([3, 2])
                with left:
                    st.markdown(f"**{i}. {r['title']}**  \n`{r['domain']}` · {r['minutes']} min · intensity "
                                f"{r['intensity']} · score **{r['score']:.3f}**")
                    for reason in r["reasons"]:
                        st.markdown(f"- {reason}")
                    cols = st.columns(4)
                    for col, outcome in zip(cols, FB.OUTCOMES):
                        col.button(OUTCOME_LABELS[outcome], key=f"fb-{pid}-{r['id']}-{outcome}",
                                   on_click=give_feedback, args=(pid, r, outcome), width="stretch")
                with right:
                    st.altair_chart(contribution_chart(r), width="stretch")
        st.caption(X.DISCLAIMER)

        with st.expander("Feedback so far and what it changed"):
            log = feedback_log(pid)
            if log:
                st.dataframe(pd.DataFrame(log), hide_index=True, width="stretch")
            else:
                st.write("No feedback yet. Use the buttons above; the list re-ranks immediately.")
            base = base_context(pid)
            caps = pd.DataFrame({"domain": list(state["caps"]),
                                 "starting cap": [base["intensity_caps"][d] for d in state["caps"]],
                                 "current cap": list(state["caps"].values())})
            st.markdown("**Intensity caps per domain** (lowered by *Too difficult*, raised by two completions at the cap)")
            st.dataframe(caps, hide_index=True)
            added = sorted(set(uc["preference_tags"]) - set(base["preference_tags"]))
            removed = sorted(set(base["preference_tags"]) - set(uc["preference_tags"]))
            st.markdown(f"**Preference tags:** {', '.join(uc['preference_tags'])}"
                        + (f"  \nlearned from feedback: +{', '.join(added)}" if added else "")
                        + (f"  \ndropped after rejections: −{', '.join(removed)}" if removed else ""))

    # ------------------------------------------------------------ timelines
    with tab_time:
        specs = metric_specs()
        available = [m for m in prof["metrics"] if m in specs]
        default = [m for m in available if prof["metrics"][m]["urgency"] >= 0.01] or available[:2]
        chosen = st.multiselect("Metrics", available, default=default, format_func=X.label, key=f"metrics-{pid}")
        st.caption("Red dashed line: threshold. Orange dotted line: projected crossing if the current trend continues "
                   "(shown when projected within 10 years). Sleep and steps are weekly means of synthetic data.")
        cols = st.columns(2)
        for i, m in enumerate(chosen):
            ch = metric_chart(pid, m, prof, data.ref)
            p = prof["metrics"][m]
            with cols[i % 2]:
                if ch is not None:
                    st.altair_chart(ch, width="stretch")
                ttt = p["time_to_threshold_years"]
                st.caption(f"Current {X.fmt_value(m, p['current'])} {p['unit']} · trend {p['trend']} "
                           f"({p['slope_per_year']:+.2f}/yr, R² {p['r2']:.2f}) · urgency {p['urgency']:.2f} "
                           f"({p['urgency_status'].replace('_', ' ')})"
                           + (f" · crosses in ~{X.fmt_years(ttt)}" if p["urgency_status"] == "approaching"
                              and np.isfinite(ttt) else ""))

    # ------------------------------------------------------------ profile
    with tab_prof:
        rows = [{"metric": X.label(m), "current": f"{X.fmt_value(m, p['current'])} {p['unit']}",
                 "last updated": p["last_updated"].strftime("%Y-%m-%d"), "threshold":
                     "–" if p["threshold"] is None else X.fmt_threshold(m, p["threshold"]),
                 "trend": p["trend"], "slope / yr": round(p["slope_per_year"], 2),
                 "variance": round(p["variance"], 2) if not np.isnan(p["variance"]) else None,
                 "urgency": round(p["urgency"], 3), "status": p["urgency_status"].replace("_", " "),
                 "history score": None if np.isnan(p["history_score"]) else round(p["history_score"], 2)}
                for m, p in prof["metrics"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        a, b = st.columns(2)
        with a:
            st.markdown("**Active conditions** (decay weight)")
            for cnd in prof["active_conditions"]:
                st.markdown(f"- {cnd['description']} · {cnd['weight']:.2f}")
            if prof["past_major_events"]:
                st.markdown("**Past major events**")
                for e in prof["past_major_events"]:
                    st.markdown(f"- {e['description']} · {e['weight']:.2f}")
        with b:
            st.markdown("**Active medications**")
            for m in sorted({m["description"] for m in prof["active_medications"]}) or ["none"]:
                st.markdown(f"- {m}")
            st.markdown("**Allergies** (weight stays at its floor however old)")
            for al in prof["allergies"] or [{"description": "none", "w_recency": float("nan")}]:
                st.markdown(f"- {al['description']}" + ("" if np.isnan(al["w_recency"]) else f" · {al['w_recency']:.2f}"))
            facts = res["safety_facts"]
            st.markdown(f"Mapped food allergens: **{', '.join(sorted(facts.allergens)) or 'none'}**")

    # ------------------------------------------------------------ safety
    with tab_safe:
        drops = res["drops"]
        st.markdown(f"**{res['n_candidates']:,}** candidates → **{res['n_after_safety']:,}** kept after the safety "
                    f"filter (**{drops['candidate_id'].nunique():,}** removed).")
        if drops.empty:
            st.success("No safety rule applies to this patient.")
        else:
            summary = (drops.groupby(["rule_id", "reason"]).agg(removed=("candidate_id", "nunique"),
                                                                  example=("title", "first")).reset_index()
                       .sort_values("removed", ascending=False))
            st.dataframe(summary, hide_index=True, width="stretch")
            with st.expander(f"All removed candidates ({len(drops):,} rule hits)"):
                st.dataframe(drops[["title", "domain", "rule_id", "reason"]], hide_index=True,
                             width="stretch", height=400)


main()
