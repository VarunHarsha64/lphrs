"""Phases 2-3: ingest, normalise, build timeline, synthetic lifestyle layers and health profiles.

Writes to data/processed/:
    patients.parquet      demographics, age at reference date, cohort flag
    timeline.parquet      one row per event (observations, lifestyle, conditions, medications, allergies)
    user_context.parquet  synthetic goals, preferences, minute budget, injuries, smoke/alco/active
    hidden_taste.parquet  simulator-only hidden vectors (never read by the ranker)
    lifestyle_params.parquet  per-patient baselines and trends used to generate sleep/steps
    build_meta.json       reference date, cohort rule, normalisation report, fitted model
    profile_metrics.parquet   per-patient, per-metric profile (trend, urgency, history score ...)
    profile_events.parquet    per-patient condition / medication / allergy decay weights
    temporal_params.json      cohort volatility and decay rate per metric

    python scripts/02_build_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lphrs import SEED  # noqa: E402
from lphrs import ingest, normalize, profile, synth_lifestyle as synth, temporal, timeline  # noqa: E402
from lphrs.config import PROCESSED_DIR, loinc_to_metric, metric_specs  # noqa: E402

ADULT_AGE = 18
LIFESTYLE_DAYS = 730

# active-condition keywords used to bias the synthetic layers (lower-case substring match)
CONDITION_FLAGS = {
    "has_hypertension": ["hypertension"],
    "has_glycemic": ["diabetes", "prediabetes"],
    "has_obesity": ["obesity", "severely obese"],
    "knee_condition": ["osteoarthritis of knee", "injury of knee", "cruciate ligament",
                       "collateral ligament of knee", "meniscus of knee"],
    "back_condition": ["chronic low back pain", "vertebral column"],
    "smoker_condition": ["smokes tobacco"],
    "alcohol_condition": ["alcoholism", "unhealthy alcohol"],
}


def active_condition_flags(cond: pd.DataFrame, ref: pd.Timestamp) -> pd.DataFrame:
    active = cond[(cond["timestamp"] <= ref) & (cond["stop"].isna() | (cond["stop"] > ref))]
    desc = active["description"].str.lower()
    flags = pd.DataFrame({"patient_id": active["patient_id"]})
    for flag, kws in CONDITION_FLAGS.items():
        flags[flag] = desc.apply(lambda d, kws=kws: any(k in d for k in kws))
    return flags.groupby("patient_id").any().reset_index()


def latest(meas: pd.DataFrame, metric: str) -> pd.Series:
    m = meas[meas["metric"] == metric]
    return m.sort_values("timestamp").groupby("patient_id")["value"].last().rename(metric)


def adult_volatility(tl: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    """Cohort volatility from readings taken at age >= 18 (child growth would inflate weight/BMI),
    using all patients including the deceased."""
    meas = profile.measurement_frame(tl).merge(patients[["patient_id", "birthdate"]], on="patient_id")
    age = (meas["timestamp"] - meas["birthdate"]).dt.days / 365.25
    return temporal.cohort_volatility(meas[age >= ADULT_AGE], profile.temporal_cfg())


def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    specs, code_map = metric_specs(), loinc_to_metric()

    print("Loading Synthea ...")
    patients = ingest.load_patients()
    encounters = ingest.load_encounters()
    obs_raw = ingest.load_observations(set(code_map))
    cond = normalize.normalize_events(ingest.load_conditions())
    meds = normalize.normalize_events(ingest.load_medications())
    alg = normalize.normalize_events(ingest.load_allergies())

    # Reference date = end of the simulated record (latest encounter start).
    ref = normalize.to_utc(encounters["timestamp"]).max()
    print(f"Reference date (latest encounter): {ref.isoformat()}")

    print("Normalising observations ...")
    meas, norm_report = normalize.normalize_health_data(obs_raw, specs, code_map)

    # Patients and cohort: living adults receive recommendations; everyone stays in the timeline.
    patients["birthdate"] = normalize.to_utc(patients["birthdate"])
    patients["deathdate"] = normalize.to_utc(patients["deathdate"])
    patients["alive"] = patients["deathdate"].isna()
    end = patients["deathdate"].fillna(ref)
    patients["age"] = ((end - patients["birthdate"]).dt.days / 365.25).round(1)
    patients["in_cohort"] = patients["alive"] & (patients["age"] >= ADULT_AGE)

    # Cohort features that condition the synthetic layers.
    cohort = patients[patients["in_cohort"]][["patient_id", "gender", "age"]].copy()
    for metric, col in (("bmi", "bmi"), ("systolic_bp", "sbp"), ("hba1c", "hba1c")):
        cohort = cohort.merge(latest(meas, metric).rename(col), left_on="patient_id",
                              right_index=True, how="left")
    cohort = cohort.merge(active_condition_flags(cond, ref), on="patient_id", how="left")
    flag_cols = list(CONDITION_FLAGS)
    cohort[flag_cols] = cohort[flag_cols].fillna(False).astype(bool)
    cohort = cohort.sort_values("patient_id").reset_index(drop=True)

    print("Generating synthetic lifestyle layers ...")
    model = synth.fit_lifestyle_model(ingest.load_sleep())
    daily, params = synth.generate_daily(cohort, model, ref, days=LIFESTYLE_DAYS, seed=SEED)
    context = synth.generate_context(cohort, params, model, seed=SEED)
    hidden = synth.generate_hidden_taste(cohort, context, seed=SEED)

    print("Building timeline ...")
    tl = timeline.build_timeline(meas, cond, meds, alg, lifestyle=daily)
    assert timeline.is_sorted(tl), "timeline is not sorted per patient"

    tl.to_parquet(PROCESSED_DIR / "timeline.parquet", index=False)
    patients.to_parquet(PROCESSED_DIR / "patients.parquet", index=False)
    context.to_parquet(PROCESSED_DIR / "user_context.parquet", index=False)
    hidden.to_parquet(PROCESSED_DIR / "hidden_taste.parquet", index=False)
    params.to_parquet(PROCESSED_DIR / "lifestyle_params.parquet", index=False)
    meta = {"seed": SEED, "reference_date": ref.isoformat(), "adult_age": ADULT_AGE,
            "lifestyle_days": LIFESTYLE_DAYS, "normalisation": norm_report, "lifestyle_model": model}
    (PROCESSED_DIR / "build_meta.json").write_text(json.dumps(meta, indent=2))

    print("Building health profiles (Phase 3) ...")
    vol = adult_volatility(tl, patients)
    goals = dict(zip(context["patient_id"], context["goals"].map(list)))
    prof_metrics, prof_events = profile.build_all_profiles(tl, ref, vol, goals, patient_ids=cohort["patient_id"])
    prof_metrics.to_parquet(PROCESSED_DIR / "profile_metrics.parquet", index=False)
    prof_events.to_parquet(PROCESSED_DIR / "profile_events.parquet", index=False)
    (PROCESSED_DIR / "temporal_params.json").write_text(vol.to_json(orient="index", indent=2))

    # ------------------------------------------------------------ summary
    print("\n" + "=" * 64)
    print("Build summary")
    print("=" * 64)
    print(f"Patients: {len(patients)} total | {patients['alive'].sum()} alive | "
          f"{(~patients['alive']).sum()} deceased | {patients['in_cohort'].sum()} in cohort (alive, 18+)")
    print(f"Patients with any timeline event: {tl['patient_id'].nunique()}")
    print("\nNormalisation of observations:")
    for k, v in norm_report.items():
        print(f"  {k:<24}{v:>10,}")
    print("\nEvents per type:")
    for k, v in tl["event_type"].value_counts().items():
        print(f"  {k:<24}{v:>10,}")
    print("\nMeasurements per metric:")
    per_metric = tl[tl["metric"].notna()].groupby(["event_type", "metric"]).agg(
        rows=("value", "size"), patients=("patient_id", "nunique"))
    for (et, m), r in per_metric.iterrows():
        print(f"  {et:<12}{m:<20}{r['rows']:>10,} rows  {r['patients']:>5} patients")
    print("\nSynthetic lifestyle (cohort):")
    print(f"  sleep baseline mean {params['sleep_base'].mean():.2f} h | "
          f"steps baseline mean {params['steps_base'].mean():,.0f} | "
          f"trending: sleep {params['sleep_trending'].mean():.0%}, steps {params['steps_trending'].mean():.0%}")
    goal_counts = pd.Series(np.concatenate(context["goals"].to_numpy())).value_counts()
    print("  goals:", ", ".join(f"{g} {c}" for g, c in goal_counts.items()))
    print("  injuries:", context["injury"].value_counts().to_dict())
    print(f"  smoke {context['smoke'].mean():.0%} | alco {context['alco'].mean():.0%} | "
          f"active {context['active'].mean():.0%} | minutes/day median {context['minutes_per_day'].median():.0f}")
    print("\nCohort volatility and decay:")
    for m, r in vol.sort_values("volatility").iterrows():
        print(f"  {m:<20} volatility {r['volatility']:5.2f}  lambda {r['lambda']:.5f}/day  "
              f"half-life {np.log(2) / r['lambda']:5.0f} days")
    print(f"\nProfiles: {prof_metrics['patient_id'].nunique()} patients, {len(prof_metrics):,} metric rows, "
          f"{len(prof_events):,} event rows")
    print(f"\nWrote {len(tl):,} timeline rows to {PROCESSED_DIR / 'timeline.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
