"""Loaders for each raw dataset. Column names are lower-cased and renamed to project names.

Synthea loaders return raw strings for dates and values; normalize.py does the parsing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import RAW_DIR

SYNTHEA_DIR = RAW_DIR / "synthea"


def _read(path: Path, usecols: list[str], **kw) -> pd.DataFrame:
    return pd.read_csv(path, usecols=usecols, dtype=str, keep_default_na=False,
                       na_values=[""], engine="pyarrow", **kw)


def load_patients(raw_dir: Path = SYNTHEA_DIR) -> pd.DataFrame:
    df = _read(raw_dir / "patients.csv", ["Id", "BIRTHDATE", "DEATHDATE", "GENDER", "RACE", "ETHNICITY"])
    return df.rename(columns={"Id": "patient_id", "BIRTHDATE": "birthdate", "DEATHDATE": "deathdate",
                              "GENDER": "gender", "RACE": "race", "ETHNICITY": "ethnicity"})


def load_observations(codes: set[str], raw_dir: Path = SYNTHEA_DIR) -> pd.DataFrame:
    """Only the rows whose LOINC code is in `codes` (observations.csv is large)."""
    df = _read(raw_dir / "observations.csv", ["DATE", "PATIENT", "CATEGORY", "CODE", "VALUE", "UNITS"])
    df = df[df["CODE"].isin(codes)]
    return df.rename(columns={"DATE": "timestamp", "PATIENT": "patient_id", "CATEGORY": "category",
                              "CODE": "code", "VALUE": "value", "UNITS": "unit"}).reset_index(drop=True)


def _load_events(fname: str, extra: dict[str, str], raw_dir: Path) -> pd.DataFrame:
    base = {"START": "timestamp", "STOP": "stop", "PATIENT": "patient_id",
            "CODE": "code", "DESCRIPTION": "description"}
    cols = {**base, **extra}
    df = _read(raw_dir / fname, list(cols))
    return df.rename(columns=cols)


def load_conditions(raw_dir: Path = SYNTHEA_DIR) -> pd.DataFrame:
    return _load_events("conditions.csv", {}, raw_dir)


def load_medications(raw_dir: Path = SYNTHEA_DIR) -> pd.DataFrame:
    return _load_events("medications.csv", {"REASONDESCRIPTION": "reason"}, raw_dir)


def load_allergies(raw_dir: Path = SYNTHEA_DIR) -> pd.DataFrame:
    return _load_events("allergies.csv", {"TYPE": "allergy_type", "CATEGORY": "allergy_category"}, raw_dir)


def load_encounters(raw_dir: Path = SYNTHEA_DIR) -> pd.DataFrame:
    df = _read(raw_dir / "encounters.csv", ["Id", "START", "STOP", "PATIENT", "ENCOUNTERCLASS"])
    return df.rename(columns={"Id": "encounter_id", "START": "timestamp", "STOP": "stop",
                              "PATIENT": "patient_id", "ENCOUNTERCLASS": "encounter_class"})


def load_sleep(raw_dir: Path = RAW_DIR / "sleep") -> pd.DataFrame:
    """Sleep Health and Lifestyle dataset (used only to fit lifestyle distributions)."""
    df = pd.read_csv(raw_dir / "Sleep_health_and_lifestyle_dataset.csv")
    df = df.rename(columns={"Gender": "gender", "Age": "age", "Sleep Duration": "sleep_hours",
                            "Stress Level": "stress", "BMI Category": "bmi_category",
                            "Daily Steps": "steps"})
    # the dataset uses both "Normal" and "Normal Weight" for the same category
    df["bmi_category"] = df["bmi_category"].replace({"Normal Weight": "Normal"})
    df["gender"] = df["gender"].map({"Male": "M", "Female": "F"})
    return df[["gender", "age", "sleep_hours", "steps", "stress", "bmi_category"]]
