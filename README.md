# LPHRS: Longitudinal Personalised Health Recommendation System

A Python research prototype that reads a patient's health record **over time**, works out which
measurements are drifting toward a problem, and suggests up to five safe, personalised lifestyle
actions with plain-language reasons.

> **Wellness guidance only. Not medical advice.** All clinical data is synthetic (Synthea), and the
> lifestyle, goals and feedback layers are simulated. Nothing here has been reviewed by a clinician.

What it does, end to end:

1. **Ingests** Synthea records (vitals, labs, conditions, medications, allergies) and builds one
   timeline per patient, plus two years of synthetic daily sleep and step data.
2. **Profiles** each patient over time: trends, time-to-threshold **urgency** (e.g. "systolic BP is rising
   ~4 mmHg/year and will reach 140 in about 3 months"), and a decay that lets old routine readings fade
   while old allergies and chronic conditions keep their weight.
3. **Scores CVD risk** with a calibrated XGBoost model trained on the Kaggle cardiovascular dataset.
4. **Generates candidates** (77 hand-written interventions + 5,000 recipes) and removes unsafe ones
   with deterministic **safety rules** (allergies, sodium/sugar caps, injuries, heart conditions,
   warfarin/vitamin K, statins/grapefruit), logging every removal.
5. **Ranks and selects** a five-item plan that fits the user's daily time budget, mixes domains and avoids
   conflicting items, and **explains** each pick.
6. **Adapts** to feedback (accept / complete / reject / too difficult), and learns ranking weights from
   simulated users.
7. **Evaluates** all of this in seven experiments and three case studies
   ([reports/results.md](reports/results.md)), and shows it in a **Streamlit dashboard**.

---

## Quick start: explore the results without any raw data

This repository already includes everything the scripts generated: the processed data (`data/processed/`),
the trained risk model (`models/`) and all reports and figures (`reports/`). You do **not** need Java,
Synthea or the Kaggle downloads to run the dashboard, the CLI or the tests.

Needs **Python 3.12 or newer** (built and tested on 3.14) and about 700 MB for packages.

```bash
git clone <this repository's URL>
cd <repository folder>
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
streamlit run app/dashboard.py
```

The browser opens at http://localhost:8501. The sidebar has three pages:

- **dashboard**: pick a patient, see the recommendations with reasons, metric timelines with projected
  threshold crossings, the safety filter's removals, and feedback buttons that re-rank the list live;
- **Results**: all seven experiments and three case studies with figures (`reports/results.md`);
- **Project Report**: a long, beginner-friendly explanation of the whole project (`reports/PROJECT_REPORT.md`).

Also without raw data:

```bash
python scripts/04_run_pipeline.py                      # list example patients
python scripts/04_run_pipeline.py --patient 355d5f82 --breakdown
python -m pytest -q                                    # 314 tests
```

Only re-running steps 2, 3, 5 or 6 needs the raw datasets (see Setup below).

---

## Setup (full rebuild from raw data)

### Requirements

| Need | Version used | Notes |
|---|---|---|
| Python | 3.14 | All packages publish 3.14 wheels. |
| Java | **17 or newer** | Only for generating Synthea data. The current Synthea release needs Java 17 (the original plan said 11). |
| Disk | ~3.2 GB | Synthea CSVs ~970 MB (+ a duplicate ~975 MB in `data/raw/synthea_build/output`, safe to delete after step 1), recipes ~280 MB, Synthea jar ~190 MB, virtual environment ~690 MB. |

### 1. Create the environment

```powershell
cd lphrs
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

`scikit-learn==1.8.0` and `scipy==1.17.0` are pinned on purpose: on this Windows machine,
**Smart App Control** blocked the newest 3.14 builds of those two packages because they were too new to
have a reputation. If a future package fails with *"An Application Control policy has blocked this
file"*, installing a slightly older version is the first thing to try.

### 2. Get the data

Run the checker first; it lists what is present and what you still need:

```powershell
python scripts/01_download.py
```

| Dataset | Where it goes | How to get it |
|---|---|---|
| Synthea (1,000 patients) | `data/raw/synthea/` | `python scripts/01_download.py --generate-synthea` downloads the jar (~200 MB, resumes if interrupted) and generates the data with seed 42. Needs Java 17+; if `java -version` shows an older one, set `$env:JAVA_HOME` to the newer JDK in that terminal first. Fallback: `--synthea-sample` (small pre-built sample). |
| `cardio_train.csv` | `data/raw/cvd/` | [Kaggle: Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) |
| `RAW_recipes.csv` | `data/raw/recipes/` | [Kaggle: Food.com Recipes and Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions). Only this one file is needed. |
| `Sleep_health_and_lifestyle_dataset.csv` | `data/raw/sleep/` | [Kaggle: Sleep Health and Lifestyle](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset) |

Kaggle downloads need a free account; unzip and place only the named CSV. With a Kaggle API token in
`~/.kaggle/kaggle.json`, `python scripts/01_download.py --kaggle` fetches them for you.

Re-run `python scripts/01_download.py` until every dataset shows **PRESENT**.

---

## Running it

Run the scripts in order from the `lphrs` folder (with the virtual environment active).
Times are from the development machine.

| Step | Command | Time | Writes |
|---|---|---|---|
| 1 | `python scripts/01_download.py` | seconds | status report (and data, with flags) |
| 2 | `python scripts/02_build_dataset.py` | ~2 min | `data/processed/`: timeline, patients, synthetic lifestyle/goals/preferences, hidden simulator vectors, health profiles, decay parameters |
| 3 | `python scripts/03_train_risk_model.py` | ~30 s | `models/cvd_risk.joblib`, `reports/risk_model_metrics.json`, calibration and feature-importance plots, `data/processed/cvd_risk.parquet` |
| 4 | `python scripts/04_run_pipeline.py --patient <id>` | ~2 s | prints 5 recommendations with reasons and the safety drops |
| 5 | `python scripts/05_simulate_feedback.py` | ~15–25 min | `config/weights_learned.yaml`, `data/processed/feedback_log.parquet`, learning-curve and weight plots |
| 6 | `python scripts/06_evaluate.py` | ~3 min | `reports/results.md`, experiment figures, case-study figures |
| — | `streamlit run app/dashboard.py` | ~5 s to load | interactive dashboard at http://localhost:8501 |

Useful options:

```powershell
python scripts/04_run_pipeline.py                         # list example patient ids
python scripts/04_run_pipeline.py --patient 233b7fd8      # an id prefix is enough
python scripts/04_run_pipeline.py --patient 233b7fd8 --breakdown        # show every score term
python scripts/04_run_pipeline.py --patient 233b7fd8 --weights learned  # use the learned weights
python scripts/05_simulate_feedback.py --patients 30 --rounds 10        # quick trial run
```

### The dashboard

`streamlit run app/dashboard.py` gives:

- a **patient picker** (775 adults, sorted by CVD risk) and a default/learned weights switch;
- **recommendations** with reasons, a per-term score breakdown and **feedback buttons that re-rank the
  list live** (accept, completed, reject, too difficult);
- **metric timelines** with threshold lines and the **projected crossing** for rising metrics;
- the **profile** (metrics, conditions, medications, allergies with their decay weights);
- the **safety filter**: what was removed and which rule removed it.

---

## Tests

```powershell
python -m pytest -q
```

314 tests: unit tests for every module, the plan's acceptance checks (e.g. BP readings 120, 124, 130, 136
over 4 years give a slope of about 5.4/year and a finite time-to-threshold; a peanut-allergic patient
with a knee injury gets zero peanut recipes and zero high-impact exercise), an equivalence test of the
fast set-selection code against a reference implementation on 200 random pools, integration tests on
the real processed data, and dashboard interaction tests. Tests that need the processed data are skipped
if it has not been built.

---

## Project layout

```
lphrs/
  config/
    metrics.yaml          per-metric LOINC codes, units, valid ranges, thresholds, decay settings
    safety_rules.yaml     hard safety rules + allergen and interaction keyword lists
    interventions.yaml    77 intervention templates, recipe filters and diet tags, preference vocabularies
    synergy.yaml          tag-pair synergy / conflict / redundancy values
    weights.yaml          default ranking weights (the live setting), K, alpha, domain cap
    weights_learned.yaml  learned on simulated users (opt-in)
  src/lphrs/
    ingest.py  normalize.py  timeline.py  synth_lifestyle.py      data layer
    temporal.py  profile.py                                       decay, trends, urgency, profiles
    risk_model.py                                                 CVD classifier
    candidates.py  safety.py  ranker.py  explain.py  pipeline.py  recommendation pipeline
    feedback.py  weight_learner.py                                feedback, simulated users, learning
    evaluate.py                                                   experiments 1-7
  scripts/01_download.py ... 06_evaluate.py
  app/dashboard.py
  tests/
  reports/  results.md, figures/, metrics JSON files
  data/raw, data/processed, models/      (git-ignored)
```

### How the scoring works

Each candidate gets seven terms in [0, 1]:

`Score = 0.25·R + 0.20·G + 0.20·U + 0.10·P + 0.10·F + 0.10·H − 0.05·C`

| Term | Meaning |
|---|---|
| R relevance | targets a metric past its threshold (or often beyond it), a cardiovascular metric when CVD risk is high, or an active chronic condition |
| G goals | share of the user's goals it serves |
| U urgency | time-to-threshold urgency of the metrics it targets |
| P preference | overlap with the user's preferred diet and exercise tags |
| F feasibility | fits the daily time budget and the domain's intensity cap |
| H history | smoothed acceptance of this item and its domain (0.5 with no feedback) |
| C cost | effort |

A greedy selector then builds the five-item plan: it adds synergy between items (e.g. evening walk +
earlier bedtime), penalises conflicts and duplicates (late caffeine + sleep hygiene, two walks), keeps to
the time budget with a look-ahead so the list can be completed, and allows at most two items per domain.

### Choices that differ from the build plan

| Area | What changed | Why |
|---|---|---|
| Java | 17+ instead of 11+ | the current Synthea release is compiled for Java 17 |
| Cohort | 775 living adults get recommendations | 225 of the 1,000 living patients are children; the risk model and lifestyle data are adult-only |
| Safety keywords | matched against the real Synthea wording | e.g. soy is recorded as "Glycine max"; "coronary" alone matched an imaging finding |
| HbA1c valid range | lower bound 2.0 % instead of 3 % | Synthea produces 2.4–2.9 % for some healthy patients |
| Volatility | made unit-free (change SD ÷ mean level) | raw SDs in different units cannot be compared |
| Trends | need readings spread over ≥ 1 year | readings from one hospital stay produced a 199 mg/dL/year "trend" |
| Conditions | split into chronic / resolved major / acute / social | social records (employment, housing) should not keep a permanent weight |
| Relevance R | also counts active chronic conditions | otherwise one abnormal metric (BMI) crowded out everything |
| Recipes | meals only, ≥ 5 min, no alcohol, no processed meat, no whole-batch nutrition | a cocktail, a salad dressing and a hot-dog recipe were ranking first |
| Set selection | budget look-ahead + redundancy penalties | plain greedy could stop at 3 items or pick two walks |
| Live weights | hand-set defaults stay live; learned weights are opt-in | learned weights maximise simulated acceptance and nearly ignore health relevance and urgency |
| Simulator | base acceptance offset −1.75 | so about half of first-round items are accepted instead of 82 % |

---

## Results at a glance

| # | Experiment | Result (775 adult patients unless stated) |
|---|---|---|
| 1 | Safety filter on vs off | top-5 lists containing an unsafe item: **7.9 % → 0.0 %** |
| 2 | Urgency term off vs on | rising-but-normal metric targeted: 72.3 % → 72.5 % overall; 63.4 % → 66.3 % where urgency ≥ 0.2 |
| 3 | Early warning (last 2 years hidden) | at similar alarm counts a simple "within 2 % of threshold" flag beat urgency ≥ 0.2 on recall (18.1 % vs 7.3 %) and precision (36.7 % vs 23.2 %); urgency warned earlier for the crossings it caught (median 36 vs 13 months) |
| 4 | Floored vs plain decay | events older than 5 years: allergies keep weight 1.00 (plain 0.0000), chronic conditions 0.80 (plain 0.0015) |
| 5 | Greedy set vs independent top-5 | conflicting pairs per list 5.06 → 0.46; domains 2.03 → 3.37; within time budget 47.2 % → 100 % |
| 6 | Learned vs fixed weights (simulated users) | acceptance in the last 5 of 30 rounds: 61.9 % (fixed) vs 63.8 % (learned), both from 56.0 % |
| 7 | CVD risk model (Kaggle test set) | ROC-AUC 0.795 (logistic regression) vs 0.806 (XGBoost); Brier 0.185 vs 0.178 |

Full methods, tables, figures and three worked case studies: **[reports/results.md](reports/results.md)**.

---

## Limitations

- **Synthetic data.** Clinical records come from Synthea. Lifestyle, goals, preferences, injuries and
  feedback are simulated; no public dataset links them to clinical records.
- **Not clinically reviewed.** Thresholds and safety rules are simplified illustrations.
- **Trend projection** assumes the recent linear trend continues. Synthea readings are roughly yearly, and
  in the hidden-future test the trend-based urgency flag did **not** beat a simple "latest value near the
  threshold" flag at predicting crossings within two years (it warned earlier for the cases it caught).
- **Learned weights** reflect the simulator's assumptions. The history term is partly self-reinforcing
  (it is computed from the same feedback the learner trains on), and maximising acceptance pushes health
  relevance and urgency down.
- **Risk model** is trained on a different population (Kaggle, ages 29–65) from the one it scores; patients
  outside that age range are treated as the nearest training age, and missing labs default to "normal".
- **Recipe allergens** are found by ingredient keyword matching. An audit found and fixed 38 untagged seafood
  recipes, but keyword matching can still miss unusual ingredient names; real use would need a curated
  ingredient–allergen database.
- **Recipe diet tags** only check sodium, sugar, protein and meat content, not overall food quality. Processed
  meats and alcohol are excluded from the pool (a hot-dog recipe once ranked first), but other low-quality
  foods could still score well.
- The system gives **wellness guidance only** and does not diagnose or prescribe.
