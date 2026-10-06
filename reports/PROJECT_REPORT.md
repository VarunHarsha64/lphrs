# LPHRS: Complete Project Report

**Longitudinal Personalised Health Recommendation System**
A full explanation of what was built, how it works, why each decision was made, and what went wrong
along the way.

> **Wellness guidance only. Not medical advice.** Everything in this project runs on synthetic
> (computer-generated) patients. Nothing has been reviewed by a doctor.

---

## How to read this report

This report assumes you know basic programming (variables, functions, loops, lists, dictionaries)
and have heard the main ideas of AI/ML (training a model, features, labels, accuracy), but have not
built a real system before. Every technical idea is explained the first time it appears, and there
is a glossary at the end.

If you are short on time, read in this order:

1. **Part A** (one-page summary) to know what the project is.
2. **Part D** (the novel algorithms) because that is what makes the project original.
3. **Part H** (results) and **Part J** (questions your teacher may ask).

Contents:

- **Part A**: The project in one page
- **Part B**: Background concepts you need
- **Part C**: Tools, languages and libraries, and why each one
- **Part D**: The data
- **Part E**: How the system is put together (architecture)
- **Part F**: The novel algorithms, explained step by step with worked examples
- **Part G**: The machine-learning models (risk model and weight learner)
- **Part H**: Evaluation: the seven experiments and what they really show
- **Part I**: The build, phase by phase, including every mistake and correction
- **Part J**: Questions your teacher may ask, with answers
- **Part K**: Limitations and future work
- **Part L**: How to demo the project
- **Part M**: Glossary

---

# Part A: The project in one page

### The problem

Most health apps give advice based on a **snapshot**: your latest blood pressure, your latest weight.
But health is a **story over time**. A blood pressure of 136 that has been rising 5 points every year
is more worrying than a blood pressure of 136 that has been stable for ten years, even though the
snapshot is identical. Likewise, a peanut allergy recorded 20 years ago is still just as dangerous today,
while a sore throat from 20 years ago is irrelevant.

### What we built

A Python system that:

1. **Reads a patient's full health history** (blood pressure, cholesterol, blood sugar, weight,
   conditions, medications, allergies) and puts it on one timeline.
2. **Works out what is changing**: for each measurement it fits a trend line and estimates **how soon
   it will cross a danger threshold** (for example, "systolic blood pressure reaches 140 in about 3
   months"). This is called **time-to-threshold urgency**.
3. **Decides how much old information still matters**, using a **decay** where old routine readings fade
   away but allergies and chronic diseases never fade below a minimum ("floor").
4. **Estimates cardiovascular (heart disease) risk** with a trained machine-learning model.
5. **Builds a list of possible actions** (77 hand-written lifestyle suggestions plus 5,000 real recipes),
   then **removes anything unsafe** for that person (allergens, too much salt for high blood pressure,
   high-impact exercise for a bad knee, and so on).
6. **Scores and picks five actions** that fit into the person's daily free time, cover different areas of
   life (food, exercise, sleep, etc.), and do not conflict with each other.
7. **Explains each pick in plain English**, for example: *"Your systolic blood pressure has risen about
   4.0 mmHg per year and is projected to reach 140 in about 3 months."*
8. **Learns from feedback** (accepted, completed, rejected, too difficult) and adapts.
9. **Shows everything in an interactive web dashboard.**
10. **Measures itself honestly** in seven experiments.

### The data

- **Synthea**: a free tool that generates realistic but fake patient records. We generated 1,000 living
  patients (plus 156 who "died" during the simulation), 15 years of history each.
- **Kaggle Cardiovascular Disease dataset**: 70,000 real (anonymised) people, used to train the heart-risk model.
- **Kaggle Food.com recipes**: 231,637 recipes with nutrition, of which we use 5,000.
- **Kaggle Sleep Health and Lifestyle dataset**: 374 people, used to make realistic synthetic sleep and step data.

### The four original ("novel") components

| # | Component | One-line idea |
|---|---|---|
| 1 | **Persistence-floored adaptive decay** | Old data fades, but how fast depends on how jumpy that measurement is, and serious things (allergies, chronic disease) never fade below a floor. |
| 2 | **Time-to-threshold urgency** | Fit a trend line to recent readings and turn "how many years until it crosses the danger line, and how sure are we" into a 0–1 urgency score. |
| 3 | **Seven-term scoring** | Each suggestion gets a score from health relevance, goals, urgency, preferences, feasibility, past success and effort. |
| 4 | **Cross-domain set selection** | Instead of "take the 5 highest scores", build the list one item at a time, rewarding items that work well together and keeping to the time budget. |

### Headline results

| Experiment | Result |
|---|---|
| Safety filter | Unsafe items in top-5 lists: **7.9 % of lists without the filter → 0 % with it** |
| Set selection | Conflicting pairs per list **5.06 → 0.46**; lists within the time budget **47 % → 100 %** |
| Decay | 5+ year-old allergies keep weight **1.00** (plain decay: 0.0000) |
| Risk model | ROC-AUC **0.806** (XGBoost) vs 0.795 (logistic regression) |
| Weight learning (simulated) | Acceptance **63.8 %** with learned weights vs 61.9 % with fixed weights |
| Early warning | **Honest negative result**: the trend-based urgency flag did *not* beat a simple "near the threshold" flag at predicting crossings within 2 years |

### Size of the project

About **4,300 lines** of Python (2,500 in the core library `src/lphrs`, 1,800 in the scripts and
dashboard), **1,400 lines** of tests (**314 automated tests**, all passing), 5 hand-written configuration
files (plus `weights_learned.yaml`, which the simulation generates), 6 scripts, 1 dashboard.

---

# Part B: Background concepts you need

This part explains every idea the project relies on. Skip anything you already know.

## B1. Health-data basics

**Longitudinal data.** Data about the same person collected repeatedly over time. "Longitudinal"
is the opposite of "cross-sectional" (one moment in time for many people). Our system is
longitudinal: each patient has years of readings.

**Vital signs and lab tests.** Vital signs are quick body measurements (blood pressure, heart rate,
weight). Lab tests need a blood sample (cholesterol, glucose, HbA1c).

**The measurements we use** and their danger thresholds:

| Measurement | What it is | Unit | Threshold | "Bad direction" |
|---|---|---|---|---|
| Systolic blood pressure | top BP number, pressure when the heart beats | mmHg | 140 | up |
| Diastolic blood pressure | bottom number, pressure between beats | mmHg | 90 | up |
| BMI | body mass index = weight(kg) ÷ height(m)² | kg/m² | 30 | up |
| HbA1c | average blood sugar over ~3 months | % | 6.5 | up |
| Glucose | blood sugar right now | mg/dL | 126 | up |
| Total cholesterol | all cholesterol in blood | mg/dL | 240 | up |
| LDL | "bad" cholesterol | mg/dL | 160 | up |
| HDL | "good" cholesterol | mg/dL | 40 | **down** (low is bad) |
| Sleep (synthetic) | hours per night | h | 6 | **down** |
| Steps (synthetic) | steps per day | steps | 5,000 | **down** |

"Bad direction" matters: for most metrics, rising is bad. For HDL, sleep and steps, *falling* is bad.
The algorithm has to handle both.

**LOINC codes.** Every lab test has a standard international code so computers agree on what was
measured. For example, systolic blood pressure is LOINC `8480-6`. We look up measurements by these codes.

**SNOMED tags.** Synthea's condition names end with a tag in brackets: `(disorder)`, `(finding)`,
`(situation)`. We use these tags as extra context.

**Synthetic data.** Data generated by a computer program to look realistic, with no real people
behind it. We use it because real health records are private and hard to get legally. The downside is
that results on synthetic data may not hold for real people.

## B2. Maths and statistics basics

**Linear regression (fitting a straight line).** Given points (time, value), find the straight line
`value = slope × time + intercept` that is closest to all points (it minimises the sum of squared
vertical distances; this is called **ordinary least squares, OLS**). The **slope** tells you how fast
the value changes per unit of time.

**R² ("R-squared").** How well the line fits, from 0 (the line explains nothing) to 1 (all points sit
exactly on the line). We use it as a measure of **how much to trust a trend**.

**Exponential decay and half-life.** A weight that shrinks over time as `w = e^(−λ·t)`, where λ
("lambda") is the decay rate. The **half-life** is the time for the weight to halve: `ln 2 / λ`.
With λ = 0.0019 per day, the half-life is about 365 days (one year).

**Sigmoid (logistic) function.** `sigmoid(z) = 1 / (1 + e^(−z))`. It squashes any number into the
range 0–1, so it can turn a score into a probability. sigmoid(0) = 0.5; big positive z → near 1;
big negative z → near 0.

**Standard deviation (SD).** How spread out numbers are. Small SD = values cluster; big SD = values vary a lot.

**Median and IQR.** The median is the middle value when sorted (less affected by extremes than the
average). The IQR (inter-quartile range) is the range covering the middle 50 % of values.

**Random seed.** Computers generate "random" numbers from a starting number called a seed. Using the
same seed (we always use **42**) gives the same "random" numbers every run, so results are reproducible.

## B3. Machine-learning basics

**Supervised learning / classification.** You give the model examples with known answers
(**features** → **label**). It learns a rule, then predicts labels for new examples. Our risk model's
features are age, blood pressure, cholesterol, etc.; the label is "has cardiovascular disease: yes/no".

**Train / validation / test split.** You split the data into three parts:

- **Training set** (70 %): the model learns from it.
- **Validation set** (15 %): used to choose settings and calibrate, like a practice exam.
- **Test set** (15 %): touched only once at the very end, like the real exam. This gives an honest
  estimate of performance on unseen data.

**Stratified split.** Each part keeps the same proportion of yes/no labels as the full data.

**Overfitting.** When a model memorises the training data instead of learning general patterns. It
looks great on training data and poor on new data. The validation/test split catches this.

**Logistic regression.** A simple classification model. It computes a weighted sum of the features
(`z = w1·x1 + w2·x2 + … + b`) and passes it through the sigmoid to get a probability. Each weight
(**coefficient**) shows how much that feature pushes the prediction up or down. It is easy to explain,
which is why it is a good **baseline**.

**Decision tree.** A flowchart of yes/no questions ("is systolic BP > 140?") ending in a prediction.

**Gradient boosting and XGBoost.** Instead of one big tree, build many small trees **one after another**,
where each new tree tries to correct the mistakes of the trees before it. The final prediction adds up
all the trees. **XGBoost** is a fast, popular library for this. Its main settings
(**hyperparameters**):

- `max_depth`: how many questions deep each tree can go (deeper = more complex).
- `learning_rate`: how much each new tree is allowed to change the prediction (smaller = slower, more careful learning).
- `n_estimators`: how many trees.

**Hyperparameter tuning.** Trying several combinations of settings and keeping the one that scores best
on the validation set (never the test set).

**Calibration.** A model might say "70 % risk" for a group of people of whom only 55 % actually have the
disease. Then its probabilities are not **calibrated**. Calibration fixes the mapping from model score to
true probability. Two methods:

- **Platt scaling (sigmoid)**: fit a logistic curve from the model's scores to the true labels.
- **Isotonic regression**: fit a step-shaped curve that only goes up (more flexible).

**Evaluation metrics for classification:**

| Metric | Plain meaning | Range / good value |
|---|---|---|
| Accuracy | share of predictions that are right | 0–1, higher better |
| Precision | of the people flagged as positive, how many really are | 0–1, higher better |
| Recall | of the real positives, how many were flagged | 0–1, higher better |
| F1 | a balance of precision and recall | 0–1, higher better |
| ROC-AUC | the chance the model ranks a random sick person above a random healthy person | 0.5 = coin flip, 1 = perfect |
| PR-AUC | area under the precision–recall curve; focuses on the positive class | higher better |
| Brier score | average squared gap between predicted probability and what happened | 0 = perfect, lower better |

**Feature importance.** How much each feature contributed to the model's decisions. For XGBoost we use
"gain": how much each feature improved the trees' accuracy when it was used for a split.

## B4. Recommender-system basics

A **recommender system** suggests items to a user (films, products, or here, health actions). Most
real recommenders follow the same pipeline, and so does ours:

1. **Candidate generation**: collect everything that could be suggested.
2. **Filtering**: remove things that must never be shown (here: unsafe items).
3. **Scoring/ranking**: give each remaining item a score for this user.
4. **Re-ranking / set selection**: build the final short list, considering how items fit together
   (variety, conflicts, time).
5. **Explanation**: say why each item was chosen.
6. **Feedback loop**: learn from what the user accepts or rejects.

A known risk of feedback loops is that the system can reinforce its own past choices. We found and
documented this effect in Part H.

## B5. Algorithm basics

**Greedy algorithm.** Build a solution step by step, always taking the option that looks best right now.
Greedy algorithms are fast but not always perfect overall. Our set selection is greedy.

**Simulation.** When you cannot test with real people, you write a program that pretends to be people
(a **simulator**), with behaviour rules, and test your system against it. Results then only show how
the system behaves *under the simulator's assumptions*.

**Common random numbers.** When comparing two versions in a simulation, give both the same random
draws, so any difference comes from the versions, not from luck.

---

# Part C: Tools, languages and libraries, and why each one

| Tool | Version | What it is | Why we used it |
|---|---|---|---|
| **Python** | 3.14.8 | programming language | The standard language for data science and ML; huge library ecosystem; readable. 3.14 was the version on your machine, and every library we needed supports it. |
| **pandas** | 3.0.6 | tables ("DataFrames") in Python | Loading CSVs, filtering, grouping and joining patient records. Almost every module uses it. |
| **NumPy** | 2.5.3 | fast maths on arrays | Trend fitting, decay formulas, vectorised scoring (doing a calculation on thousands of items at once instead of in a slow loop). |
| **PyArrow / Parquet** | 25.0.1 | fast column-based file format | Our processed data (1.3 million timeline rows) is stored as `.parquet` files: much smaller and faster to read than CSV, and it keeps data types (dates stay dates). The plan required "Parquet files, no database server". |
| **scikit-learn** | 1.8.0 | classic ML library | Logistic regression, train/test splitting, calibration, metrics (ROC-AUC, Brier, etc.). Industry standard, well documented. |
| **XGBoost** | 3.4.1 | gradient-boosted trees | Usually the strongest model for table-shaped data like the CVD dataset. |
| **SciPy** | 1.17.0 | scientific maths | Required by scikit-learn. |
| **PyYAML** | 6.0.3 | reads `.yaml` config files | All thresholds, rules, templates and weights live in human-readable YAML files, so they can be changed without touching code. |
| **Matplotlib** | 3.11.2 | plotting | All report figures (calibration curve, learning curve, case-study timelines). |
| **Altair** | 6.3.0 | interactive charts | Charts inside the dashboard (comes with Streamlit). |
| **Streamlit** | 1.65.0 | turns a Python script into a web app | The dashboard, without writing HTML/JavaScript. |
| **pytest** | 9.1.1 | testing framework | 314 automated tests that check every part still works after each change. |
| **joblib** | 1.6.0 | saves Python objects to disk | Saving the trained risk model (`models/cvd_risk.joblib`). |
| **Java** (Temurin 21) | 21 | programming language runtime | Only needed to run **Synthea**, which is written in Java. |
| **Synthea** | latest release | synthetic patient generator (by MITRE) | Realistic fake medical records without privacy issues. |
| **Git Bash / PowerShell** | — | command-line shells on Windows | Running scripts. |
| **venv** | built into Python | an isolated folder of packages | Keeps this project's library versions separate from the rest of the computer. |

Two packages in `requirements.txt` are listed because the build plan asked for them but are not
essential: **kaggle** (only used if you have a Kaggle API token, for automatic download) and **tqdm**
(progress bars; not actually used in the final code).

### Why configuration files (YAML) instead of hard-coding?

Things like "the sodium cap for people with hypertension" or "the list of fish names" are **knowledge**,
not logic. Putting them in YAML means:

- a non-programmer (or a doctor) could review and edit them;
- the code stays general;
- every rule is in one place, which made our many corrections easy (Part I).

The five config files: `metrics.yaml` (metrics, thresholds, decay settings), `safety_rules.yaml`
(rules and allergen keywords), `interventions.yaml` (77 suggestions, recipe filters, diet tags),
`synergy.yaml` (which items go well or badly together), `weights.yaml` (scoring weights).

### Why "pure functions" and lots of tests?

A **pure function** takes inputs and returns outputs without changing anything else (no hidden
side effects). The build plan asked for this. Pure functions are easy to test: give known input,
check the output. That let us write 314 tests, which caught many bugs (Part I).

---

# Part D: The data

## D1. Synthea (main clinical data)

- Generated on your machine with `java -jar synthea-with-dependencies.jar -p 1000 -s 42
  --exporter.csv.export=true --exporter.fhir.export=false --exporter.years_of_history=15`.
  `-p 1000` = 1,000 living patients; `-s 42` = random seed; 15 years of history.
- Result: **1,156 patients** (1,000 alive + 156 who died during the simulated years).
- Six tables used: `patients`, `encounters` (visits), `conditions`, `observations` (measurements),
  `medications`, `allergies`. Synthea also wrote 12 other tables (claims, procedures, etc.) that we don't use.
- `observations.csv` alone is 192 MB.

**Who gets recommendations?** Of the 1,000 living patients, **225 are children**. Adult lifestyle
advice and the adult heart-risk model don't fit children, so the recommendation group (the
**cohort**) is the **775 living adults (18+)**. Children and deceased patients stay in the data and
help with population-level statistics.

## D2. Kaggle Cardiovascular Disease dataset (trains the risk model)

- 70,000 people, columns: age (in **days**), gender (1 = female, 2 = male; we confirmed this because
  group 1 averages 161 cm and group 2 averages 170 cm), height, weight, systolic (`ap_hi`) and diastolic
  (`ap_lo`) blood pressure, cholesterol level (1/2/3), glucose level (1/2/3), smoking, alcohol,
  physical activity, and the label `cardio` (has cardiovascular disease). 49.97 % have the disease (34,979 of 70,000).
- It contains entry errors, e.g. blood pressure of 16,020 or −150, height 55 cm. Cleaning is in Part G.

## D3. Kaggle Food.com recipes

- 231,637 recipes with ingredients, minutes, tags and nutrition. Nutrition is given as
  **% of daily value per serving** (e.g. sodium 30 = 30 % of a day's recommended sodium).
- After filtering (Part F, section F3), **5,000 meals** are sampled with seed 42.

## D4. Kaggle Sleep Health and Lifestyle dataset

- 374 people aged 27–59, with sleep duration, daily steps, stress level, BMI category, gender.
- Used **only** to make the synthetic sleep and step data realistic.
- Quirks found: only 20 different step values, only 10 obese people, and two labels ("Normal" and
  "Normal Weight") for the same BMI category, which we merged.

## D5. Synthetic layers (things no public dataset provides)

No public dataset links a person's medical record to their sleep, steps, goals, preferences and
feedback. So, as the build plan required, we **generated** these, and the report says so clearly.
For each of the 775 adults, `synth_lifestyle.py` creates:

1. **Two years of daily sleep and steps.**
   - A **linear regression** fitted on the sleep dataset predicts each person's typical sleep, steps
     and stress from their age, gender and BMI category.
   - Each person gets a personal baseline (prediction + random personal difference).
   - **30 %** of people get a real **trend** (e.g. steps falling 1,000–2,500 per year; 60 % of trends go in the bad direction).
   - Daily noise is **autocorrelated** (today looks a bit like yesterday), weekends differ
     (+0.4 h sleep, −15 % steps), and **8 %** of days are missing (wearable not worn).
2. **Goals** (1–3 of: weight loss, lower BP, better sleep, more activity, glucose control), biased by
   their real record: e.g. someone with hypertension is much more likely to pick "lower BP".
3. **Preferences**: diet style, liked exercise types (older people lean toward walking, swimming,
   tai chi; younger toward running, HIIT, strength), and **minutes available per day** (15–90).
4. **Injuries** (none / knee / back), more likely if the record shows a knee or back problem.
5. **Smoking, alcohol, activity** flags for the risk model.
6. **A hidden "true taste" vector**, used **only** by the feedback simulator (Part G), never shown to
   the recommender. It is stored in a separate file (`hidden_taste.parquet`) so the ranking code
   cannot accidentally read it.

---

# Part E: How the system is put together

## E1. The pipeline

```
 RAW DATA                     PROCESSING                                   OUTPUT
 ─────────                    ──────────                                   ──────
 Synthea CSVs ──► ingest ──► normalize ──► timeline (1.3 M rows) ─┐
 Sleep dataset ─► synth_lifestyle (sleep, steps, goals, prefs) ───┤
                                                                  ▼
                                      temporal engine (decay, trends, urgency)
                                                                  ▼
 CVD dataset ──► risk_model (XGBoost) ─────────────► health profile ◄── user context
                                                                  ▼
 77 templates + 5,000 recipes ─────────────────────► candidates (5,077)
                                                                  ▼
                                                  safety filter (9 rules)  ──► drop log
                                                                  ▼
                                                  ranker: 7-term score
                                                                  ▼
                                                  greedy set selection (5 items)
                                                                  ▼
                                                  explanations ──► CLI / dashboard
                                                                  ▲
                                         feedback ──► adapt profile ──┘
```

## E2. The files

| File | Job |
|---|---|
| `scripts/01_download.py` | checks/fetches the four datasets |
| `src/lphrs/ingest.py` | reads the raw CSV files |
| `src/lphrs/normalize.py` | cleans measurements (units, dates, impossible values, duplicates) |
| `src/lphrs/timeline.py` | builds one sorted event list per patient |
| `src/lphrs/synth_lifestyle.py` | generates the synthetic layers |
| `src/lphrs/temporal.py` | **novel**: decay, trend fitting, time-to-threshold urgency |
| `src/lphrs/profile.py` | builds the health profile and user context |
| `src/lphrs/risk_model.py` | trains and applies the CVD model |
| `src/lphrs/candidates.py` | builds the 5,077 candidates, tags recipes |
| `src/lphrs/safety.py` | the safety filter |
| `src/lphrs/ranker.py` | **novel**: seven-term scoring and set selection |
| `src/lphrs/explain.py` | plain-language reasons |
| `src/lphrs/pipeline.py` | runs everything for one patient |
| `src/lphrs/feedback.py` | feedback rules, simulated users, fast batch scorer |
| `src/lphrs/weight_learner.py` | learns scoring weights from feedback |
| `src/lphrs/evaluate.py` | the seven experiments |
| `scripts/02` … `06` | run each stage and save outputs |
| `app/dashboard.py` | the Streamlit dashboard |
| `tests/` | 314 automated tests |

## E3. What a "timeline" row looks like

Every event, whether a blood pressure reading, a diagnosis or a day of step counting, becomes one row
with the same columns:

| column | example |
|---|---|
| patient_id | `355d5f82-…` |
| timestamp | 2024-10-27 (always in UTC) |
| stop | end date for conditions/medications, empty if ongoing |
| event_type | observation / lifestyle / condition / medication / allergy |
| metric | systolic_bp |
| value | 138.0 |
| unit | mmHg |
| code | 8480-6 |
| description | "Essential hypertension (disorder)" |
| source | synthea / synthetic |
| context | vital-signs, disorder, food:allergy, wearable… |
| confidence | 1.0 for measured data, 0.8 for synthetic data |

Having one shape for everything means the later algorithms can treat all history uniformly.

Final timeline: **1,316,866 rows** (1,041,422 synthetic daily lifestyle rows; 157,767 clinical
measurements; 64,478 medication events; 51,949 condition events; 1,250 allergy events).

## E4. Data cleaning ("NormalizeHealthData")

From 161,937 raw measurements:

- **Units**: converted to one standard unit per metric (e.g. pounds → kg, mmol/L → mg/dL). Synthea
  already used standard units, so no values changed; the 45,462 blood-pressure rows written as `mm[Hg]`
  were relabelled `mmHg` (factor 1.0). The code and tests handle real conversions too.
- **Dates**: all converted to UTC timestamps in ISO-8601 format, so dates from different sources compare correctly.
- **Impossible values dropped**: 13 (e.g. negative LDL, systolic BP of 47).
- **BP pairs** where diastolic ≥ systolic (physically impossible): 46 readings dropped.
- **Duplicates merged**: 4,111 (mostly glucose recorded under two codes from the same blood test).
- **Result**: 157,767 clean measurements.
- No "Z-scoring" (standardising) at this stage: values stay in real units so explanations can say
  "139 mmHg". Scaling happens only inside models.

---

# Part F: The novel algorithms, step by step

This is the heart of the project. Each algorithm is explained in plain words, then as a formula, then
with a worked example using **real numbers from our data**.

## F1. Persistence-floored adaptive decay (build plan 5.1)

### The idea in plain words

When deciding what matters about a patient *today*, old information should count less than new
information. A cholesterol test from 8 years ago says little about today. But some old facts are just
as important as ever: an egg allergy diagnosed 23 years ago is still an egg allergy. And some
measurements naturally jump around a lot between visits (so an old reading goes out of date quickly),
while others barely change (so an old reading stays informative for longer).

So our decay has two twists compared with ordinary ("plain") exponential decay:

1. **Adaptive**: the decay speed depends on how *volatile* (jumpy) each measurement is across the population.
2. **Persistence-floored**: serious, persistent events never drop below a minimum weight (the *floor*).

### The formulas

```
lambda_m  = lambda_base × clip(volatility_m, 0.25, 4.0)     ← decay speed for metric m
w_recency = max(floor_e, exp(−lambda × days_since_event))    ← how much the event still counts
floor_e   = severity_e × persistence_e                       ← minimum weight (0 to 1)
W_e       = w_recency × goal_relevance_e                     ← final weight
```

Which λ is used: **measurement readings** (blood pressure, cholesterol…) decay with their own
volatility-scaled `lambda_m`. **Conditions, medications and allergies** have no volatility of their
own, so they use `lambda_base` directly (and readings have floor 0).

- `lambda_base = 0.0019` per day (about a one-year half-life for an average metric).
- `clip(x, 0.25, 4.0)` keeps the multiplier between 0.25× and 4× so no metric decays absurdly fast or slow.
- `goal_relevance_e` is 1.0 if the event relates to one of the user's goals, otherwise 0.6.
  **Allergies always get 1.0**, whatever the goals, because they are safety-critical.

### Measuring volatility (and a problem we had to solve)

The build plan said: volatility = "cohort-level standard deviation of visit-to-visit change for metric m,
divided by the median across metrics". But blood pressure changes are in mmHg, HbA1c in %, steps in
thousands. Comparing their standard deviations directly is meaningless (steps would always look
"most volatile" just because the numbers are bigger).

**Our fix**: make it unit-free. For each metric we divide the SD of visit-to-visit changes by the
**average level** of that metric (giving "typical change as a fraction of typical value"), and only then
compare to the median. A test checks that multiplying a metric's values by 1,000 (a unit change) gives
exactly the same volatility.

We also calculate volatility only from readings taken when patients were adults, because children's
growth would make weight and BMI look very volatile.

**Result on our data:**

| Metric | Volatility (relative to median) | Half-life of a reading |
|---|---|---|
| Body weight, BMI | 0.17 (clipped up to 0.25) | 1,459 days (~4 years) |
| HbA1c | 0.48 | 762 days |
| Diastolic / systolic BP | 0.56 / 0.57 | ~650 days |
| Sleep | 0.61 | 595 days |
| Glucose | 1.39 | 263 days |
| Total cholesterol / HDL | 1.64 / 1.89 | 223 / 193 days |
| Steps, heart rate | 1.92 / 1.94 | ~190 days |
| LDL | 2.70 | 135 days |

(Weight and BMI are so stable that their volatility, 0.17, falls below the 0.25 minimum, so they are
clipped to 0.25. Without the clip their half-life would be about 2,150 days.)

This makes sense: weight changes slowly, so an old weight is still informative; cholesterol jumps around
between tests, so an old result goes stale quickly.

### Floors: which events never fade?

We had to classify Synthea's conditions, which turned out to need care (see Part I, Phase 3).

| Event class | How we recognise it | severity | persistence | floor |
|---|---|---|---|---|
| Allergy | from the allergies table | 1.0 | 1.0 | **1.0** |
| Chronic condition | condition still ongoing (no stop date) | 0.8 | 1.0 | **0.8** |
| Resolved major event | ended, and is a heart attack, fracture, stroke, transplant…; or a "History of …" record | 0.6 | 0.5 | **0.3** |
| Acute condition | ended, minor (e.g. sinusitis) | 0.3 | 0 | 0 |
| Social finding | employment, education, housing, "medication review due"… | 0 | 0 | 0 |
| Medication | weight 1 while active, then decays | 0.5 | 0 | 0 |
| Ordinary reading | a measurement | 0 | 0 | 0 |

### Worked example (real patients from our data)

**An egg allergy recorded in May 2003.** At the reference date (5 October 2026) it is 8,559 days old.

- Plain decay: `exp(−0.0019 × 8,559) = exp(−16.26) ≈ 0.000000087`. Effectively forgotten.
- Floored decay: `max(1.0, 0.000000087) = 1.0`. **Fully remembered.**

**Chronic neck pain, ongoing, first recorded 2,976 days ago.**

- Plain decay: `exp(−0.0019 × 2,976) ≈ 0.0035`.
- Floored: `max(0.8, 0.0035) = 0.8`.
- This patient's goals do not include anything related to neck pain, so goal relevance = 0.6:
  final weight `W = 0.8 × 0.6 = 0.48`.

**A sinusitis that ended 8 years ago** has floor 0, so it fades to near zero under both schemes. That is
what we want: old minor illnesses are irrelevant.

### The "weighted history score"

For each metric we also compute a **history score**: the decay-weighted share of past readings that were
beyond the threshold. A patient whose blood pressure was above 140 at most recent visits gets a score
near 1, even if today's reading happens to be 138. This feeds the relevance term R (section F4).

---

## F2. Time-to-threshold urgency (build plan 5.2)

### The idea in plain words

Look at the last 5 years of readings, fit a straight line, and ask:

- **Is the value already past the danger threshold?** Then urgency is maximum (1.0).
- **Is it moving away from the threshold?** Then urgency is 0.
- **Otherwise**, how long until the line hits the threshold (**time-to-threshold**, TTT), and how much do
  we trust the line (**confidence**)? Sooner and more trustworthy means more urgent.

### The formula

```
slope, r², n = straight-line fit of value on time (years), last 5 years
if already past threshold θ:          urgency = 1.0     (checked first: even 1 reading is enough)
elif fewer than 3 points:             urgency = 0.0     ("insufficient data")
elif slope points away from θ:        urgency = 0.0
else:
    ttt_years  = (θ − x_now) / slope
    confidence = r² × min(1, n / 5)
    urgency    = confidence / (1 + ttt_years)
```

Why `confidence / (1 + ttt)`?

- If the crossing is imminent (ttt ≈ 0), urgency ≈ confidence.
- If it's 1 year away, urgency is halved; 3 years away, quartered. Far-off crossings matter less.
- `r²` punishes a messy, unreliable trend; `min(1, n/5)` punishes trends based on only 3 or 4 points.
- The result is always between 0 and 1.

The "bad direction" from the config decides what "past" and "toward" mean: for HDL, sleep and steps,
*falling* is bad, so the comparisons flip.

### Worked example 1: the build plan's acceptance test

Blood pressure readings of 120, 124, 130, 136 taken one year apart (years −3, −2, −1, 0):

1. Line fit: mean time = −1.5, mean value = 127.5. Slope = Σ(t−t̄)(v−v̄) / Σ(t−t̄)²
   = [(−1.5)(−7.5) + (−0.5)(−3.5) + (0.5)(2.5) + (1.5)(8.5)] / [2.25+0.25+0.25+2.25]
   = 27 / 5 = **5.4 mmHg per year**. ✔ (the plan expected "about 5.4")
2. Fitted values: 119.4, 124.8, 130.2, 135.6. Residuals 0.6, −0.8, −0.2, 0.4.
   r² = 1 − (sum of squared residuals 1.2) / (total variation 147) = **0.992**.
3. ttt = (140 − 136) / 5.4 = **0.74 years** (about 9 months). Finite ✔.
4. confidence = 0.992 × min(1, 4/5) = 0.794.
5. urgency = 0.794 / (1 + 0.74) = **0.456**.

The test file `tests/test_temporal.py` checks exactly this.

### Worked example 2: a real patient (case study 1, patient 355d5f82)

Systolic readings in the last 5 years:

| Date | Years before today | Systolic BP |
|---|---|---|
| 2021-10-17 | −4.97 | 122 |
| 2021-10-31 | −4.93 | 123 |
| 2022-10-23 | −3.95 | 137 |
| 2023-10-29 | −2.93 | 136 |
| 2024-10-27 | −1.94 | 138 |
| 2024-12-22 | −1.79 | 138 |
| 2025-11-09 | −0.90 | 139 |

- Line fit: slope **3.97 mmHg/year**, r² **0.735**, n = 7 points.
- Current value 139, threshold 140: ttt = (140 − 139) / 3.97 = **0.25 years (3 months)**.
- confidence = 0.735 × min(1, 7/5) = 0.735.
- urgency = 0.735 / 1.25 = **0.587**.

This becomes the explanation the system printed for this patient: *"Your systolic blood pressure has
risen about 4.0 mmHg per year and is projected to reach 140 in about 3 months."*

### Extra rules we added (and why)

- **Readings must span at least 1 year.** One patient had six glucose readings all taken during a single
  month-long hospital stay. A line through them gave a slope of **199 mg/dL per year**, an absurd
  "trend", and urgency 0.83. Now, if the readings span less than a year, we treat the trend as
  "insufficient data". (This affected only 3 of 1,265 trends, but they were very wrong.)
- **Sleep and steps use weekly averages.** Daily wearable data is very noisy; weekly averages give
  sensible trends.
- **Trend labels** (rising / falling / stable / insufficient) need a change of at least 2 % of the current
  value per year and a minimal fit (r² ≥ 0.05). "Worsening" or "improving" then depends on the bad direction.

---

## F3. Building the candidates and making them safe

### The 77 hand-written suggestions ("templates")

Spread across six areas (**domains**): nutrition 16, fitness 17, sleep 11, lifestyle 12,
preventive care 11, wellness 10. Each has:

- **tags** (e.g. `walking`, `low_impact`, `evening_walk`)
- **intensity** 1–5
- **minutes** per day
- **effort** cost 0–1
- **targets** (which metrics it helps, e.g. systolic BP, glucose)
- **goals** it serves
- **contraindication tags** (e.g. `high_impact`, `allergen:peanut`)

Examples: "Take a 10-minute walk after dinner" (fitness, intensity 1, 10 min, targets glucose, steps
and sleep); "Do a 20-minute HIIT session" (intensity 5, tagged `high_impact`); "Have a small handful of
unsalted peanuts" (tagged `allergen:peanut`, partly there to test the allergy rule).

### The 5,000 recipes

From 231,637 recipes we keep only sensible everyday meals:

| Step | Recipes left |
|---|---|
| All recipes | 231,637 |
| 5–60 minutes to make (shorter ones were mostly drinks and garnishes) | 163,849 |
| No alcohol and no processed meat | 136,126 |
| Tagged as a meal (main dish, breakfast, lunch, side, salad, soup…), not dessert/drink/sauce/dip/snack | 67,419 |
| ≤ 1,500 calories per serving (larger values are whole-batch figures) | 66,713 |
| Random sample (seed 42) | **5,000** |

Each recipe gets tags automatically from its nutrition and ingredients:

- **Diet tags**: `low_sodium` (sodium ≤ 10 % of daily value), `low_sugar` (≤ 10 %), `high_protein`
  (≥ 30 %), `vegetarian` (no meat or fish words). These decide which metrics and goals the recipe targets.
- **Allergen tags**: by keyword matching on ingredients for 8 allergens (peanut, tree nut, milk, egg,
  wheat, soy, fish, shellfish), with **exclusions** so that "eggplant" is not egg, "peanut butter" is not
  milk, and "rice flour" is not wheat.
- **Interaction tags**: `high_vitamin_k` (kale, spinach, broccoli…), `grapefruit`.

### The safety filter: nine deterministic rules

"Deterministic" means fixed rules, no machine learning: the same input always gives the same result,
and every decision can be traced. For safety that matters more than cleverness.

| Rule | Applies to people with | Removes |
|---|---|---|
| allergy_match | any mapped food allergy | anything with that allergen |
| hypertension_sodium_cap | hypertension | recipes with > 25 % daily sodium per serving |
| diabetes_sugar_cap | diabetes or prediabetes | recipes with > 25 % daily sugar |
| joint_high_impact | knee or back injury, knee/hip osteoarthritis, knee ligament injuries, chronic low back pain | high-impact exercise |
| knee_load | knee injury or knee problems | knee-loading exercise (squats, stairs, jogging) |
| back_heavy_load | back injury, chronic low back pain | heavy spinal loading (heavy weights) |
| cardiac_intensity_cap | heart attack, heart failure, coronary disease, atrial fibrillation… | anything with intensity ≥ 4 |
| warfarin_vitamin_k | taking warfarin (a blood thinner) | high vitamin K foods |
| statin_grapefruit | taking simvastatin, atorvastatin, etc. | grapefruit |

How the patient side works: Synthea allergy descriptions such as *"Peanut (substance)"*, *"Cow's milk
(substance)"* or *"Glycine max (substance)"* (the Latin name for soy!) are mapped to our allergen names.
Conditions and medications are matched by keywords against the active records.

**Every removal is logged** with the rule and a plain reason ("Contains an ingredient you are allergic
to"). The dashboard shows this log, and experiment 1 uses it.

**Example (case study 3):** a 40-year-old woman allergic to cow's milk, fish, peanut and shellfish, with
a back injury and prediabetes. Of 5,077 candidates, **4,042 are removed** and 1,035 remain. Her final
list contains no fish, dairy, peanut or high-impact items.

---

## F4. Seven-term scoring (build plan 5.3)

### The formula

```
Score = 0.25·R + 0.20·G + 0.20·U + 0.10·P + 0.10·F + 0.10·H − 0.05·C
```

Every term is a number between 0 and 1:

| Term | Name | How it is computed |
|---|---|---|
| **R** | Health relevance | the highest of: (a) for each metric the item targets, 1 if that metric is past its threshold, otherwise its history score; (b) the patient's CVD risk, if the item targets a heart-related metric (BP, cholesterol); (c) the decay weight of an active chronic condition linked to one of the item's goals (e.g. sleep apnoea → sleep items) |
| **G** | Goal alignment | share of the user's goals the item serves (2 of 3 goals → 0.67) |
| **U** | Urgency | the highest time-to-threshold urgency among the metrics the item targets |
| **P** | Preference | 1 shared preference tag → 0.5; 2 or more → 1.0 |
| **F** | Feasibility | time fit (full marks up to half the daily budget, falling to 0 at the budget) × intensity fit (each level above the user's cap for that domain costs 0.5) |
| **H** | History | smoothed past acceptance of this item and its domain; 0.5 when there is no feedback yet |
| **C** | Cost | the item's effort; **subtracted** |

The weights (0.25, 0.20 …) are the hand-set defaults from the build plan. R, G and U together carry
65 % of the score, so health need and the user's own goals dominate.

### Worked example (case study 1, patient 355d5f82, top pick)

"Try 15 minutes of progressive muscle relaxation":

| Term | Value | Weight | Contribution |
|---|---|---|---|
| R | 1.00 (sleep 5.96 h, below the 6 h threshold) | 0.25 | 0.250 |
| G | 1.00 (serves both goals: lower BP, better sleep) | 0.20 | 0.200 |
| U | 1.00 (sleep is past its threshold) | 0.20 | 0.200 |
| P | 0.00 | 0.10 | 0.000 |
| F | 1.00 (15 min of a 30 min budget) | 0.10 | 0.100 |
| H | 0.50 (no feedback yet) | 0.10 | 0.050 |
| C | 0.15 | −0.05 | −0.0075 |
| **Score** | | | **0.7925** |

The same patient's fifth pick, "Keep alcohol to at most one drink a day", shows the urgency term working:
R = 0.85 (his CVD risk of 85 %, since alcohol reduction targets BP) and U = 0.587 (the systolic trend from
F2): 0.25·0.85 + 0.20·0.5 + 0.20·0.587 + 0 + 0.10·1 + 0.10·0.5 − 0.05·0.4 = **0.560**.

---

## F5. Cross-domain set selection (build plan 5.4)

### Why not just take the top 5?

Experiment 5 measured what happens if you simply take the five highest scores:

- the five often come from one or two domains (2.03 different domains on average);
- they often clash or repeat (5.06 conflicting or redundant pairs per list, e.g. two different walks, or
  an afternoon green tea next to "no caffeine after 2 pm");
- more than half the time (52.8 %) they don't fit in the user's daily time budget.

### The greedy algorithm

```
selected = []
budget   = user's minutes per day
repeat until 5 items are chosen:
    for every remaining item that fits the time budget and whose domain has < 2 items so far:
        gain(item) = Score(item) + 0.3 × Σ synergy(item, s) for every s already selected
    take the item with the highest gain (stop if the best gain ≤ 0)
    subtract its minutes from the budget
```

**Synergy** comes from `synergy.yaml`: 33 tag pairs with values between −1 and +1, for example:

| Tag pair | Value | Why |
|---|---|---|
| evening walk + earlier bedtime | +0.5 | an evening walk helps you sleep |
| wind-down routine + screen limit | +0.4 | they support each other |
| HIIT + calorie deficit | −0.6 | hard training on too little food |
| late caffeine + sleep hygiene | −0.8 | contradictory advice |
| late caffeine + caffeine cutoff | −1.0 | directly contradictory |
| walking + walking | −0.4 | redundant (two walks) |
| recipe + recipe | −0.2 | mild redundancy |

The synergy between two items is the sum over all their tag pairs, clipped to the range −1 to +1.
α = 0.3 controls how much synergy matters compared with the score.

### Our improvement: budget look-ahead

Plain greedy has a flaw we found in testing. A patient with a 30-minute budget got three 10-minute items
first, used up the whole budget, and ended with **only 3 recommendations**, even though many 1–2 minute
habits ("no caffeine after 2 pm") would have fitted.

**Fix**: before taking an item, check that the *remaining* budget can still be filled with the shortest
available items (respecting the 2-per-domain cap). Items are tried in order of gain, and the first one
that leaves room is taken. If 5 items cannot fit at all, the algorithm aims for the longest list that can.
Afterwards every one of the 775 patients received exactly 5 items, all within budget.

### Worked example (case study 1)

Pick order, with each item's own score and its gain (score + synergy bonus):

| Pick | Item | Score | Gain | Synergy bonus |
|---|---|---|---|---|
| 1 | Progressive muscle relaxation (15 min) | 0.793 | 0.793 | 0 (nothing selected yet) |
| 2 | One screen-free evening a week (5 min) | 0.685 | 0.805 | +0.12 |
| 3 | Screens away 60 min before bed (5 min) | 0.680 | 0.800 | +0.12 |
| 4 | Avoid alcohol within 3 h of bedtime (1 min) | 0.685 | 0.865 | +0.18 |
| 5 | Keep alcohol to ≤ 1 drink a day (1 min) | 0.560 | 0.830 | +0.27 |

Total 27 of 30 minutes, three domains, a coherent "better sleep, lower BP" plan.

### Making it fast without changing the answer

The feedback simulation (Part G2) runs this selection 46,500 times. The first version of the selection
step alone took about 235 ms per patient (the whole pipeline about 350 ms). A shortcut that only considered the top 300 items was tried and **rejected**: it changed the
list for 22 of 60 test patients (because many recipes tie on score). Instead the algorithm was rewritten
with matrix maths (synergy as a tag-by-tag matrix) and a smarter look-ahead. Scoring plus selection now
takes about 38 ms per patient (roughly 6× faster) with **identical** results. This is checked by an
automated test that runs the old and new versions on 200 random candidate pools and requires exactly the
same picks and gains, and by a one-off check (not part of the test suite) on 60 real patients, where all
60 lists matched the full pipeline.

---

## F6. Explanations

For each chosen item, the system looks at the term contributions (weight × value) and turns the **top
two** positive ones into sentences:

| Term | Example sentence |
|---|---|
| U | "Your systolic blood pressure has risen about 4.0 mmHg per year and is projected to reach 140 in about 3 months." |
| R (metric) | "Your latest BMI is 30.1 kg/m2, above the 30 threshold." |
| R (lifestyle) | "Your daily steps recently average 2,919, below the 5,000 threshold." |
| R (condition) | "Your record lists prediabetes; this supports glucose control." |
| R (heart risk) | "Your estimated cardiovascular risk is 85%, and this supports heart health." |
| G | "Supports your goals: lower blood pressure, better sleep." |
| P | "Matches your preferences (walking)." |
| F | "Fits your day: 10 of your 30 available minutes." |
| H | "Similar suggestions have worked well for you before." (only once real feedback exists) |

Safeguards we added after seeing bad output:

- A trend sentence needs urgency ≥ 0.02 and a projection within 10 years. (We once printed
  *"projected to reach 6 in about 1,473 years"*.)
- Contributions below 0.01 are not explained.
- If rounding would make a value look equal to its threshold ("6.0 hours, below the 6-hour threshold"),
  an extra decimal is shown ("5.96 hours").
- Every list and item carries **"Wellness guidance only. Not medical advice."**

---

## F7. Learning from feedback ("AdaptProfileFromFeedback")

Each user has a feedback **state**: per-item and per-domain counts, per-domain intensity caps and
tag likings ("affinity"). Four kinds of feedback:

| Feedback | Effect |
|---|---|
| ACCEPTED | item and domain: +1 accepted, +1 shown; the item's tags gain affinity +1 |
| COMPLETED | counted twice (+2/+2, affinity +2); two completions at a domain's intensity cap raise that cap by 1 (max 5) |
| REJECTED | +1 shown; tags lose affinity −1 |
| TOO_DIFFICULT | +1 shown; that domain's intensity cap becomes max(1, min(current cap, item intensity − 1)), so it never rises and never goes below 1; the completion counter for that domain resets |

Tags whose affinity reaches +2 join the user's preferences (raising P); at −2 they leave. The generic
`recipe` tag is ignored here, so accepting one recipe doesn't make the user "like all recipes". The H term
uses the counts: `H = (accepted_item + 0.5·accepted_domain + 1) / (shown_item + 0.5·shown_domain + 2)`,
which starts at 0.5 and moves toward the real acceptance rate as evidence builds up.
The dashboard's buttons call exactly this function and re-rank the list immediately.

---

# Part G: The machine-learning models

## G1. The cardiovascular risk model

### Cleaning the training data

From 70,000 rows:

- removed **1,334** with impossible blood pressure (systolic outside 70–250, diastolic outside 40–150,
  or diastolic ≥ systolic), as the build plan specified;
- removed **56** more with implausible height (outside 120–220 cm) or weight (outside 30–200 kg). This
  was our addition, agreed with you, because heights like 55 cm produce absurd BMIs;
- converted age from days to years and added BMI.

Result: **68,610 rows**, 49.5 % with disease.

### Training

- **Split** 70/15/15, stratified: 48,027 train, 10,291 validation, 10,292 test.
- **12 features**: age, gender, height, weight, BMI, systolic BP, diastolic BP, cholesterol level,
  glucose level, smoking, alcohol, activity.
- **Baseline**: logistic regression, with features standardised first (rescaled to mean 0, SD 1 so their
  coefficients are comparable).
- **XGBoost**: 16 combinations tried (depth 3/4/5/6 × learning rate 0.05/0.1 × 200/400 trees); best on
  validation was depth 4, learning rate 0.05, 200 trees.
- **Model choice**: XGBoost had the higher validation ROC-AUC (0.8005 vs 0.7921), so it was chosen.
- **Calibration**: isotonic and Platt both fitted on the validation set; isotonic had the lower Brier
  score (0.1792 vs 0.1805) and was kept.

### Results on the untouched test set

| Model | ROC-AUC | PR-AUC | Accuracy | F1 | Brier |
|---|---|---|---|---|---|
| Logistic regression | 0.7945 | 0.7724 | 0.7314 | 0.7136 | 0.1854 |
| XGBoost | 0.8063 | 0.7887 | 0.7421 | 0.7289 | 0.1780 |
| XGBoost + isotonic calibration (saved) | 0.8059 | 0.7817 | 0.7414 | 0.7294 | 0.1781 |

The plan expected ROC-AUC around 0.78–0.80; we got 0.806. XGBoost was already well calibrated, so
calibration changed little (the calibration plot shows its curve almost on the diagonal).

**Feature importance:** systolic blood pressure dominates (about 60 % of XGBoost's gain), followed by
cholesterol and age. In logistic regression, smoking and alcohol get slightly *negative* coefficients.
That looks wrong medically but is a known quirk of this dataset, where these are self-reported.

### Applying the model to Synthea patients

The Kaggle columns had to be rebuilt from Synthea records:

- **age, gender**: from the patient table (gender F → 1, M → 2).
- **weight, BMI, systolic, diastolic**: latest readings.
- **height**: Synthea's height wasn't in our metric list, so we derived it: height (m) = √(weight ÷ BMI).
- **cholesterol level**: total cholesterol < 200 → 1, 200–239 → 2, ≥ 240 → 3.
- **glucose level**: < 100 → 1, 100–125 → 2, ≥ 126 → 3.
- **smoking, alcohol, activity**: from the synthetic lifestyle layer (Synthea rarely records smoking: only 7 patients).
- Missing labs default to level 1 and are flagged: cholesterol imputed for 194 patients, glucose for 342.

Sanity checks on the cohort's predicted risk: median risk rises with age (0.10 for ages 18–28, 0.20 for
29–44, 0.45 for 45–64, 0.59 for 65+); people with systolic ≥ 140 have median risk 0.81 vs 0.25 below
120; patients with hypertension, heart disease or diabetes average about 0.56 vs about 0.35 without.

Two honest caveats:

- **319 of 775** patients are outside the training ages (29–65). XGBoost simply treats them as the
  nearest training age.
- Isotonic calibration produces a step function, so the cohort has only **44 distinct risk values**.

## G2. The simulated users and the weight learner

### Why simulate?

The build plan asks to learn the scoring weights (w1…w7) from feedback, but no real users exist.
So we built **simulated users** whose behaviour follows stated rules, and used them to test whether the
learning mechanism works.

### The simulated user

```
P(accept) = sigmoid( offset + bias + a · [R, G, U, P, F, H, C] + taste_match − difficulty_penalty + noise )
```

- `a`: the user's hidden sensitivity to each term. Average [0.6, 1.0, 0.4, 0.8, 1.0, 0.6, −0.8], plus
  personal variation (SD 0.3). So, on average, simulated users care most about goals and feasibility,
  and dislike effort.
- `bias`: personal tendency to accept anything (average −1.0).
- `offset = −1.75`: a calibration we added. Without it, users accepted **82 %** of first-round items,
  leaving no room to show improvement. We tested offsets on 100 patients (0 → 82 %, −1.5 → 57 %,
  −2.0 → 47 %) and chose −1.75 (about 50 %).
- `taste_match`: the user's hidden liking for the item's domain and exercise types.
- `difficulty_penalty = 0.8 × max(0, intensity − (capacity − 1))`, where capacity (1–5) is the user's
  hidden fitness level, based on age and activity.
- If intensity > capacity, the answer is always **TOO_DIFFICULT**. Of accepted items, 70 % become
  **COMPLETED**.

### The experiment

- **30 rounds** × **775 users** × **5 items** × **2 arms** = **232,500 recommendations**.
- **Fixed arm**: the default weights throughout.
- **Learned arm**: starts with the defaults. Every 5 rounds a **logistic regression** is fitted on this
  arm's own log (did the user accept? vs the item's seven term values). The coefficients become the new
  weights:
  - negative coefficients → 0 (except cost, which uses the size of its negative coefficient);
  - then scaled so the weights sum to 1.
- Both arms adapt each user's state (history, caps, likings) after every answer.
- **Common random numbers**: both arms get the same random draws for the same user, round and list
  position, so differences come from the weights, not luck.

### Results

| | Fixed weights | Learned weights |
|---|---|---|
| Acceptance, rounds 1–5 (identical by design) | 56.0 % | 56.0 % |
| Acceptance, last 5 rounds | 61.9 % | **63.8 %** |
| TOO_DIFFICULT answers | 813 | **234** |

How the learned weights changed:

| Round | R | G | U | P | F | H | C |
|---|---|---|---|---|---|---|---|
| 0 (default) | 0.250 | 0.200 | 0.200 | 0.100 | 0.100 | 0.100 | 0.050 |
| 5 | 0.055 | 0.099 | 0.042 | 0.094 | 0.268 | 0.328 | 0.114 |
| 25 | 0.043 | 0.089 | 0.042 | 0.065 | 0.171 | 0.540 | 0.050 |
| 30 (final, saved) | 0.043 | 0.088 | 0.039 | 0.062 | 0.157 | 0.566 | 0.045 |

Note: rounds 26–30, which give the 63.8 %, were run with the **round-25** weights. The round-30 refit
happens after the last round, so it is saved to `config/weights_learned.yaml` (and used in experiment 6
and the `--weights learned` option) but never ran inside the simulation.

- Both arms improve from 51.0 % in round 1 to 61.9 % (fixed) and 63.8 % (learned) in the last five rounds,
  because the **adaptation** (caps, history, likings) works.
- The learned arm is about **2 points better**, and it almost eliminates "too difficult" answers, because
  it puts more weight on feasibility F (which penalises items above the user's adapted intensity cap).

### Three caveats (important when presenting)

1. **It only shows the mechanism works on simulated users.** The learned weights reflect the simulator's
   assumptions, not real people.
2. **The history term H is inflated by a feedback loop.** H is computed from past acceptances. Users who
   like the system have high H *and* keep accepting, so the learner credits H for acceptances that really
   come from the user's own taste. That's why H grows at every refit (0.33 → 0.57).
3. **Learned weights nearly ignore health (R 0.04, U 0.04).** Simulated users mostly respond to goals,
   feasibility and taste, so a system that only maximises acceptance learns to stop caring whether the
   advice fits the patient's health. **So the hand-set defaults remain the live setting**; the learned
   weights are available as an option (`--weights learned`, or the dashboard switch). This was your decision.

---

# Part H: Evaluation: the seven experiments and what they really show

All numbers come from `scripts/06_evaluate.py`, which writes `reports/results.md` with full tables and
figures. Every experiment changes **one thing** and keeps everything else equal, so differences can be
traced to that one thing (a **controlled comparison**, or **ablation**).

## Experiment 1: Safety filter on vs off

**Question:** does the filter really keep unsafe items out of the final lists, and how often would they
otherwise appear?

**Method:** run all 775 patients twice, with and without the filter; check every final item against all
rules.

| | Lists | Items | Unsafe items | Lists with ≥ 1 unsafe item |
|---|---|---|---|---|
| With filter | 775 | 3,875 | **0** | **0 %** |
| Without filter | 775 | 3,875 | 64 | 7.9 % |

Without the filter, the rules most often broken were: high-impact exercise for injured joints (25
lists), allergens (19), high intensity for heart patients (12), sodium (8), knee load (3), sugar (1).

**Meaning:** the filter meets the plan's target of 0 %. Without it, about 1 in 13 patients would get
something unsafe, because unsafe items can score well (e.g. high-intensity exercise scores well for
weight loss).

## Experiment 2: Snapshot vs longitudinal (urgency off vs on)

**Question:** does adding the urgency term make the system act on metrics that are still normal but
heading the wrong way?

**Method:** find patient-metric pairs that are *rising but still normal* (509 pairs from 367 patients).
Check whether the final list targets that metric, with U switched off ("snapshot") and on ("longitudinal").

| | U off | U on |
|---|---|---|
| All 509 pairs | 72.3 % | 72.5 % |
| Pairs with urgency ≥ 0.2 (101) | 63.4 % | 66.3 % |
| Pairs with urgency < 0.05 (208) | 76.9 % | 76.4 % |

**Meaning (honest):** the overall effect is very small. Most rising metrics have tiny urgency (slow
trends far from the threshold, or messy fits), and many are targeted anyway through relevance or goals
(walking targets BP, BMI and glucose). Where urgency is high, U helps a bit (+2.9 points). U also boosts
items for metrics *already* past their threshold (urgency 1), which compete for the same five slots, so
for some individual metrics the effect is negative.

## Experiment 3: Early warning (the hidden-future test)

**Question:** does urgency predict which metrics will actually cross their threshold, and how early?

**Method:**

1. Hide every patient's last 2 years of data (cut date 5 October 2024).
2. Using only the earlier data, flag metrics that are not yet past the threshold.
3. Look at the hidden 2 years: did they cross?

We compare the urgency flag (urgency ≥ 0.1, 0.2 or 0.3) with a simple **baseline**: "latest value within
2 %, 5 % or 10 % of the threshold". There were 3,559 pairs, and 259 crossed.

| Method | Alarms | Recall | Precision | Median lead (detected) |
|---|---|---|---|---|
| Urgency ≥ 0.1 | 184 | 11.6 % | 16.3 % | 36 months |
| Urgency ≥ 0.2 | 82 | 7.3 % | 23.2 % | 36 months |
| Urgency ≥ 0.3 | 36 | 4.2 % | 30.6 % | 24 months |
| Latest within 2 % | 128 | 18.1 % | 36.7 % | 13 months |
| Latest within 5 % | 465 | 46.3 % | 25.8 % | 36 months |
| Latest within 10 % | 1,146 | 62.2 % | 14.0 % | 146 months |

**Meaning (honest negative result):** at a similar number of alarms (urgency ≥ 0.2: 82; within 2 %: 128),
the simple baseline has **both higher recall and higher precision**. Urgency's only advantage is earlier
warning for the crossings it catches (36 vs 13 months). On this synthetic data, most crossings are not
preceded by a steady, well-fitted trend in roughly yearly readings, so a trend-based flag misses them.
This is a real finding, and reporting it honestly is part of good science.

## Experiment 4: Plain vs floored decay

**Method:** take every condition, medication and allergy older than 5 years in the 775 profiles and
compare their weights under both schemes.

| Event class | Events | Floored mean | Plain mean |
|---|---|---|---|
| Allergy | 826 | **1.000** | 0.0000 |
| Chronic condition | 3,236 | **0.800** | 0.0015 |
| Resolved major | 722 | **0.300** | 0.0031 |
| Acute condition | 7,946 | 0.005 | 0.0054 |
| Medication | 26,394 | 0.005 | 0.0051 |
| Social finding | 12,830 | 0.005 | 0.0047 |

**Meaning:** plain decay would forget every old allergy and chronic disease. The floored version keeps
them while letting irrelevant old events fade exactly as before. This is the property the build plan asked for.

## Experiment 5: Independent top-5 vs greedy set

| | Top-5 | Greedy set |
|---|---|---|
| Conflicting/redundant pairs per list | 5.06 | **0.46** |
| Lists with any conflict | 84.1 % | 36.0 % |
| Different domains per list | 2.03 | **3.37** |
| Lists within time budget | 47.2 % | **100 %** |
| Summed score | 3.529 | 3.230 |

**Meaning:** the greedy set is far more usable: varied, coherent and doable. The price is a slightly
lower total score, which is expected: top-5 maximises the sum by definition but ignores everything else.

## Experiment 6: Fixed vs learned weights

See Part G2: 61.9 % vs 63.8 % acceptance in the last 5 rounds, with the three caveats. We also measured
what the learned weights do to real lists: the average urgency of recommended items drops from 0.699 to
0.578 and relevance from 0.882 to 0.812. That confirms caveat 3.

## Experiment 7: Logistic regression vs XGBoost

See Part G1: ROC-AUC 0.795 vs 0.806, Brier 0.185 vs 0.178. XGBoost wins on every metric, by a modest
margin, which is typical for this dataset.

## The three case studies (in `reports/results.md`)

1. **Rising blood pressure** (64-year-old man, CVD risk 85 %, systolic 139 rising ~4/year): the plan
   focuses on sleep and BP, and its fifth item carries the trend explanation "projected to reach 140 in
   about 3 months".
2. **Prediabetes without diabetes** (47-year-old woman, HbA1c 6.4 % rising 0.15/year, projected to cross
   6.5 in about 8 months): two low-sugar recipes, an after-dinner walk (her steps are falling below
   5,000), a medication reminder and "ask your clinician whether an HbA1c test is due".
3. **Allergies and injury** (40-year-old woman with four food allergies, a back injury and prediabetes):
   4,042 of 5,077 candidates removed by the safety filter; her plan is walking, resistance bands, step
   goals and two allergen-free recipes.

---

# Part I: The build, phase by phase, including every mistake and correction

The project followed the build plan's 10 phases. Before each phase a plan was agreed with you; after
each phase its acceptance check was run. This section records what happened, including everything that
went wrong. Good engineering means finding problems, explaining them and fixing them at the root.

## Phase 1: Repository, configs, download script

**Built:** the folder layout, `requirements.txt`, the five config files, and `01_download.py`. The
download script is check-only by default, validates each file's columns, and records where the data came
from in `SOURCE.json`.

**Problems and fixes:**

1. **Wrong Python version.** The plan said 3.11; your machine had only 3.14. You chose 3.14. Before writing
   anything, we checked that every package published a 3.14 version.
2. **Kaggle links looked broken** (404 errors). Kaggle refuses "HEAD" requests; with a normal request the
   pages loaded fine (200). So the links were OK.
3. **The Synthea download crashed at 23 %** ("connection forcibly closed by the remote host"). We added
   automatic retry and **resume** (HTTP "Range" requests continue from the last byte received), tested it
   on a smaller file, and the next run completed.
4. **Synthea refused to run with Java 11** ("class file version 61.0"). Version 61 means Java 17: the plan's
   "Java 11+" was out of date. The script now requires Java 17 and honours `JAVA_HOME`. You installed
   Java 21. Setting `JAVA_HOME` didn't change what `java --version` printed (that follows PATH), but the
   script uses `JAVA_HOME`, so it worked.
5. Small: file sizes under 1 MB printed as "0.0 MB"; changed to KB.

**Acceptance check passed:** all four datasets reported PRESENT.

## Phase 2: Ingest, normalise, timeline, synthetic lifestyle

**Built:** `ingest.py`, `normalize.py`, `timeline.py`, `synth_lifestyle.py`, `02_build_dataset.py`.

**Problems and fixes (found by checking the real data before writing code):**

6. **All 10 LOINC codes existed**, but glucose also appeared under a second code (`2345-7`, 7,854 rows).
   It was added as an alias.
7. **Safety keywords didn't match Synthea's wording:**
   - soy allergy is recorded as **"Glycine max (substance)"**;
   - the keyword "coronary" also matched an imaging finding ("Abnormal findings diagnostic imaging
     heart+coronary circulat") present in 179 patients, which would have wrongly applied the heart rule;
   - "osteoarthritis" matched hand osteoarthritis (irrelevant for knee-loading exercise);
   - knee and back injuries also exist as real conditions.

   All were fixed in `safety_rules.yaml`.
8. **225 of the 1,000 living patients are children.** With your agreement, recommendations go to the 775 adults.
9. **HbA1c lower limit too strict.** Synthea produces 2.4–2.9 % for some healthy patients; our 3 % limit
   dropped 1,286 real readings and biased HbA1c upward. Lowered to 2 %. Weight limit lowered to 0.5 kg
   so low-birth-weight newborns aren't dropped.
10. **Sleep dataset quirks:** "Normal" and "Normal Weight" merged; ages outside 27–59 use the nearest edge
    rather than extrapolating.
11. **Back injuries are common (230 of 775)** because Synthea records active chronic low back pain for
    204 of the 775 cohort adults. You agreed to keep this.

**Acceptance check passed:** `timeline.parquet` exists; tests pass for unit conversion, sort order and
range validation; the summary printed the patient count and events per type.

## Phase 3: Temporal engine and profiles

**Built:** `temporal.py` and `profile.py`.

**Problems and fixes:**

12. **Social records were "chronic conditions".** Many ongoing Synthea conditions are not illnesses:
    "Full-time employment", "Received higher education", "Medication review due". As chronic conditions
    they would keep a permanent 0.8 weight. We added a `social_finding` class with no floor, and treated
    "History of …" records as resolved major events.
13. **Volatility made unit-free** (Part F1).
14. **A weekly-average date could fall after the reference date** (a week's end date in the future);
    fixed to use the last real day in the week.
15. **Library changes in pandas 3:** it returns read-only arrays (our code tried to modify one), and an
    all-empty date column came out without a time zone. Both fixed; the second also made the timeline
    builder force UTC everywhere.
16. **Two of my test cases had wrong expected values.** The code was right; the made-up test data was not.
    Fixed the tests, not the code.
17. **The hospital-stay "trend"** (glucose slope 199/year): we added the 1-year minimum span rule.

**Acceptance check passed:** the BP example gives slope 5.4 and a finite time-to-threshold; an old
allergy keeps weight ≥ its floor.

## Phase 4: Candidates and safety filter

**Built:** `candidates.py` and `safety.py`, plus tests including the acceptance test.

**Checks:** every rule fired on real patients; the worst-affected patient still kept hundreds of candidates.
The acceptance check passed on test data **and** on all 5 real patients with a peanut allergy and a knee injury.

## Phase 5: CVD risk model

**Built:** `risk_model.py` and `03_train_risk_model.py`.

**Problems and fixes:**

18. **Windows Smart App Control blocked scikit-learn and SciPy.** These are compiled files, and the newest
    builds for Python 3.14 were too new for Microsoft to trust ("An Application Control policy has blocked
    this file"). We **did not bypass the security feature**. Instead we found older versions
    (scikit-learn 1.8.0, SciPy 1.17.0) that load fine and pinned them, with a comment explaining why.
19. **A test that didn't test what it claimed.** It tried to swap in a small hyperparameter grid, but
    Python fixes default argument values when the function is defined, so the swap had no effect and the
    test silently ran the full grid. The grid became an explicit argument.
20. **Three patient groups had identical median risk (0.546).** Suspicious, so we checked the means
    (0.56, 0.58, 0.55): different. Isotonic calibration outputs steps, so medians landed on the same step.
    Not a bug.

**Acceptance check passed:** metrics and plots saved; model file saved; all 775 profiles have `cvd_risk`.

## Phase 6: Scoring, set selection, explanations, full pipeline

**Built:** `ranker.py`, `explain.py`, `pipeline.py`, `04_run_pipeline.py`.

This phase had the most fixes, because looking at real output revealed design gaps:

21. **A 1-minute cocktail ("zorbatini") was the top recommendation.** Added a 5-minute minimum and
    excluded alcohol.
22. **Every reason was about BMI.** One abnormal metric gave R = 1 and U = 1 to everything targeting it.
    R now also counts active chronic conditions (sleep apnoea → sleep items, etc.).
23. **A salad dressing was ranked first.** Recipes are now limited to real meals using Food.com's meal tags.
24. **Only 3 recommendations** for a 30-minute budget: the budget look-ahead was added (Part F5). The first
    version ignored the 2-per-domain cap, which a test caught; then the fallback was improved to build the
    longest list that fits.
25. **Two walks in one list**: added redundancy (same-tag) penalties to `synergy.yaml`.
26. **"Projected to reach 6 in about 1,473 years"**: added minimum urgency and maximum horizon for trend sentences.
27. Wording fixes: "Body mass index … 30.0 threshold" → "BMI … 30"; "daily steps … steps a day";
    condition reasons now name only the goals related to that condition.

**Acceptance check passed:** `04_run_pipeline.py --patient <id>` prints 5 recommendations with reasons
and the disclaimer. Across all 775 patients: always 5 items, never over budget, zero safety violations,
every item explained.

## Phase 7: Feedback loop, simulator, weight learner

**Built:** `feedback.py`, `weight_learner.py`, `05_simulate_feedback.py`.

28. **Speed.** 46,500 full pipeline runs at ~350 ms each would take about 4.5 hours. Fixed parts are now computed
    once per patient; only F, H and P are recomputed each round.
29. **A tempting shortcut was rejected.** Limiting selection to the top 300 items changed 22 of 60 lists.
    Instead the selection was rewritten to be about 6× faster and **proven identical** (Part F5).
30. **Simulated users accepted 82 % of everything**, leaving no room to show learning. Calibrated with an
    offset of −1.75 (≈ 50 % at the start).
31. **Found the H feedback loop and the "learned weights ignore health" problem**; you decided to keep the
    defaults live.

**Acceptance check passed:** learning-curve plot saved; learned weights written to `config/weights_learned.yaml`.

## Phase 8: Evaluation

**Built:** `evaluate.py`, `06_evaluate.py`, `reports/results.md`.

32. **Experiment 2 showed almost no effect**, so we added a breakdown by urgency level to show where U
    does and doesn't matter.
33. **Experiment 3's first comparison was unfair** (a 10 % band raises 14× more alarms than urgency
    ≥ 0.2), and my first written reading of it was **wrong**: it implied urgency was more precise. We
    added 2 % and 5 % bands, compared at similar alarm counts, and the interpretation text is now
    **generated from the numbers** so it can't contradict them. An unverified claim was removed.
34. **A real safety bug:** a fish-allergic patient was recommended *"poached basa fillets"*. Basa is a fish,
    but it wasn't on our keyword list. An **audit** against Food.com's own fish/seafood tags found **38
    seafood recipes** our keywords missed (hake, flounder, grouper, orange roughy, sole, mahi mahi, squid…).
    We extended the lists, excluded two false matches ("spike seasoning", "seafood seasoning"), added
    regression tests, and **re-ran the simulation and evaluation** so every number reflects the fix.
35. **Case study 2 wasn't a prediabetes case** (the patient also had type 2 diabetes): now excluded.
36. **Hot dogs ranked first for a prediabetic patient** (low sugar, high protein). With your agreement,
    processed meats are now excluded from recipes.
37. **The alcohol filter had a big bug.** Plain substring matching meant "gin" matched **gin**ger and extra
    vir**gin** olive oil, "rum" matched bread**crum**bs and d**rum**sticks, and "wine" matched wine vinegar.
    It had wrongly removed **24,493 recipes** (15 % of eligible ones). We added an exceptions list (also so
    "graham crackers" don't count as ham) with tests, and re-ran everything again.
38. Small: "6.0 hours below the 6-hour threshold" now shows 5.96; a hard-coded "319" is now computed;
    "±0.02" label now "±2 %"; an outdated limitation sentence was rewritten.

**Acceptance check passed:** `reports/results.md` contains all 7 experiments with figures and 3 case studies.

## Phase 9: Dashboard

**Built:** `app/dashboard.py` and `tests/test_dashboard.py`.

39. Streamlit's test tool couldn't handle a formatted drop-down list, so the patient picker now uses the
    labels directly.
40. Replaced a deprecated Streamlit option; turned off Streamlit's usage-statistics collection.

**Acceptance check passed:** the dashboard shows the patient picker, timelines with thresholds and
projected crossings, profile, safety drops with reasons, recommendations with score breakdowns, and
feedback buttons that re-rank live. Automated tests click the buttons and confirm the re-ranking. The page
served correctly from a real server. The visual layout was not reviewed in a browser from the build environment.

## Phase 10: README

`README.md` covers setup (including the Java 17 and Smart App Control notes), how to run every script,
the design, deviations from the plan, results and limitations. Its first draft understated disk space
(said 1.5 GB; measured 3.2 GB, because Synthea leaves a ~1 GB duplicate copy that can be deleted).

## What the corrections teach

- **Look at real output, not just whether code runs.** Many issues (the cocktail, the hot dog, "1,473
  years", basa) only showed up by reading actual recommendations.
- **Check the data before trusting assumptions.** The LOINC codes, Synthea wording and HbA1c ranges all
  needed adjusting.
- **Be suspicious of round or identical numbers.** Identical medians, identical tag counts (1,647
  low-sodium and 1,647 low-sugar recipes turned out to be a coincidence; we checked) and 82 % acceptance
  were all investigated.
- **Report negative results honestly** (experiments 2 and 3).
- **Never trade correctness for speed without proof** (the rejected shortlist; the proven-identical rewrite).
- **When a fix changes the data, re-run everything downstream**, so the report never mixes old and new numbers.

---

# Part J: Questions your teacher may ask, with answers

**1. What is novel about this project?**
Four things working together: (a) decay whose speed depends on each measurement's volatility and which
never forgets allergies or chronic disease below a floor; (b) urgency based on projected time-to-threshold,
weighted by how reliable the trend is; (c) a seven-term score combining health, goals, urgency,
preferences, feasibility, history and effort; (d) building the list as a coherent set (synergy,
conflicts, time budget, domain variety) rather than taking the top 5.

**2. Why synthetic data?** Real health records are private and legally protected. Synthea produces
realistic records with no privacy risk. The trade-off is that results may not transfer to real patients,
which we state clearly.

**3. Why not use a neural network?** For table-shaped data with 12 features, gradient-boosted trees
(XGBoost) usually match or beat neural networks, train in seconds and are easier to inspect. The ranking
itself is deliberately a transparent formula so every decision can be explained, which matters in health.

**4. Why is the safety filter rule-based and not learned?** Safety must be predictable and auditable. A
learned model could occasionally let an allergen through; a rule never does (as long as the keyword list
is complete, which is why we audited it).

**5. How do you know the risk model is good?** On 10,292 test rows it never saw, ROC-AUC is 0.806 (a
random guess is 0.5) and the calibration curve lies on the diagonal, so a predicted 30 % risk means about
30 % of such people really have the disease.

**6. What does calibration mean and why does it matter here?** It means predicted probabilities match
real frequencies. We use the risk as a number inside the score (R), so it must mean what it says.

**7. Why split into train, validation and test?** Training data teaches the model, validation picks the
settings and calibrates, test gives an honest final grade. Using test data for decisions would make the
grade over-optimistic.

**8. Explain the urgency formula.** Fit a line to the last 5 years; if already past the threshold,
urgency 1; if moving away, 0; otherwise urgency = (r² × min(1, n/5)) / (1 + years to threshold). Reliable
trends that cross soon score high.

**9. Explain the decay with an example.** A 23-year-old egg allergy: plain decay gives 0.000000087, which
means forgotten; ours gives max(1.0, …) = 1.0. A sinusitis from 8 years ago fades to near 0 under both.

**10. Why is the set selection greedy? Is it optimal?** Choosing the best possible 5 out of thousands, with
interactions between items, is a hard combinatorial problem. Greedy is fast and gave large improvements
(conflicts 5.06 → 0.46). It isn't guaranteed optimal.

**11. Did urgency actually help?** Honestly, only a little. In experiment 2 it raised targeting of
high-urgency rising metrics by about 3 points; in experiment 3 a simple near-threshold flag predicted
crossings better. Urgency gives earlier warnings for the crossings it does catch. These are real findings
on this synthetic data.

**12. Why didn't you use the learned weights?** They maximise acceptance by simulated users, who mostly
care about convenience, so they reduce health relevance and urgency. A health system shouldn't optimise
only for "what people click". Also, the history weight is inflated by a feedback loop.

**13. What is a feedback loop in a recommender?** When the system's own past choices influence the data it
learns from. Our H term is built from past acceptances, so the learner over-credits it.

**14. How do you make results reproducible?** Seed 42 everywhere (data generation, splits, sampling,
simulation), configs in files, scripts that regenerate every output, and 314 automated tests.

**15. How did you test it?** Unit tests for every function (e.g. unit conversion, sort order, the BP
example, allergen exclusions), acceptance tests from the plan, an equivalence test proving the fast
selection matches the slow one on 200 random cases, integration tests on real data, and dashboard tests
that click the buttons.

**16. What was the hardest bug?** Probably the substring matching in ingredient filters. It caused a missed
fish allergen (basa) and 24,493 wrongly removed recipes (gin in ginger, rum in breadcrumbs). It shows why
real systems need curated ingredient databases.

**17. What are the limitations?** Synthetic data; rules not clinically reviewed; linear trends; simulated
feedback; the risk model trained on a different population and ages; keyword-based allergen detection;
diet tags that don't judge food quality. See Part K.

**18. What would you do next?** Test with real (consented, anonymised) data and clinicians; replace keyword
allergen matching with a curated database; try non-linear or probabilistic trend models; correct the H
feedback loop (e.g. with randomised exploration); learn weights that balance acceptance *and* health benefit.

**19. Why Parquet and not CSV or a database?** Parquet is compressed, fast and keeps data types; the plan
wanted no database server. The 1.3 million timeline rows load in under 0.2 seconds.

**20. Why Python 3.14 and pinned versions?** It was the installed version. Two libraries were pinned to
slightly older releases because Windows Smart App Control blocked the newest builds; we chose not to
disable a security feature.

**21. What does R² of 0.735 mean in case 1?** The straight line explains 73.5 % of the variation in his
blood pressure readings. Fairly reliable, so confidence is 0.735.

**22. Why only 5 recommendations?** The plan specifies K = 5: enough to cover several areas, few enough to
actually do. The time budget keeps the total realistic.

---

# Part K: Limitations and future work

| Limitation | Effect | What would fix it |
|---|---|---|
| Synthetic clinical and lifestyle data | results may not hold for real people | evaluation on real anonymised records |
| Rules and thresholds not clinically reviewed | simplified medicine | clinician review |
| Linear trends, yearly readings | misses sudden changes; experiment 3 negative | non-linear / probabilistic models, more frequent data |
| Simulated feedback | learned weights reflect simulator assumptions | real user study |
| H feedback loop | history over-credited | exploration, or debiasing techniques |
| Risk model population mismatch (ages 29–65, different country) | less reliable for 319 patients | train on a matching population |
| Keyword allergen matching | can miss unusual names (basa) | curated ingredient–allergen database |
| Diet tags ignore food quality | some poor recipes can score well | nutrition quality scores |
| Greedy selection | not guaranteed optimal | exact or approximate optimisation |

---

# Part L: How to demo the project

From the `lphrs` folder, with the virtual environment active (`.venv\Scripts\activate`):

1. **Show the data is in place:** `python scripts/01_download.py` (all PRESENT).
2. **Show one patient in the terminal:**
   `python scripts/04_run_pipeline.py --patient 355d5f82 --breakdown`
   Point out the trend sentence ("projected to reach 140 in about 3 months"), the safety removals, the
   score terms and the disclaimer.
3. **Show the learned-weights difference:** `python scripts/04_run_pipeline.py --patient 355d5f82 --weights learned`.
   Note how the reasons shift toward convenience.
4. **Open the dashboard:** `streamlit run app/dashboard.py`.
   - Pick patient `355d5f82` and open **Metric timelines** to show the systolic chart with the
     threshold line and projected crossing.
   - Open **Safety filter** to show what was removed and why.
   - Back in **Recommendations**, click *Too difficult* on an exercise and show the re-ranked list
     and the lowered intensity cap in the feedback panel.
5. **Show the evaluation:** open `reports/results.md` (summary table, then experiments 1, 4 and 5 as
   clear wins, and 3 as the honest negative result).
6. **Show the tests:** `python -m pytest -q` (314 passed).

---

# Part M: Glossary

| Term | Meaning |
|---|---|
| Acceptance check | the "done when" test for each phase in the build plan |
| Ablation | turning one component off to measure its effect |
| Allergen tag | a label like `allergen:peanut` attached to a recipe |
| α (alpha) | how much synergy counts in set selection (0.3) |
| AUC | area under a curve; see ROC-AUC |
| Baseline | a simple method to compare against |
| Brier score | mean squared error of predicted probabilities; lower is better |
| Calibration | making predicted probabilities match real frequencies |
| Candidate | anything that could be recommended (template or recipe) |
| Cohort | the group of patients studied (775 living adults) |
| Common random numbers | giving compared methods the same random draws |
| Confidence (urgency) | r² × min(1, n/5); how much to trust a trend |
| Contraindication | a reason something should not be given to someone |
| CVD | cardiovascular disease (heart and blood vessel disease) |
| Decay / λ | how fast old information loses weight; half-life = ln 2 / λ |
| Deterministic | same input always gives the same output |
| Domain | category of advice (nutrition, fitness, sleep, lifestyle, preventive care, wellness) |
| Feature | an input variable to a model |
| Floor | minimum weight an event keeps however old (severity × persistence) |
| Greedy algorithm | builds a solution by taking the best-looking option at each step |
| Hyperparameter | a setting chosen before training (tree depth, learning rate) |
| Intensity cap | the highest exercise intensity currently suggested in a domain |
| Isotonic regression | a step-shaped, only-increasing curve used for calibration |
| K | number of recommendations (5) |
| Label | the answer a model learns to predict |
| Logistic regression | a linear model passed through a sigmoid to give a probability |
| LOINC | standard codes for lab tests and measurements |
| Longitudinal | data about the same person over time |
| OLS | ordinary least squares: the standard straight-line fit |
| Parquet | a compressed, typed, column-based file format |
| Platt scaling | sigmoid-based calibration |
| Precision / recall | of those flagged, share correct / of true cases, share flagged |
| Pure function | output depends only on inputs; no side effects |
| R² | how well a line fits (0–1) |
| ROC-AUC | chance a random positive is ranked above a random negative |
| Seed | the starting number for reproducible randomness (42) |
| Sigmoid | S-shaped function mapping any number to 0–1 |
| Stratified split | each part keeps the same class proportions |
| Synergy | how well two items go together (−1 to +1) |
| Synthea | open-source synthetic patient generator |
| Threshold (θ) | the danger line for a metric (e.g. 140 mmHg) |
| Time-to-threshold (TTT) | projected years until a trend crosses the threshold |
| Urgency | 0–1 score from time-to-threshold and confidence |
| UTC / ISO-8601 | a universal time zone / a standard date format |
| Volatility | how much a metric typically changes between visits, relative to its size |
| XGBoost | a fast library for gradient-boosted decision trees |

---

*All numbers in this report were produced by the code in this repository (`scripts/02`–`06`, seed 42).
Full tables and figures: `reports/results.md`. Setup and usage: `README.md`.*

*Wellness guidance only. Not medical advice.*
