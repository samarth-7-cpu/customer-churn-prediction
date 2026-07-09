# Customer Churn Prediction — Team Roadmap (3 Members)

**Dataset:** Bank Customer Churn (Kaggle) — unmodified, 80:20 split, `random_state=42`
**Metric:** F1 score only — `f1_score(y_test, model.predict(X_test))`, default 0.5 threshold
**Hard rule:** `Complain` is target leakage — drop it. Also drop `RowNumber`, `CustomerId`, `Surname`. Test set is touched exactly once, at the very end.

---

## Role Split

| Role | Owner | Focus |
|---|---|---|
| A — Data & Pipeline Lead | e.g. you (Samarth) | Preprocessing, feature engineering, the shared train/test split |
| B — EDA & Insights Lead | Teammate 1 | Exploratory analysis, visualizations, business insight narrative |
| C — Modeling & Evaluation Lead | Teammate 2 | Model training, comparison, final web app (optional) |

Person A works first and fastest — everyone else depends on their split + cleaned features, so front-load their part.

---

## Phase 0 — Setup (Day 1, morning)

- [ ] All three: download dataset, read guidelines PDF together, agree on repo structure (`/data`, `/notebooks`, `/src`, `/reports`)
- [ ] Person A: create the **canonical train/test split** first thing — `train_test_split(..., test_size=0.2, random_state=42)` — and commit `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` (or pickle) so **everyone works off the identical partition**. This is the single most important step for compliance — do it before any EDA.
- [ ] Person A: drop `RowNumber`, `CustomerId`, `Surname`, `Complain` right after the split (before anyone touches features), and document why in a markdown cell (cite the leakage explanation from the guidelines).

## Phase 1 — EDA (Day 1–2)

**Owner: Person B**, working only on the **training set** (never touch `X_test`/`y_test` here).

- [ ] Class balance of `Exited` (churn is usually imbalanced — flag this early since it affects F1 later)
- [ ] Univariate distributions: Age, Balance, CreditScore, EstimatedSalary, Tenure
- [ ] Categorical breakdowns: Geography, Gender, NumOfProducts, HasCrCard, IsActiveMember vs churn rate
- [ ] Correlation heatmap (numeric features) + churn correlation ranking
- [ ] Segment analysis: e.g. churn rate by tenure bucket, by product count, by geography × gender
- [ ] Write 5–6 bullet "insight" takeaways as you go — these become the presentation's backbone later

## Phase 2 — Feature Engineering (Day 2–3)

**Owner: Person A**, informed by Person B's findings, still training-set-only.

- [ ] Encode categoricals (Geography, Gender) — one-hot or ordinal depending on model family
- [ ] Scale numeric features (StandardScaler/MinMax) — fit **only on train**, then apply same transform to test later
- [ ] Engineered features to consider: tenure buckets, balance-to-salary ratio, active-member × product-count interaction, zero-balance flag
- [ ] Build this as a **sklearn Pipeline/ColumnTransformer** (not manual df edits) — makes it trivial to apply identically to the held-out test set at the end and avoids leakage bugs
- [ ] Handoff to Person C: a fitted pipeline object + processed `X_train`

## Phase 3 — Modeling (Day 3–5)

**Owner: Person C**, using cross-validation on train only.

- [ ] Baseline: Logistic Regression
- [ ] Tree-based: Decision Tree, Random Forest
- [ ] Boosted: XGBoost
- [ ] SVM (optional if time allows — slower on this dataset size)
- [ ] Use **stratified k-fold CV on the training set** to tune hyperparameters and compare models — **never** touch `X_test` for this
- [ ] Track CV F1 (mean ± std) per model in a comparison table
- [ ] Handle class imbalance if EDA flagged it (class_weight='balanced', or SMOTE — but only inside the CV folds/pipeline, not applied globally before splitting)
- [ ] Pick the best model by CV F1, retrain it on full training set

## Phase 4 — Mid-Evaluation Checkpoint (Day 5)

**No test-set access at this stage.** The mid-eval is judged on progress and rigor, not final score — treat it as a checkpoint, not the finish line.

**Prep deck (5–6 slides):**

- [ ] **Slide 1 — Problem framing + compliance recap.** State the `Complain` leakage explanation, confirm the shared 80:20 split with `random_state=42`. Cheap to include, signals rigor immediately.
- [ ] **Slides 2–3 — EDA highlights** (Person B). Don't show every chart — pick 4–5 sharpest insights: class imbalance in `Exited`, strongest churn correlate, one segment finding (e.g. geography × churn or product count × churn). Prioritize interpretation over volume.
- [ ] **Slide 4 — Pipeline + feature engineering** (Person A). Diagram the ColumnTransformer/Pipeline, list engineered features, state explicitly it's fit on train only.
- [ ] **Slides 5–6 — Preliminary modeling** (Person C). Report **cross-validation F1 (mean ± std)** only, across folds, for models tried so far (Logistic Regression, Random Forest, XGBoost, etc.). Frame as "in-progress comparison" — not final numbers.
- [ ] **Slide 7 — What's left + timeline.** Remaining tuning, the single final test-set evaluation (not yet done), notebook cleanup, optional web app.

**Delivery:**

- [ ] Each person presents their own section live — don't let all three speak generically
- [ ] Narrate insights, not code/syntax, during the walkthrough
- [ ] Do **not** run `f1_score(y_test, ...)` before or during this round — save that single shot for the final evaluation

## Phase 5 — Final Evaluation (Day 6–7, once only)

**Owner: Person C, reviewed by all three.**

- [ ] Finish remaining hyperparameter tuning using CV on train only (based on mid-eval feedback, if any)
- [ ] Apply Person A's fitted preprocessing pipeline to `X_test` (transform only, never fit)
- [ ] Single, final call: `f1_score(y_test, model.predict(X_test))`
- [ ] Confusion matrix + classification report for the writeup (secondary, F1 is what's judged)
- [ ] **Do not** go back and re-tune after seeing this number — that's a disqualification risk per the guidelines

## Phase 6 — Deliverables (Day 7–8)

- [ ] **Notebook** (all three, merged): EDA → pipeline → results, clean markdown commentary throughout — this is graded on clarity too, not just code
- [ ] **Presentation** (Person B leads, others contribute slides): insights from EDA, modeling approach, final F1, top churn drivers, business recommendations
- [ ] **Optional web app** (Person C, if time permits): simple Streamlit/FastAPI form that takes customer features and returns churn risk + probability, using the saved pipeline + model

---

## Timeline Snapshot

| Day | Milestone |
|---|---|
| 1 | Split finalized + committed, EDA underway |
| 2 | EDA complete, feature engineering started |
| 3 | Pipeline finalized, modeling started |
| 4 | CV-based model comparison in progress |
| 5 | **Mid-evaluation checkpoint** — present EDA, pipeline, CV results, plan (no test-set touch) |
| 6–7 | Finish tuning based on feedback, single final test-set evaluation |
| 7–8 | Notebook assembly, presentation polish, optional web app, submission |

## Compliance Checklist (review together before submitting)

- [ ] Same 80:20 split, `random_state=42`, used by all three
- [ ] `Complain`, `RowNumber`, `CustomerId`, `Surname` dropped before modeling
- [ ] Test set never used in training, CV, or tuning — only the final single F1 call
- [ ] F1 computed via `f1_score(y_test, model.predict(X_test))` at default threshold — no custom threshold tuning
- [ ] Notebook includes EDA + pipeline + results in one place
