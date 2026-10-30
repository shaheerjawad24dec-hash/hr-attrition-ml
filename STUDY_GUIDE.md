# Study guide: understand every part of the project

Read this guide, run the app, then explain each file in your own words. If
your brief requires original code, rebuild the parts yourself and use this
project only as a reference.

## 1. The big picture

```
Kaggle CSV ──► train_models.py ──► models/*.joblib + metrics.json
                                         │ loaded once at startup
Browser form ──JSON──► FastAPI (app/main.py) ──► model.predict ──JSON──► page updates
```

- **Training** happens once, offline, with `python train_models.py`.
- **Prediction** happens on every button click. Nothing is retrained then; the
  saved model is reused.
- The page and the API come from the same server (port 8000), so JavaScript can
  call `/predict/...` with relative paths and no CORS setup is needed.

## 2. The ML workflow used (the professional checklist)

1. **Define the business question and success metric first**
   (`docs/BUSINESS_PROBLEM.md`).
2. **Understand the data:** 1,470 rows, 35 columns, no missing values, 16%
   attrition (imbalanced).
3. **Clean and select features:** drop constant or ID columns, drop protected
   attributes, keep a usable form.
4. **Split before anything is learned:** 80% train, 20% test. Use `stratify` for
   classification so both parts contain 16% leavers.
5. **Preprocess inside a Pipeline:** scaling and one-hot encoding are fitted on
   training folds only, which prevents data leakage.
6. **Compare candidate models with 5-fold cross-validation** on the training set.
7. **Select by a rule decided in advance:** the simplest model within 1% of the best.
8. **Tune the decision threshold** on out-of-fold training predictions.
9. **Evaluate once on the untouched test set.** Report several metrics and
   compare with a naive baseline.
10. **Save** the fitted pipeline with joblib and **serve** it through an API with
    input validation.
11. **Test** everything automatically.

## 3. Key concepts

**Classification vs regression.** Classification predicts a category (leave or
stay). Regression predicts a number (monthly income in USD).

**Train/test split.** The model must be judged on employees it has never seen,
or the score is over-optimistic.

**Cross-validation (5-fold).** The training set is cut into 5 parts. The model
trains on 4 and is scored on the 5th, five times, and the scores are averaged.
This is more reliable than one split and is used to *choose* a model without
touching the test set.

**Pipeline.** `Pipeline([("prep", ...), ("model", ...)])` chains preprocessing
and the model. `fit` learns both, `predict` applies both. Saving the pipeline
guarantees the API transforms inputs exactly as in training.

**ColumnTransformer.** Applies `StandardScaler` to numeric columns and
`OneHotEncoder` to text columns, then joins the results side by side.

**StandardScaler.** `z = (value - training mean) / training std`. Logistic
regression works better when features have comparable scales, and coefficients
become comparable.

**OneHotEncoder.** Turns `OverTime = Yes/No` into two 0/1 columns, because models
need numbers. `handle_unknown="ignore"` avoids a crash on an unseen category.

**Imbalanced classes.** Only 16% left. A model that always says "stays" is 84%
accurate and useless. `class_weight="balanced"` makes each leaver count about
5x more during training.

### Classification metrics (from the confusion matrix)

|  | Predicted stay | Predicted leave |
| --- | --- | --- |
| Actually stayed | TN = 200 | FP = 47 |
| Actually left | FN = 18 | TP = 29 |

- **Accuracy** = (TP+TN)/all = 229/294 = **77.9%**. This is misleading when
  classes are imbalanced.
- **Recall** = TP/(TP+FN) = 29/47 = **61.7%**, the share of leavers caught.
  This is the business priority.
- **Precision** = TP/(TP+FP) = 29/76 = **38.2%**, the share of flagged employees
  who really left (the base rate is 16%).
- **F1** = 2·P·R/(P+R) = **0.472**, which balances precision and recall.
- **ROC-AUC = 0.769**: the probability that a random leaver gets a higher risk
  score than a random stayer. 0.5 is random guessing and 1.0 is perfect. It does
  not depend on the threshold.

**Decision threshold.** `predict_proba` gives a probability. We choose the
cut-off: a lower threshold catches more leavers but raises more false alarms.
The rule used is: *the highest threshold that still catches 70% of leavers in
cross-validation on training data*, which gives **0.55**. On the test set it
caught 61.7%. Test scores can be lower than CV scores, and this difference is
reported rather than hidden.

### Regression metrics

- **MAE** (mean absolute error) = average |actual - predicted| = **$802/month**.
  It is easy to explain in money terms.
- **RMSE** = sqrt(mean of squared errors) = **$1,096**. It punishes big errors more.
- **R²** = 1 - SSE/SST = **0.945**: the model explains 94.5% of the variance in
  pay on the test set. It is not a percentage of correct answers.
- **Baseline:** always predicting the training average gives MAE $3,387.

### Why these models won

- **Attrition:** Logistic regression (CV AUC 0.823) and Gradient Boosting
  (0.825) were within 1%, so the simpler, explainable logistic model was kept.
  Its coefficients show *why* someone is at risk.
- **Salary:** Linear regression's MAE ($859) was 12% worse than the tree
  ensembles (about $765), because pay jumps in steps between job levels and
  trees capture steps. Gradient Boosting and Random Forest tied, and Gradient
  Boosting is listed first and saves to a much smaller file.

## 4. Walk through train_models.py

| Code | Purpose |
| --- | --- |
| Imports | `pandas` reads the CSV; `sklearn` provides the models, metrics, splitters and preprocessing; `joblib` saves models; `json` writes metrics |
| `ROOT`, `DATA_PATH`, `MODEL_DIR` | Paths built from the script's own folder, so the script works from any working directory |
| `SEED = 42` | Repeatable splits and models |
| `TEST_SIZE`, `TOLERANCE`, `TARGET_RECALL` | Settings in one place: 20% test, 1% "simplest wins" margin, 70% recall goal |
| Feature lists | Which columns each model uses; protected attributes are deliberately absent |

**`load_data()`** checks the CSV exists and reads it with `encoding="utf-8-sig"`.
The Kaggle file starts with a BOM character that would otherwise corrupt the
first column name. It then checks that every needed column is present.

**`preprocessor(numeric, categorical)`** returns a ColumnTransformer that scales
the numeric columns and one-hot encodes the text columns.

**`choose(candidates, scores, higher_is_better)`** finds the best score, then
walks the candidates from simplest to most complex and returns the first one
within 1% of the best.

**`train_attrition(df)`**
1. `y = (df["Attrition"] == "Yes").astype(int)` converts Yes/No to 1/0.
2. `train_test_split(..., stratify=y)` keeps 16% leavers in both parts.
3. Three candidate models go in a dictionary. The `for` loop wraps each in a
   pipeline and runs `cross_validate(..., scoring="roc_auc")` on training data only.
4. `choose(...)` picks the model.
5. `cross_val_predict(..., method="predict_proba")` gives each training employee
   a probability from a model that did *not* see them. Thresholds from 0.05 to
   0.95 are tried, and the highest one that keeps recall >= 70% is kept.
6. `model.fit(X_train, y_train)` trains the final model on all training data.
7. On the test set, `predict_proba(...)[:, 1]` takes the probability of class 1
   ("leaves"). `>= threshold` turns it into a yes/no.
8. `confusion_matrix(...).ravel()` gives TN, FP, FN, TP.
9. `feature_means` stores the average transformed value of each column. The API
   uses it to explain predictions.
10. `joblib.dump({...})` saves a bundle: the pipeline, threshold, feature order
    and means.
11. It returns a dictionary of metrics for `metrics.json`.

**`top_drivers(model)`** applies only if the model has coefficients (`coef_`).
It sorts them by size and returns the 6 strongest. A positive coefficient raises
risk and a negative one lowers it. `clean_name` turns `cat__OverTime_Yes` into
`OverTime = Yes`.

**`train_salary(df)`** follows the same pattern with `KFold` (no stratify for a
number), scoring on MAE and R². It saves the pipeline plus `test_mae`, which the
API uses to show a ± range.

**`main()`** creates `models/`, loads the data, trains both models and saves
everything to `metrics.json`, including the drop-down choices and library
versions. It prints a summary. `if __name__ == "__main__":` means the script
only trains when you run it, not when another file imports it.

## 5. Walk through app/main.py

- **`lifespan`** runs once at startup. It checks the three files exist (with a
  helpful error otherwise), loads them with `joblib.load` and keeps them in
  `app.state`. `yield` hands control to the server.
- **`JobRole = Literal[...]`** means only these 9 strings are accepted.
- **`ValidatedInput`** uses `extra="forbid"` to reject unknown keys and
  `allow_inf_nan=False` to reject NaN or infinity.
- **`EmployeeInput` / `SalaryInput`**: each line is a field with a type and
  bounds. `ge` is >= and `le` is <=. `strict=True` means `2.5` is not accepted
  as an integer.
- **`@model_validator(mode="after")`** adds cross-field rules, for example years
  at the company cannot exceed total working years. Raising `ValueError` makes
  FastAPI return 422.
- **`EMPLOYEE_COLUMNS` / `SALARY_COLUMNS`** map API names (snake_case) to the
  Kaggle column names the pipeline was trained on.
- **`to_frame`** builds a one-row pandas DataFrame in the exact training column
  order. Models expect a table, even for one row.
- **`explain`**: for a linear model, log-odds = intercept + Σ coefᵢ·xᵢ. So
  coefᵢ·(xᵢ − averageᵢ) is how much feature i moves *this* employee's risk away
  from the average employee. One-hot columns of the same feature are summed back
  together, and the 3 largest positive effects are returned.
- **`/predict/attrition`** takes the probability from `predict_proba(row)[0, 1]`,
  compares it with the saved threshold and sets the risk level (High >=
  threshold, Medium >= half of it, otherwise Low).
- **`/predict/salary`** returns the estimate, ± MAE as a typical range and, if a
  current salary was given, the % gap and Below/Within/Above benchmark.

## 6. Walk through frontend/index.html

- `<head>`: charset, the mobile viewport, a title and the CSS in `<style>`.
  `:root` holds the colour variables. Grid and flex lay out the cards and form
  columns, and media queries stack them on phones.
- Forms: each input's `name` equals the API field name. `label for` matches the
  input `id`. `min`, `max`, `step` and `required` give browser-side validation.
  The backend validates again, because anyone can call the API directly.
- `role="status" aria-live="polite"` makes screen readers announce new results.
- JavaScript:
  - `ROLES` / `SCALE` fill the repeated drop-downs. The roles must match the
    `Literal` in `main.py`.
  - `connectForm(formId, route, resultId, render)` is reusable submit handling.
    It calls `preventDefault()` to stop a page reload. It loops over `FormData`,
    skipping empty optional fields and converting numbers with `Number()`.
    `fetch` sends a POST with a JSON body. If `!response.ok`, it formats
    FastAPI's 422 `detail` list into readable text. `finally` re-enables the button.
  - The attrition renderer shows the probability, a coloured badge, a bar with a
    threshold marker and the top factors. It uses `textContent` (not
    `innerHTML`) for server data, which is safer.
  - The salary renderer shows the estimate, range and gap.
  - `loadMetrics()` calls `GET /metrics` once at page load to fill the score
    tiles, the CV tables and the confusion matrix.

## 7. Supporting files

- `setup_project.py`: checks Python >= 3.12, creates `.venv` with the `venv`
  module and runs pip, training and tests through `subprocess.run`. It stops
  with a clear message if a step fails.
- `run_app.py`: starts `uvicorn app.main:app --reload` with the `.venv` Python
  and opens the browser after 2.5 seconds.
- `.bat` / `.sh` files: double-click wrappers around those two scripts.
- `.vscode/launch.json`: press **F5** and choose "Run FastAPI app" or "Train
  models" to run with the debugger.
- `tests/test_app.py`: a `client` fixture starts the app once. The tests check
  the page, high vs low risk, that overtime raises risk, the salary range and
  gap, that a higher level pays more, 422 errors for bad input, that the scores
  in metrics.json can be reproduced, that no protected attributes are used, and
  the missing-model error.

## 8. Viva questions

**Why this dataset?** It is a real business problem (staff turnover) with both
a category target (Attrition) and a numeric target (MonthlyIncome) in one table.

**Why not use accuracy for attrition?** With 84% stayers, "everyone stays"
scores 84% accuracy and catches no one. Recall, precision and ROC-AUC show
whether the model actually finds leavers.

**Why is the threshold not 0.5?** 0.5 is arbitrary. The business chose a recall
target, so the threshold was set to meet it, using training data only.

**Why is test AUC (0.769) lower than CV AUC (0.823)?** 294 employees with 47
leavers is a small sample, so scores vary between splits. Tuning on the test
set to "fix" this would be cheating (leakage).

**Why logistic regression if Gradient Boosting scored higher in CV?** The gap
(0.0014) is within noise. Logistic regression is simpler, faster and
explainable to HR.

**Why Gradient Boosting for salary?** Pay jumps in steps by job level. Trees
model steps naturally, and the linear model was 12% worse in MAE.

**What is data leakage and how did you prevent it?** It means information from
the test set influencing training. Here, the split comes first; scaling,
encoding, model choice and threshold are all learned inside training folds;
and the test set is used once.

**Why exclude gender, age and marital status?** Using protected
characteristics for HR decisions is unfair and often illegal. The model
remained just as good without them.

**Does "OverTime raises risk" mean overtime causes leaving?** No, it is an
association. It is a good starting point for an HR investigation, not proof.

**What would you improve?** Use real company data with exit dates (survival
analysis), add calibration, track fairness metrics by group, retrain regularly
and monitor for drift.

## 9. Practice tasks

1. Change `TARGET_RECALL` to 0.8, retrain, and explain how precision and the
   number flagged change.
2. Add `Department` to the salary features, retrain and compare MAE.
3. In `/docs`, send `"over_time": "Maybe"` and explain the 422 response.
4. Set `TOLERANCE = 0` and see which attrition model is chosen.
5. Add a fourth candidate (for example `DecisionTreeClassifier`) to the
   comparison.
6. Restyle the page yourself, keeping the input `name` attributes unchanged.
