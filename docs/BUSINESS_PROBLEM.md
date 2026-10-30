# Business problem: TalentKeep retention and pay benchmarking

## Situation

TalentKeep is a fictional mid-size pharmaceutical company with 1,470 employees
in Sales, Research & Development and Human Resources. The HR dataset (IBM HR
Analytics, Kaggle) shows **237 of 1,470 employees (16.1%) left**.

Every departure costs the company recruitment fees, interview time, onboarding,
lost productivity while the new hire ramps up, and lost team knowledge. HR has a
limited retention budget (stay interviews, career moves, overtime reduction, pay
adjustments) and cannot spend it on everyone.

## Questions to answer

| # | Business question | ML task | Target column | Who uses it |
| --- | --- | --- | --- | --- |
| 1 | Which current employees are most likely to leave, and why? | Binary **classification** | `Attrition` (Yes/No) | HR business partners, line managers |
| 2 | What monthly pay is typical for a role, level and experience? Is this person under- or over-paid relative to it? | **Regression** | `MonthlyIncome` (USD) | Compensation team, recruiters |

The two answers work together. A high-risk employee who is also **below the
pay benchmark** is a clear case for a pay review. A high-risk employee paid
fairly points to other causes, such as overtime, travel or satisfaction.

## Success criteria (set before training)

- **Attrition:** catch at least ~70% of leavers on training cross-validation
  (recall). Missing a leaver costs much more than a retention conversation that
  turns out to be unnecessary. Ranking quality is measured with ROC-AUC, and the
  model must beat the 16% base rate in precision.
- **Salary:** an average error (MAE) much lower than simply predicting the
  company average, and R² above 0.9.

## Data decisions

- **Dropped** constant or ID columns: `EmployeeCount`, `Over18`,
  `StandardHours`, `EmployeeNumber`.
- **Excluded for fairness:** `Gender`, `Age`, `MaritalStatus`. HR actions must
  not depend on protected characteristics. A test confirms the models never use them.
- **Feature selection:** 15 features for attrition and 6 for salary, so the
  web form stays usable. In cross-validation, the 15-feature attrition model
  scored ROC-AUC 0.823 against 0.827 for all 30 usable columns, so almost nothing
  was lost.
- **No leakage:** the scaler, encoder, model choice and decision threshold are
  all learned from the training 80% only. The test 20% is used once, at the end.

## Results on the 294 held-out employees

**Attrition (LogisticRegression, threshold 0.55)**

|  | Predicted stay | Predicted leave |
| --- | --- | --- |
| Actually stayed | 200 | 47 |
| Actually left | 18 | **29** |

- 29 of 47 leavers caught (recall 61.7%). This is below the 70% target reached
  in cross-validation, because the test split proved harder: test ROC-AUC 0.769
  against 0.823 in CV. This is reported honestly rather than tuned on the test set.
- HR would review 76 employees (26%) instead of all 294. Of those, 38% actually
  left, compared with 16% for a random pick, so the model gives **2.4x better
  targeting**.

**Salary (GradientBoostingRegressor)**

- R² 0.945, MAE $802/month (typical monthly income ranges from $1,009 to
  $19,999). Predicting the average would give MAE $3,387.
- Job level dominates pay: it accounts for about 95% of the model's feature
  importance. Job role and total working years refine the estimate.

## Main drivers of attrition (logistic coefficients, standardized)

Higher risk: Sales Representative and Laboratory Technician roles, overtime,
frequent travel and few total working years. Lower risk: Research Director
role, no travel, no overtime and longer careers.

**Recommendations:** review overtime load in sales and lab roles, and offer
career-path conversations to early-career staff. Use the pay benchmark to fix
below-benchmark pay for high-risk employees first.

## Limitations

- The data is fictional (created by IBM data scientists) and is a single
  snapshot. Patterns must be revalidated on the company's real data.
- 1,470 rows is small. Test scores vary by split, so CV and test scores are
  both reported.
- The model shows association, not causation. Use it to start a conversation,
  never to make automated decisions about a person.
- Retrain periodically, because the reasons people leave change over time.
