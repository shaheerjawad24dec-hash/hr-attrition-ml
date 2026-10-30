# API examples

Start the server (see README), open http://127.0.0.1:8000/docs, expand a POST
operation, click **Try it out**, paste a JSON example and click **Execute**.

## Attrition risk (classification)

`POST /predict/attrition`

```json
{
  "job_role": "Sales Representative", "job_level": 1, "monthly_income": 2200,
  "total_working_years": 2, "years_at_company": 1, "years_with_curr_manager": 0,
  "num_companies_worked": 3, "over_time": "Yes", "business_travel": "Travel_Frequently",
  "distance_from_home": 20, "stock_option_level": 0, "job_satisfaction": 1,
  "environment_satisfaction": 1, "work_life_balance": 1, "job_involvement": 2
}
```

Actual response (HTTP 200):

```json
{
  "attrition_probability": 0.9974,
  "at_risk": true,
  "risk_level": "High",
  "threshold": 0.55,
  "top_risk_factors": [
    {"feature": "JobRole", "value": "Sales Representative", "effect": 1.465},
    {"feature": "OverTime", "value": "Yes", "effect": 1.072},
    {"feature": "TotalWorkingYears", "value": 2, "effect": 0.981}
  ]
}
```

- `risk_level`: **High** if probability >= threshold, **Medium** if at least
  half the threshold, otherwise **Low**.
- `effect` = coefficient x (this employee's value - average employee), on the
  log-odds scale. Only factors that raise risk are listed (top 3).

A low-risk example (a Research Director with level 4, $16,000/month, 20 years'
experience, no overtime and high satisfaction) returns
`"attrition_probability": 0.0071, "risk_level": "Low"`.

| Field | Allowed values |
| --- | --- |
| job_role | Healthcare Representative, Human Resources, Laboratory Technician, Manager, Manufacturing Director, Research Director, Research Scientist, Sales Executive, Sales Representative |
| job_level | integer 1-5 |
| monthly_income | 1000-20000 |
| total_working_years, years_at_company, years_with_curr_manager | integers 0-40; years_with_curr_manager <= years_at_company <= total_working_years |
| num_companies_worked | integer 0-10 |
| over_time | "Yes" or "No" |
| business_travel | "Non-Travel", "Travel_Rarely", "Travel_Frequently" |
| distance_from_home | integer 1-30 |
| stock_option_level | integer 0-3 |
| job_satisfaction, environment_satisfaction, work_life_balance, job_involvement | integer 1-4 |

## Salary benchmark (regression)

`POST /predict/salary`

```json
{"job_role": "Sales Executive", "job_level": 2, "total_working_years": 8,
 "years_at_company": 5, "years_in_current_role": 3, "education": 3,
 "current_monthly_income": 4500}
```

Actual response (HTTP 200):

```json
{
  "predicted_monthly_income": 5923.72,
  "typical_range": [5121.63, 6725.82],
  "unit": "USD per month",
  "current_monthly_income": 4500.0,
  "gap_percent": -24.0,
  "pay_position": "Below benchmark"
}
```

`current_monthly_income` is optional. Without it, the last three keys are not
returned. `typical_range` is the estimate ± the model's test MAE ($802).
`pay_position` is "Below benchmark", "Within benchmark range" or "Above benchmark".
Rules: `education` is 1-5, and `years_in_current_role <= years_at_company <= total_working_years`.

## Errors and scores

Missing fields, extra fields, wrong types, unknown options, NaN, out-of-range
values and inconsistent years return HTTP **422**, for example:

```json
{"detail": [{"type": "literal_error", "loc": ["body", "over_time"],
             "msg": "Input should be 'Yes' or 'No'", "input": "Maybe"}]}
```

`GET /metrics` returns everything in `models/metrics.json`: the CV comparison,
chosen models, threshold, test scores, confusion matrix and top drivers.
