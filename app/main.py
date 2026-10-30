"""Load the saved HR models and serve the HTML page and prediction API."""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"

JobRole = Literal[
    "Healthcare Representative", "Human Resources", "Laboratory Technician",
    "Manager", "Manufacturing Director", "Research Director",
    "Research Scientist", "Sales Executive", "Sales Representative",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load once at startup instead of loading again for every request."""
    required = ["attrition_classifier.joblib", "salary_regressor.joblib", "metrics.json"]
    missing = [name for name in required if not (MODEL_DIR / name).exists()]
    if missing:
        raise RuntimeError(f"Missing {missing}. Run: python train_models.py")
    app.state.attrition = joblib.load(MODEL_DIR / "attrition_classifier.joblib")
    app.state.salary = joblib.load(MODEL_DIR / "salary_regressor.joblib")
    app.state.metrics = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))
    yield  # Requests are served after this point.


app = FastAPI(title="TalentKeep HR Analytics API", version="1.0.0", lifespan=lifespan)


class ValidatedInput(BaseModel):
    # Reject unknown keys and non-finite numbers such as NaN or infinity.
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class EmployeeInput(ValidatedInput):
    """One current employee. Ranges match the values seen in the training data."""
    job_role: JobRole
    job_level: int = Field(ge=1, le=5, strict=True)
    monthly_income: float = Field(ge=1000, le=20000)
    total_working_years: int = Field(ge=0, le=40, strict=True)
    years_at_company: int = Field(ge=0, le=40, strict=True)
    years_with_curr_manager: int = Field(ge=0, le=40, strict=True)
    num_companies_worked: int = Field(ge=0, le=10, strict=True)
    over_time: Literal["Yes", "No"]
    business_travel: Literal["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
    distance_from_home: int = Field(ge=1, le=30, strict=True)
    stock_option_level: int = Field(ge=0, le=3, strict=True)
    job_satisfaction: int = Field(ge=1, le=4, strict=True)
    environment_satisfaction: int = Field(ge=1, le=4, strict=True)
    work_life_balance: int = Field(ge=1, le=4, strict=True)
    job_involvement: int = Field(ge=1, le=4, strict=True)

    @model_validator(mode="after")
    def years_are_consistent(self):
        if self.years_at_company > self.total_working_years:
            raise ValueError("years_at_company cannot exceed total_working_years")
        if self.years_with_curr_manager > self.years_at_company:
            raise ValueError("years_with_curr_manager cannot exceed years_at_company")
        return self


class SalaryInput(ValidatedInput):
    """A role and experience profile, plus an optional current salary to compare."""
    job_role: JobRole
    job_level: int = Field(ge=1, le=5, strict=True)
    total_working_years: int = Field(ge=0, le=40, strict=True)
    years_at_company: int = Field(ge=0, le=40, strict=True)
    years_in_current_role: int = Field(ge=0, le=40, strict=True)
    education: int = Field(ge=1, le=5, strict=True)
    current_monthly_income: Optional[float] = Field(default=None, ge=1000, le=20000)

    @model_validator(mode="after")
    def years_are_consistent(self):
        if self.years_at_company > self.total_working_years:
            raise ValueError("years_at_company cannot exceed total_working_years")
        if self.years_in_current_role > self.years_at_company:
            raise ValueError("years_in_current_role cannot exceed years_at_company")
        return self


# API field name (snake_case) -> column name used in the Kaggle CSV.
EMPLOYEE_COLUMNS = {
    "job_role": "JobRole", "job_level": "JobLevel", "monthly_income": "MonthlyIncome",
    "total_working_years": "TotalWorkingYears", "years_at_company": "YearsAtCompany",
    "years_with_curr_manager": "YearsWithCurrManager",
    "num_companies_worked": "NumCompaniesWorked", "over_time": "OverTime",
    "business_travel": "BusinessTravel", "distance_from_home": "DistanceFromHome",
    "stock_option_level": "StockOptionLevel", "job_satisfaction": "JobSatisfaction",
    "environment_satisfaction": "EnvironmentSatisfaction",
    "work_life_balance": "WorkLifeBalance", "job_involvement": "JobInvolvement",
}
SALARY_COLUMNS = {
    "job_role": "JobRole", "job_level": "JobLevel",
    "total_working_years": "TotalWorkingYears", "years_at_company": "YearsAtCompany",
    "years_in_current_role": "YearsInCurrentRole", "education": "Education",
}


def to_frame(data: BaseModel, columns: dict, order: list) -> pd.DataFrame:
    """Build a one-row table with the exact column names and order used in training."""
    values = data.model_dump()
    row = {csv_name: values[api_name] for api_name, csv_name in columns.items()}
    return pd.DataFrame([row])[order]


def explain(bundle: dict, row: pd.DataFrame, top: int = 3):
    """Return the factors that push this employee's risk up the most.

    Only possible for a linear model: each factor's effect is
    coefficient x (employee's value - average employee's value), on the log-odds scale.
    """
    model = bundle["model"]
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "coef_") or "feature_means" not in bundle:
        return []
    prep = model.named_steps["prep"]
    x = np.asarray(prep.transform(row)).ravel()
    effects = estimator.coef_[0] * (x - bundle["feature_means"])
    categorical = list(prep.transformers_[1][2])
    grouped = {}
    for name, effect in zip(prep.get_feature_names_out(), effects):
        column = name.split("__", 1)[1]
        for cat in categorical:  # Merge one-hot columns back into one feature.
            if column.startswith(cat + "_"):
                column = cat
        grouped[column] = grouped.get(column, 0.0) + float(effect)
    ranked = sorted(grouped.items(), key=lambda item: item[1], reverse=True)
    factors = []
    for col, effect in ranked[:top]:
        if effect <= 0:  # Only report factors that increase risk.
            continue
        value = row.iloc[0][col]
        value = value.item() if hasattr(value, "item") else value  # NumPy -> Python.
        factors.append({"feature": col, "value": value, "effect": round(effect, 3)})
    return factors


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/metrics")
def metrics():
    """Saved evaluation results, shown on the page."""
    return app.state.metrics


@app.post("/predict/attrition")
def predict_attrition(data: EmployeeInput):
    bundle = app.state.attrition
    row = to_frame(data, EMPLOYEE_COLUMNS, bundle["features"])
    probability = float(bundle["model"].predict_proba(row)[0, 1])
    threshold = bundle["threshold"]
    if probability >= threshold:
        level = "High"
    elif probability >= threshold / 2:
        level = "Medium"
    else:
        level = "Low"
    return {
        "attrition_probability": round(probability, 4),
        "at_risk": probability >= threshold,
        "risk_level": level,
        "threshold": threshold,
        "top_risk_factors": explain(bundle, row),
    }


@app.post("/predict/salary")
def predict_salary(data: SalaryInput):
    bundle = app.state.salary
    row = to_frame(data, SALARY_COLUMNS, bundle["features"])
    estimate = float(bundle["model"].predict(row)[0])
    mae = bundle["test_mae"]
    result = {
        "predicted_monthly_income": round(estimate, 2),
        "typical_range": [round(max(estimate - mae, 0), 2), round(estimate + mae, 2)],
        "unit": "USD per month",
    }
    if data.current_monthly_income is not None:
        current = data.current_monthly_income
        gap = (current - estimate) / estimate * 100
        if current < estimate - mae:
            position = "Below benchmark"
        elif current > estimate + mae:
            position = "Above benchmark"
        else:
            position = "Within benchmark range"
        result.update({"current_monthly_income": current,
                       "gap_percent": round(gap, 1), "pay_position": position})
    return result
