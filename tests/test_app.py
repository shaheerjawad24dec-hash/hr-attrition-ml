"""Check the saved models, HTTP responses and invalid-input handling."""

import shutil

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.metrics import mean_absolute_error, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

import app.main as backend
from app.main import ROOT, app

HIGH_RISK = {
    "job_role": "Sales Representative", "job_level": 1, "monthly_income": 2200,
    "total_working_years": 2, "years_at_company": 1, "years_with_curr_manager": 0,
    "num_companies_worked": 3, "over_time": "Yes", "business_travel": "Travel_Frequently",
    "distance_from_home": 20, "stock_option_level": 0, "job_satisfaction": 1,
    "environment_satisfaction": 1, "work_life_balance": 1, "job_involvement": 2,
}
LOW_RISK = {
    "job_role": "Research Director", "job_level": 4, "monthly_income": 16000,
    "total_working_years": 20, "years_at_company": 10, "years_with_curr_manager": 7,
    "num_companies_worked": 1, "over_time": "No", "business_travel": "Travel_Rarely",
    "distance_from_home": 3, "stock_option_level": 1, "job_satisfaction": 4,
    "environment_satisfaction": 4, "work_life_balance": 3, "job_involvement": 3,
}
SALARY = {
    "job_role": "Sales Executive", "job_level": 2, "total_working_years": 8,
    "years_at_company": 5, "years_in_current_role": 3, "education": 3,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_page_and_api_schema(client):
    page = client.get("/")
    assert page.status_code == 200
    assert 'id="attrition-form"' in page.text and 'id="salary-form"' in page.text
    paths = client.get("/openapi.json").json()["paths"]
    for route in ["/predict/attrition", "/predict/salary"]:
        assert "post" in paths[route]


def test_high_and_low_risk_employees(client):
    high = client.post("/predict/attrition", json=HIGH_RISK)
    low = client.post("/predict/attrition", json=LOW_RISK)
    assert high.status_code == low.status_code == 200
    high, low = high.json(), low.json()
    assert high["at_risk"] is True and high["risk_level"] == "High"
    assert low["at_risk"] is False and low["risk_level"] == "Low"
    assert 0 <= low["attrition_probability"] < high["attrition_probability"] <= 1
    assert any(f["feature"] == "OverTime" for f in high["top_risk_factors"])


def test_overtime_raises_risk(client):
    no_overtime = {**HIGH_RISK, "over_time": "No"}
    with_ot = client.post("/predict/attrition", json=HIGH_RISK).json()
    without = client.post("/predict/attrition", json=no_overtime).json()
    assert with_ot["attrition_probability"] > without["attrition_probability"]


def test_salary_estimate_and_pay_gap(client):
    response = client.post("/predict/salary", json=SALARY)
    assert response.status_code == 200
    data = response.json()
    assert data["unit"] == "USD per month"
    low, high = data["typical_range"]
    assert low < data["predicted_monthly_income"] < high
    assert "pay_position" not in data  # Not requested without a current salary.

    gap = client.post("/predict/salary", json={**SALARY, "current_monthly_income": 3000}).json()
    assert gap["pay_position"] == "Below benchmark" and gap["gap_percent"] < 0


def test_higher_job_level_pays_more(client):
    junior = client.post("/predict/salary", json={**SALARY, "job_level": 1}).json()
    senior = client.post("/predict/salary", json={**SALARY, "job_level": 4}).json()
    assert senior["predicted_monthly_income"] > junior["predicted_monthly_income"]


@pytest.mark.parametrize("change", [
    {"job_level": 6},                                  # Above allowed range.
    {"job_level": 2.5},                                # Must be an integer.
    {"monthly_income": "NaN"},                         # Non-finite number.
    {"over_time": "Maybe"},                            # Not an allowed option.
    {"job_role": "Astronaut"},                         # Unknown role.
    {"years_at_company": 5, "total_working_years": 2},  # Inconsistent years.
    {"unexpected": 1},                                 # Extra field.
])
def test_invalid_employee_input(client, change):
    response = client.post("/predict/attrition", json={**HIGH_RISK, **change})
    assert response.status_code == 422


def test_missing_and_invalid_salary_input(client):
    assert client.post("/predict/salary", json={}).status_code == 422
    bad = {**SALARY, "years_in_current_role": 9}  # More than years at company.
    assert client.post("/predict/salary", json=bad).status_code == 422


def test_reported_scores_match_saved_models(client):
    """Rebuild the test split and score the saved models independently."""
    metrics = client.get("/metrics").json()
    df = pd.read_csv(ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv",
                     encoding="utf-8-sig")

    bundle = app.state.attrition
    y = (df["Attrition"] == "Yes").astype(int)
    _, X_test, _, y_test = train_test_split(
        df[bundle["features"]], y, test_size=0.2, random_state=42, stratify=y)
    proba = bundle["model"].predict_proba(X_test)[:, 1]
    assert roc_auc_score(y_test, proba) == pytest.approx(metrics["attrition"]["test"]["roc_auc"])
    assert recall_score(y_test, proba >= bundle["threshold"]) == pytest.approx(
        metrics["attrition"]["test"]["recall"])

    salary = app.state.salary
    _, X_test, _, y_test = train_test_split(
        df[salary["features"]], df["MonthlyIncome"], test_size=0.2, random_state=42)
    assert mean_absolute_error(y_test, salary["model"].predict(X_test)) == pytest.approx(
        metrics["salary"]["test"]["mae"])


def test_protected_attributes_not_used(client):
    for name in ["attrition", "salary"]:
        features = getattr(app.state, name)["features"]
        assert not {"Gender", "Age", "MaritalStatus"} & set(features)


@pytest.mark.parametrize("filename", ["attrition_classifier.joblib", "salary_regressor.joblib"])
def test_missing_model_has_helpful_error(filename, monkeypatch, tmp_path):
    for path in (ROOT / "models").iterdir():
        if path.name != filename:
            shutil.copy(path, tmp_path / path.name)
    monkeypatch.setattr(backend, "MODEL_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="python train_models.py"):
        with TestClient(app):
            pass
