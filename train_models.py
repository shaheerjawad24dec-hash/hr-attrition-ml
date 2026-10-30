"""Train, compare, evaluate and save both HR models. Run: python train_models.py

Business problem (see docs/BUSINESS_PROBLEM.md):
  1. Classification - which employees are at risk of leaving (Attrition = Yes)?
  2. Regression     - what monthly income does a role/experience profile earn?
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    precision_score, r2_score, recall_score, roc_auc_score, root_mean_squared_error,
)
from sklearn.model_selection import (
    KFold, StratifiedKFold, cross_val_predict, cross_validate, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
MODEL_DIR = ROOT / "models"
SEED = 42          # Same split and same results every run.
TEST_SIZE = 0.2    # 20% of rows are kept aside and never used for training.
TOLERANCE = 0.01   # Prefer a simpler model if it is within 1% of the best score.
TARGET_RECALL = 0.70  # Business rule: flag enough employees to catch 70% of leavers.

# Gender, Age and MaritalStatus are deliberately NOT used: HR decisions must
# not be driven by protected personal characteristics.
ATTRITION_NUMERIC = [
    "JobLevel", "MonthlyIncome", "TotalWorkingYears", "YearsAtCompany",
    "YearsWithCurrManager", "StockOptionLevel", "JobSatisfaction",
    "EnvironmentSatisfaction", "WorkLifeBalance", "JobInvolvement",
    "DistanceFromHome", "NumCompaniesWorked",
]
ATTRITION_CATEGORICAL = ["OverTime", "BusinessTravel", "JobRole"]

SALARY_NUMERIC = [
    "JobLevel", "TotalWorkingYears", "YearsAtCompany", "YearsInCurrentRole", "Education",
]
SALARY_CATEGORICAL = ["JobRole"]


def load_data():
    """Read the Kaggle CSV and check it is the expected file."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Download it from Kaggle "
            "(see README) and place the CSV in the data/ folder."
        )
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")  # utf-8-sig removes the BOM.
    needed = set(ATTRITION_NUMERIC + ATTRITION_CATEGORICAL + SALARY_NUMERIC
                 + SALARY_CATEGORICAL + ["Attrition"])
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    return df


def preprocessor(numeric, categorical):
    """Scale numeric columns and one-hot encode text columns."""
    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])


def choose(candidates, scores, higher_is_better):
    """Return the first (simplest) candidate within TOLERANCE of the best score."""
    best = max(scores.values()) if higher_is_better else min(scores.values())
    for name in candidates:
        gap = (best - scores[name]) if higher_is_better else (scores[name] - best)
        if gap <= abs(best) * TOLERANCE:
            return name
    return None  # Not reached: the best model is always within tolerance.


def train_attrition(df):
    features = ATTRITION_NUMERIC + ATTRITION_CATEGORICAL
    X = df[features]
    y = (df["Attrition"] == "Yes").astype(int)  # 1 = employee left.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    # Candidates listed from simplest / most explainable to most complex.
    # class_weight="balanced" compensates for only ~16% of employees leaving.
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced",
            random_state=SEED, n_jobs=-1),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=SEED),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = {}
    for name, estimator in candidates.items():
        pipe = Pipeline([("prep", preprocessor(ATTRITION_NUMERIC, ATTRITION_CATEGORICAL)),
                         ("model", estimator)])
        result = cross_validate(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        cv_scores[name] = float(result["test_score"].mean())
    chosen = choose(list(candidates), cv_scores, higher_is_better=True)

    model = Pipeline([("prep", preprocessor(ATTRITION_NUMERIC, ATTRITION_CATEGORICAL)),
                      ("model", candidates[chosen])])

    # Pick the decision threshold from out-of-fold TRAINING predictions only.
    # A retention conversation is cheap; losing an employee is expensive. So we
    # use the HIGHEST threshold that still catches TARGET_RECALL of leavers.
    oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    ok = [t for t in thresholds if recall_score(y_train, oof >= t) >= TARGET_RECALL]
    threshold = float(max(ok))

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    # Average of each transformed column in the training data. The API uses it
    # to explain a prediction: coefficient x (employee value - average value).
    transformed = model.named_steps["prep"].transform(X_train)
    feature_means = np.asarray(transformed.mean(axis=0)).ravel()
    joblib.dump({"model": model, "threshold": threshold, "features": features,
                 "feature_means": feature_means},
                MODEL_DIR / "attrition_classifier.joblib")
    return {
        "task": "classification",
        "question": "Will this employee leave? (Attrition = Yes)",
        "chosen_model": chosen,
        "selection_rule": "highest 5-fold CV ROC-AUC; simplest model within 1% wins",
        "cv_roc_auc": cv_scores,
        "decision_threshold": threshold,
        "threshold_rule": f"highest threshold reaching {TARGET_RECALL:.0%} recall "
                          "on out-of-fold training predictions",
        "numeric_features": ATTRITION_NUMERIC,
        "categorical_features": ATTRITION_CATEGORICAL,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_positive_rate": float(y_test.mean()),
        "test": {
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred)),
            "recall": float(recall_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred)),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "baseline_accuracy_always_stay": float(1 - y_test.mean()),
        },
        "top_drivers": top_drivers(model),
    }


def top_drivers(model, n=6):
    """For a linear model, list the features that most raise or lower risk."""
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "coef_"):
        return []
    names = model.named_steps["prep"].get_feature_names_out()
    coefs = estimator.coef_[0]
    order = np.argsort(-np.abs(coefs))[:n]
    return [{"feature": clean_name(names[i]), "coefficient": round(float(coefs[i]), 3)}
            for i in order]


def clean_name(raw):
    """'cat__OverTime_Yes' -> 'OverTime = Yes', 'num__JobLevel' -> 'JobLevel'."""
    kind, name = raw.split("__", 1)
    if kind == "cat":
        for col in ATTRITION_CATEGORICAL + SALARY_CATEGORICAL:
            if name.startswith(col + "_"):
                return f"{col} = {name[len(col) + 1:]}"
    return name


def train_salary(df):
    features = SALARY_NUMERIC + SALARY_CATEGORICAL
    X = df[features]
    y = df["MonthlyIncome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )
    candidates = {
        "LinearRegression": LinearRegression(),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=SEED),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_mae, cv_r2 = {}, {}
    for name, estimator in candidates.items():
        pipe = Pipeline([("prep", preprocessor(SALARY_NUMERIC, SALARY_CATEGORICAL)),
                         ("model", estimator)])
        result = cross_validate(pipe, X_train, y_train, cv=cv,
                                scoring=["neg_mean_absolute_error", "r2"])
        cv_mae[name] = float(-result["test_neg_mean_absolute_error"].mean())
        cv_r2[name] = float(result["test_r2"].mean())
    chosen = choose(list(candidates), cv_mae, higher_is_better=False)

    model = Pipeline([("prep", preprocessor(SALARY_NUMERIC, SALARY_CATEGORICAL)),
                      ("model", candidates[chosen])])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, pred))

    joblib.dump({"model": model, "features": features, "test_mae": mae},
                MODEL_DIR / "salary_regressor.joblib")
    return {
        "task": "regression",
        "question": "What monthly income (USD) fits this role and experience?",
        "chosen_model": chosen,
        "selection_rule": "lowest 5-fold CV MAE; simplest model within 1% wins",
        "cv_mae": cv_mae,
        "cv_r2": cv_r2,
        "numeric_features": SALARY_NUMERIC,
        "categorical_features": SALARY_CATEGORICAL,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test": {
            "r2": float(r2_score(y_test, pred)),
            "mae": mae,
            "rmse": float(root_mean_squared_error(y_test, pred)),
            "baseline_mae_predict_mean": float(
                mean_absolute_error(y_test, np.full(len(y_test), y_train.mean()))),
        },
    }


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    df = load_data()
    metrics = {
        "dataset": "IBM HR Analytics Employee Attrition & Performance (Kaggle)",
        "dataset_url": "https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset",
        "rows": int(len(df)),
        "random_state": SEED,
        "test_fraction": TEST_SIZE,
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "joblib_version": joblib.__version__,
        # Valid choices for the frontend drop-downs come from the data itself.
        "choices": {col: sorted(df[col].unique().tolist())
                    for col in ["OverTime", "BusinessTravel", "JobRole"]},
        "attrition": train_attrition(df),
        "salary": train_salary(df),
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n",
                                            encoding="utf-8")
    a, s = metrics["attrition"], metrics["salary"]
    print(f"Attrition model: {a['chosen_model']}  threshold={a['decision_threshold']:.2f}")
    print(f"  test ROC-AUC={a['test']['roc_auc']:.3f}  recall={a['test']['recall']:.2%}  "
          f"precision={a['test']['precision']:.2%}  accuracy={a['test']['accuracy']:.2%}")
    print(f"Salary model: {s['chosen_model']}")
    print(f"  test R2={s['test']['r2']:.3f}  MAE=${s['test']['mae']:,.0f}/month")
    print("Saved both models and metrics.json in models/.")


if __name__ == "__main__":
    main()
