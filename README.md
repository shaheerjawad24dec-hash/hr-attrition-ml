# TalentKeep: Employee Attrition & Salary Benchmark (Full-Stack ML)

A business problem solved with machine learning on a Kaggle dataset: one
**classification** model, one **regression** model, a Python **FastAPI** backend
and an **HTML/CSS/JavaScript** frontend with a form for each model.

**Dataset (Kaggle):** IBM HR Analytics Employee Attrition & Performance
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
(already included as `data/WA_Fn-UseC_-HR-Employee-Attrition.csv`).

## Business problem

A 1,470-person company loses about **16%** of its employees. HR wants to:

1. **Find employees at risk of leaving** early enough to act (classification:
   `Attrition` Yes/No).
2. **Benchmark fair monthly pay** for a role and experience level, so offers
   and pay reviews are consistent (regression: `MonthlyIncome`). Underpaid
   staff are a retention risk, which links the two models.

Full write-up: [`docs/BUSINESS_PROBLEM.md`](docs/BUSINESS_PROBLEM.md).

## Quick start (installs every library for you)

You need **Python 3.12 or 3.13** ([download](https://www.python.org/downloads/);
on Windows tick *"Add python.exe to PATH"*). Open this folder in VS Code, open a
terminal (**Terminal > New Terminal**) and run:

| | Windows | macOS / Linux |
| --- | --- | --- |
| 1. Install libraries, train, test | `py setup_project.py` (or double-click `setup_windows.bat`) | `python3 setup_project.py` |
| 2. Start the app | `py run_app.py` (or double-click `run_windows.bat`) | `python3 run_app.py` |

Then open **http://127.0.0.1:8000/**. The API docs are at **http://127.0.0.1:8000/docs**.
Stop the server with `Ctrl+C`.

`setup_project.py` creates a `.venv` folder, downloads every library in
`requirements-dev.txt`, trains both models and runs the tests. In VS Code, press
`Ctrl+Shift+P`, choose **Python: Select Interpreter** and pick `.venv`.

### Manual install (same result)

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python train_models.py
python -m uvicorn app.main:app --reload
```

If PowerShell blocks `activate`, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use
`setup_windows.bat`, which does not need activation.

## Models and measured results

Both tasks use an 80/20 train/test split (`random_state=42`, stratified for
classification). Three algorithms were compared with **5-fold cross-validation
on the training set only**. The simplest one within 1% of the best score was
kept and then scored **once** on the 294 untouched test employees.

| Task | Chosen model | CV comparison | Held-out test result |
| --- | --- | --- | --- |
| Attrition (classification) | StandardScaler + OneHot + **LogisticRegression** (balanced) | ROC-AUC: Logistic 0.823, Random Forest 0.810, Gradient Boosting 0.825 | ROC-AUC **0.769**, recall **61.7% (29/47 leavers)**, precision 38.2%, accuracy 77.9% |
| Salary (regression) | OneHot + **GradientBoostingRegressor** | MAE: Linear $859, Gradient Boosting $765, Random Forest $765 | R² **0.945**, MAE **$802/month**, RMSE $1,096 |

- **Decision threshold = 0.55**, chosen on training data to catch about 70% of
  leavers. On the test set it caught 29 of 47 leavers while flagging 76 of 294
  employees. Flagged employees leave at 38%, versus a 16% base rate (2.4x).
- Always predicting "stays" would give 84% accuracy and catch **zero** leavers.
  That is why accuracy is not the main metric here.
- Always predicting the average salary would give MAE $3,387, so the regressor
  cuts the error by 76%.
- **Gender, Age and MaritalStatus are not used** by either model (fairness).

All numbers are saved in `models/metrics.json` and shown on the web page.

## Endpoints

| Method and path | Input / purpose |
| --- | --- |
| `POST /predict/attrition` | 15 employee fields. Returns probability, risk level and the top factors raising risk |
| `POST /predict/salary` | role, level, experience, education, optional current salary. Returns estimate, range and pay gap |
| `GET /metrics` | Saved evaluation results |
| `GET /` | The HTML frontend |

Invalid inputs return HTTP 422. See [`docs/API.md`](docs/API.md) for examples.

## Project structure

```
train_models.py        Load the Kaggle CSV, compare models, evaluate, save with joblib
app/main.py            FastAPI: load models once, validate input, predict, explain
frontend/index.html    Both forms, CSS and JavaScript fetch calls
models/                Saved models (.joblib) and metrics.json
data/                  Kaggle CSV
tests/test_app.py      17 automated checks
setup_project.py       Creates .venv and installs every library, trains, tests
run_app.py             Starts the server and opens the browser
setup_windows.bat / run_windows.bat / setup_mac_linux.sh   Double-click helpers
.vscode/               Run/debug configs ("Run FastAPI app", "Train models") and pytest setup
docs/                  Business problem, API examples, Kaggle link, GitHub steps, verification
STUDY_GUIDE.md         Line-by-line explanation, metrics, viva questions, practice tasks
```

## Tests

```bash
python -m pytest -q
```

Recorded result: **17 passed** (see `docs/verification.json`). A harmless
Starlette/httpx deprecation warning may appear.

## Learning note

If your brief requires your own code, use this project as a worked reference.
Study `STUDY_GUIDE.md` and be ready to explain every line. Only load `.joblib`
files you trust. Rerun `python train_models.py` if your library versions differ.
