# Create your submission repository

1. Extract the ZIP and open the `hr-attrition-ml` folder in VS Code.
2. Run `py setup_project.py` (Windows) or `python3 setup_project.py`, then start
   the app and try both forms.
3. On GitHub, choose **New repository** and name it, for example,
   `hr-attrition-ml`. Set the visibility your instructor asks for.
4. Choose **uploading an existing file** (or **Add file > Upload files**).
5. Upload the **contents** of the folder: `app`, `frontend`, `models`, `data`,
   `docs`, `tests`, `.vscode` and the root files. Do **not** upload `.venv`,
   `__pycache__` or `.pytest_cache`.
6. Commit, check that the README shows on the repository page, and copy the URL.

Or with Git from the project folder (replace `YOUR_USERNAME`):

```bash
git init
git add .
git commit -m "Add HR attrition and salary ML project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/hr-attrition-ml.git
git push -u origin main
```

`.gitignore` already excludes `.venv` and caches. If you change features or
parameters, rerun `python train_models.py` and `python -m pytest -q`, then commit
the source, the `.joblib` files and `metrics.json` together.
