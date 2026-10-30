"""One-command setup: create .venv, download all libraries, train models, run tests.

Run from this folder (Windows, macOS or Linux):
    python setup_project.py        (Windows can also use: py setup_project.py)

What it does, step by step:
  1. Checks your Python version (3.12 or newer is needed).
  2. Creates a virtual environment in .venv (an isolated folder of libraries).
  3. Upgrades pip and downloads every library in requirements-dev.txt.
  4. Trains both models (python train_models.py).
  5. Runs the automated tests (python -m pytest -q).
"""

import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
MIN_PYTHON = (3, 12)


def venv_python() -> Path:
    """Path to the Python inside .venv (different folder names on Windows)."""
    if sys.platform == "win32":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def run(step: str, *args) -> None:
    print(f"\n=== {step} ===", flush=True)
    result = subprocess.run([str(a) for a in args], cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"\nFAILED during: {step}. Read the error above, fix it, then rerun "
                 "python setup_project.py")


def main() -> None:
    if sys.version_info < MIN_PYTHON:
        sys.exit(
            f"Python {sys.version.split()[0]} found, but {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ "
            "is required.\nInstall Python 3.12 or 3.13 from https://www.python.org/downloads/ "
            "(Windows: tick 'Add python.exe to PATH'), then run this script again."
        )
    print(f"Using Python {sys.version.split()[0]} at {sys.executable}")

    if not venv_python().exists():
        print("\n=== Creating virtual environment in .venv ===", flush=True)
        venv.create(VENV, with_pip=True)
    else:
        print("\n.venv already exists; reusing it.")

    py = venv_python()
    run("Upgrading pip", py, "-m", "pip", "install", "--upgrade", "pip")
    run("Downloading and installing libraries (this can take a few minutes)",
        py, "-m", "pip", "install", "-r", "requirements-dev.txt")
    run("Training both models", py, "train_models.py")
    run("Running automated tests", py, "-m", "pytest", "-q")

    print("\nSetup complete. Start the app with:")
    print("    python run_app.py" if sys.platform != "win32" else "    py run_app.py")
    print("then open http://127.0.0.1:8000/")


if __name__ == "__main__":
    main()
