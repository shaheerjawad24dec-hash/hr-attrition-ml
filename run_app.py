"""Start the web app with the .venv libraries and open it in your browser.

Run: python run_app.py   (Windows: py run_app.py)   Stop with Ctrl+C.
"""

import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
URL = "http://127.0.0.1:8000/"

python = ROOT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
if not python.exists():
    sys.exit("No .venv found. Run first: python setup_project.py")
if not (ROOT / "models" / "metrics.json").exists():
    sys.exit("Models not trained. Run: python setup_project.py")

print(f"Starting server at {URL}  (press Ctrl+C to stop)")
threading.Timer(2.5, lambda: webbrowser.open(URL)).start()  # Open after startup.
try:
    subprocess.run([str(python), "-m", "uvicorn", "app.main:app", "--reload"], cwd=ROOT)
except KeyboardInterrupt:
    print("\nServer stopped.")
