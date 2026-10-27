"""Simple, transparent setup helper for FruitVision AI.

Run this file yourself from the project folder with:
    python scripts/setup_and_train.py

It creates a virtual environment, installs the pinned dependencies, prepares
three fruit classes, and trains the CNN. No shell commands are downloaded or
executed from the internet by this script.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
PYTHON = VENV / ("Scripts" if sys.platform == "win32" else "bin") / ("python.exe" if sys.platform == "win32" else "python")


def run(*args: str) -> None:
    print("\n>", " ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    print("FruitVision AI - setup and training")
    print("This script only runs commands that are explicitly listed below.")

    if not VENV.exists():
        run(sys.executable, "-m", "venv", str(VENV))

    run(str(PYTHON), "-m", "pip", "install", "-r", "requirements.txt")
    run(str(PYTHON), "scripts/prepare_dataset.py")
    run(str(PYTHON), "train.py")

    print("\nTraining completed.")
    print("Expected model: models/best_model.keras")
    print("To start Streamlit:")
    print(f'  "{PYTHON}" -m streamlit run app.py')


if __name__ == "__main__":
    main()
