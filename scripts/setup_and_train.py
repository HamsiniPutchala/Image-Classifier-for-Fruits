"""Transparent local setup helper for FruitVision AI.

Run from the repository root with:
    python scripts/setup_and_train.py

The script creates a local virtual environment, installs the project's pinned
Python dependencies, prepares the real Fruits-360 subset, and trains the CNN.
It does not download or execute arbitrary shell scripts.

Recommended Python version: 3.11.x (TensorFlow 2.15 compatibility).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
PYTHON = VENV / ("Scripts" if sys.platform == "win32" else "bin") / (
    "python.exe" if sys.platform == "win32" else "python"
)


def run(*args: str) -> None:
    print("\n>", " ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def check_python_version() -> None:
    if sys.version_info[:2] != (3, 11):
        print(
            "Unsupported Python version: "
            f"{sys.version_info.major}.{sys.version_info.minor}"
        )
        print("Please install Python 3.11.x and run this setup again.")
        raise SystemExit(1)


def main() -> None:
    print("FruitVision AI - setup and training")
    print("This script only runs commands explicitly defined by this project.")
    print("Recommended runtime: Python 3.11.x")

    check_python_version()

    if not VENV.exists():
        run(sys.executable, "-m", "venv", str(VENV))

    if not PYTHON.exists():
        raise RuntimeError(f"Virtual-environment Python was not created: {PYTHON}")

    run(str(PYTHON), "-m", "pip", "install", "--upgrade", "pip")
    run(str(PYTHON), "-m", "pip", "install", "-r", "requirements.txt")
    run(str(PYTHON), "scripts/prepare_dataset.py")
    run(str(PYTHON), "train.py")

    print("\nTraining completed successfully.")
    print("Expected models:")
    print("  models/best_model.keras")
    print("  models/fruit_classifier.keras")
    print("\nTo start Streamlit:")
    print(f'  "{PYTHON}" -m streamlit run app.py')


if __name__ == "__main__":
    main()
