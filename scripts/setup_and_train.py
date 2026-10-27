"""Local setup helper for FruitVision AI training.

Run from the repository root with:
    python scripts/setup_and_train.py

Requires Python 3.11.x for the pinned TensorFlow 2.15 training environment.
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
REQUIREMENTS = ROOT / "requirements-train.txt"


def run(*args: str) -> None:
    print("\n>", " ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def check_python_version() -> None:
    if sys.version_info[:2] != (3, 11):
        raise SystemExit(
            f"Unsupported Python version {sys.version_info.major}.{sys.version_info.minor}. "
            "Please install Python 3.11.x for the training environment."
        )


def main() -> None:
    check_python_version()
    print("FruitVision AI - local training setup")

    if not VENV.exists():
        run(sys.executable, "-m", "venv", str(VENV))
    if not PYTHON.exists():
        raise RuntimeError(f"Virtual-environment Python was not created: {PYTHON}")

    run(str(PYTHON), "-m", "pip", "install", "--upgrade", "pip")
    run(str(PYTHON), "-m", "pip", "install", "-r", str(REQUIREMENTS))
    run(str(PYTHON), "scripts/prepare_dataset.py")
    run(str(PYTHON), "train.py")

    print("\nTraining completed successfully.")
    print("Generated files:")
    print("  models/best_model.keras")
    print("  models/fruit_classifier.keras")
    print("  models/class_names.json")
    print("  results/training_history.png")
    print("\nStart the app with:")
    print(f'  "{PYTHON}" -m streamlit run app.py')


if __name__ == "__main__":
    main()
