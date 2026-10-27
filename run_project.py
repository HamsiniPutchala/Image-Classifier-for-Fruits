"""
FruitVision AI - Safe Project Launcher

This is a transparent Python launcher for the Fruit Image Classifier project.
It does not download or execute unknown programs or scripts.
It only runs the project's own setup/training script and then starts Streamlit.
"""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SETUP = ROOT / "scripts" / "setup_and_train.py"
APP = ROOT / "app.py"


def run(command):
    print("\n> " + " ".join(map(str, command)))
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        print("\nThe previous step failed. Check the message above.")
        input("Press Enter to close...")
        raise SystemExit(result.returncode)


def main():
    print("=" * 58)
    print(" FruitVision AI - Safe Project Launcher")
    print("=" * 58)
    print("This launcher runs only files that are part of this project.")
    print()

    if not SETUP.exists():
        print(f"Missing setup file: {SETUP}")
        input("Press Enter to close...")
        return

    print("[1/2] Setting up the dataset and training the CNN...")
    run([sys.executable, str(SETUP)])

    if not APP.exists():
        print(f"Missing Streamlit app: {APP}")
        input("Press Enter to close...")
        return

    print("\n[2/2] Starting the Streamlit application...")
    run([sys.executable, "-m", "streamlit", "run", str(APP)])


if __name__ == "__main__":
    main()
