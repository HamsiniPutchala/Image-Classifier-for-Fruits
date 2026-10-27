"""Download and prepare a three-class fruit dataset automatically.

Run from the repository root:
    python scripts/prepare_dataset.py

The script downloads the Fruits-360 dataset through KaggleHub, finds the
Apple, Banana and Orange classes, and creates the dataset structure expected
by train.py:

    dataset/train/<class>
    dataset/validation/<class>

The original dataset remains in KaggleHub's cache; only the selected images
are copied into this project.
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path

import kagglehub

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "dataset"
TRAIN_ROOT = DATASET_ROOT / "train"
VAL_ROOT = DATASET_ROOT / "validation"
CLASSES = ["Apple", "Banana", "Orange"]
TRAIN_PER_CLASS = 500
VAL_PER_CLASS = 100
SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def locate_source(root: Path, class_name: str) -> Path:
    matches = [p for p in root.rglob(class_name) if p.is_dir()]
    if not matches:
        raise FileNotFoundError(
            f"Could not find class folder '{class_name}' in downloaded dataset at {root}"
        )
    # Prefer a directory whose parent is Training or Test, otherwise first match.
    matches.sort(key=lambda p: (0 if p.parent.name.lower() in {"training", "train"} else 1, len(p.parts)))
    return matches[0]


def copy_selected(source: Path, destination: Path, count: int, rng: random.Random) -> int:
    images = [p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if len(images) < count:
        raise RuntimeError(f"Only {len(images)} images found in {source}; need {count}.")
    rng.shuffle(images)
    destination.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images[:count], start=1):
        shutil.copy2(image, destination / f"{index:04d}_{image.name}")
    return count


def main() -> None:
    print("Downloading Fruits-360 through KaggleHub...")
    downloaded = Path(kagglehub.dataset_download("moltean/fruits"))
    print(f"Dataset cache: {downloaded}")

    rng = random.Random(SEED)
    if DATASET_ROOT.exists():
        shutil.rmtree(DATASET_ROOT)
    for class_name in CLASSES:
        source = locate_source(downloaded, class_name)
        print(f"Preparing {class_name} from {source}")
        copy_selected(source, TRAIN_ROOT / class_name, TRAIN_PER_CLASS, rng)
        # Draw validation images from the same source without overlap.
        images = [p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        rng.shuffle(images)
        val_dir = VAL_ROOT / class_name
        val_dir.mkdir(parents=True, exist_ok=True)
        # Use a different deterministic slice for validation.
        for index, image in enumerate(images[TRAIN_PER_CLASS:TRAIN_PER_CLASS + VAL_PER_CLASS], start=1):
            shutil.copy2(image, val_dir / f"{index:04d}_{image.name}")
        print(f"  train={TRAIN_PER_CLASS}, validation={VAL_PER_CLASS}")

    print("\nDataset preparation complete.")
    print(f"Created: {DATASET_ROOT}")
    print("Classes: " + ", ".join(CLASSES))
    print("Next step: python train.py")


if __name__ == "__main__":
    main()
