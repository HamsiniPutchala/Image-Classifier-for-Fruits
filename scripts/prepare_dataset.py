"""Download and prepare a three-class fruit dataset automatically.

Run from the repository root:
    python scripts/prepare_dataset.py

The script downloads the Fruits-360 dataset through KaggleHub, finds the
Apple, Banana and Orange classes, and creates the dataset structure expected
by train.py. It deliberately keeps the downloaded source outside the repo
and copies only a compact, deterministic subset into dataset/.
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
    matches.sort(
        key=lambda p: (
            0 if p.parent.name.lower() in {"training", "train"} else 1,
            len(p.parts),
        )
    )
    return matches[0]


def image_files(source: Path) -> list[Path]:
    return [
        p for p in source.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def copy_images(images: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images, start=1):
        shutil.copy2(image, destination / f"{index:04d}_{image.name}")


def main() -> None:
    print("Downloading Fruits-360 through KaggleHub...")
    downloaded = Path(kagglehub.dataset_download("moltean/fruits"))
    print(f"Dataset cache: {downloaded}")

    rng = random.Random(SEED)
    if DATASET_ROOT.exists():
        shutil.rmtree(DATASET_ROOT)

    for class_name in CLASSES:
        source = locate_source(downloaded, class_name)
        images = image_files(source)
        required = TRAIN_PER_CLASS + VAL_PER_CLASS
        if len(images) < required:
            raise RuntimeError(
                f"{class_name}: found {len(images)} images but need at least {required}."
            )

        rng.shuffle(images)
        train_images = images[:TRAIN_PER_CLASS]
        validation_images = images[TRAIN_PER_CLASS:required]

        copy_images(train_images, TRAIN_ROOT / class_name)
        copy_images(validation_images, VAL_ROOT / class_name)
        print(
            f"{class_name}: train={len(train_images)}, "
            f"validation={len(validation_images)}"
        )

    print("\nDataset preparation complete.")
    print(f"Created: {DATASET_ROOT}")
    print("Classes: " + ", ".join(CLASSES))
    print("Next step: python train.py")


if __name__ == "__main__":
    main()
