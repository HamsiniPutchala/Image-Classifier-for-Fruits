"""Prepare a clean 3-class fruit dataset from the real Fruits-360 dataset.

The public Fruits-360 dataset stores varieties such as "Apple Crimson Snow",
"Apple Granny Smith", "Banana Red", etc. as separate folders. This script
correctly combines those varieties into the three project classes:
Apple, Banana and Orange.

It uses the source Training split for project training images and the source
Test split for project validation images, so the same source image is never
copied into both sets. Only a compact subset is copied into this repository's
local ``dataset/`` directory; the full KaggleHub cache remains outside the
repository.

Run from the repository root:
    python scripts/prepare_dataset.py
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


def normalize_name(name: str) -> str:
    """Normalize a dataset folder name for reliable matching."""
    return " ".join(name.lower().replace("_", " ").replace("-", " ").split())


def class_matches(folder_name: str, class_name: str) -> bool:
    """Return True for a fruit class and all of its named varieties."""
    name = normalize_name(folder_name)
    target = class_name.lower()
    return name == target or name.startswith(target + " ")


def find_split_root(root: Path, split_names: set[str]) -> Path:
    """Find the Training/Test directory in the KaggleHub download."""
    candidates = [
        p for p in root.rglob("*")
        if p.is_dir() and normalize_name(p.name) in split_names
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find a Training/Test folder in downloaded dataset at {root}"
        )
    candidates.sort(key=lambda p: len(p.parts))
    return candidates[0]


def find_class_folders(split_root: Path, class_name: str) -> list[Path]:
    """Find every source variety folder belonging to one project class."""
    matches = [
        p for p in split_root.rglob("*")
        if p.is_dir() and class_matches(p.name, class_name)
    ]
    # Keep only the nearest matching folders so nested directories cannot be
    # counted twice.
    matches = [
        p for p in matches
        if not any(parent in matches for parent in p.parents)
    ]
    return sorted(matches, key=lambda p: p.name.lower())


def image_files(folders: list[Path]) -> list[Path]:
    """Collect image files from all matching variety folders."""
    images: list[Path] = []
    for folder in folders:
        images.extend(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    return sorted(images, key=lambda p: str(p).lower())


def copy_images(images: list[Path], destination: Path) -> None:
    """Copy images with deterministic unique filenames."""
    destination.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images, start=1):
        destination_name = f"{index:04d}_{image.stem}{image.suffix.lower()}"
        shutil.copy2(image, destination / destination_name)


def main() -> None:
    print("Downloading the real Fruits-360 dataset through KaggleHub...")
    downloaded = Path(kagglehub.dataset_download("moltean/fruits"))
    print(f"Dataset cache: {downloaded}")

    source_train = find_split_root(downloaded, {"training", "train"})
    source_test = find_split_root(downloaded, {"test", "testing"})
    print(f"Source training split: {source_train}")
    print(f"Source test split:     {source_test}")

    rng = random.Random(SEED)
    if DATASET_ROOT.exists():
        shutil.rmtree(DATASET_ROOT)

    for class_name in CLASSES:
        train_folders = find_class_folders(source_train, class_name)
        test_folders = find_class_folders(source_test, class_name)
        train_images = image_files(train_folders)
        test_images = image_files(test_folders)

        if len(train_images) < TRAIN_PER_CLASS:
            raise RuntimeError(
                f"{class_name}: only {len(train_images)} source training images "
                f"found; need {TRAIN_PER_CLASS}."
            )
        if len(test_images) < VAL_PER_CLASS:
            raise RuntimeError(
                f"{class_name}: only {len(test_images)} source test images "
                f"found; need {VAL_PER_CLASS}."
            )

        rng.shuffle(train_images)
        rng.shuffle(test_images)
        selected_train = train_images[:TRAIN_PER_CLASS]
        selected_val = test_images[:VAL_PER_CLASS]

        copy_images(selected_train, TRAIN_ROOT / class_name)
        copy_images(selected_val, VAL_ROOT / class_name)

        variety_names = [p.name for p in train_folders]
        print(
            f"{class_name}: train={len(selected_train)}, "
            f"validation={len(selected_val)}, "
            f"varieties={len(variety_names)}"
        )
        print("  Included varieties: " + ", ".join(variety_names))

    print("\nDataset preparation complete.")
    print(f"Created: {DATASET_ROOT}")
    print("Classes: " + ", ".join(CLASSES))
    print(
        f"Images: {TRAIN_PER_CLASS * len(CLASSES)} training + "
        f"{VAL_PER_CLASS * len(CLASSES)} validation"
    )
    print("Next step: python train.py")


if __name__ == "__main__":
    main()
