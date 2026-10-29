"""Prepare a balanced 10-class fruit dataset from Fruits-360.

The project combines matching varieties into these ten fruit classes:
Apple, Banana, Grape, Kiwi, Lime, Mango, Orange, Pineapple, Strawberry,
and Watermelon.
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
CLASSES = [
    "Apple", "Banana", "Grape", "Kiwi", "Lime",
    "Mango", "Orange", "Pineapple", "Strawberry", "Watermelon",
]
TRAIN_PER_CLASS = 500
VAL_PER_CLASS = 100
SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def class_matches(folder_name: str, class_name: str) -> bool:
    name = normalize_name(folder_name)
    target = normalize_name(class_name)
    return name == target or name.startswith(target)


def find_split_root(root: Path, split_names: set[str]) -> Path:
    candidates = [
        p for p in root.rglob("*")
        if p.is_dir() and normalize_name(p.name) in {normalize_name(x) for x in split_names}
    ]
    if not candidates:
        raise FileNotFoundError(f"Could not find a split folder under {root}")
    candidates.sort(
        key=lambda p: sum(
            1 for f in p.rglob("*")
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ),
        reverse=True,
    )
    return candidates[0]


def find_class_folders(split_root: Path, class_name: str) -> list[Path]:
    matches: list[Path] = []
    for folder in split_root.rglob("*"):
        if not folder.is_dir() or not class_matches(folder.name, class_name):
            continue
        if any(f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS for f in folder.iterdir()):
            matches.append(folder)
    return sorted(set(matches), key=lambda p: str(p).lower())


def image_files(folders: list[Path]) -> list[Path]:
    images: list[Path] = []
    for folder in folders:
        images.extend(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    return sorted(set(images), key=lambda p: str(p).lower())


def copy_images(images: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images, start=1):
        shutil.copy2(image, destination / f"{index:04d}_{image.stem}{image.suffix.lower()}")


def main() -> None:
    print("Downloading Fruits-360 through KaggleHub...")
    downloaded = Path(kagglehub.dataset_download("moltean/fruits"))
    source_train = find_split_root(downloaded, {"training", "train"})
    source_test = find_split_root(downloaded, {"test", "testing"})

    rng = random.Random(SEED)
    if DATASET_ROOT.exists():
        shutil.rmtree(DATASET_ROOT)

    for class_name in CLASSES:
        train_images = image_files(find_class_folders(source_train, class_name))
        test_images = image_files(find_class_folders(source_test, class_name))
        print(f"{class_name}: discovered {len(train_images)} training and {len(test_images)} test images")

        if len(train_images) < TRAIN_PER_CLASS:
            raise RuntimeError(f"{class_name}: only {len(train_images)} source training images found; need {TRAIN_PER_CLASS}.")
        if len(test_images) < VAL_PER_CLASS:
            raise RuntimeError(f"{class_name}: only {len(test_images)} source test images found; need {VAL_PER_CLASS}.")

        rng.shuffle(train_images)
        rng.shuffle(test_images)
        copy_images(train_images[:TRAIN_PER_CLASS], TRAIN_ROOT / class_name)
        copy_images(test_images[:VAL_PER_CLASS], VAL_ROOT / class_name)

    print("\n10-class dataset preparation complete.")
    print("Classes: " + ", ".join(CLASSES))
    print(f"Images: {TRAIN_PER_CLASS * len(CLASSES)} training + {VAL_PER_CLASS * len(CLASSES)} validation")


if __name__ == "__main__":
    main()
