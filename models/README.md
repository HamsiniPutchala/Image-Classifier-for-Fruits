# FruitVision AI model

The training pipeline creates the trained CNN as:

`models/best_model.keras`

It uses the fixed three-class order: `Apple`, `Banana`, `Orange`.

## Automatic setup

From the repository root on Windows, double-click:

`scripts/setup_and_train.bat`

or run:

```bash
python scripts/prepare_dataset.py
python train.py
```

The preparation script downloads the public Fruits-360 dataset through KaggleHub and copies only the Apple, Banana and Orange images needed by this project. The full source dataset is not stored in this repository.

The trained `.keras` file is generated locally. If you want to deploy it directly from GitHub, keep the model below GitHub's normal file-size limits; otherwise use Git LFS or external model hosting.
