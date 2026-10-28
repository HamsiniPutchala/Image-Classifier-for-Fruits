# 🍎 Fruit Image Classifier using MobileNetV2

An end-to-end deep learning project that classifies fruit images into **Apple, Banana, or Orange** using **MobileNetV2 transfer learning**, TensorFlow/Keras, and an interactive Streamlit application.

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=flat-square&logo=tensorflow)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-red?style=flat-square&logo=streamlit)](https://image-classifier-for-fruits-ufsztygrsajk3tknxhm9mr.streamlit.app/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

## 🌐 Live Demo

**Try the deployed Streamlit application:**

https://image-classifier-for-fruits-ufsztygrsajk3tknxhm9mr.streamlit.app/

> The deployed application recognizes **Apple, Banana, and Orange**. Images that do not pass the application's confidence and safety checks are rejected instead of being recorded as a supported-fruit prediction.

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Supported Classes](#-supported-classes)
- [Model Architecture](#-model-architecture)
- [Data Preprocessing](#-data-preprocessing)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Unsupported Image Handling](#-unsupported-image-handling)
- [Streamlit Application](#-streamlit-application)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Tech Stack](#-tech-stack)
- [References](#-references)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

## 🎯 Overview

This project implements a complete image-classification workflow from dataset preparation and transfer-learning model training to evaluation, interactive prediction, and cloud deployment.

The classifier uses a **MobileNetV2 backbone pretrained on ImageNet**. The backbone is frozen during training and a lightweight classification head is added for the three project classes. The application then applies additional inference checks to reduce the chance of confidently accepting unsupported images.

The project is designed as an **AI/ML internship and academic portfolio project**, demonstrating practical skills in deep learning, transfer learning, model evaluation, Streamlit development, and deployment.

## ✨ Key Features

### 🤖 Deep Learning Classification
- MobileNetV2 transfer learning with ImageNet weights
- Frozen pretrained feature extractor for efficient training
- Global Average Pooling classification head
- Dense layer with 128 units and ReLU activation
- Dropout regularization (0.30)
- Three-class softmax output

### 🖼️ Image Processing
- RGB image conversion
- Resize to **128 × 128 pixels**
- Pixel normalization to the `[0, 1]` range
- Training augmentation using rotation, zoom, shifts, shear, and horizontal flipping

### ⚙️ Training Optimization
- Adam optimizer
- Learning rate: `0.001`
- Categorical cross-entropy loss
- EarlyStopping
- ModelCheckpoint
- ReduceLROnPlateau
- Best model saved as `models/best_model.keras`

### 📊 Evaluation
- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix

> **Evaluation note:** the current `evaluate.py` evaluates the **validation split**, not a separately held-out test set. Therefore, metrics produced by the script should be described as **validation metrics** unless a separate test dataset is created and evaluated.

### 🌐 Streamlit Application
- Single-image upload
- Batch image prediction
- Camera input
- Prediction confidence visualization
- Ranked class probabilities
- Prediction history
- CSV export
- Dashboard with prediction statistics and charts
- Model setup and architecture information
- Presentation-friendly interface

## 🍎 Supported Classes

The current trained model supports exactly three classes:

| Class | Supported |
|---|---|
| 🍎 Apple | Yes |
| 🍌 Banana | Yes |
| 🍊 Orange | Yes |

The model is **not trained as a general fruit detector**. An image of another object or unsupported fruit can otherwise be forced into one of the three softmax classes, so the application includes additional rejection checks.

## 🧠 Model Architecture

The actual training implementation uses **MobileNetV2 transfer learning**, not a custom multi-block CNN.

```text
Input Image (128 × 128 × 3)
        ↓
Pixel values [0, 1]
        ↓
MobileNetV2 input scaling [-1, 1]
        ↓
MobileNetV2 backbone
(ImageNet pretrained, frozen)
        ↓
Global Average Pooling 2D
        ↓
Dense(128, ReLU)
        ↓
Dropout(0.30)
        ↓
Dense(3, Softmax)
        ↓
Apple / Banana / Orange
```

### Model Configuration

| Component | Configuration |
|---|---|
| Backbone | MobileNetV2 |
| Pretrained weights | ImageNet |
| Backbone training | Frozen |
| Input size | 128 × 128 × 3 |
| Pooling | Global Average Pooling 2D |
| Classifier layer | Dense(128, ReLU) |
| Regularization | Dropout(0.30) |
| Output layer | Dense(3, Softmax) |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | Categorical Cross-Entropy |
| Batch size | 32 |
| Maximum epochs | 10 |

## 🔄 Data Preprocessing

Training images are loaded from class-specific directories and processed using TensorFlow/Keras image generators.

### Training augmentation
- Rotation range: ±20°
- Zoom range: 20%
- Horizontal flipping
- Width shift: 20%
- Height shift: 20%
- Shear: 20%
- Nearest-neighbor fill mode

Validation images use resizing and normalization without training augmentation.

## 🏋️ Training

Run:

```bash
python train.py
```

The training script:

1. Loads the training and validation datasets.
2. Builds the MobileNetV2 transfer-learning model.
3. Freezes the ImageNet pretrained backbone.
4. Compiles the model using Adam and categorical cross-entropy.
5. Trains for up to 10 epochs with callbacks.
6. Saves the best checkpoint to `models/best_model.keras`.
7. Saves the final trained model to `models/fruit_classifier.keras`.
8. Saves the class mapping to `models/class_names.json`.
9. Generates the training-history visualization.

## 📊 Evaluation

Run:

```bash
python evaluate.py
```

The script loads `models/best_model.keras` and evaluates it on the repository's **validation directory**.

It generates:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix

The confusion matrix is saved under the `results/` directory.

### Important metric terminology

The current project does **not** use a separately held-out test set in `evaluate.py`. Avoid describing these results as **Test Accuracy**. Use **Validation Accuracy** or **Validation Performance** unless a dedicated test split is added later.

## 🚫 Unsupported Image Handling

A normal three-class softmax model always selects one of its known classes, even when an image is unrelated to those classes. This project therefore adds an inference-time safety layer in the Streamlit application.

The current rejection logic combines:

- **Confidence threshold:** prediction confidence must reach at least 92%.
- **Prediction margin:** the top class must sufficiently exceed the second-best class.
- **Multi-view agreement:** the prediction is checked across the original, mirrored, and center-cropped views.
- **Visual compatibility checks:** simple color-profile checks help reject obviously incompatible inputs.

When an image fails these checks, the application displays:

> 🚫 No supported fruit exists in this image.

The rejected image is not added to prediction history.

### Important limitation

These checks improve practical rejection behavior, but they do not provide mathematically guaranteed open-set recognition. A closed-set classifier can still misclassify some visually similar unsupported images. A dedicated out-of-distribution detector or a separately trained reject/unknown model would be a stronger future enhancement.

## 🌐 Streamlit Application

Start the application locally with:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

### Application sections

- **Home** — project overview and model readiness
- **Classifier** — upload and classify images
- **Camera** — capture an image using the camera input
- **Dashboard** — prediction statistics, charts, history, and CSV export
- **Model Setup** — model configuration and inference pipeline
- **About** — project and technology information

## 📦 Installation

### Prerequisites

- Python 3.11
- pip
- Git
- A virtual environment is recommended

### 1. Clone the repository

```bash
git clone https://github.com/HamsiniPutchala/Image-Classifier-for-Fruits.git
cd Image-Classifier-for-Fruits
```

### 2. Create and activate a virtual environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The deployment requirements currently use TensorFlow CPU 2.15.0, NumPy 1.26.4, Pandas 2.2.3, Pillow 10.4.0, and Streamlit 1.50.0.

## 🚀 Usage

### Train the model

```bash
python train.py
```

### Evaluate the model

```bash
python evaluate.py
```

### Run command-line prediction

```bash
python predict.py <image_path>
```

### Run the web application

```bash
streamlit run app.py
```

For the easiest experience, use the live Streamlit deployment listed at the top of this README.

## 📁 Project Structure

```text
Image-Classifier-for-Fruits/
│
├── dataset/
│   ├── train/
│   │   ├── Apple/
│   │   ├── Banana/
│   │   └── Orange/
│   └── validation/
│       ├── Apple/
│       ├── Banana/
│       └── Orange/
│
├── models/
│   ├── best_model.keras
│   ├── fruit_classifier.keras
│   └── class_names.json
│
├── results/
│   ├── training_history.png
│   └── confusion_matrix.png
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── plotting.py
│   ├── metrics.py
│   └── helper.py
│
├── train.py
├── evaluate.py
├── predict.py
├── app.py
├── requirements.txt
├── runtime.txt
├── setup.py
├── README.md
├── LICENSE
└── .gitignore
```

## 📈 Results

The repository includes the trained model and evaluation pipeline. Run `python evaluate.py` to generate the current validation metrics and confusion matrix from the bundled model and validation dataset.

Because the evaluation script uses the validation split, this README intentionally does not label those measurements as test-set performance.

For an internship or academic report, report the exact metrics produced by the current run and identify the evaluated split clearly.

## ⚠️ Limitations

- The model currently recognizes only Apple, Banana, and Orange.
- The classifier is a closed-set softmax model, so unsupported-image rejection is implemented as an additional inference safety layer rather than a true unknown-class detector.
- The current evaluation pipeline uses a validation split rather than a dedicated held-out test set.
- The visual compatibility checks are heuristic and should not be treated as a replacement for a dedicated out-of-distribution detection method.
- Real-world performance can vary with lighting, background, image quality, camera angle, occlusion, and fruit appearance.

## 🔮 Future Improvements

### Model and robustness
- Fine-tune selected MobileNetV2 layers after the initial transfer-learning stage.
- Compare the current model with other lightweight transfer-learning architectures.
- Add Grad-CAM or another explainability method.
- Improve robustness to lighting, background, occlusion, and image-quality variations.
- Add a dedicated out-of-distribution detection or rejection model.

### Evaluation
- Create a genuinely held-out test dataset.
- Add cross-validation or repeated evaluation where appropriate.
- Track per-class confusion and confidence calibration.
- Add automated regression tests for inference behavior.

### Deployment
- Add a REST API for programmatic inference.
- Containerize the application with Docker.
- Optimize the model for lower-latency or edge deployment.
- Add monitoring for prediction confidence and rejected inputs.

## 🛠️ Tech Stack

### AI / Machine Learning
- **Python 3.11**
- **TensorFlow / Keras**
- **MobileNetV2**
- **NumPy**
- **Scikit-learn**

### Web Application
- **Streamlit**
- **Pillow**
- **Pandas**

### Development
- **Git / GitHub**
- **VS Code**
- **pip**

## 📚 References

- [TensorFlow Documentation](https://www.tensorflow.org/guide)
- [Keras MobileNetV2 Documentation](https://keras.io/api/applications/mobilenet/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Fruits-360 Dataset](https://www.kaggle.com/datasets/moltean/fruits)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make and test your changes.
4. Commit your changes with a meaningful message.
5. Push the branch and open a Pull Request.

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 👨‍💻 Author

**Hamsini Putchala**

- GitHub: https://github.com/HamsiniPutchala
