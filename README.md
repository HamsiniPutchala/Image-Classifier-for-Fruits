# 🍎 Fruit Image Classifier using MobileNetV2

An end-to-end deep learning project that classifies fruit images into **10 fruit classes** using **MobileNetV2 transfer learning**, TensorFlow/Keras, and an interactive Streamlit application.

## 🌐 Live Demo

https://image-classifier-for-fruits-ufsztygrsajk3tknxhm9mr.streamlit.app/

> The application is being upgraded from the original three-class model to a ten-class model. The training workflow automatically rebuilds the model and class metadata after the ten-class dataset preparation is committed.

## 🎯 Supported Classes

The upgraded classifier recognizes:

1. 🍎 Apple
2. 🍌 Banana
3. 🫐 Blackberry
4. 🍇 Grape
5. 🥝 Kiwi
6. 🥭 Mango
7. 🍊 Orange
8. 🍍 Pineapple
9. 🍓 Strawberry
10. 🍉 Watermelon

## 🧠 Model Architecture

The project continues to use **MobileNetV2 transfer learning** rather than a CNN built from scratch.

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
Dense(10, Softmax)
        ↓
Apple / Banana / Blackberry / Grape / Kiwi
Mango / Orange / Pineapple / Strawberry / Watermelon
```

The output layer is created dynamically from the number of dataset classes, so expanding the dataset from 3 to 10 classes does not require changing the MobileNetV2 backbone.

## 🖼️ Preprocessing

- RGB conversion
- Resize to **128 × 128** pixels
- Pixel normalization to `[0, 1]`
- Training augmentation using rotation, zoom, shifts, shear, and horizontal flipping
- Validation data uses resizing and normalization without training augmentation.

## 🏋️ Training

The training configuration remains:

| Component | Configuration |
|---|---|
| Backbone | MobileNetV2 |
| Pretrained weights | ImageNet |
| Backbone | Frozen |
| Input | 128 × 128 × 3 |
| Pooling | Global Average Pooling 2D |
| Dense layer | 128 units, ReLU |
| Dropout | 0.30 |
| Output | 10 units, Softmax |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | Categorical Cross-Entropy |
| Batch size | 32 |
| Maximum epochs | 10 |

The training script determines the number of output classes from the dataset, saves the best model, and writes the class-to-index mapping to `models/class_names.json`.

## 📊 Evaluation

Run:

```bash
python evaluate.py
```

The evaluation pipeline now works with a variable number of classes and checks that the model output count, validation dataset classes, and class metadata agree before generating accuracy, precision, recall, F1-score, classification report, and confusion matrix.

The current evaluation terminology remains **Validation Performance**, because the project evaluates the validation directory rather than a separately held-out test set.

## 🛡️ Unsupported Image Handling

The Streamlit application still uses inference-time safety gates:

- Confidence threshold: **92%**
- Prediction margin threshold
- Agreement across original, mirrored, and center-cropped views
- Lightweight visual compatibility checks

These checks reduce forced predictions but are not a mathematically guaranteed OOD detector.

## 🌐 Streamlit Features

- Single-image upload
- Batch image prediction
- Camera input
- Top-3 predictions
- Probability comparison
- Image-quality checks
- Prediction history
- CSV export
- Analytics dashboard
- Model Setup page
- About page
- Ten-class fruit display

The app now checks that the bundled model output count matches the class metadata before allowing classification, preventing an old three-class model from being used with ten-class labels.

## 📁 Project Structure

```text
Image-Classifier-for-Fruits/
├── dataset/
│   ├── train/
│   │   ├── Apple/
│   │   ├── Banana/
│   │   ├── Blackberry/
│   │   ├── Grape/
│   │   ├── Kiwi/
│   │   ├── Mango/
│   │   ├── Orange/
│   │   ├── Pineapple/
│   │   ├── Strawberry/
│   │   └── Watermelon/
│   └── validation/
│       └── same ten class folders
├── models/
│   ├── best_model.keras
│   ├── fruit_classifier.keras
│   └── class_names.json
├── results/
├── utils/
├── scripts/
│   └── prepare_dataset.py
├── train.py
├── evaluate.py
├── predict.py
├── app.py
├── requirements.txt
├── runtime.txt
└── README.md
```

## 📦 Dataset Preparation

The preparation script downloads Fruits-360 through KaggleHub and creates balanced project folders for the ten selected classes. It uses source training images for training and source test images for validation, helping prevent train/validation leakage.

Run:

```bash
python scripts/prepare_dataset.py
python train.py
python evaluate.py
streamlit run app.py
```

## 🤖 Automatic Training

The GitHub Actions workflow now verifies all ten classes, trains MobileNetV2, validates the generated class metadata, evaluates the model, uploads the model as an artifact, and publishes the new model and `class_names.json` to the repository.

## ⚠️ Limitations

- Performance depends on the quality and diversity of the ten-class training data.
- The current model is still a closed-set softmax classifier.
- Unsupported-image rejection is a safety layer, not a guaranteed unknown detector.
- Real-world lighting, backgrounds, occlusion, camera quality, and fruit varieties can affect predictions.
- A dedicated held-out test set is still recommended for final performance reporting.

## 🔮 Future Improvements

- Fine-tune selected MobileNetV2 layers
- Add Grad-CAM explainability
- Add a dedicated OOD detector
- Add a separate held-out test set
- Add more fruit categories
- Support multiple fruits in one image using object detection
- Add REST API and Docker deployment
- Add model monitoring and confidence calibration

## 🛠️ Tech Stack

- Python 3.11
- TensorFlow / Keras
- MobileNetV2
- NumPy
- Pillow
- Pandas
- Scikit-learn
- Streamlit
- Git / GitHub

## 📚 References

- TensorFlow Documentation — https://www.tensorflow.org/
- Keras MobileNetV2 Documentation — https://keras.io/api/applications/mobilenet/
- Streamlit Documentation — https://docs.streamlit.io/
- Fruits-360 Dataset — https://www.kaggle.com/datasets/moltean/fruits
- Scikit-learn Documentation — https://scikit-learn.org/

## 👨‍💻 Author

**Hamsini Putchala**

GitHub: https://github.com/HamsiniPutchala
