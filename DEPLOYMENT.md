# Streamlit Community Cloud Deployment

## Repository

- Repository: `HamsiniPutchala/Image-Classifier-for-Fruits`
- Branch: `main`
- Entrypoint: `app.py`

## Before deployment

1. Make sure `requirements.txt` is present in the repository root.
2. Make sure a trained model is available at `models/best_model.keras` or `models/fruit_classifier.keras`.
3. If the model is not committed to GitHub, use the **Model Setup** page in the app for temporary testing.

## Deploy

1. Open Streamlit Community Cloud: https://share.streamlit.io/
2. Sign in with GitHub.
3. Select **Create app** → **Yup, I have an app**.
4. Select this repository, branch `main`, and file `app.py`.
5. Open **Advanced settings** and select **Python 3.11**. TensorFlow 2.15 is used by this project and should be deployed with a compatible Python version.
6. Click **Deploy**.

## After deployment

The application provides:

- Single-image classification
- Multiple-image batch classification
- Webcam prediction
- Confidence and top-3 predictions
- Probability visualization
- Image-quality analysis
- Prediction history
- Analytics dashboard
- CSV export
- Temporary model upload for testing

## If deployment fails

Open the Streamlit app logs and check the dependency installation error first. Most Community Cloud build failures come from incompatible or missing dependencies. The dependency versions are pinned in `requirements.txt`.

## Model note

A trained `.keras` model is not currently stored in the repository. The app can start without the model, but prediction requires either a model in `models/` or a temporary `.keras` upload through **Model Setup**.
