# Model file

Place the trained Keras model here as:

`best_model.keras`

The Streamlit application automatically checks for `models/best_model.keras` and `models/fruit_classifier.keras`.

## Important

The repository currently does not contain a trained `.keras` model. Uploading a model is required for permanent predictions after deployment. The app also provides a **Model Setup** page where a `.keras` file can be uploaded temporarily for testing.

If the model is larger than GitHub's normal file limit, use Git LFS or a model-hosting service and update `app.py` to load it from that location.