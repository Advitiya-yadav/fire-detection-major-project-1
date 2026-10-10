"""Streamlit interface for classifying fire, nonfire, and smoke images."""

from pathlib import Path
from huggingface_hub import hf_hub_download
import streamlit as st
import tensorflow as tf

from data_pipeline import CLASS_NAMES, NUM_CLASSES
from prediction import (
    FIRE_THRESHOLD_FILENAME,
    MIN_CONFIDENCE_THRESHOLD,
    download_fire_threshold,
    load_fire_threshold,
    predict_probabilities,
    preprocess_image,
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_REPO_ID = "advitiya21/fire-detection-model"
MODEL_FILENAME = "best_model.keras"
MODEL_CACHE_DIR = PROJECT_DIR / "training_output"
DEFAULT_FIRE_THRESHOLD = 0.80


@st.cache_resource
def load_model():
    model_path = MODEL_CACHE_DIR / MODEL_FILENAME
    downloaded_model = False
    if not model_path.is_file():
        try:
            model_path = Path(
                hf_hub_download(
                    repo_id=MODEL_REPO_ID,
                    filename=MODEL_FILENAME,
                    local_dir=MODEL_CACHE_DIR,
                )
            )
            downloaded_model = True
        except Exception as exc:
            raise RuntimeError(
                f"Could not download '{MODEL_FILENAME}' from Hugging Face repo "
                f"'{MODEL_REPO_ID}'. Check your internet connection and that the "
                "repository is public or correctly authenticated."
            ) from exc

    if downloaded_model and not (
        MODEL_CACHE_DIR / FIRE_THRESHOLD_FILENAME
    ).is_file():
        download_fire_threshold(MODEL_REPO_ID, MODEL_CACHE_DIR)

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    if model.output_shape[-1] != NUM_CLASSES:
        raise ValueError(
            f"The model outputs {model.output_shape[-1]} classes, but the app "
            f"expects {NUM_CLASSES}: {', '.join(CLASS_NAMES)}."
        )
    fire_threshold, calibrated = load_fire_threshold(MODEL_CACHE_DIR)
    return model, fire_threshold, calibrated


def main():
    st.set_page_config(page_title="Major Project", page_icon="")
    st.title("🔥 UAV FIRE IMAGE CLASSIFIER")
    st.write("Upload an aerial image to check whether it contains fire, smoke, or neither.")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp", "gif", "webp"],
        help="The uploaded image is processed locally by the model.",
    )

    if uploaded_file is None:
        st.info("Upload an image to get started.")
        return

    st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)
    if not st.button("Classify image", type="primary"):
        return

    try:
        model, fire_threshold, calibrated = load_model()
        image = preprocess_image(uploaded_file.getvalue())
        probabilities = predict_probabilities(model, image)[0]
    except (OSError, tf.errors.OpError, ValueError, ImportError) as exc:
        st.error(f"Could not classify this image: {exc}")
        return
    except RuntimeError as exc:
        st.error(str(exc))
        return

    ranked_predictions = sorted(
        zip(CLASS_NAMES, probabilities),
        key=lambda prediction: float(prediction[1]),
        reverse=True,
    )
    predicted_class, confidence = ranked_predictions[0]

    if predicted_class == "fire" and confidence < fire_threshold:
        st.warning(
            f"Uncertain result: the top guess is **fire** "
            f"({float(confidence):.1%}), below the fire decision threshold "
            f"({fire_threshold:.1%})."
        )
    elif confidence < MIN_CONFIDENCE_THRESHOLD:
        st.warning(
            f"Uncertain result: the top guess is **{predicted_class}** "
            f"({float(confidence):.1%}; review threshold "
            f"{MIN_CONFIDENCE_THRESHOLD:.0%})."
        )
    else:
        st.success(
            f"Prediction: **{predicted_class}** "
            f"({float(confidence):.1%} confidence)"
        )

    st.subheader("Class scores")
    for class_name, score in ranked_predictions:
        st.progress(float(score), text=f"{class_name}: {float(score):.1%}")

    st.caption(
        (
            f"Fire threshold calibrated to validation data: {fire_threshold:.1%}. "
            if calibrated
            else f"No calibration file found; using fallback fire threshold "
            f"{DEFAULT_FIRE_THRESHOLD:.0%}. "
        )
        + "Scores are not guarantees. Validation false-positive rates do not "
        "guarantee performance on new images or make this a safety system."
    )


if __name__ == "__main__":
    main()
