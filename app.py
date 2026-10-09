"""Streamlit interface for classifying fire, nonfire, and smoke images."""

from pathlib import Path

import streamlit as st
import tensorflow as tf

from data_pipeline import CLASS_NAMES, NUM_CLASSES
from model import build_hff_model
from prediction import (
    MIN_CONFIDENCE_THRESHOLD,
    predict_probabilities,
    preprocess_image,
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "training_output" / "best_model.keras"


def _load_model_file(model_path: Path):
    candidates = [model_path]
    if model_path.suffix.lower() == ".keras":
        candidates.append(model_path.with_suffix(".weights.h5"))

    last_error = None
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            return tf.keras.models.load_model(candidate, compile=False, safe_mode=False)
        except Exception as exc:  # pragma: no cover - fallback to weights path below
            last_error = exc

    if not last_error:
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run train.py first."
        )
    raise last_error


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run train.py first."
        )

    try:
        model = _load_model_file(MODEL_PATH)
    except Exception:
        weights_path = MODEL_PATH.with_suffix(".weights.h5")
        if not weights_path.is_file():
            raise
        model, _ = build_hff_model()
        model.load_weights(weights_path)

    if model.output_shape[-1] != NUM_CLASSES:
        raise ValueError(
            f"The model outputs {model.output_shape[-1]} classes, but the app "
            f"expects {NUM_CLASSES}: {', '.join(CLASS_NAMES)}."
        )
    return model


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
        model = load_model()
        image = preprocess_image(uploaded_file.getvalue())
        probabilities = predict_probabilities(model, image)[0]
    except (OSError, tf.errors.OpError, ValueError, ImportError) as exc:
        st.error(f"Could not classify this image: {exc}")
        return

    ranked_predictions = sorted(
        zip(CLASS_NAMES, probabilities),
        key=lambda prediction: float(prediction[1]),
        reverse=True,
    )
    predicted_class, confidence = ranked_predictions[0]

    if confidence < MIN_CONFIDENCE_THRESHOLD:
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
        "Confidence is the model's score for this image, not a guarantee or "
        "the model's overall accuracy. The review threshold is not calibrated "
        "and is not a safety guarantee."
    )


if __name__ == "__main__":
    main()
