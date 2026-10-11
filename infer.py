"""Predict fire or nonfire for one image."""

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download
import tensorflow as tf

from data_pipeline import CLASS_NAMES, NUM_CLASSES
from model import build_hff_model
from prediction import (
    DEFAULT_FIRE_THRESHOLD,
    FIRE_THRESHOLD_FILENAME,
    MIN_CONFIDENCE_THRESHOLD,
    download_fire_threshold,
    load_fire_threshold,
    predict_probabilities,
    preprocess_image,
)

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_DIR / "training_output" / "best_model.keras"
MODEL_REPO_ID = "advitiya21/fire-detection-model"
MODEL_FILENAME = "best_model.keras"


def _load_model(model_path: Path):
    candidates = [model_path]
    if model_path.suffix.lower() == ".keras":
        candidates.append(model_path.with_suffix(".weights.h5"))

    last_error = None
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            return tf.keras.models.load_model(candidate, compile=False, safe_mode=False)
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise FileNotFoundError(
        f"Trained model not found: {model_path}. Run train.py first."
    )


def predict_image(
    image_path: Path,
    model_path: Path | None = None,
    min_confidence: float = MIN_CONFIDENCE_THRESHOLD,
) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    downloaded_model = False
    if model_path is None:
        model_path = DEFAULT_MODEL
        if not model_path.is_file():
            model_path = Path(
                hf_hub_download(
                    repo_id=MODEL_REPO_ID,
                    filename=MODEL_FILENAME,
                    local_dir=DEFAULT_MODEL.parent,
                )
            )
            downloaded_model = True
    if downloaded_model and not (
        DEFAULT_MODEL.parent / FIRE_THRESHOLD_FILENAME
    ).is_file():
        download_fire_threshold(MODEL_REPO_ID, DEFAULT_MODEL.parent)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. Run train.py first."
        )

    try:
        model = _load_model(model_path)
    except Exception:
        weights_path = model_path.with_suffix(".weights.h5")
        if not weights_path.is_file():
            raise
        model, _ = build_hff_model()
        model.load_weights(weights_path)

    if model.output_shape[-1] != NUM_CLASSES:
        raise ValueError(
            f"Model outputs {model.output_shape[-1]} classes, "
            f"but this project expects {NUM_CLASSES}: {CLASS_NAMES}"
        )

    image = preprocess_image(tf.io.read_file(str(image_path)))
    probabilities = predict_probabilities(model, image)[0]
    fire_threshold, calibrated = load_fire_threshold(model_path.parent)

    ranked_predictions = sorted(
        zip(CLASS_NAMES, probabilities),
        key=lambda prediction: float(prediction[1]),
        reverse=True,
    )
    predicted_class, confidence = ranked_predictions[0]
    if predicted_class == "fire" and confidence < fire_threshold:
        print(
            f"Uncertain: top guess is fire ({float(confidence):.1%}), "
            f"below the fire decision threshold ({fire_threshold:.1%})."
        )
    elif confidence < min_confidence:
        print(
            f"Uncertain: top guess is {predicted_class} "
            f"({float(confidence):.1%}; threshold {min_confidence:.1%})."
        )
    else:
        print(f"Prediction: {predicted_class} ({float(confidence):.1%})")
    print("Class scores:")
    for class_name, score in ranked_predictions:
        print(f"  {class_name}: {float(score):.1%}")
    if not calibrated:
        print(
            f"Warning: no calibration file found; using fallback fire "
            f"threshold {DEFAULT_FIRE_THRESHOLD:.0%}."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to the image to classify.")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=(
            "Local Keras model path. By default, uses the local trained model "
            f"or downloads from Hugging Face ({MODEL_REPO_ID}/{MODEL_FILENAME})."
        ),
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=MIN_CONFIDENCE_THRESHOLD,
        help=(
            "Below this score, label the result uncertain "
            f"(default: {MIN_CONFIDENCE_THRESHOLD:.2f})."
        ),
    )
    args = parser.parse_args()
    if not 0 <= args.min_confidence <= 1:
        parser.error("--min-confidence must be between 0 and 1.")
    predict_image(args.image, args.model, args.min_confidence)


if __name__ == "__main__":
    main()
