"""Shared image preprocessing and test-time augmentation for predictions."""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from data_pipeline import CLASS_NAMES, IMG_SIZE

MIN_CONFIDENCE_THRESHOLD = 0.60
DEFAULT_FIRE_THRESHOLD = 0.80
FIRE_THRESHOLD_FILENAME = "fire_threshold.json"
FIRE_CLASS_INDEX = CLASS_NAMES.index("fire")


def preprocess_image(image_bytes):
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    return tf.ensure_shape(image, (*IMG_SIZE, 3))


def predict_probabilities(model, images):
    """Average original and horizontally flipped predictions."""
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    if images.shape.rank == 3:
        images = tf.expand_dims(images, axis=0)

    original_predictions = model(images, training=False)
    flipped_predictions = model(
        tf.image.flip_left_right(images),
        training=False,
    )
    return ((original_predictions + flipped_predictions) / 2).numpy()


def calibrate_fire_threshold(labels, probabilities, max_false_positive_rate):
    """Choose the lowest fire-score threshold meeting the validation FPR limit."""
    if not 0 <= max_false_positive_rate <= 1:
        raise ValueError("max_false_positive_rate must be between 0 and 1.")

    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(CLASS_NAMES):
        raise ValueError(
            f"Expected probabilities shaped (samples, {len(CLASS_NAMES)})."
        )
    if labels.ndim != 1 or len(labels) != len(probabilities):
        raise ValueError("Expected one class label per probability row.")
    if np.any(labels < 0) or np.any(labels >= len(CLASS_NAMES)):
        raise ValueError("Validation labels contain an unknown class index.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Validation probabilities must all be finite.")

    negative_scores = probabilities[labels != FIRE_CLASS_INDEX, FIRE_CLASS_INDEX]
    if not len(negative_scores):
        raise ValueError(
            "Cannot calibrate the fire threshold without nonfire validation images."
        )

    allowed_false_positives = int(
        np.floor(max_false_positive_rate * len(negative_scores))
    )
    if allowed_false_positives == len(negative_scores):
        threshold = 0.0
    else:
        descending_scores = np.sort(negative_scores)[::-1]
        threshold = float(
            np.nextafter(
                descending_scores[allowed_false_positives], np.inf
            )
        )

    measured_false_positive_rate = float(
        np.mean(negative_scores >= threshold)
    )
    return threshold, measured_false_positive_rate


def save_fire_threshold(path, threshold, max_false_positive_rate, measured_rate):
    """Persist the validation-calibrated fire decision threshold."""
    path = Path(path)
    path.write_text(
        json.dumps(
            {
                "class_order": CLASS_NAMES,
                "fire_threshold": float(threshold),
                "max_false_positive_rate": float(max_false_positive_rate),
                "validation_false_positive_rate": float(measured_rate),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def load_fire_threshold(directory):
    """Load a model's calibration artifact or return the conservative fallback."""
    path = Path(directory) / FIRE_THRESHOLD_FILENAME
    if not path.is_file():
        return DEFAULT_FIRE_THRESHOLD, False

    try:
        calibration = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read fire calibration file '{path}': {exc}") from exc

    if not isinstance(calibration, dict):
        raise ValueError(f"Fire calibration file '{path}' must contain a JSON object.")
    threshold = calibration.get("fire_threshold")
    if calibration.get("class_order") != CLASS_NAMES:
        raise ValueError(
            f"Fire calibration file '{path}' has a different class order."
        )
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not np.isfinite(threshold)
        or threshold < 0
    ):
        raise ValueError(
            f"Fire calibration file '{path}' contains an invalid threshold."
        )
    return float(threshold), True


def download_fire_threshold(repo_id, directory):
    """Download a hosted calibration artifact when that model repo provides one."""
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=FIRE_THRESHOLD_FILENAME,
            local_dir=directory,
        )
    except EntryNotFoundError:
        return False
    return True
