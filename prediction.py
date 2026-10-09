"""Shared image preprocessing and test-time augmentation for predictions."""

import tensorflow as tf

from data_pipeline import IMG_SIZE

MIN_CONFIDENCE_THRESHOLD = 0.60


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
