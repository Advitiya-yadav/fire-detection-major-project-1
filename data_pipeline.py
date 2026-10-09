
import os
from pathlib import Path

import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

# Fixed class order -- index 0/1/2 maps to the model's softmax head
CLASS_NAMES = ["nonfire", "fire", "smoke"]
NUM_CLASSES = len(CLASS_NAMES)


def _augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    return image, label


def _resolve_split_directory(data_dir, split):
    root = Path(data_dir)
    candidates = (root / split, root / "merged_dataset" / split)
    for directory in candidates:
        if directory.is_dir():
            return directory

    expected = ", ".join(str(directory) for directory in candidates)
    raise FileNotFoundError(
        f"Target split directory not found for '{split}'. Checked: {expected}"
    )


def get_class_weights(data_dir):
    """Return inverse-frequency weights based on the training image counts."""
    train_dir = _resolve_split_directory(data_dir, "train")
    class_counts = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = train_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Training class directory not found: {class_dir}"
            )
        count = sum(
            1
            for image_path in class_dir.rglob("*")
            if image_path.is_file()
            and image_path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if count == 0:
            raise ValueError(f"No training images found for class '{class_name}'.")
        class_counts[class_index] = count

    total_images = sum(class_counts.values())
    class_weights = {
        class_index: total_images / (NUM_CLASSES * count)
        for class_index, count in class_counts.items()
    }
    return class_counts, class_weights


def load_dataset(data_dir="merged_dataset", split="train", batch_size=BATCH_SIZE, augment=False, shuffle=True):
    """
    split: 'train' | 'val' | 'test'
    Loads images and labels from the split folder or a nested merged_dataset folder.
    Pixels remain in [0, 255] because EfficientNetB2 handles internal normalization.
    """
    directory_path = _resolve_split_directory(data_dir, split)

    ds = tf.keras.utils.image_dataset_from_directory(
        os.fspath(directory_path),
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=42,
    )

    if augment:
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)

    return ds.prefetch(AUTOTUNE)


if __name__ == "__main__":
    train_ds = load_dataset("merged_dataset", "train", augment=True)
    for imgs, labels in train_ds.take(1):
        print("Batch image shape:", imgs.shape)
        print("Batch label shape:", labels.shape)
        print("Pixel value range:", float(tf.reduce_min(imgs)), "to", float(tf.reduce_max(imgs)))