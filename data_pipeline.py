
import csv
import os
from collections import Counter
from pathlib import Path

import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

# Fixed class order: index 0/1 maps to the model's two-output softmax head.
CLASS_NAMES = ["fire", "nonfire"]
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


def _get_aerial_split_files(data_dir, split):
    split_dir = _resolve_split_directory(data_dir, split)
    dataset_dir = split_dir.parent
    manifest_path = dataset_dir / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Aerial-only training requires the merge manifest: {manifest_path}. "
            "Run merge_datasets.py to create it."
        )

    split_files = []
    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        for row in csv.DictReader(manifest_file):
            if row.get("split") != split or row.get("is_aerial", "").lower() != "true":
                continue
            class_name = row.get("class")
            if class_name == "smoke":
                continue
            if class_name not in CLASS_NAMES:
                raise ValueError(
                    f"Unknown class {class_name!r} in manifest '{manifest_path}'."
                )
            image_path = dataset_dir / row["dest_path"]
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image listed in manifest does not exist: {image_path}"
                )
            split_files.append((os.fspath(image_path), CLASS_NAMES.index(class_name)))

    if not split_files:
        raise ValueError(
            f"No aerial images found for split '{split}' in '{manifest_path}'."
        )
    return split_files


def _load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image.set_shape((None, None, 3))
    image = tf.image.resize(image, IMG_SIZE)
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label


def _get_split_files(data_dir, split):
    split_dir = _resolve_split_directory(data_dir, split)
    split_files = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"'{split}' class directory not found: {class_dir}"
            )
        split_files.extend(
            (os.fspath(image_path), class_index)
            for image_path in class_dir.rglob("*")
            if image_path.is_file()
            and image_path.suffix.lower() in IMAGE_EXTENSIONS
        )
    if not split_files:
        raise ValueError(f"No fire/nonfire images found for split '{split}'.")
    return split_files


def get_class_weights(data_dir, aerial_only=False):
    """Return inverse-frequency weights based on the training image counts."""
    if aerial_only:
        counts = Counter(
            label for _, label in _get_aerial_split_files(data_dir, "train")
        )
        class_counts = {}
        for class_index, class_name in enumerate(CLASS_NAMES):
            count = counts[class_index]
            if count == 0:
                raise ValueError(
                    f"No aerial training images found for class '{class_name}'."
                )
            class_counts[class_index] = count

        total_images = sum(class_counts.values())
        class_weights = {
            class_index: total_images / (NUM_CLASSES * count)
            for class_index, count in class_counts.items()
        }
        return class_counts, class_weights

    train_files = _get_split_files(data_dir, "train")
    class_counts = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        count = sum(
            label == class_index for _, label in train_files
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


def load_dataset(
    data_dir="merged_dataset",
    split="train",
    batch_size=BATCH_SIZE,
    augment=False,
    shuffle=True,
    aerial_only=False,
):
    """
    split: 'train' | 'val' | 'test'
    Loads images and labels from the split folder or a nested merged_dataset folder.
    Pixels remain in [0, 255] because EfficientNetB2 handles internal normalization.
    """
    if aerial_only:
        split_files = _get_aerial_split_files(data_dir, split)
        paths, labels = zip(*split_files)
        ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
        if shuffle:
            ds = ds.shuffle(len(paths), seed=42, reshuffle_each_iteration=True)
        ds = ds.map(_load_image, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size)
    else:
        split_files = _get_split_files(data_dir, split)
        paths, labels = zip(*split_files)
        ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
        if shuffle:
            ds = ds.shuffle(len(paths), seed=42, reshuffle_each_iteration=True)
        ds = ds.map(_load_image, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size)

    if augment:
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)

    return ds.prefetch(AUTOTUNE)


if __name__ == "__main__":
    train_ds = load_dataset("merged_dataset", "train", augment=True)
    for imgs, labels in train_ds.take(1):
        print("Batch image shape:", imgs.shape)
        print("Batch label shape:", labels.shape)
        print("Pixel value range:", float(tf.reduce_min(imgs)), "to", float(tf.reduce_max(imgs)))