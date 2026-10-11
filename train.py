"""Train the HFF fire-detection classifier."""

import argparse
import importlib.util
import math
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from data_pipeline import (
    BATCH_SIZE,
    CLASS_NAMES,
    IMG_SIZE,
    NUM_CLASSES,
    get_class_weights,
    load_dataset,
)
from prediction import (
    FIRE_CLASS_INDEX,
    calibrate_fire_threshold,
    predict_probabilities,
    save_fire_threshold,
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [PROJECT_DIR / "model.py", PROJECT_DIR / "model (1).py"]


def _resolve_model_file() -> Path:
    for candidate in MODEL_CANDIDATES:
        if candidate.is_file():
            return candidate

    searched = ", ".join(str(path) for path in MODEL_CANDIDATES)
    raise FileNotFoundError(
        "Could not locate the model definition file. "
        f"Looked for: {searched}. Expected the main model builder in 'model.py'."
    )


MODEL_FILE = _resolve_model_file()


def _load_model_builder():
    spec = importlib.util.spec_from_file_location("fire_detection_model", MODEL_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load model definition from '{MODEL_FILE}'.")

    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    return model_module.build_hff_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_DIR / "merged_dataset",
        help="Dataset root containing train, val, and test splits.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "training_output",
        help="Directory for the best model checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=5,
        help="Additional low-learning-rate epochs for the pretrained backbone.",
    )
    parser.add_argument(
        "--fine-tune-layers",
        type=int,
        default=30,
        help="Number of final backbone layers to unfreeze during fine-tuning.",
    )
    parser.add_argument(
        "--max-fire-fpr",
        type=float,
        default=0.01,
        help=(
            "Maximum fire false-positive rate on nonfire validation "
            "images when calibrating the fire decision threshold."
        ),
    )
    parser.add_argument(
        "--aerial-only",
        action="store_true",
        help=(
            "Train, calibrate, and evaluate using only aerial images marked "
            "in merged_dataset/manifest.csv."
        ),
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("--epochs, --batch-size, and --patience must be positive.")
    if args.fine_tune_epochs < 0 or args.fine_tune_layers < 1:
        parser.error(
            "--fine-tune-epochs must be non-negative and "
            "--fine-tune-layers positive."
        )
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive.")
    if not math.isfinite(args.max_fire_fpr) or not 0 <= args.max_fire_fpr <= 1:
        parser.error("--max-fire-fpr must be between 0 and 1.")

    tf.keras.utils.set_random_seed(42)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    class_counts, class_weights = get_class_weights(
        args.data_dir, aerial_only=args.aerial_only
    )
    print("Training images by class:")
    for class_index, class_name in enumerate(CLASS_NAMES):
        print(
            f"  {class_name}: {class_counts[class_index]} "
            f"(weight {class_weights[class_index]:.3f})"
        )

    train_ds = load_dataset(
        args.data_dir,
        "train",
        batch_size=args.batch_size,
        augment=True,
        shuffle=True,
        aerial_only=args.aerial_only,
    )
    val_ds = load_dataset(
        args.data_dir,
        "val",
        batch_size=args.batch_size,
        augment=False,
        shuffle=False,
        aerial_only=args.aerial_only,
    )
    test_ds = load_dataset(
        args.data_dir,
        "test",
        batch_size=args.batch_size,
        augment=False,
        shuffle=False,
        aerial_only=args.aerial_only,
    )

    build_hff_model = _load_model_builder()
    model, backbone = build_hff_model(
        input_shape=(*IMG_SIZE, 3),
        num_classes=NUM_CLASSES,
        freeze_backbone=True,
    )

    checkpoint_path = args.output_dir / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    if args.fine_tune_epochs:
        if args.fine_tune_layers > len(backbone.layers):
            parser.error(
                f"--fine-tune-layers cannot exceed the {len(backbone.layers)} "
                "layers in the backbone."
            )

        backbone.trainable = True
        for layer in backbone.layers[:-args.fine_tune_layers]:
            layer.trainable = False
        for layer in backbone.layers[-args.fine_tune_layers:]:
            layer.trainable = not isinstance(
                layer, tf.keras.layers.BatchNormalization
            )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=args.learning_rate * 0.1
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=args.epochs,
            epochs=args.epochs + args.fine_tune_epochs,
            callbacks=callbacks,
            class_weight=class_weights,
        )

    model = tf.keras.models.load_model(checkpoint_path, compile=False)
    weights_path = args.output_dir / "best_model.weights.h5"
    model.save_weights(weights_path)
    validation_labels = []
    validation_probabilities = []
    for images, labels in val_ds:
        validation_probabilities.extend(predict_probabilities(model, images))
        validation_labels.extend(tf.argmax(labels, axis=1).numpy().tolist())

    fire_threshold, validation_fire_fpr = calibrate_fire_threshold(
        validation_labels,
        validation_probabilities,
        args.max_fire_fpr,
    )
    save_fire_threshold(
        args.output_dir / "fire_threshold.json",
        fire_threshold,
        args.max_fire_fpr,
        validation_fire_fpr,
    )
    print(
        "\nFire decision threshold calibrated on validation nonfire "
        f"images: {fire_threshold:.4f} "
        f"(false-positive rate {validation_fire_fpr:.2%}; "
        f"target <= {args.max_fire_fpr:.2%})"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=args.learning_rate * 0.1
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("\nEvaluating best checkpoint on the held-out test set (single-view):")
    test_metrics = model.evaluate(test_ds, return_dict=True)
    for name, value in test_metrics.items():
        print(f"{name}: {value:.4f}")

    true_labels = []
    predicted_labels = []
    test_fire_scores = []
    for images, labels in test_ds:
        probabilities = predict_probabilities(model, images)
        true_labels.extend(tf.argmax(labels, axis=1).numpy().tolist())
        predicted_labels.extend(probabilities.argmax(axis=1).tolist())
        test_fire_scores.extend(probabilities[:, FIRE_CLASS_INDEX].tolist())

    print("\nPer-class metrics (flip-averaged predictions):")
    print(
        classification_report(
            true_labels,
            predicted_labels,
            labels=range(NUM_CLASSES),
            target_names=CLASS_NAMES,
            zero_division=0,
            digits=4,
        )
    )
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(true_labels, predicted_labels, labels=range(NUM_CLASSES)))
    test_negative_scores = [
        score
        for label, score in zip(true_labels, test_fire_scores)
        if label != FIRE_CLASS_INDEX
    ]
    if test_negative_scores:
        test_false_positive_rate = float(
            np.mean(np.asarray(test_negative_scores) >= fire_threshold)
        )
        print(
            "Held-out test fire false-positive rate at calibrated threshold: "
            f"{test_false_positive_rate:.2%}"
        )
    print(f"Best model saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
