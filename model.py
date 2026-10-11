"""Binary fire-detection model following the provided DFF/FGA/HFF workflow."""

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import EfficientNetB2

IMG_SIZE = (224, 224, 3)
NUM_CLASSES = 2


def build_dff_backbone(input_shape=IMG_SIZE, freeze=True):
    """Return the discriminative fire feature extractor from the paper."""
    backbone = EfficientNetB2(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    backbone.trainable = not freeze
    return backbone


def fire_guided_attention(fm, name="fga"):
    """Fuse multiscale 7x7, 5x5, and 3x3 fire-attention feature maps."""
    fm_reduced = layers.Conv2D(
        256,
        1,
        padding="same",
        activation="relu",
        name=f"{name}_reduce",
    )(fm)
    fm_reduced = layers.BatchNormalization(name=f"{name}_bn")(fm_reduced)

    branch7 = layers.Conv2D(128, 7, padding="same", activation="relu", name=f"{name}_conv7")(fm_reduced)
    branch5 = layers.Conv2D(128, 5, padding="same", activation="relu", name=f"{name}_conv5")(fm_reduced)
    branch3 = layers.Conv2D(128, 3, padding="same", activation="relu", name=f"{name}_conv3")(fm_reduced)
    multiscale = layers.Concatenate(name=f"{name}_multiscale")(
        [branch7, branch5, branch3]
    )
    attention = layers.Conv2D(
        384,
        1,
        padding="same",
        activation="sigmoid",
        name=f"{name}_attention",
    )(multiscale)
    attended = layers.Multiply(name=f"{name}_apply_attention")(
        [multiscale, attention]
    )
    return layers.Conv2D(
        256, 1, padding="same", activation="relu", name=f"{name}_out"
    )(attended)


def hierarchical_feature_fusion(fga_features, direct_features, name="hff"):
    """Hierarchically concatenate attended FGA and direct DFF feature maps."""
    fga_features = layers.Conv2D(
        128, 1, padding="same", activation="relu", name=f"{name}_fga_projection"
    )(fga_features)
    direct_features = layers.Conv2D(
        128, 1, padding="same", activation="relu", name=f"{name}_direct_projection"
    )(direct_features)
    level1 = layers.Concatenate(name=f"{name}_level1")(
        [fga_features, direct_features]
    )
    level2 = layers.Conv2D(
        128, 3, padding="same", activation="relu", name=f"{name}_level2"
    )(level1)
    level3 = layers.Concatenate(name=f"{name}_level3")([level1, level2])
    return layers.Conv2D(
        256, 1, padding="same", activation="relu", name=f"{name}_out"
    )(level3)


def build_hff_model(input_shape=IMG_SIZE, num_classes=NUM_CLASSES, freeze_backbone=True):
    """Build DFF with parallel FGA/direct paths, merged by HFF."""
    inputs = layers.Input(shape=input_shape, name="input_tensor")
    backbone = build_dff_backbone(input_shape=input_shape, freeze=freeze_backbone)
    dff_features = backbone(inputs)

    fga_features = fire_guided_attention(dff_features, name="fga")
    direct_features = layers.Conv2D(
        256,
        1,
        padding="same",
        activation="relu",
        name="direct_branch",
    )(dff_features)
    fused = hierarchical_feature_fusion(
        fga_features, direct_features, name="hff"
    )

    fused = layers.BatchNormalization(name="classifier_bn")(fused)
    fused = layers.GlobalAveragePooling2D(name="classifier_pool")(fused)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="predictions",
    )(fused)

    model = Model(inputs, outputs, name="HFF_FireDetection")
    return model, backbone


def build_paper_model(input_shape=IMG_SIZE, num_classes=NUM_CLASSES, freeze_backbone=True):
    """Build the binary DFF, parallel FGA/direct, HFF architecture."""
    return build_hff_model(
        input_shape=input_shape,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
    )


if __name__ == "__main__":
    model, _ = build_paper_model()
    print(f"Total params: {model.count_params():,}")
