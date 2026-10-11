"""
Paper-aligned HFF fire detection model.

This keeps the project architecture aligned with the reference paper:
  1. DFF: EfficientNetB2 backbone (ImageNet pretrained)
  2. FGA: fire-guided attention with multiscale context
  3. HFF: hierarchical feature fusion combining localized and contextual cues
"""

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import EfficientNetB2

IMG_SIZE = (224, 224, 3)
NUM_CLASSES = 3


def build_dff_backbone(input_shape=IMG_SIZE, freeze=True):
    """Return the discriminative fire feature extractor from the paper."""
    backbone = EfficientNetB2(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    backbone.trainable = not freeze
    return backbone


def _global_descriptor(x, filters=64, kernel_size=1, name=None):
    x = layers.Conv2D(filters, kernel_size, padding="same", activation="relu", name=name)(x)
    x = layers.GlobalAveragePooling2D()(x)
    return layers.BatchNormalization()(x)


def fire_guided_attention(fm, name="fga"):
    """
    Fire-Guided Attention (FGA) module.

    This follows the paper's design closely:
    - 1x1 bottleneck reduction to prevent parameter explosion
    - multiscale receptive fields (7x7, 5x5, 3x3)
    - per-branch descriptors using global pooling and batchnorm
    - context fusion with the original feature map
    """
    fm_reduced = layers.Conv2D(
        256,
        1,
        padding="same",
        activation="relu",
        name=f"{name}_reduce",
    )(fm)
    fm_reduced = layers.BatchNormalization(name=f"{name}_bn")(fm_reduced)
    fm_reduced = layers.Dropout(0.25, name=f"{name}_drop")(fm_reduced)

    branch7 = layers.Conv2D(128, 7, padding="same", activation="relu", name=f"{name}_conv7")(fm_reduced)
    branch5 = layers.Conv2D(128, 5, padding="same", activation="relu", name=f"{name}_conv5")(fm_reduced)
    branch3 = layers.Conv2D(128, 3, padding="same", activation="relu", name=f"{name}_conv3")(fm_reduced)

    g1 = _global_descriptor(branch7, filters=64, kernel_size=1, name=f"{name}_g1")
    g2 = _global_descriptor(branch5, filters=64, kernel_size=1, name=f"{name}_g2")
    g3 = _global_descriptor(branch3, filters=64, kernel_size=1, name=f"{name}_g3")
    multiscale = layers.Concatenate(name=f"{name}_multiscale")([g1, g2, g3])

    local_context = layers.GlobalAveragePooling2D()(
        layers.Conv2D(64, 1, padding="same", activation="relu", name=f"{name}_local_ctx")(fm)
    )
    global_context = layers.GlobalAveragePooling2D(name=f"{name}_global_ctx")(fm)

    return layers.Concatenate(name=f"{name}_out")([multiscale, local_context, global_context])


def hierarchical_feature_fusion(fm, name="hff"):
    """
    Hierarchical Feature Fusion (HFF) module.

    The paper's HFF mechanism combines multiple multiscale and hierarchical feature maps
    while preserving local structure and high-level semantic cues.
    """
    fm_reduced = layers.Conv2D(
        256,
        1,
        padding="same",
        activation="relu",
        name=f"{name}_reduce",
    )(fm)

    f7 = layers.Conv2D(128, 7, padding="same", activation="relu", name=f"{name}_f7")(fm_reduced)
    f5 = layers.Conv2D(96, 5, padding="same", activation="relu", name=f"{name}_f5")(f7)
    f3 = layers.Conv2D(96, 3, padding="same", activation="relu", name=f"{name}_f3")(f7)

    d1 = _global_descriptor(f7, filters=64, kernel_size=1, name=f"{name}_d1")
    d2 = _global_descriptor(f5, filters=64, kernel_size=1, name=f"{name}_d2")
    d3 = _global_descriptor(f3, filters=64, kernel_size=1, name=f"{name}_d3")

    # Hierarchical concatenation in the same spirit as the paper: progressively combine
    # higher-level semantic branches with spatially aware feature maps.
    h1 = layers.Concatenate(name=f"{name}_h1")([d1, d2])
    h2 = layers.Concatenate(name=f"{name}_h2")([h1, d3])

    context = layers.GlobalAveragePooling2D()(
        layers.Conv2D(64, 1, padding="same", activation="relu", name=f"{name}_context")(fm)
    )
    return layers.Concatenate(name=f"{name}_out")([h2, context])


def build_hff_model(input_shape=IMG_SIZE, num_classes=NUM_CLASSES, freeze_backbone=True):
    """
    DFF -> FGA -> HFF pipeline used by the paper.

    1) DFF: EfficientNetB2 feature extractor
    2) FGA: fire-guided attention
    3) HFF: hierarchical feature fusion
    """
    inputs = layers.Input(shape=input_shape, name="input_tensor")
    backbone = build_dff_backbone(input_shape=input_shape, freeze=freeze_backbone)
    fm = backbone(inputs)

    fga_out = fire_guided_attention(fm, name="fga")
    hff_out = hierarchical_feature_fusion(fm, name="hff")

    fused = layers.Concatenate(name="dff_fga_hff_fusion")([fga_out, hff_out])
    fused = layers.BatchNormalization()(fused)
    fused = layers.Dense(256, activation="relu")(fused)
    fused = layers.Dropout(0.4)(fused)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="predictions",
    )(fused)

    model = Model(inputs, outputs, name="HFF_FireDetection")
    return model, backbone


def build_paper_model(input_shape=IMG_SIZE, num_classes=NUM_CLASSES, freeze_backbone=True):
    """Explicit paper-aligned alias for the DFF -> FGA -> HFF architecture."""
    return build_hff_model(
        input_shape=input_shape,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
    )


if __name__ == "__main__":
    model, _ = build_paper_model()
    print(f"Total params: {model.count_params():,}")
