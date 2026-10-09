"""
HFF Fire Detection Model (3-Class: nonfire, fire, smoke)
Paper Reference: Danish et al., IEEE JSTARS, 2026.
Modules:
  1. DFF: EfficientNetB2 Backbone (ImageNet Pretrained)
  2. FGA: Fire-Guided Attention (1x1 bottleneck + multiscale conv cascade)
  3. HFF: Hierarchical Feature Fusion (Cascaded feature integration)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB2

IMG_SIZE = (224, 224, 3)
NUM_CLASSES = 3  # Matches len(data_pipeline.CLASS_NAMES)


def build_dff_backbone(input_shape=IMG_SIZE, freeze=True):
    """
    Discriminative Fire Feature (DFF) extractor:
    ImageNet-pretrained EfficientNetB2 backbone without top dense layers.
    """
    backbone = EfficientNetB2(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    backbone.trainable = not freeze
    return backbone


def fire_guided_attention(fm, name="fga"):
    """
    FGA Module (Eq. 3-8 in paper):
    1. 1x1 conv bottleneck reduces 1408 -> 256 channels to prevent parameter explosion.
    2. Multiscale receptive field cascade: 7x7 -> 5x5 -> 3x3.
    3. Per-scale descriptors (Conv + GlobalAveragePooling + BatchNorm).
    4. Contextual integration via 1x1 projection + global pooling.
    """
    # 1x1 bottleneck reduction: 1408 -> 256
    fm_reduced = layers.Conv2D(256, 1, padding="same", activation="relu", name=f"{name}_reduce")(fm)

    x0 = layers.BatchNormalization()(fm_reduced)
    x0 = layers.Dropout(0.3)(x0)

    # Hierarchical multiscale cascade
    f1 = layers.Conv2D(128, 7, padding="same", activation="relu", name=f"{name}_conv7")(x0)
    f2 = layers.Conv2D(128, 5, padding="same", activation="relu", name=f"{name}_conv5")(f1)
    f3 = layers.Conv2D(128, 3, padding="same", activation="relu", name=f"{name}_conv3")(f2)

    # Per-scale descriptors: Conv -> GAP -> BN
    def scale_descriptor(t, k, filters=64):
        t = layers.Conv2D(filters, k, padding="same", activation="relu")(t)
        t = layers.GlobalAveragePooling2D()(t)
        return layers.BatchNormalization()(t)

    g1 = scale_descriptor(f1, 7)
    g2 = scale_descriptor(f2, 5)
    g3 = scale_descriptor(f3, 3)
    f_ms = layers.Concatenate(name=f"{name}_fms")([g1, g2, g3])

    # Context projection & global pooling descriptor
    p1 = layers.GlobalAveragePooling2D()(
        layers.Conv2D(64, 1, padding="same", activation="relu")(fm)
    )
    global_ctx = layers.GlobalAveragePooling2D()(fm)

    fga_out = layers.Concatenate(name=f"{name}_out")([f_ms, p1, global_ctx])
    return fga_out


def hierarchical_feature_fusion(fm, name="hff"):
    """
    HFF Module (Section II-E, Eq. 15):
    Extracts multiscale spatial representations (7x7, 5x5, 3x3, 1x1)
    and merges them via chained cumulative concatenation.
    """
    # 1x1 bottleneck reduction: 1408 -> 256
    fm_reduced = layers.Conv2D(256, 1, padding="same", activation="relu", name=f"{name}_reduce")(fm)

    f = layers.Conv2D(128, 7, padding="same", activation="relu")(fm_reduced)
    k = layers.Conv2D(64, 1, padding="same", activation="relu")(fm_reduced)
    l = layers.Conv2D(64, 1, padding="same", activation="relu")(fm_reduced)
    g = layers.Conv2D(96, 5, padding="same", activation="relu")(f)
    h = layers.Conv2D(96, 3, padding="same", activation="relu")(f)

    def ds_bn(t, filters=64):
        t = layers.Conv2D(filters, 1, padding="same", activation="relu")(t)
        t = layers.GlobalAveragePooling2D()(t)
        return layers.BatchNormalization()(t)

    x1, x2, x3, x4, x5 = ds_bn(f), ds_bn(g), ds_bn(h), ds_bn(k), ds_bn(l)

    # Chained concatenation: F1 = X1 (+) X2; F2 = F1 (+) X3; F3 = F2 (+) X4
    f1 = layers.Concatenate()([x1, x2])
    f2 = layers.Concatenate()([f1, x3])
    f3 = layers.Concatenate()([f2, x4])

    global_ctx = layers.GlobalAveragePooling2D()(
        layers.Conv2D(64, 1, padding="same", activation="relu")(fm)
    )
    hff_out = layers.Concatenate(name=f"{name}_out")([f3, x5, global_ctx])
    return hff_out


def build_hff_model(input_shape=IMG_SIZE, num_classes=NUM_CLASSES, freeze_backbone=True):
    inputs = layers.Input(shape=input_shape, name="input_tensor")
    backbone = build_dff_backbone(input_shape=input_shape, freeze=freeze_backbone)
    fm = backbone(inputs)

    fga_out = fire_guided_attention(fm)
    hff_out = hierarchical_feature_fusion(fm)

    fused = layers.Concatenate(name="dff_fga_hff_fusion")([fga_out, hff_out])
    fused = layers.BatchNormalization()(fused)
    fused = layers.Dense(256, activation="relu")(fused)
    fused = layers.Dropout(0.4)(fused)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(fused)

    model = Model(inputs, outputs, name="HFF_FireDetection")
    return model, backbone


if __name__ == "__main__":
    model, backbone = build_hff_model()
    print(f"\nTotal params: {model.count_params():,}")
