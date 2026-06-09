"""Model factory — baselines (timm) + proposed hybrid and DEKAN variants.

Supported names (passed to training/train.py --model):

  Baselines (timm, ImageNet-1k pretrained):
    resnet18            ResNet-18,              ~11.7M params,  ~1.8 GFLOPs
    mobilenetv2_100     MobileNetV2 (w=1.0),    ~3.5M params,   ~0.3 GFLOPs
    efficientnet_b0     EfficientNet-B0,        ~5.3M params,   ~0.39 GFLOPs
    mobilevit_s         MobileViT-S (hybrid),   ~5.6M params,   ~1.4 GFLOPs
    efficientformer_l1  EfficientFormer-L1,     ~12.3M params,  ~1.3 GFLOPs

  Hybrid / proposed lightweight (see models/hybrid.py):
    hybrid_full         CNN + ViT + metadata fusion (proposed, <6M)
    hybrid_no_meta      CNN + ViT, no metadata    (ablation)
    hybrid_cnn_only     CNN stem + GAP + head      (ablation)
    hybrid_vit_only     Pure ViT-Tiny on patches   (ablation)

  DEKAN flagship (see models/dekan.py, ~12M params accuracy-oriented tier):
    dekan_full          DenseNet + EffNet + TinyViT trunk + metadata + KAN  (proposed)
    dekan_no_meta       same, no metadata                                   (ablation)
    dekan_linear        same as full but nn.Linear head                     (ablation: KAN)
    dekan_densenet_only DenseNet only + metadata + KAN                      (ablation: fusion)
    dekan_effnet_only   EfficientNet-B0 only + metadata + KAN               (ablation: fusion)

Baselines load ImageNet-1k weights and replace the classifier with an 8-class
head. Hybrid/DEKAN variants load ImageNet pretrained weights only for their CNN
stems; the transformer trunk and metadata head are initialised from scratch.

`needs_metadata(name)` is the single authoritative check used by both
training/train.py and training/eval.py to decide whether to request the metadata
dict from the DataLoader.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

from models.hybrid import (
    create_hybrid,
    is_hybrid,
    needs_metadata as _hybrid_needs_metadata,
)
from models.dekan import create_dekan, dekan_needs_metadata, is_dekan

NUM_CLASSES = 8


def needs_metadata(name: str) -> bool:
    """True if the named model consumes the clinical metadata dict.

    Single source of truth: training/train.py and training/eval.py both import
    this function so adding a new metadata-consuming variant here is sufficient.
    """
    return _hybrid_needs_metadata(name) or dekan_needs_metadata(name)


# Map CLI short-name → timm model id.  Add new baselines here.
MODEL_REGISTRY: dict[str, str] = {
    "resnet18":           "resnet18",
    "mobilenetv2_100":    "mobilenetv2_100",
    "mobilenetv2":        "mobilenetv2_100",      # alias
    "efficientnet_b0":    "efficientnet_b0",
    "mobilevit_s":        "mobilevit_s",
    "efficientformer_l1": "efficientformer_l1",
}


def create_model(
    name: str,
    pretrained: bool = True,
    num_classes: int = NUM_CLASSES,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
) -> nn.Module:
    """Create a timm model with an 8-class head.

    Args:
        name:           Short name key from MODEL_REGISTRY, or a raw timm model id.
        pretrained:     Load ImageNet-1k weights.
        num_classes:    Output dimension. Default 8 for ISIC-2019.
        drop_rate:      Classifier dropout.
        drop_path_rate: Stochastic depth rate (ViT / hybrid models).

    Returns:
        nn.Module ready for training. Call
          model.to(device, memory_format=torch.channels_last)
        after this to enable channels-last layout for ~15% speedup.
    """
    # Dispatch to in-house model factories before falling through to timm.
    if is_hybrid(name):
        return create_hybrid(name, pretrained=pretrained, num_classes=num_classes)
    if is_dekan(name):
        return create_dekan(name, pretrained=pretrained, num_classes=num_classes)

    timm_id = MODEL_REGISTRY.get(name, name)

    kwargs: dict = dict(
        model_name=timm_id,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    # drop_rate is supported by most timm models; silently skip if not.
    if drop_rate > 0.0:
        kwargs["drop_rate"] = drop_rate
    if drop_path_rate > 0.0:
        kwargs["drop_path_rate"] = drop_path_rate

    model = timm.create_model(**kwargs)
    return model


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}