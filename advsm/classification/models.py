"""Load ImageNet classifiers from ``configs/classifiers.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import timm
import torch.nn as nn
import yaml
from robustbench import load_model

from advsm._paths import repo_root

_DEFAULT_CONFIG = Path(repo_root()) / "configs" / "classifiers.yaml"

_CLASSIFIER_MODEL_KEYS = [
    "ResNet-50",
    "ConvNeXt-B",
    "ViT-B",
    "Swin-B",
    "ResNet-50 (Robust)",
    "ConvNeXt-B (Robust)",
    "ViT-B (Robust)",
    "Swin-B (Robust)",
]


def load_classifiers_config(path: str | Path | None = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else _DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("models", data)


def load_classifier(name: str, config_path: str | Path | None = None) -> nn.Module:
    models = load_classifiers_config(config_path)
    if name not in models:
        known = ", ".join(models.keys())
        raise KeyError(f"Unknown classifier {name!r}. Known: {known}")
    spec = models[name]
    dataset = spec.get("dataset", "imagenet")
    if dataset != "imagenet":
        raise ValueError(f"Only imagenet classifiers are configured; got {dataset!r} for {name!r}")

    kind = spec["kind"]
    if kind == "timm":
        return timm.create_model(spec["timm_name"], pretrained=True, num_classes=1000).eval()
    if kind == "robustbench":
        return load_model(spec["robustbench_id"], dataset="imagenet", threat_model="Linf").eval()
    raise ValueError(f"Unsupported classifier kind {kind!r} for {name!r}")
