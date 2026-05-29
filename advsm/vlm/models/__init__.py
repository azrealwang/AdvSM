from __future__ import annotations

from pathlib import Path

from advsm.vlm.utils.config import load_yaml
from advsm.vlm.models.base import BaseRobustVQAModel
from advsm.vlm.models.robust_llava import RobustLLaVAWrapper
from advsm.vlm.models.llava_encoder_replace import LlavaEncoderReplaceWrapper


def load_model_from_config(model_key: str, models_yaml: str | Path) -> BaseRobustVQAModel:
    cfg = load_yaml(models_yaml)
    models = cfg.get("models", cfg)
    if model_key not in models:
        raise KeyError(f"Model {model_key!r} not found in {models_yaml}")
    mcfg = models[model_key]
    typ = mcfg.get("type", "robust_llava")
    if typ in ("robust_llava", "llava"):
        return RobustLLaVAWrapper(model_key, mcfg)
    if typ == "llava_encoder_replace":
        return LlavaEncoderReplaceWrapper(model_key, mcfg)
    raise ValueError(f"Unknown model type: {typ!r}")
