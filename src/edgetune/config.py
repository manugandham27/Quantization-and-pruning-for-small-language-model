"""
Configuration loader for EdgeTune YAML configurations.
"""

import os
from typing import Dict, Any
import yaml

from edgetune.schemas import (
    ModelConfig,
    DatasetConfig,
    LoRAConfigSchema,
    TrainingConfig,
    QuantizationConfigSchema,
    PruningConfigSchema,
)


def load_yaml(filepath: str) -> Dict[str, Any]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_base_config(config_path: str = "configs/base_model.yaml") -> Dict[str, Any]:
    raw_data = load_yaml(config_path)
    model_cfg = ModelConfig(**raw_data.get("model", {}))
    dataset_cfg = DatasetConfig(**raw_data.get("dataset", {}))
    hardware_cfg = raw_data.get("hardware", {})
    return {
        "model": model_cfg,
        "dataset": dataset_cfg,
        "hardware": hardware_cfg,
    }


def load_lora_config(config_path: str = "configs/lora.yaml") -> Dict[str, Any]:
    raw_data = load_yaml(config_path)
    lora_cfg = LoRAConfigSchema(**raw_data.get("lora", {}))
    training_cfg = TrainingConfig(**raw_data.get("training", {}))
    return {"lora": lora_cfg, "training": training_cfg}


def load_qlora_config(config_path: str = "configs/qlora.yaml") -> Dict[str, Any]:
    raw_data = load_yaml(config_path)
    lora_cfg = LoRAConfigSchema(**raw_data.get("qlora", {}))
    training_cfg = TrainingConfig(**raw_data.get("training", {}))
    quant_bits = raw_data.get("quantization_bits", 4)
    quant_type = raw_data.get("quant_type", "nf4")
    return {
        "qlora": lora_cfg,
        "training": training_cfg,
        "quantization_bits": quant_bits,
        "quant_type": quant_type,
    }


def load_quant_config(config_path: str = "configs/quantization_gptq.yaml") -> QuantizationConfigSchema:
    raw_data = load_yaml(config_path)
    return QuantizationConfigSchema(**raw_data.get("quantization", {}))


def load_prune_config(config_path: str = "configs/pruning.yaml") -> PruningConfigSchema:
    raw_data = load_yaml(config_path)
    return PruningConfigSchema(**raw_data.get("pruning", {}))
