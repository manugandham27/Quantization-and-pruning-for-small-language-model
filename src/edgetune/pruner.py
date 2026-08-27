"""
Pruning Engine supporting Unstructured Magnitude Pruning and Structured Head/Channel Pruning.
"""

from typing import Any

import torch
from torch import nn
from torch.nn.utils import prune
from transformers import PreTrainedModel

from edgetune.schemas import PruningConfigSchema


def is_linear_module(module: nn.Module) -> bool:
    if isinstance(module, nn.Linear):
        return True
    try:
        from transformers.pytorch_utils import Conv1D
        if isinstance(module, Conv1D):
            return True
    except ImportError:
        pass
    return hasattr(module, "weight") and isinstance(module.weight, torch.Tensor) and module.weight.ndim == 2 and not isinstance(module, (nn.Embedding, nn.LayerNorm))


def calculate_model_sparsity(model: nn.Module) -> tuple[float, int, int]:
    """
    Computes total non-zero vs zero weight parameters in linear layers.
    Returns (sparsity_ratio, zero_params, total_params).
    """
    total_params = 0
    zero_params = 0

    for name, module in model.named_modules():
        if is_linear_module(module):
            weight = module.weight.data
            total_params += weight.numel()
            zero_params += (weight == 0).sum().item()

    sparsity_ratio = (zero_params / total_params) if total_params > 0 else 0.0
    return round(sparsity_ratio, 4), zero_params, total_params


class Pruner:
    """
    Pruner engine managing magnitude-based unstructured and structured pruning.
    """

    def __init__(self, config: PruningConfigSchema):
        self.config = config

    def apply_unstructured_pruning(
        self,
        model: PreTrainedModel,
    ) -> tuple[PreTrainedModel, dict[str, Any]]:
        print(f"[Pruner] Applying L1 Unstructured Magnitude Pruning (target sparsity: {self.config.sparsity * 100:.1f}%)...")
        pruned_modules = []

        for name, module in model.named_modules():
            if is_linear_module(module):

                matched = any(t in name for t in self.config.target_modules)
                if matched or not self.config.target_modules:
                    prune.l1_unstructured(module, name="weight", amount=self.config.sparsity)
                    if self.config.make_permanent:
                        prune.remove(module, "weight")
                    pruned_modules.append(name)

        sparsity_ratio, zero_params, total_params = calculate_model_sparsity(model)
        print(f"[Pruner] Applied to {len(pruned_modules)} linear layers. Measured Sparsity: {sparsity_ratio * 100:.2f}%")

        metadata = {
            "method": "unstructured_magnitude",
            "target_sparsity": self.config.sparsity,
            "measured_sparsity": sparsity_ratio,
            "zero_parameters": zero_params,
            "total_parameters": total_params,
            "pruned_layers": len(pruned_modules),
        }
        return model, metadata

    def apply_structured_pruning(
        self,
        model: PreTrainedModel,
    ) -> tuple[PreTrainedModel, dict[str, Any]]:
        print(f"[Pruner] Applying Structured Channel/Head Pruning (target sparsity: {self.config.sparsity * 100:.1f}%)...")
        pruned_modules = []

        for name, module in model.named_modules():
            if is_linear_module(module):

                matched = any(t in name for t in self.config.target_modules)
                if matched or not self.config.target_modules:
                    # Prune channels along dim 0 (output features)
                    prune.ln_structured(module, name="weight", amount=self.config.sparsity, n=2, dim=0)
                    if self.config.make_permanent:
                        prune.remove(module, "weight")
                    pruned_modules.append(name)

        sparsity_ratio, zero_params, total_params = calculate_model_sparsity(model)
        print(f"[Pruner] Applied structured pruning to {len(pruned_modules)} layers. Measured Sparsity: {sparsity_ratio * 100:.2f}%")

        metadata = {
            "method": "structured_head_channel",
            "target_sparsity": self.config.sparsity,
            "measured_sparsity": sparsity_ratio,
            "zero_parameters": zero_params,
            "total_parameters": total_params,
            "pruned_layers": len(pruned_modules),
        }
        return model, metadata

    def prune(self, model: PreTrainedModel) -> tuple[PreTrainedModel, dict[str, Any]]:
        if self.config.method == "structured_head_channel":
            return self.apply_structured_pruning(model)
        else:
            return self.apply_unstructured_pruning(model)


def apply_pruning(
    model: PreTrainedModel,
    config: PruningConfigSchema,
) -> tuple[PreTrainedModel, dict[str, Any]]:
    pruner = Pruner(config)
    return pruner.prune(model)
