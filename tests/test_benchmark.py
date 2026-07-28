"""
Unit tests for EdgeTune benchmark metrics and helper logic.
"""

import pytest
import torch
import torch.nn as nn
from edgetune.schemas import BenchmarkMetrics
from edgetune.model_loader import get_model_size_mb, get_directory_size_mb
from edgetune.pruner import calculate_model_sparsity


class SimpleToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc2(self.fc1(x))


def test_get_model_size_mb():
    model = SimpleToyModel()
    size_mb = get_model_size_mb(model)
    assert isinstance(size_mb, float)
    assert size_mb > 0.0


def test_calculate_model_sparsity():
    model = SimpleToyModel()
    # Zero out half of fc1 weights
    with torch.no_grad():
        model.fc1.weight.data[:32, :] = 0.0

    sparsity, zero_params, total_params = calculate_model_sparsity(model)
    assert sparsity > 0.0
    assert zero_params > 0
    assert total_params == (64 * 128 + 128 * 10)


def test_benchmark_metrics_schema():
    metrics = BenchmarkMetrics(
        variant_name="TestVariant",
        base_model="Qwen2.5-0.5B",
        task_name="samsum",
        device="cpu",
        disk_size_mb=250.0,
        peak_memory_mb=400.0,
        ttft_ms=15.2,
        tokens_per_sec=45.0,
        rouge1=42.5,
        rouge2=18.3,
        rougeL=35.1,
        compression_ratio=4.0,
    )
    assert metrics.variant_name == "TestVariant"
    assert metrics.compression_ratio == 4.0
