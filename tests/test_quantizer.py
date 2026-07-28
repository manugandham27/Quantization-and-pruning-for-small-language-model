"""
Unit tests for Groupwise 4-Bit Linear Layer and Quantization module.
"""

import pytest
import torch
import torch.nn as nn
from edgetune.quantizer import Groupwise4BitLinear, Groupwise4BitQuantizer
from edgetune.schemas import QuantizationConfigSchema


def test_groupwise_4bit_linear_packing_and_forward():
    in_features = 128
    out_features = 64
    group_size = 32

    float_layer = nn.Linear(in_features, out_features, bias=True)
    # Initialize with non-trivial weights
    torch.nn.init.kaiming_uniform_(float_layer.weight)

    q_layer = Groupwise4BitLinear.from_float_linear(float_layer, group_size=group_size)

    assert q_layer.in_features == in_features
    assert q_layer.out_features == out_features
    # Check packed weights shape: (out_features, in_features // 2)
    assert q_layer.qweights.shape == (out_features, in_features // 2)
    # Check scales shape: (out_features, in_features // group_size)
    assert q_layer.scales.shape == (out_features, in_features // group_size)

    # Test forward pass with dummy input
    x = torch.randn(4, in_features)
    out_float = float_layer(x)
    out_q = q_layer(x)

    assert out_q.shape == (4, out_features)
    # 4-bit approximation error check (cosine similarity or small MSE difference)
    cos_sim = torch.nn.functional.cosine_similarity(out_float.flatten(), out_q.flatten(), dim=0)
    assert cos_sim.item() > 0.90


def test_groupwise_quantizer_on_model():
    class DummyTransformerBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(128, 128)
            self.v_proj = nn.Linear(128, 128)

        def forward(self, x):
            return self.v_proj(self.q_proj(x))

    model = DummyTransformerBlock()
    config = QuantizationConfigSchema(
        method="gptq_groupwise",
        bits=4,
        group_size=64,
        target_modules=["q_proj", "v_proj"],
    )

    quantizer = Groupwise4BitQuantizer(config)
    model, metadata = quantizer.quantize(model, None)

    assert metadata["bits"] == 4
    assert metadata["quantized_layers"] == 2
    assert isinstance(model.q_proj, Groupwise4BitLinear)
    assert isinstance(model.v_proj, Groupwise4BitLinear)
