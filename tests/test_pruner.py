"""
Unit tests for Pruner module (unstructured & structured pruning).
"""

import pytest
from torch import nn

from edgetune.pruner import Pruner, calculate_model_sparsity
from edgetune.schemas import PruningConfigSchema


class DummyLinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(128, 128)
        self.v_proj = nn.Linear(128, 128)

    def forward(self, x):
        return self.v_proj(self.q_proj(x))


def test_unstructured_pruning():
    net = DummyLinearNet()
    config = PruningConfigSchema(
        method="unstructured_magnitude",
        sparsity=0.50,
        target_modules=["q_proj", "v_proj"],
        make_permanent=True,
    )
    pruner = Pruner(config)
    net, meta = pruner.prune(net)

    assert meta["method"] == "unstructured_magnitude"
    assert meta["pruned_layers"] == 2
    sparsity, _zero_params, _total_params = calculate_model_sparsity(net)
    assert pytest.approx(sparsity, abs=0.05) == 0.50


def test_structured_pruning():
    net = DummyLinearNet()
    config = PruningConfigSchema(
        method="structured_head_channel",
        sparsity=0.25,
        target_modules=["q_proj"],
        make_permanent=True,
    )
    pruner = Pruner(config)
    net, meta = pruner.prune(net)

    assert meta["method"] == "structured_head_channel"
    assert meta["pruned_layers"] == 1
