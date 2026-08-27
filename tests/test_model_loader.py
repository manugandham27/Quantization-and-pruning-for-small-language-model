"""
Unit tests for model_loader module functions.
"""

import os
import tempfile

import torch
from torch import nn

from edgetune.model_loader import (
    get_directory_size_mb,
    get_model_size_mb,
    get_optimal_device,
    get_torch_dtype,
)


def test_get_optimal_device():
    device = get_optimal_device()
    assert isinstance(device, torch.device)
    assert device.type in ["cuda", "mps", "cpu"]


def test_get_torch_dtype():
    assert get_torch_dtype("float16") == torch.float16
    assert get_torch_dtype("fp16") == torch.float16
    assert get_torch_dtype("bfloat16") == torch.bfloat16
    assert get_torch_dtype("bf16") == torch.bfloat16
    assert get_torch_dtype("float32") == torch.float32
    assert get_torch_dtype("fp32") == torch.float32
    assert get_torch_dtype("unknown") == torch.float16


def test_get_model_size_mb():
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1000, 1000)  # 1,000,000 floats ~ 4MB in FP32

    model = DummyModel()
    size_mb = get_model_size_mb(model)
    assert size_mb > 3.5 and size_mb < 4.5


def test_get_directory_size_mb():
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = os.path.join(tmpdir, "test1.bin")
        file2 = os.path.join(tmpdir, "test2.bin")
        # Write 1MB to each file
        with open(file1, "wb") as f:
            f.write(b"\x00" * (1024 * 1024))
        with open(file2, "wb") as f:
            f.write(b"\x00" * (1024 * 1024))

        dir_size = get_directory_size_mb(tmpdir)
        assert dir_size == 2.0

    assert get_directory_size_mb("/non/existent/path/for/testing") == 0.0
