"""
Base model and tokenizer loader with multi-hardware support (CUDA, MPS, CPU).
"""

import os
import torch
from typing import Tuple, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from edgetune.schemas import ModelConfig


def get_optimal_device() -> torch.device:
    """
    Detects best hardware device: CUDA GPU, Apple Silicon MPS, or CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(dtype_str.lower(), torch.float16)


def load_model_and_tokenizer(
    model_config: ModelConfig,
    device_override: Optional[torch.device] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, torch.device]:
    """
    Loads pretrained causal LLM and tokenizer with specified precision and hardware device.
    """
    device = device_override or get_optimal_device()
    torch_dtype = get_torch_dtype(model_config.torch_dtype)

    print(f"[ModelLoader] Loading model '{model_config.name_or_path}'...")
    print(f"[ModelLoader] Target Device: {device}, Precision: {torch_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.name_or_path,
        trust_remote_code=model_config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    quantization_config = None
    if load_in_4bit or load_in_8bit:
        try:
            from transformers import BitsAndBytesConfig
            if load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                )
            elif load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        except Exception as e:
            print(f"[ModelLoader] BitsAndBytesConfig warning: {e}. Falling back to standard precision.")

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": model_config.trust_remote_code,
        "torch_dtype": torch_dtype,
    }
    if device.type == "mps":
        model_kwargs["attn_implementation"] = "eager"


    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = model_config.device_map
    elif device.type in ["cuda", "mps"]:
        model = AutoModelForCausalLM.from_pretrained(
            model_config.name_or_path,
            **model_kwargs
        )
        model = model.to(device)
        return model, tokenizer, device
    else:
        model_kwargs["device_map"] = "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_config.name_or_path,
        **model_kwargs
    )

    return model, tokenizer, device


def get_model_size_mb(model: torch.nn.Module) -> float:
    """
    Calculates size of model parameters in memory in MB.
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / (1024 ** 2)
    return round(size_all_mb, 2)


def get_directory_size_mb(directory: str) -> float:
    """
    Calculates total disk footprint of saved checkpoint files in MB.
    """
    if not os.path.exists(directory):
        return 0.0
    total_bytes = 0
    for root, dirs, files in os.walk(directory):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                total_bytes += os.path.getsize(fp)
    return round(total_bytes / (1024 ** 2), 2)
