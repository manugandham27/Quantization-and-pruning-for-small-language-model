"""
Pydantic schemas for EdgeTune configurations, model variants, and benchmark metrics.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    torch_dtype: str = "float16"
    device_map: str = "auto"
    trust_remote_code: bool = True
    max_length: int = 512


class DatasetConfig(BaseModel):
    name: str = "samsum"
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"
    max_train_samples: int = 500
    max_eval_samples: int = 100
    text_column: str = "dialogue"
    summary_column: str = "summary"
    max_length: int = 256



class LoRAConfigSchema(BaseModel):
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = Field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


class TrainingConfig(BaseModel):
    learning_rate: float = 3e-4
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_strategy: str = "epoch"
    output_dir: str = "results/checkpoints/lora"


class QuantizationConfigSchema(BaseModel):
    method: str = "bitsandbytes"  # 'bitsandbytes', 'gptq_groupwise', 'pytorch_dynamic'
    bits: int = 4
    quant_type: str = "nf4"
    double_quant: bool = True
    compute_dtype: str = "float16"
    group_size: int = 128
    sym: bool = True
    desc_act: bool = False
    target_modules: List[str] = Field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    output_dir: str = "results/checkpoints/quantized"


class PruningConfigSchema(BaseModel):
    method: str = "unstructured_magnitude"  # 'unstructured_magnitude', 'structured_head_channel'
    sparsity: float = 0.50
    target_modules: List[str] = Field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    make_permanent: bool = True
    output_dir: str = "results/checkpoints/pruned"


class BenchmarkMetrics(BaseModel):
    variant_name: str
    base_model: str
    task_name: str
    device: str
    disk_size_mb: float
    peak_memory_mb: float
    ttft_ms: float  # Time To First Token in milliseconds
    tokens_per_sec: float
    rouge1: float
    rouge2: float
    rougeL: float
    perplexity: Optional[float] = None
    compression_ratio: float  # vs FP16 baseline
    effective_sparsity: float = 0.0
    effective_bits: float = 16.0


class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9


class GenerationResponse(BaseModel):
    generated_text: str
    time_to_first_token_ms: float
    tokens_per_second: float
    total_tokens: int
    peak_memory_mb: float
    model_variant: str
