"""
Comprehensive Benchmark Harness measuring Disk Footprint, Peak Memory,
TTFT (ms), Tokens/sec, ROUGE-1/2/L, Perplexity, and Compression Ratios.
"""

import os
import time

import evaluate
import psutil
import torch
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from edgetune.model_loader import (
    get_directory_size_mb,
    get_model_size_mb,
)
from edgetune.pruner import calculate_model_sparsity
from edgetune.schemas import BenchmarkMetrics


def get_peak_memory_mb(device: torch.device) -> float:
    """
    Returns peak memory allocated on GPU/MPS/CPU in MB.
    """
    if device.type == "cuda" and torch.cuda.is_available():
        peak_bytes = torch.cuda.max_memory_allocated(device)
        return round(peak_bytes / (1024 ** 2), 2)
    elif device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        try:
            peak_bytes = torch.mps.current_allocated_memory()
            return round(peak_bytes / (1024 ** 2), 2)
        except (AttributeError, RuntimeError):
            pass

    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024 ** 2), 2)


def measure_generation_performance(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    prompts: list[str],
    max_new_tokens: int = 64,
    warmup_runs: int = 1,
) -> tuple[float, float]:
    """
    Measures Time-To-First-Token (TTFT in ms) and Generation Speed (Tokens/sec).
    """
    model.eval()

    # Warmup run
    if prompts and warmup_runs > 0:
        warmup_input = tokenizer(prompts[0], return_tensors="pt").to(device)
        with torch.no_grad():
            model.generate(**warmup_input, max_new_tokens=8, do_sample=False)

    ttft_list: list[float] = []
    tokens_per_sec_list: list[float] = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

        # 1. Measure TTFT (generating exactly 1 token)
        t_start = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=1, min_new_tokens=1, do_sample=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        t_ttft = (time.perf_counter() - t_start) * 1000.0  # ms
        ttft_list.append(t_ttft)

        # 2. Measure full generation speed (max_new_tokens)
        t_gen_start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens, do_sample=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        t_gen_end = time.perf_counter()

        gen_tokens = outputs.shape[1] - input_len
        gen_time = max(t_gen_end - t_gen_start, 1e-5)
        tps = gen_tokens / gen_time
        tokens_per_sec_list.append(tps)

    avg_ttft = round(sum(ttft_list) / len(ttft_list), 2) if ttft_list else 0.0
    avg_tps = round(sum(tokens_per_sec_list) / len(tokens_per_sec_list), 2) if tokens_per_sec_list else 0.0

    return avg_ttft, avg_tps


def evaluate_rouge(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    val_dataset: Dataset,
    max_samples: int = 50,
    max_new_tokens: int = 64,
) -> tuple[float, float, float]:
    """
    Evaluates ROUGE-1, ROUGE-2, and ROUGE-L scores on dialogue summarization validation split.
    """
    model.eval()
    rouge = evaluate.load("rouge")

    predictions = []
    references = []

    eval_subset = val_dataset.select(range(min(len(val_dataset), max_samples)))

    for example in eval_subset:
        dialogue = example["dialogue"]
        reference = example["summary"]
        prompt = f"Summarize dialogue:\n{dialogue}\n\nSummary:"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        generated_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        predictions.append(generated_text.strip() or " ")
        references.append(reference.strip() or " ")

    if not predictions or not references:
        return 0.0, 0.0, 0.0

    results = rouge.compute(predictions=predictions, references=references)
    r1 = round(float(results.get("rouge1", 0.0)) * 100, 2)
    r2 = round(float(results.get("rouge2", 0.0)) * 100, 2)
    rL = round(float(results.get("rougeL", 0.0)) * 100, 2)

    return r1, r2, rL


def benchmark_model_variant(
    variant_name: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    val_dataset: Dataset,
    baseline_disk_size_mb: float | None = None,
    checkpoint_dir: str | None = None,
    effective_bits: float = 16.0,
) -> BenchmarkMetrics:
    """
    Runs full benchmark sweep on a model variant and produces a structured BenchmarkMetrics object.
    """
    print("\n========================================================")
    print(f"[Benchmark] Running Benchmark for Variant: '{variant_name}'")
    print("========================================================")

    # 1. Disk Size & Sparsity
    in_memory_mb = get_model_size_mb(model)
    if checkpoint_dir and os.path.exists(checkpoint_dir):
        disk_size_mb = get_directory_size_mb(checkpoint_dir)
        if disk_size_mb == 0.0:
            disk_size_mb = in_memory_mb
    else:
        disk_size_mb = in_memory_mb

    sparsity_ratio, _zero_params, _total_params = calculate_model_sparsity(model)

    # 2. Peak Memory
    peak_mem_mb = get_peak_memory_mb(device)

    # 3. Latency Benchmark
    test_prompts = [
        "Summarize dialogue:\nAlice: Hey Bob, are we still meeting for lunch at 12?\nBob: Yes! See you at the diner.\n\nSummary:",
        "Summarize dialogue:\nManager: Team, please submit your weekly reports by 5 PM today.\nDev: On it, will submit shortly.\n\nSummary:",
    ]
    ttft_ms, tokens_per_sec = measure_generation_performance(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompts=test_prompts,
        max_new_tokens=48,
    )

    # 4. ROUGE Evaluation
    r1, r2, rL = evaluate_rouge(
        model=model,
        tokenizer=tokenizer,
        device=device,
        val_dataset=val_dataset,
        max_samples=25,
        max_new_tokens=48,
    )

    # 5. Compression Ratio vs Baseline
    ref_baseline_mb = baseline_disk_size_mb if baseline_disk_size_mb else disk_size_mb
    comp_ratio = round(ref_baseline_mb / max(disk_size_mb, 0.1), 2)

    metrics = BenchmarkMetrics(
        variant_name=variant_name,
        base_model=getattr(model.config, "_name_or_path", "unknown"),
        task_name="samsum_summarization",
        device=str(device.type),
        disk_size_mb=disk_size_mb,
        peak_memory_mb=peak_mem_mb,
        ttft_ms=ttft_ms,
        tokens_per_sec=tokens_per_sec,
        rouge1=r1,
        rouge2=r2,
        rougeL=rL,
        compression_ratio=comp_ratio,
        effective_sparsity=sparsity_ratio,
        effective_bits=effective_bits,
    )

    print(f"| Variant Name       | {metrics.variant_name}")
    print(f"| Disk Size (MB)     | {metrics.disk_size_mb} MB")
    print(f"| Peak Memory (MB)   | {metrics.peak_memory_mb} MB")
    print(f"| TTFT (ms)          | {metrics.ttft_ms} ms")
    print(f"| Tokens / sec       | {metrics.tokens_per_sec} tok/s")
    print(f"| ROUGE-1 / 2 / L    | {metrics.rouge1} / {metrics.rouge2} / {metrics.rougeL}")
    print(f"| Compression Ratio  | {metrics.compression_ratio}x")
    print("========================================================\n")

    return metrics
