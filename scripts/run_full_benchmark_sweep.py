#!/usr/bin/env python3
"""
Full Benchmark Sweep Script: Evaluates baseline FP16, LoRA, QLoRA, Pruned,
Quantized, and Combined pipelines end-to-end and generates complete report artifacts.
"""

import copy
import json
import os
import sys
from typing import Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from export_report import (
    generate_csv_report,
    generate_markdown_table,
    generate_pareto_chart,
)

from edgetune.benchmark import benchmark_model_variant
from edgetune.config import (
    load_base_config,
    load_lora_config,
    load_prune_config,
    load_qlora_config,
    load_quant_config,
)
from edgetune.model_loader import load_model_and_tokenizer
from edgetune.peft_trainer import prepare_samsum_dataset, train_peft_model
from edgetune.pruner import apply_pruning
from edgetune.quantizer import apply_quantization


def main():
    print("======================================================================")
    print("      EdgeTune: Full Compression & Evaluation Benchmark Sweep         ")
    print("======================================================================")

    os.makedirs("results/checkpoints", exist_ok=True)
    base_cfg = load_base_config("configs/base_model.yaml")

    # Load validation split once
    model_cfg = base_cfg["model"]
    dataset_cfg = base_cfg["dataset"]
    
    # Pre-load tokenizer and validation dataset for evaluation
    _, tokenizer, device = load_model_and_tokenizer(model_cfg)
    _, _, val_dataset = prepare_samsum_dataset(dataset_cfg, tokenizer)



    benchmark_results: list[dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Stage 1: Baseline FP16 Model Evaluation
    # -------------------------------------------------------------------------
    print("\n[Sweep Stage 1/6] Evaluated Baseline FP16 Model...")
    base_model, _, _ = load_model_and_tokenizer(model_cfg, device_override=device)
    fp16_metrics = benchmark_model_variant(
        variant_name="Baseline (FP16)",
        model=base_model,
        tokenizer=tokenizer,
        device=device,
        val_dataset=val_dataset,
        effective_bits=16.0,
    )
    baseline_disk_size = fp16_metrics.disk_size_mb
    benchmark_results.append(fp16_metrics.model_dump())
    del base_model

    # -------------------------------------------------------------------------
    # Stage 2: LoRA Fine-Tuning
    # -------------------------------------------------------------------------
    print("\n[Sweep Stage 2/6] Training & Evaluating LoRA Variant...")
    lora_dict = load_lora_config("configs/lora.yaml")
    lora_dir = train_peft_model(
        model_cfg=model_cfg,
        dataset_cfg=dataset_cfg,
        lora_cfg=lora_dict["lora"],
        training_cfg=lora_dict["training"],
        use_qlora=False,
    )
    lora_model_cfg = copy.deepcopy(model_cfg)
    lora_model_cfg.name_or_path = lora_dir
    lora_model, _, _ = load_model_and_tokenizer(lora_model_cfg, device_override=device)
    lora_metrics = benchmark_model_variant(
        variant_name="LoRA Fine-tuned",
        model=lora_model,
        tokenizer=tokenizer,
        device=device,
        val_dataset=val_dataset,
        baseline_disk_size_mb=baseline_disk_size,
        checkpoint_dir=lora_dir,
        effective_bits=16.0,
    )
    benchmark_results.append(lora_metrics.model_dump())

    # -------------------------------------------------------------------------
    # Stage 3: QLoRA Fine-Tuning
    # -------------------------------------------------------------------------
    print("\n[Sweep Stage 3/6] Training & Evaluating QLoRA Variant...")
    qlora_dict = load_qlora_config("configs/qlora.yaml")
    qlora_dir = train_peft_model(
        model_cfg=model_cfg,
        dataset_cfg=dataset_cfg,
        lora_cfg=qlora_dict["qlora"],
        training_cfg=qlora_dict["training"],
        use_qlora=True,
    )
    qlora_model_cfg = copy.deepcopy(model_cfg)
    qlora_model_cfg.name_or_path = qlora_dir
    qlora_model, _, _ = load_model_and_tokenizer(qlora_model_cfg, device_override=device, load_in_4bit=True)
    qlora_metrics = benchmark_model_variant(
        variant_name="QLoRA Fine-tuned (4-bit)",
        model=qlora_model,
        tokenizer=tokenizer,
        device=device,
        val_dataset=val_dataset,
        baseline_disk_size_mb=baseline_disk_size,
        checkpoint_dir=qlora_dir,
        effective_bits=4.0,
    )
    benchmark_results.append(qlora_metrics.model_dump())
    del qlora_model

    # -------------------------------------------------------------------------
    # Stage 4: Pruned Variant (LoRA + 50% Sparsity)
    # -------------------------------------------------------------------------
    print("\n[Sweep Stage 4/6] Applying Pruning (50% Sparsity)...")
    prune_cfg = load_prune_config("configs/pruning.yaml")
    stage4_model, _, _ = load_model_and_tokenizer(lora_model_cfg, device_override=device)
    pruned_model, p_meta = apply_pruning(stage4_model, prune_cfg)
    pruned_metrics = benchmark_model_variant(
        variant_name="LoRA + Pruned (50% Sparsity)",
        model=pruned_model,
        tokenizer=tokenizer,
        device=device,
        val_dataset=val_dataset,
        baseline_disk_size_mb=baseline_disk_size,
        effective_bits=16.0 * (1 - p_meta["measured_sparsity"]),
    )
    benchmark_results.append(pruned_metrics.model_dump())

    # -------------------------------------------------------------------------
    # Stage 5: Quantized Variant (LoRA + 4-Bit Groupwise)
    # -------------------------------------------------------------------------
    print("\n[Sweep Stage 5/6] Applying Quantization (4-Bit Groupwise)...")
    quant_cfg = load_quant_config("configs/quantization_gptq.yaml")
    stage5_model, _, _ = load_model_and_tokenizer(lora_model_cfg, device_override=device)
    quant_model, _q_meta = apply_quantization(stage5_model, tokenizer, quant_cfg)
    quant_metrics = benchmark_model_variant(
        variant_name="LoRA + Quantized (4-bit)",
        model=quant_model,
        tokenizer=tokenizer,
        device=device,
        val_dataset=val_dataset,
        baseline_disk_size_mb=baseline_disk_size,
        effective_bits=4.0,
    )
    benchmark_results.append(quant_metrics.model_dump())

    # -------------------------------------------------------------------------
    # Stage 6: Combined Compression Pipeline (LoRA -> Pruned -> Quantized)
    # -------------------------------------------------------------------------
    print("\n[Sweep Stage 6/6] Applying Combined Pipeline (LoRA -> Prune -> Quantize)...")
    stage6_model, _, _ = load_model_and_tokenizer(lora_model_cfg, device_override=device)
    combined_model, _ = apply_pruning(stage6_model, prune_cfg)
    combined_model, _ = apply_quantization(combined_model, tokenizer, quant_cfg)
    combined_metrics = benchmark_model_variant(
        variant_name="Combined (LoRA + Pruned + 4-bit Quant)",
        model=combined_model,
        tokenizer=tokenizer,
        device=device,
        val_dataset=val_dataset,
        baseline_disk_size_mb=baseline_disk_size,
        effective_bits=2.0,
    )

    benchmark_results.append(combined_metrics.model_dump())

    # -------------------------------------------------------------------------
    # Export Benchmark Results & Generate Visual Reports
    # -------------------------------------------------------------------------
    json_path = "results/benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\n[Sweep Complete] Raw benchmark metrics written to '{json_path}'")

    generate_csv_report(benchmark_results)
    generate_markdown_table(benchmark_results)
    generate_pareto_chart(benchmark_results)

    print("\n======================================================================")
    print("   EdgeTune Benchmark Sweep Successfully Completed All Stages!        ")
    print("======================================================================")


if __name__ == "__main__":
    main()
