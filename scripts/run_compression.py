#!/usr/bin/env python3
"""
Standalone execution script for quantization and pruning.
"""

import argparse
import os
from edgetune.config import load_base_config, load_quant_config, load_prune_config
from edgetune.model_loader import load_model_and_tokenizer
from edgetune.quantizer import apply_quantization
from edgetune.pruner import apply_pruning


def main():
    parser = argparse.ArgumentParser(description="Run Model Compression (Quantization / Pruning)")
    parser.add_argument("--model-path", type=str, default=None, help="Path to base model or fine-tuned checkpoint")
    parser.add_argument("--quant-config", type=str, default="configs/quantization_gptq.yaml")
    parser.add_argument("--prune-config", type=str, default="configs/pruning.yaml")
    parser.add_argument("--action", type=str, choices=["quantize", "prune", "combined"], default="combined")
    args = parser.parse_args()

    base_cfg = load_base_config()["model"]
    if args.model_path:
        base_cfg.name_or_path = args.model_path

    print(f"Loading model from '{base_cfg.name_or_path}'...")
    model, tokenizer, device = load_model_and_tokenizer(base_cfg)

    if args.action in ["prune", "combined"]:
        p_cfg = load_prune_config(args.prune_config)
        model, p_meta = apply_pruning(model, p_cfg)
        print(f"Pruning complete: {p_meta}")

    if args.action in ["quantize", "combined"]:
        q_cfg = load_quant_config(args.quant_config)
        model, q_meta = apply_quantization(model, tokenizer, q_cfg)
        print(f"Quantization complete: {q_meta}")

    save_dir = "results/checkpoints/compressed_output"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Compressed model saved to: {save_dir}")


if __name__ == "__main__":
    main()
