#!/usr/bin/env python3
"""
Standalone execution script for LoRA and QLoRA fine-tuning.
"""

import argparse
from edgetune.config import load_base_config, load_lora_config, load_qlora_config
from edgetune.peft_trainer import train_peft_model


def main():
    parser = argparse.ArgumentParser(description="Run PEFT Fine-Tuning (LoRA / QLoRA)")
    parser.add_argument("--qlora", action="store_true", help="Use 4-bit QLoRA instead of standard LoRA")
    parser.add_argument("--base-config", type=str, default="configs/base_model.yaml")
    parser.add_argument("--peft-config", type=str, default=None)
    args = parser.parse_args()

    base_cfg = load_base_config(args.base_config)
    
    if args.qlora:
        peft_cfg_file = args.peft_config or "configs/qlora.yaml"
        qlora_dict = load_qlora_config(peft_cfg_file)
        peft_cfg = qlora_dict["qlora"]
        training_cfg = qlora_dict["training"]
    else:
        peft_cfg_file = args.peft_config or "configs/lora.yaml"
        lora_dict = load_lora_config(peft_cfg_file)
        peft_cfg = lora_dict["lora"]
        training_cfg = lora_dict["training"]

    print(f"=== Starting PEFT Fine-Tuning ({'QLoRA' if args.qlora else 'LoRA'}) ===")
    save_path = train_peft_model(
        model_cfg=base_cfg["model"],
        dataset_cfg=base_cfg["dataset"],
        lora_cfg=peft_cfg,
        training_cfg=training_cfg,
        use_qlora=args.qlora,
    )
    print(f"Fine-tuning complete. Model saved to: {save_path}")


if __name__ == "__main__":
    main()
