#!/usr/bin/env python3
"""
Initializes a local base model checkpoint to ensure 100% offline, lightning-fast evaluation sweeps.
"""

import os

from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel


def init_local_base_model(output_dir: str = "models/base_model"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[InitLocalModel] Creating base model architecture at '{output_dir}'...")

    config = GPT2Config(
        vocab_size=50257,
        n_positions=512,
        n_ctx=512,
        n_embd=256,
        n_layer=4,
        n_head=4,
        bos_token_id=50256,
        eos_token_id=50256,
    )

    model = GPT2LMHeadModel(config)
    model.save_pretrained(output_dir)

    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except Exception as e:
        print(f"[InitLocalModel] Offline tokenizer fallback: {e}")
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=False)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(output_dir)
    print(f"[InitLocalModel] Local base model successfully created at '{output_dir}'!")
    return output_dir


if __name__ == "__main__":
    init_local_base_model()
