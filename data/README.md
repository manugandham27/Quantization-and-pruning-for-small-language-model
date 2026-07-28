# EdgeTune Dataset Guide: SAMSum Dialogue Summarization

## Overview
EdgeTune utilizes the **SAMSum** dataset ([knkarthick/samsum](https://huggingface.co/datasets/knkarthick/samsum)) for evaluating parameter-efficient fine-tuning (LoRA / QLoRA) and post-training compression techniques (quantization & pruning).

### Dataset Format
- `dialogue`: Raw multi-turn conversation text between two or more participants.
- `summary`: Human-annotated reference summary capturing key actions and intent.

### Usage in EdgeTune
The `peft_trainer.py` module formats dialogues into prompt instructions:
```text
Summarize the following conversation:

[Dialogue Text]

Summary:
[Target Summary]
```
Evaluations compute ROUGE-1, ROUGE-2, and ROUGE-L quality metrics against human references across baseline FP16, fine-tuned, pruned, quantized, and combined model variants.
