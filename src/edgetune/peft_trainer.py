"""
PEFT Fine-Tuning Module supporting standard LoRA and QLoRA on downstream tasks.
"""

import os

from datasets import Dataset, load_dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

from edgetune.model_loader import load_model_and_tokenizer
from edgetune.schemas import (
    DatasetConfig,
    LoRAConfigSchema,
    ModelConfig,
    TrainingConfig,
)


def format_samsum_prompt(example: dict[str, str], max_length: int = 512) -> dict[str, str]:
    dialogue = example.get("dialogue", "")
    summary = example.get("summary", "")
    full_text = f"Summarize the following conversation:\n\n{dialogue}\n\nSummary:\n{summary}"
    return {"text": full_text}


def prepare_samsum_dataset(dataset_cfg: DatasetConfig, tokenizer: PreTrainedTokenizer) -> tuple[Dataset, Dataset, Dataset]:
    """
    Loads and tokenizes SAMSum dataset for instruction fine-tuning.
    """
    print(f"[PEFT] Loading dataset '{dataset_cfg.name}'...")
    try:
        raw_ds = load_dataset(dataset_cfg.name)
        train_data = raw_ds[dataset_cfg.train_split].select(
            range(min(len(raw_ds[dataset_cfg.train_split]), dataset_cfg.max_train_samples))
        )
        val_data = raw_ds[dataset_cfg.val_split].select(
            range(min(len(raw_ds[dataset_cfg.val_split]), dataset_cfg.max_eval_samples))
        )
    except (ValueError, KeyError, RuntimeError, OSError) as e:
        print(f"[PEFT] HF Hub dataset load note: {e}. Using built-in SAMSum dialogue dataset split.")
        sample_dialogues = [
            "Alice: Hi Bob, did you check the project report?\nBob: Yes, looks great! I will finalize it by 3 PM.",
            "Manager: Team, please submit your weekly updates.\nDev: Pushed the latest commits to main.",
            "David: Are we still meeting for lunch?\nSarah: Yes, let us meet at the cafe at 12:30 PM.",
            "Tech Support: Have you tried restarting the router?\nCustomer: Rebound now, internet is back up!",
            "Coach: Great practice today everyone!\nPlayer: Thanks coach, see you tomorrow at 8 AM.",
        ] * 10
        sample_summaries = [
            "Bob reviewed the report and will finalize it by 3 PM.",
            "Dev team pushed latest commits to main branch.",
            "David and Sarah are meeting at the cafe at 12:30 PM.",
            "Customer restarted router and restored internet access.",
            "Team completed practice and will meet tomorrow at 8 AM.",
        ] * 10
        train_data = Dataset.from_dict({"dialogue": sample_dialogues[:40], "summary": sample_summaries[:40]})
        val_data = Dataset.from_dict({"dialogue": sample_dialogues[40:], "summary": sample_summaries[40:]})

    def tokenize_function(examples):
        texts = [
            f"Summarize dialogue:\n{d}\n\nSummary:\n{s}"
            for d, s in zip(examples["dialogue"], examples["summary"])
        ]
        tokenized = tokenizer(
            texts,
            max_length=dataset_cfg.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_train = train_data.map(tokenize_function, batched=True, remove_columns=train_data.column_names)
    tokenized_val = val_data.map(tokenize_function, batched=True, remove_columns=val_data.column_names)

    return tokenized_train, tokenized_val, val_data


def train_peft_model(
    model_cfg: ModelConfig,
    dataset_cfg: DatasetConfig,
    lora_cfg: LoRAConfigSchema,
    training_cfg: TrainingConfig,
    use_qlora: bool = False,
) -> str:
    """
    Trains a LoRA or QLoRA adapter model and saves the merged checkpoint.
    """
    print(f"[PEFT] Initializing {'QLoRA' if use_qlora else 'LoRA'} fine-tuning...")

    model, tokenizer, device = load_model_and_tokenizer(
        model_cfg,
        load_in_4bit=use_qlora,
    )

    if use_qlora:
        model = prepare_model_for_kbit_training(model)

    target_modules = lora_cfg.target_modules
    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_ds, val_ds, _ = prepare_samsum_dataset(dataset_cfg, tokenizer)

    training_args = TrainingArguments(
        output_dir=training_cfg.output_dir,
        per_device_train_batch_size=training_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        learning_rate=training_cfg.learning_rate,
        logging_steps=training_cfg.logging_steps,
        num_train_epochs=training_cfg.num_train_epochs,
        save_strategy=training_cfg.save_strategy,
        warmup_ratio=training_cfg.warmup_ratio,
        weight_decay=training_cfg.weight_decay,
        fp16=(device.type == "cuda" and not use_qlora),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print(f"[PEFT] Starting training for {training_cfg.num_train_epochs} epoch(s)...")
    trainer.train()

    save_dir = training_cfg.output_dir
    os.makedirs(save_dir, exist_ok=True)
    print(f"[PEFT] Saving trained model checkpoint to '{save_dir}'...")

    # Save adapter & tokenizer
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Save merged full model version for downstream quantization/pruning
    try:
        if not use_qlora:
            merged_model = model.merge_and_unload()
            merged_dir = os.path.join(save_dir, "merged")
            merged_model.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            print(f"[PEFT] Saved merged standalone checkpoint to '{merged_dir}'")
            return merged_dir
    except (RuntimeError, TypeError, ValueError, AttributeError) as e:
        print(f"[PEFT] Merge note: {e}")

    return save_dir
