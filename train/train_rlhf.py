from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DPOTrainer



@dataclass
class DPOConfig:
    model_name_or_path: str = "mistralai/Mistral-7B-Instruct-v0.2"

    ref_model_name_or_path: str = "mistralai/Mistral-7B-Instruct-v0.2"

    train_file: str = "dpo_wiki_pairs.jsonl"
    val_file: str | None = None

    output_dir: str = "./dpo-wiki-mistral"

    max_prompt_length: int = 2048
    max_target_length: int = 2048

    use_4bit: bool = True

    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 3
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0

    logging_steps: int = 10
    save_strategy: str = "epoch"
    eval_strategy: str = "no"

    bf16: bool = True
    seed: int = 42

    beta: float = 0.1


cfg = DPOConfig()


# ---------------- Prompt  ---------------- #

def build_dpo_example(example: Dict) -> Dict:

    user_prompt = example["prompt"]
    chosen = example["chosen"]
    rejected = example["rejected"]

    def wrap(p: str, a: str) -> str:
        return (
            "<s>[INST] You are a wiki page generator specialized in code repositories.\n\n"
            f"{p}\n"
            "[/INST]"
            f"{a}</s>"
        )

    chosen_full = wrap(user_prompt, chosen)
    rejected_full = wrap(user_prompt, rejected)

    return {
        "prompt": user_prompt,
        "chosen": chosen_full,
        "rejected": rejected_full,
    }


def main():
    print("Loading DPO dataset...")
    data_files = {"train": cfg.train_file}
    if cfg.val_file:
        data_files["validation"] = cfg.val_file

    raw_datasets = load_dataset("json", data_files=data_files)

    dpo_datasets = raw_datasets.map(
        build_dpo_example,
        remove_columns=raw_datasets["train"].column_names,
    )

    # tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if cfg.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    print("Loading policy model...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        trust_remote_code=True,
    )

    print("Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        cfg.ref_model_name_or_path,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        trust_remote_code=True,
    )

    peft_config = None

    # TrainingArguments
    print("Preparing TrainingArguments...")
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        evaluation_strategy=cfg.eval_strategy,
        bf16=cfg.bf16,
        lr_scheduler_type="cosine",
        report_to="none",
        seed=cfg.seed,
    )

    # DPOTrainer
    print("Initializing DPOTrainer...")
    dpo_trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=training_args,
        beta=cfg.beta,
        train_dataset=dpo_datasets["train"],
        eval_dataset=dpo_datasets.get("validation"),
        tokenizer=tokenizer,
        max_length=cfg.max_prompt_length + cfg.max_target_length,
        max_target_length=cfg.max_target_length,
        max_prompt_length=cfg.max_prompt_length,
        peft_config=peft_config,
    )


    print("Starting DPO training...")
    dpo_trainer.train()

    print("Saving DPO-tuned model...")
    dpo_trainer.model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    print("Done. DPO model saved at:", cfg.output_dir)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()