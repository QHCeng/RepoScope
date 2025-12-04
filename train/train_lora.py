from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer


# ------------- Config ------------- #

@dataclass
class TrainConfig:
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    train_file: str = "train_wiki.jsonl"
    val_file: str | None = None

    output_dir: str = "./lora-wiki-mistral"
    max_seq_length: int = 4096
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03

    # LoRA config
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules = ("q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj")  # 对 Mistral/LLaMA 系列比较合适

    use_4bit: bool = True


cfg = TrainConfig()


# ------------- Prompt  ------------- #

def build_chat_prompt(example: Dict) -> str:
    user = example["input"]
    assistant = example["output"]

    prompt = (
        "<s>[INST] You are a wiki page generator specialized for code repositories.\n\n"
        f"{user}\n[/INST]"
        f"{assistant}</s>"
    )
    return prompt



def main():
    print("Loading dataset...")
    data_files = {"train": cfg.train_file}
    if cfg.val_file:
        data_files["validation"] = cfg.val_file

    raw_datasets = load_dataset("json", data_files=data_files)

    def map_to_text(example):
        return {"text": build_chat_prompt(example)}

    tokenized_datasets = raw_datasets.map(
        map_to_text,
        remove_columns=raw_datasets["train"].column_names,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4bit
    quant_config = None
    if cfg.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        device_map="auto",
        quantization_config=quant_config,
        trust_remote_code=True,
    )

    # ----- LoRA -----
    print("Configuring LoRA...")
    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ----- SFTTrainer -----
    print("Initializing trainer...")

    training_args = dict(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.micro_batch_size,
        per_device_eval_batch_size=cfg.micro_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no" if "validation" not in tokenized_datasets else "epoch",
        bf16=True,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        peft_config=peft_config,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation"),
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        packing=True,
        **training_args,
    )

    print("Start training...")
    trainer.train()
    
    print("Saving LoRA adapter...")
    trainer.model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    print("Done. LoRA weights saved at:", cfg.output_dir)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
