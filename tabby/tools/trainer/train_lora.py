import os
from dataclasses import dataclass, field
from typing import List

import peft
import torch
import torch.nn as nn
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from .dataset import load_dataset


@dataclass
class TrainLoraArguments:
    data_path: str = field(metadata={"help": "Dataset dir for training / eval "})
    output_dir: str = field(metadata={"help": "Output dir for checkpoint"})
    base_model: str = field(
        default="TabbyML/J-350M", metadata={"help": "Base model for fine-tuning"}
    )

    batch_size: int = 128
    micro_batch_size: int = 4
    num_epochs: int = 3
    learning_rate: float = 3e-4
    cutoff_len: int = 256

    # Evaluations
    val_set_size: int = 2000
    eval_steps: int = 200

    # Lora Hyperparams
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = (
        [
            "q_proj",
            "v_proj",
        ],
    )
    resume_from_checkpoint: str = None  # either training checkpoint or final adapter


def parse_args() -> TrainLoraArguments:
    parser = HfArgumentParser(TrainLoraArguments)
    return parser.parse_args()


def train(args: TrainLoraArguments):
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    config = peft.LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=peft.TaskType.CAUSAL_LM,
    )
    model = peft.get_peft_model(model, config)

    data = load_dataset(tokenizer, args.data_path, seq_length=args.cutoff_len)

    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = False  # So the trainer won't try loading its state
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = peft.set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    train_val = data.train_test_split(
        test_size=args.val_set_size, shuffle=True, seed=42
    )
    train_data = train_val["train"].shuffle()
    val_data = train_val["test"].shuffle()

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            fp16=True,
            logging_steps=10,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=args.eval_steps,
            save_steps=args.eval_steps,
            output_dir=args.output_dir,
            save_total_limit=3,
            load_best_model_at_end=True,
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: peft.get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(args.output_dir)

    print("\n If there's a warning about missing keys above, please disregard :)")


if __name__ == "__main__":
    args = parse_args()
    train(args)
