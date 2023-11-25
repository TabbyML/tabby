import os
import glob
from dataclasses import dataclass, field
from typing import List

import peft
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset, load_dataset


class ConstantLengthDataset:
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        content_field="content",
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.content_field = content_field

    def __call__(self):
        def gen():
            for x in self:
                yield x

        return gen()

    def __iter__(self):
        for buffer in self._read_dataset_into_buffer():
            yield from self._tokenize(buffer)

    def _tokenize(self, buffer):
        tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]

        all_token_ids = []
        for tokenized_input in tokenized_inputs:
            all_token_ids.extend(tokenized_input + [self.concat_token_id])

        for i in range(0, len(all_token_ids), self.seq_length):
            input_ids = all_token_ids[i : i + self.seq_length]

            if len(input_ids) < self.seq_length:
                input_ids = all_token_ids[-self.seq_length :]

            if len(input_ids) == self.seq_length:
                self.current_size += 1
                yield dict(input_ids=input_ids, labels=input_ids)

    def _read_dataset_into_buffer(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)[self.content_field])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            yield buffer


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
    half: bool = True


def parse_args() -> TrainLoraArguments:
    parser = HfArgumentParser(TrainLoraArguments)
    return parser.parse_args()


def train(args: TrainLoraArguments):
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float16 if args.half else torch.float32
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

    data_files = glob.glob(os.path.join(args.data_path, "*.jsonl"))
    print("Collected data files...", data_files)
    dataset = load_dataset("json", data_files=data_files)["train"]
    data = Dataset.from_generator(ConstantLengthDataset(tokenizer, dataset))

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

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            fp16=args.half,
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
