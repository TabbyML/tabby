use std::{collections::HashMap, fs};

use anyhow::Result;
use lazy_static::lazy_static;
use tabby_common::{config::Config, path::index_dir, SourceFile};
use tantivy::{
    directory::MmapDirectory,
    doc,
    schema::{Schema, STORED, STRING, TEXT},
    Index,
};

pub fn index_repositories(_config: &Config) -> Result<()> {
    let mut builder = Schema::builder();

    let field_git_url = builder.add_text_field("git_url", STRING | STORED);
    let field_filepath = builder.add_text_field("filepath", STRING | STORED);
    let field_language = builder.add_text_field("language", STRING | STORED);
    let field_name = builder.add_text_field("name", STRING | STORED);
    let field_kind = builder.add_text_field("kind", STRING | STORED);
    let field_body = builder.add_text_field("body", TEXT | STORED);

    let schema = builder.build();

    fs::create_dir_all(index_dir())?;
    let directory = MmapDirectory::open(index_dir())?;
    let index = Index::open_or_create(directory, schema)?;
    let mut writer = index.writer(10_000_000)?;
    writer.delete_all_documents()?;

    for file in SourceFile::all()? {
        for doc in from_source_file(file) {
            writer.add_document(doc!(
                    field_git_url => doc.git_url,
                    field_filepath => doc.filepath,
                    field_language => doc.language,
                    field_name => doc.name,
                    field_body => doc.body,
                    field_kind => doc.kind,
            ))?;
        }
    }

    writer.commit()?;

    Ok(())
}

/// Atomic repository document in index.
struct IndexedDocument {
    git_url: String,
    filepath: String,
    language: String,
    name: String,
    body: String,
    kind: String,
}

fn from_source_file(file: SourceFile) -> impl Iterator<Item = IndexedDocument> {
    file.tags.into_iter().map(move |tag| {
        let name = file.content.get(tag.name_range).unwrap().to_owned();
        let body = file.content.get(tag.range).unwrap().to_owned();

        let language = reduce_language_if_needed(&file.language).to_owned();
        IndexedDocument {
            git_url: file.git_url.clone(),
            filepath: file.filepath.clone(),
            language,
            name,
            body,
            kind: tag.syntax_type_name,
        }
    })
}

fn reduce_language_if_needed(language: &str) -> &str {
    if ["javascript", "jsx", "typescript", "tsx"].contains(&language) {
        "javascript-typescript"
    } else {
        language
    }
}

lazy_static! {
    static ref LANGUAGE_NAME_BLACKLIST: HashMap<&'static str, Vec<&'static str>> =
        HashMap::from([("python", vec!["__init__"])]);
}

#[cfg(test)]
mod tests {
    use serde_json::{from_value, json};

    use super::*;

    fn test_source_file() -> SourceFile {
        from_value(json!(
            {
                "git_url": "https://fake.com/tabbyml.git",
                "filepath": "python/tabby/trainer.py",
                "content": "import os\nimport glob\nfrom dataclasses import dataclass, field\nfrom typing import List\n\nimport peft\nimport torch\nfrom transformers import (\n    AutoModelForCausalLM,\n    AutoTokenizer,\n    HfArgumentParser,\n    Trainer,\n    TrainingArguments,\n)\nfrom datasets import Dataset, load_dataset\n\n\nclass ConstantLengthDataset:\n    \"\"\"\n    Iterable dataset that returns constant length chunks of tokens from stream of text files.\n        Args:\n            tokenizer (Tokenizer): The processor used for proccessing the data.\n            dataset (dataset.Dataset): Dataset with text files.\n            infinite (bool): If True the iterator is reset after dataset reaches end else stops.\n            seq_length (int): Length of token sequences to return.\n            num_of_sequences (int): Number of token sequences to keep in buffer.\n            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.\n    \"\"\"\n\n    def __init__(\n        self,\n        tokenizer,\n        dataset,\n        infinite=False,\n        seq_length=1024,\n        num_of_sequences=1024,\n        chars_per_token=3.6,\n        content_field=\"content\",\n    ):\n        self.tokenizer = tokenizer\n        self.concat_token_id = tokenizer.eos_token_id\n        self.dataset = dataset\n        self.seq_length = seq_length\n        self.infinite = infinite\n        self.current_size = 0\n        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences\n        self.content_field = content_field\n\n    def __call__(self):\n        def gen():\n            for x in self:\n                yield x\n\n        return gen()\n\n    def __iter__(self):\n        for buffer in self._read_dataset_into_buffer():\n            yield from self._tokenize(buffer)\n\n    def _tokenize(self, buffer):\n        tokenized_inputs = self.tokenizer(buffer, truncation=False)[\"input_ids\"]\n\n        all_token_ids = []\n        for tokenized_input in tokenized_inputs:\n            all_token_ids.extend(tokenized_input + [self.concat_token_id])\n\n        for i in range(0, len(all_token_ids), self.seq_length):\n            input_ids = all_token_ids[i : i + self.seq_length]\n\n            if len(input_ids) < self.seq_length:\n                input_ids = all_token_ids[-self.seq_length :]\n\n            if len(input_ids) == self.seq_length:\n                self.current_size += 1\n                yield dict(input_ids=input_ids, labels=input_ids)\n\n    def _read_dataset_into_buffer(self):\n        iterator = iter(self.dataset)\n        more_examples = True\n        while more_examples:\n            buffer, buffer_len = [], 0\n            while True:\n                if buffer_len >= self.max_buffer_size:\n                    break\n                try:\n                    buffer.append(next(iterator)[self.content_field])\n                    buffer_len += len(buffer[-1])\n                except StopIteration:\n                    if self.infinite:\n                        iterator = iter(self.dataset)\n                    else:\n                        more_examples = False\n                        break\n            yield buffer\n\n\n@dataclass\nclass TrainLoraArguments:\n    data_path: str = field(metadata={\"help\": \"Dataset dir for training / eval \"})\n    output_dir: str = field(metadata={\"help\": \"Output dir for checkpoint\"})\n    base_model: str = field(\n        default=\"TabbyML/J-350M\", metadata={\"help\": \"Base model for fine-tuning\"}\n    )\n\n    batch_size: int = 128\n    micro_batch_size: int = 4\n    num_epochs: int = 3\n    learning_rate: float = 3e-4\n    cutoff_len: int = 256\n\n    # Evaluations\n    val_set_size: int = 2000\n    eval_steps: int = 200\n\n    # Lora Hyperparams\n    lora_r: int = 8\n    lora_alpha: int = 16\n    lora_dropout: float = 0.05\n    lora_target_modules: List[str] = (\n        [\n            \"q_proj\",\n            \"v_proj\",\n        ],\n    )\n    resume_from_checkpoint: str = None  # either training checkpoint or final adapter\n    half: bool = True\n\n\ndef parse_args() -> TrainLoraArguments:\n    parser = HfArgumentParser(TrainLoraArguments)\n    return parser.parse_args()\n\n\ndef train(args: TrainLoraArguments):\n    gradient_accumulation_steps = args.batch_size // args.micro_batch_size\n\n    model = AutoModelForCausalLM.from_pretrained(\n        args.base_model, torch_dtype=torch.float16 if args.half else torch.float32\n    )\n\n    tokenizer = AutoTokenizer.from_pretrained(args.base_model)\n\n    config = peft.LoraConfig(\n        r=args.lora_r,\n        lora_alpha=args.lora_alpha,\n        target_modules=args.lora_target_modules,\n        lora_dropout=args.lora_dropout,\n        bias=\"none\",\n        task_type=peft.TaskType.CAUSAL_LM,\n    )\n    model = peft.get_peft_model(model, config)\n\n    data_files = glob.glob(os.path.join(args.data_path, \"*.jsonl\"))\n    print(\"Collected data files...\", data_files)\n    dataset = load_dataset(\"json\", data_files=data_files)[\"train\"]\n    data = Dataset.from_generator(ConstantLengthDataset(tokenizer, dataset))\n\n    resume_from_checkpoint = args.resume_from_checkpoint\n    if resume_from_checkpoint:\n        # Check the available weights and load them\n        checkpoint_name = os.path.join(\n            resume_from_checkpoint, \"pytorch_model.bin\"\n        )  # Full checkpoint\n        if not os.path.exists(checkpoint_name):\n            checkpoint_name = os.path.join(\n                resume_from_checkpoint, \"adapter_model.bin\"\n            )  # only LoRA model - LoRA config above has to fit\n            resume_from_checkpoint = False  # So the trainer won't try loading its state\n        # The two files above have a different name depending on how they were saved, but are actually the same.\n        if os.path.exists(checkpoint_name):\n            print(f\"Restarting from {checkpoint_name}\")\n            adapters_weights = torch.load(checkpoint_name)\n            model = peft.set_peft_model_state_dict(model, adapters_weights)\n        else:\n            print(f\"Checkpoint {checkpoint_name} not found\")\n\n    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.\n\n    train_val = data.train_test_split(\n        test_size=args.val_set_size, shuffle=True, seed=42\n    )\n    train_data = train_val[\"train\"].shuffle()\n    val_data = train_val[\"test\"].shuffle()\n\n    trainer = Trainer(\n        model=model,\n        train_dataset=train_data,\n        eval_dataset=val_data,\n        args=TrainingArguments(\n            per_device_train_batch_size=args.micro_batch_size,\n            gradient_accumulation_steps=gradient_accumulation_steps,\n            warmup_steps=100,\n            num_train_epochs=args.num_epochs,\n            learning_rate=args.learning_rate,\n            fp16=args.half,\n            logging_steps=10,\n            evaluation_strategy=\"steps\",\n            save_strategy=\"steps\",\n            eval_steps=args.eval_steps,\n            save_steps=args.eval_steps,\n            output_dir=args.output_dir,\n            save_total_limit=3,\n            load_best_model_at_end=True,\n        ),\n    )\n    model.config.use_cache = False\n\n    old_state_dict = model.state_dict\n    model.state_dict = (\n        lambda self, *_, **__: peft.get_peft_model_state_dict(self, old_state_dict())\n    ).__get__(model, type(model))\n\n    model = torch.compile(model)\n\n    trainer.train(resume_from_checkpoint=resume_from_checkpoint)\n\n    model.save_pretrained(args.output_dir)\n\n    print(\"\\n If there's a warning about missing keys above, please disregard :)\")\n\n\nif __name__ == \"__main__\":\n    args = parse_args()\n    train(args)\n",
                "language": "python",
                "max_line_length": 115,
                "avg_line_length": 32.388393,
                "alphanum_fraction": 0.6066319,
                "tags": [
                  {
                    "range": {
                      "start": 290,
                      "end": 3094
                    },
                    "name_range": {
                      "start": 296,
                      "end": 317
                    },
                    "line_range": {
                      "start": 290,
                      "end": 318
                    },
                    "is_definition": true,
                    "syntax_type_name": "class"
                  },
                  {
                    "range": {
                      "start": 953,
                      "end": 1507
                    },
                    "name_range": {
                      "start": 957,
                      "end": 965
                    },
                    "line_range": {
                      "start": 953,
                      "end": 966
                    },
                    "is_definition": true,
                    "syntax_type_name": "function"
                  },
                ]
              })).unwrap()
    }

    #[test]
    fn it_create_documents() {
        let source_file: SourceFile = test_source_file();
        let docs: Vec<_> = from_source_file(source_file).collect();
        assert_eq!(docs.len(), 2);

        assert_eq!(docs[0].name, "ConstantLengthDataset");
        assert_eq!(docs[0].kind, "class");

        assert_eq!(docs[1].name, "__init__");
        assert_eq!(docs[1].kind, "function");
    }
}
