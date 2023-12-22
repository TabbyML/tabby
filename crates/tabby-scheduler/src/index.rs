use std::{fs, io::IsTerminal};

use anyhow::Result;
use kdam::BarExt;
use tabby_common::{
    config::Config,
    index::{register_tokenizers, CodeSearchSchema},
    path::index_dir,
    SourceFile,
};
use tantivy::{directory::MmapDirectory, doc, Index};

use crate::utils::tqdm;

// Magic numbers
static MAX_LINE_LENGTH_THRESHOLD: usize = 300;
static AVG_LINE_LENGTH_THRESHOLD: f32 = 150f32;
static MAX_BODY_LINES_THRESHOLD: usize = 15;

pub fn index_repositories(_config: &Config) -> Result<()> {
    let code = CodeSearchSchema::new();

    fs::create_dir_all(index_dir())?;
    let directory = MmapDirectory::open(index_dir())?;
    let index = Index::open_or_create(directory, code.schema)?;
    register_tokenizers(&index);

    // Initialize the search index writer with an initial arena size of 150 MB.
    let mut writer = index.writer(150_000_000)?;
    writer.delete_all_documents()?;

    let mut pb = std::io::stdout()
        .is_terminal()
        .then(SourceFile::all)
        .transpose()?
        .map(|iter| tqdm(iter.count()));
    for file in SourceFile::all()? {
        pb.as_mut().map(|b| b.update(1)).transpose()?;

        if file.max_line_length > MAX_LINE_LENGTH_THRESHOLD {
            continue;
        }

        if file.avg_line_length > AVG_LINE_LENGTH_THRESHOLD {
            continue;
        }

        for doc in from_source_file(file) {
            writer.add_document(doc!(
                    code.field_git_url => doc.git_url,
                    code.field_filepath => doc.filepath,
                    code.field_language => doc.language,
                    code.field_name => doc.name,
                    code.field_body => doc.body,
                    code.field_kind => doc.kind,
            ))?;
        }
    }

    writer.commit()?;
    writer.wait_merging_threads()?;

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
    file.tags.into_iter().filter_map(move |tag| {
        let name = file.content.get(tag.name_range).unwrap().to_owned();
        let body = file.content.get(tag.range).unwrap().to_owned();

        if body.lines().collect::<Vec<_>>().len() > MAX_BODY_LINES_THRESHOLD {
            return None;
        }

        Some(IndexedDocument {
            git_url: file.git_url.clone(),
            filepath: file.filepath.clone(),
            language: file.language.clone(),
            name,
            body,
            kind: tag.syntax_type_name,
        })
    })
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
                "content": "import os\nimport glob\nfrom dataclasses import dataclass, field\nfrom typing import List\n\nimport peft\nimport torch\nfrom transformers import (\n    AutoModelForCausalLM,\n    AutoTokenizer,\n    HfArgumentParser,\n    Trainer,\n    TrainingArguments,\n)\nfrom datasets import Dataset, load_dataset\n\n\nclass ConstantLengthDataset:\n    \"\"\"\n    Iterable dataset that returns constant length chunks of tokens from stream of text files.\n        Args:\n            tokenizer (Tokenizer): The processor used for proccessing the data.\n            dataset (dataset.Dataset): Dataset with text files.\n            infinite (bool): If True the iterator is reset after dataset reaches end else stops.\n            seq_length (int): Length of token sequences to return.\n            num_of_sequences (int): Number of token sequences to keep in buffer.\n            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.\n    \"\"\"\n\n    def __init__(\n        self,\n        tokenizer,\n        dataset,\n        infinite=False,\n        seq_length=1024,\n        num_of_sequences=1024,\n        chars_per_token=3.6,\n        content_field=\"content\",\n    ):\n        self.tokenizer = tokenizer\n        self.concat_token_id = tokenizer.eos_token_id\n        self.dataset = dataset\n        self.seq_length = seq_length\n        self.infinite = infinite\n        self.current_size = 0\n        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences\n        self.content_field = content_field\n\n    def __call__(self):\n        def gen():\n            for x in self:\n                yield x\n\n        return gen()\n\n    def __iter__(self):\n        for buffer in self._read_dataset_into_buffer():\n            yield from self._tokenize(buffer)\n\n    def _tokenize(self, buffer):\n        tokenized_inputs = self.tokenizer(buffer, truncation=False)[\"input_ids\"]\n\n        all_token_ids = []\n        for tokenized_input in tokenized_inputs:\n            all_token_ids.extend(tokenized_input + [self.concat_token_id])\n\n        for i in range(0, len(all_token_ids), self.seq_length):\n            input_ids = all_token_ids[i : i + self.seq_length]\n\n            if len(input_ids) < self.seq_length:\n                input_ids = all_token_ids[-self.seq_length :]\n\n            if len(input_ids) == self.seq_length:\n                self.current_size += 1\n                yield dict(input_ids=input_ids, labels=input_ids)\n\n    def _read_dataset_into_buffer(self):\n        iterator = iter(self.dataset)\n        more_examples = True\n        while more_examples:\n            buffer, buffer_len = [], 0\n            while True:\n                if buffer_len >= self.max_buffer_size:\n                    break\n                try:\n                    buffer.append(next(iterator)[self.content_field])\n                    buffer_len += len(buffer[-1])\n                except StopIteration:\n                    if self.infinite:\n                        iterator = iter(self.dataset)\n                    else:\n                        more_examples = False\n                        break\n            yield buffer\n\n\n",
                "language": "python",
                "max_line_length": 115,
                "avg_line_length": 32.388393,
                "alphanum_fraction": 0.6066319,
                "tags": [
                  {
                    "range": {
                      "start": 290,
                      "end": 320
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
                      "end": 970
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
        assert!(
            docs[0].body.starts_with("class ConstantLengthDataset"),
            "body: {:?}",
            docs[0].body
        );

        assert_eq!(docs[1].name, "__init__");
        assert_eq!(docs[1].kind, "function");
        assert!(
            docs[1].body.starts_with("def __init__"),
            "body: {:?}",
            docs[1].body
        );
    }
}
