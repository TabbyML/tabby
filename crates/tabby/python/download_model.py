#!/usr/bin/env python3

from dataclasses import dataclass, field

from huggingface_hub import snapshot_download
from transformers import HfArgumentParser


@dataclass
class Arguments:
    repo_id: str = field(
        metadata={"help": "Huggingface model repository id, e.g TabbyML/NeoX-160M"}
    )
    device: str = field(metadata={"help": "Device type for inference: cpu / cuda"})
    output_dir: str = field(metadata={"help": "Output directory"})


def parse_args():
    parser = HfArgumentParser(Arguments)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Loading {args.repo_id}, this will take a while...")
    snapshot_download(
        local_dir=args.output_dir,
        repo_id=args.repo_id,
        allow_patterns=[f"ctranslate2/{args.device}/*", "tokenizer.json"],
    )
    print(f"Loaded {args.repo_id} !")
