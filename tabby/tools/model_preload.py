from dataclasses import dataclass, field

from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser


@dataclass
class Arguments:
    repo_id: str = field(
        metadata={"help": "Huggingface model repository id, e.g TabbyML/NeoX-160M"}
    )
    backend: str = "python"


def parse_args():
    parser = HfArgumentParser(Arguments)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Loading {args.repo_id} ...")

    AutoTokenizer.from_pretrained(args.repo_id)
    print("Tokenizer done")

    AutoModelForCausalLM.from_pretrained(args.repo_id)
    print("Model done")

    snapshot_download(repo_id=args.repo_id, allow_patterns="triton/**/*")
    print("triton done")
