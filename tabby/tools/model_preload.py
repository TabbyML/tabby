from dataclasses import dataclass, field

from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser


@dataclass
class Arguments:
    repo_id: str = field(
        metadata={"help": "Huggingface model repository id, e.g TabbyML/NeoX-160M"}
    )
    backend: str = "python"
    prefer_local_files: bool = True


def parse_args():
    parser = HfArgumentParser(Arguments)
    return parser.parse_args()


def preload(local_files_only=False):
    AutoTokenizer.from_pretrained(args.repo_id, local_files_only=local_files_only)
    AutoModelForCausalLM.from_pretrained(
        args.repo_id, local_files_only=local_files_only
    )
    snapshot_download(
        repo_id=args.repo_id,
        allow_patterns="triton/**/*",
        local_files_only=local_files_only,
    )


if __name__ == "__main__":
    args = parse_args()
    print(f"Loading {args.repo_id} ...")
    try:
        preload(local_files_only=args.prefer_local_files)
    except Exception as e:
        if "offline" in str(e):
            preload(local_files_only=False)
        else:
            raise e
    print(f"Loaded {args.repo_id} !")
