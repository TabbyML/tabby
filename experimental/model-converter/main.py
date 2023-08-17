from args import make_parser
import json
import os
import shutil

from ctranslate2.converters.transformers import TransformersConverter
from huggingface_hub import snapshot_download
from transformers.convert_slow_tokenizers_checkpoints_to_fast import (
    convert_slow_checkpoint_to_fast,
)


class InvalidConvertionException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def convert_tokenizer():
    if os.path.exists("./tokenizer.json"):
        print("found tokenizer.json, skipping tokenizer conversion")
        return

    # Infer tokenizer name
    if not os.path.isfile("tokenizer_config.json"):
        raise InvalidConvertionException(
            "cannot find tokenizer_config.json, unable to infer tokenizer name"
        )

    data = {}
    with open("tokenizer_config.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    tokenizer_name = data["tokenizer_class"]

    convert_tmp_dir = "./convert_tmp"

    # Start to convert
    convert_slow_checkpoint_to_fast(
        tokenizer_name=tokenizer_name,
        checkpoint_name="./",
        dump_path=convert_tmp_dir,
        force_download=True,
    )

    # After successful conversion, copy file from ./convert_tmp to ./
    for root, dirs, files in os.walk(convert_tmp_dir):
        for f in files:
            fpath = os.path.join(root, f)
            shutil.copy2(fpath, "./")
        for d in dirs:
            dpath = os.path.join(root, d)
            shutil.copy2(dpath, "./")
    shutil.rmtree(convert_tmp_dir)


def generate_tabby_json(args):
    if os.path.exists("./tabby.json"):
        print("found tabby.json, skipping tabby.json generation")
        return

    data = {}
    data["auto_model"] = (
        "AutoModelForCausalLM"
        if args.inference_mode == "causallm"
        else "AutoModelForSeq2SeqLM"
    )
    if args.prompt_template:
        data["prompt_template"] = args.prompt_template
    with open("tabby.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def main():
    # Set up args
    parser = make_parser()

    args = parser.parse_args()

    # Check out model
    model_path = snapshot_download(
        repo_id=args.model,
        cache_dir=args.output_dir,
        force_download=False,
    )

    os.chdir(model_path)
    convert_output_dir = os.path.join(model_path, "ctranslate2")

    # Convert model into ctranslate
    converter = TransformersConverter(
        model_name_or_path=model_path,
        load_as_float16=True,
        trust_remote_code=True,
    )
    converter.convert(
        output_dir=convert_output_dir, vmap=None, quantization="float16", force=True
    )

    # Convert model with fast tokenizer
    convert_tokenizer()

    # Generate tabby.json
    generate_tabby_json(args)


if __name__ == "__main__":
    main()
