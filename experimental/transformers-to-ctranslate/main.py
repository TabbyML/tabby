from args import make_parser
import json
import os

from ctranslate2.converters.transformers import TransformersConverter
from huggingface_hub import snapshot_download
from transformers.convert_slow_tokenizers_checkpoints_to_fast import convert_slow_checkpoint_to_fast

class InvalidConvertionException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

def convert_tokenizer():
    # Infer tokenizer name
    if not os.path.isfile('tokenizer_config.json'):
        raise InvalidConvertionException("cannot find tokenizer_config.json, unable to infer tokenizer name")

    data = {}
    with open('tokenizer_config.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    tokenizer_name = data["tokenizer_class"]

    # Start to convert
    convert_slow_checkpoint_to_fast(
        tokenizer_name=tokenizer_name,
        checkpoint_name='./',
        dump_path='./',
        force_download=True,
    )

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
    convert_output_dir = os.path.join(model_path, "ctranslate2")

    # Convert model into ctranslate
    converter = TransformersConverter(
        model_name_or_path=args.model,
        load_as_float16=args.quantization in ("float16", "int8_float16"),
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )
    converter.convert(
        output_dir=convert_output_dir,
        vmap=None,
        quantization=args.quantization,
        force=True
    )

    # Convert model with fast tokenizer
    if not os.path.exists(os.path.join(model_path, "tokenizer2.json")):
        os.chdir(model_path)
        convert_tokenizer()
    else:
        print("found tokenizer.json, skipping tokenizer conversion")

if __name__ == "__main__":
    main()
