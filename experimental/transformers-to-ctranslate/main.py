import os
from args import make_parser

from dotenv import load_dotenv
from ctranslate2.converters.transformers import TransformersConverter
from huggingface_hub import snapshot_download
from transformers.convert_slow_tokenizers_checkpoints_to_fast import convert_slow_checkpoint_to_fast
from huggingface_hub import login, HfApi

if __name__ == "__main__":
    # Set up args
    parser = make_parser()

    args = parser.parse_args()

    # Load env
    load_dotenv()

    # Check out model
    model_path = snapshot_download(
        repo_id=args.model,
        cache_dir=args.output_dir,
        force_download=True
    )
    convert_output_dir = os.path.join(model_path, "ctranslate2")

    # Convert model into ctranslate
    converter = TransformersConverter(
        args.model,
        activation_scales=args.activation_scales,
        copy_files=args.copy_files,
        load_as_float16=args.quantization in ("float16", "int8_float16"),
        revision=args.revision,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        trust_remote_code=args.trust_remote_code,
    )
    converter.convert(
        output_dir=convert_output_dir,
        vmap=None,
        quantization=args.quantization,
        force=True
    )

    # Convert model with fast tokenizer
    os.chdir(model_path)
    convert_slow_checkpoint_to_fast(
        tokenizer_name=args.tokenizer_name,
        checkpoint_name='./',
        dump_path='./',
        force_download=False,
    )

    # Upload folder to huggingface
    token = os.getenv("HF_TOKEN")
    login(token)
    api = HfApi()
    api.upload_folder(
        folder_path=model_path,
        repo_id=args.model,
        repo_type='model',
    )