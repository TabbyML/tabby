import argparse
from ctranslate2.converters.converter import Converter

def make_parser():    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Name of the pretrained model to download, "
            "or path to a directory containing the pretrained model."
        ),
    )
    parser.add_argument(
        "--activation_scales",
        help=(
            "Path to the pre-computed activation scales. Models may "
            "use them to rescale some weights to smooth the intermediate activations "
            "and improve the quantization accuracy. See "
            "https://github.com/mit-han-lab/smoothquant."
        ),
    )
    parser.add_argument(
        "--copy_files",
        nargs="+",
        help=(
            "List of filenames to copy from the Hugging Face model to the converted "
            "model directory."
        ),
    )
    parser.add_argument(
        "--revision",
        help="Revision of the model to download from the Hugging Face Hub.",
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help="Enable the flag low_cpu_mem_usage when loading the model with from_pretrained.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow converting models using custom code.",
    )

    parser.add_argument(
        "--tokenizer_name",
        required=True,
        help="Tokenizer name of the model.",
    )

    Converter.declare_arguments(parser)
    return parser