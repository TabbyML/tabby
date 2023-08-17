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

    Converter.declare_arguments(parser)
    return parser