import argparse

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
        "--output_dir",
        required=True,
        help="Output model directory."
    )
    parser.add_argument(
        "--inference_mode",
        required=True,
        choices=["causallm", "seq2seq"],
        help="Model inference mode. ",
    )
    parser.add_argument(
        "--prompt_template",
        default=None,
        help="prompt template for fim"
    )

    return parser