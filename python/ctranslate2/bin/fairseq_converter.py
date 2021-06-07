import argparse

from ctranslate2 import converters


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_path", required=True, help="Model path.")
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Data directory containing the source and target vocabularies.",
    )
    converters.Converter.declare_arguments(parser)
    args = parser.parse_args()
    converters.FairseqConverter(args.model_path, args.data_dir).convert_from_args(args)


if __name__ == "__main__":
    main()
