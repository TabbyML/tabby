import argparse

from ctranslate2 import converters


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_path", required=True, help="Model path.")
    converters.Converter.declare_arguments(parser)
    args = parser.parse_args()
    converters.OpenNMTPyConverter(args.model_path).convert_from_args(args)


if __name__ == '__main__':
    main()
