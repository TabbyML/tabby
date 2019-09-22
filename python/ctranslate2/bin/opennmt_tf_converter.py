import argparse

from ctranslate2 import converters


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_path", required=True,
                        help="Model path (a checkpoint, a checkpoint directory, or a SavedModel).")
    parser.add_argument("--src_vocab", default=None,
                        help="Source vocabulary file (required if converting a checkpoint).")
    parser.add_argument("--tgt_vocab", default=None,
                        help="Target vocabulary file (required if converting a checkpoint).")
    converters.Converter.declare_arguments(parser)
    args = parser.parse_args()
    converters.OpenNMTTFConverter(
        args.model_path,
        src_vocab=args.src_vocab,
        tgt_vocab=args.tgt_vocab).convert_from_args(args)


if __name__ == '__main__':
    main()
