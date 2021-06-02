import argparse
import opennmt

from ctranslate2 import converters


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Model path (a checkpoint or a checkpoint directory).",
    )
    parser.add_argument(
        "--model_type",
        required=True,
        help="The model type used in OpenNMT-tf training.",
    )
    parser.add_argument(
        "--src_vocab",
        required=True,
        help="Source vocabulary file.",
    )
    parser.add_argument(
        "--tgt_vocab",
        required=True,
        help="Target vocabulary file.",
    )
    converters.Converter.declare_arguments(parser)
    args = parser.parse_args()
    model = opennmt.models.get_model_from_catalog(args.model_type)
    model_spec = model.ctranslate2_spec
    if model_spec is None:
        raise NotImplementedError("Model %s is not supported" % args.model_type)
    converters.OpenNMTTFConverter(
        model_spec,
        args.src_vocab,
        args.tgt_vocab,
        model_path=args.model_path,
    ).convert_from_args(args)


if __name__ == "__main__":
    main()
