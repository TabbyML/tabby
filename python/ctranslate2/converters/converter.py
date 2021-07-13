import abc
import os
import shutil

from ctranslate2.specs.model_spec import ModelSpec


class Converter(abc.ABC):
    @staticmethod
    def declare_arguments(parser):
        parser.add_argument(
            "--output_dir", required=True, help="Output model directory."
        )
        parser.add_argument(
            "--vocab_mapping", default=None, help="Vocabulary mapping file (optional)."
        )
        parser.add_argument(
            "--quantization",
            default=None,
            choices=["int8", "int8_float16", "int16", "float16"],
            help="Weight quantization type.",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Force conversion even if the output directory already exists.",
        )
        return parser

    def convert_from_args(self, args):
        return self.convert(
            args.output_dir,
            vmap=args.vocab_mapping,
            quantization=args.quantization,
            force=args.force,
        )

    def convert(self, output_dir, vmap=None, quantization=None, force=False):
        if os.path.exists(output_dir) and not force:
            raise RuntimeError(
                "output directory %s already exists, use --force to override"
                % output_dir
            )

        model_spec = self._load()
        if model_spec is None:
            raise NotImplementedError(
                "This model is not supported by CTranslate2 or this converter"
            )
        if vmap is not None:
            model_spec.register_vocabulary_mapping(vmap)

        model_spec.validate()
        model_spec.optimize(quantization=quantization)

        # Create model directory.
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        model_spec.save(output_dir)
        return output_dir

    @abc.abstractmethod
    def _load(self):
        raise NotImplementedError()
