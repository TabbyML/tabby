import abc
import inspect
import os
import shutil

from ctranslate2.specs import catalog
from ctranslate2.specs.model_spec import ModelSpec


def _list_specs():
    return {
        symbol: getattr(catalog, symbol)
        for symbol in dir(catalog)
        if inspect.isclass(getattr(catalog, symbol)) and not symbol.startswith("_")
    }


class Converter(abc.ABC):
    @staticmethod
    def declare_arguments(parser):
        parser.add_argument(
            "--output_dir", required=True, help="Output model directory."
        )
        parser.add_argument(
            "--model_spec",
            required=True,
            choices=list(_list_specs().keys()),
            help="Type of model to convert.",
        )
        parser.add_argument(
            "--vocab_mapping", default=None, help="Vocabulary mapping file (optional)."
        )
        parser.add_argument(
            "--quantization",
            default=None,
            choices=["int8", "int16", "float16"],
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
            args.model_spec,
            vmap=args.vocab_mapping,
            quantization=args.quantization,
            force=args.force,
        )

    def convert(
        self, output_dir, model_spec, vmap=None, quantization=None, force=False
    ):
        if os.path.exists(output_dir) and not force:
            raise RuntimeError(
                "output directory %s already exists, use --force to override"
                % output_dir
            )
        if isinstance(model_spec, str):
            spec_class = _list_specs()[model_spec]
            model_spec = spec_class()
        if not isinstance(model_spec, ModelSpec):
            raise TypeError("model_spec should extend ctranslate2.specs.ModelSpec")
        try:
            self._load(model_spec)
        except NotImplementedError:
            raise NotImplementedError(
                "This converter does not support the model %s" % model_spec
            )
        model_spec.validate()
        model_spec.optimize(quantization=quantization)

        # Create model directory.
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        model_spec.save(output_dir)
        if vmap is not None:
            shutil.copy(vmap, os.path.join(output_dir, "vmap.txt"))
        return output_dir

    @abc.abstractmethod
    def _load(self, model_spec):
        raise NotImplementedError()
