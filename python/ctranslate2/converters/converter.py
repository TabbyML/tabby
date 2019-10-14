import abc
import filecmp
import inspect
import os
import shutil
import six

from ctranslate2.specs import catalog


def _list_specs():
    return {symbol:getattr(catalog, symbol) for symbol in dir(catalog)
            if inspect.isclass(getattr(catalog, symbol)) and not symbol.startswith("_")}


@six.add_metaclass(abc.ABCMeta)
class Converter(object):

    @staticmethod
    def declare_arguments(parser):
        parser.add_argument("--output_dir", required=True,
                            help="Output model directory.")
        parser.add_argument("--model_spec", required=True, choices=list(six.iterkeys(_list_specs())),
                            help="Type of model to convert.")
        parser.add_argument("--vocab_mapping", default=None,
                            help="Vocabulary mapping file (optional).")
        parser.add_argument("--quantization", default=None, choices=["int8", "int16"],
                            help="Weight quantization type.")
        parser.add_argument("--force", action="store_true",
                            help="Force conversion even if the output directory already exists.")
        return parser

    def convert_from_args(self, args):
        return self.convert(
            args.output_dir,
            args.model_spec,
            vmap=args.vocab_mapping,
            quantization=args.quantization,
            force=args.force)

    def convert(self, output_dir, model_spec, vmap=None, quantization=None, force=False):
        if os.path.exists(output_dir):
            if not force:
                raise RuntimeError(
                    "output directory %s already exists, use --force to override" % output_dir)
            else:
                shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        if isinstance(model_spec, six.string_types):
          spec_class = _list_specs()[model_spec]
          model_spec = spec_class()
        try:
            src_vocab, tgt_vocab = self._load(model_spec)
        except NotImplementedError:
            raise NotImplementedError("This converter does not support the model %s" % model_spec)
        model_spec.validate()
        if quantization is not None:
            model_spec.quantize(quantization)
        model_spec.serialize(os.path.join(output_dir, "model.bin"))
        if vmap is not None:
            shutil.copy(vmap, os.path.join(output_dir, "vmap.txt"))
        src_vocab_path = os.path.join(output_dir, "source_vocabulary.txt")
        tgt_vocab_path = os.path.join(output_dir, "target_vocabulary.txt")
        self._save_vocabulary(src_vocab, src_vocab_path)
        self._save_vocabulary(tgt_vocab, tgt_vocab_path)
        # For shared vocabularies, keep a single file in the model directory.
        if filecmp.cmp(src_vocab_path, tgt_vocab_path, shallow=False):
            os.remove(tgt_vocab_path)
            os.rename(src_vocab_path, os.path.join(output_dir, "shared_vocabulary.txt"))
        return output_dir

    @abc.abstractmethod
    def _load(self, model_spec):
        raise NotImplementedError()

    @abc.abstractmethod
    def _save_vocabulary(self, vocab, destination):
        raise NotImplementedError()
