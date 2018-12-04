from __future__ import print_function

import abc
import os
import shutil
import six
import struct

import numpy as np


_BINARY_VERSION = 1


@six.add_metaclass(abc.ABCMeta)
class Converter(object):

    @staticmethod
    def declare_arguments(parser):
        parser.add_argument("--output_dir", required=True,
                            help="Output model directory.")
        parser.add_argument("--model_type", default="transformer", choices=["transformer"],
                            help="Type of model to convert.")
        parser.add_argument("--vocab_mapping", default=None,
                            help="Vocabulary mapping file (optional).")
        parser.add_argument("--quantization", default=None, choices=["int16"],
                            help="Weight quantization type.")
        parser.add_argument("--force", action="store_true",
                            help="Force conversion even if the output directory already exists.")
        return parser

    def convert_from_args(self, args):
        return self.convert(
            args.output_dir,
            args.model_type,
            vmap=args.vocab_mapping,
            quantization=args.quantization,
            force=args.force)

    def convert(self, output_dir, model_type, vmap=None, quantization=None, force=False):
        if os.path.exists(output_dir):
            if not force:
                raise RuntimeError(
                    "output directory %s already exists, use --force to override" % output_dir)
            else:
                shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        spec, src_vocab, tgt_vocab = self._load(model_type)
        spec.validate()
        if quantization is not None:
            spec.quantize(quantization)
        if vmap is not None:
            shutil.copy(vmap, os.path.join(output_dir, "vmap.txt"))
        self._serialize_variables(output_dir, spec.variables())
        self._save_vocabulary(src_vocab, os.path.join(output_dir, "source_vocabulary.txt"))
        self._save_vocabulary(tgt_vocab, os.path.join(output_dir, "target_vocabulary.txt"))
        return output_dir

    @abc.abstractmethod
    def _load(self, model_type):
        raise NotImplementedError()

    @abc.abstractmethod
    def _save_vocabulary(self, vocab, destination):
        raise NotImplementedError()

    def _serialize_variables(self, output_dir, variables):
        with open(os.path.join(output_dir, "model.bin"), "wb") as model:
            model.write(struct.pack("I", _BINARY_VERSION))
            model.write(struct.pack("I", len(variables)))
            for name, value in six.iteritems(variables):
                print("Saving %s %s [%s]" % (name, value.shape, value.dtype))
                model.write(struct.pack("H", len(name) + 1))
                model.write(six.b(name))
                model.write(struct.pack('B', 0))
                model.write(struct.pack("B", len(value.shape)))
                for dim in value.shape:
                    model.write(struct.pack("I", dim))
                model.write(struct.pack("B", value.dtype.itemsize))
                model.write(struct.pack("I", value.size))
                model.write(value.tobytes())
