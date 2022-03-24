"""Specifications declare the expected variables layout of CTranslate2 models
that do not load a computation graph. The model converter should make sure that
each required variable of the specification is set.
"""

import os
import shutil
import struct
import numpy as np

OPTIONAL = "__optional"
CURRENT_BINARY_VERSION = 5


def _join_scope(scope, name):
    if not scope:
        return name
    return "%s/%s" % (scope, name)


def _split_scope(scope):
    return scope.split("/")


def _parent_scope(scope):
    keys = _split_scope(scope)
    scope, attr = keys[:-1], keys[-1]
    return "/".join(scope), attr


def visit_spec(spec, fn, scope=""):
    """Recursively visits a layer spec."""
    for name, value in list(spec.__dict__.items()):
        if name.startswith("_"):
            continue
        if isinstance(value, list):
            for i, elem in enumerate(value):
                visit_spec(elem, fn, scope=_join_scope(scope, "%s_%d" % (name, i)))
        elif isinstance(value, LayerSpec):
            visit_spec(value, fn, scope=_join_scope(scope, name))
        else:
            fn(spec, _join_scope(scope, name), value)


def index_spec(spec, index):
    if not index:
        return spec
    keys = _split_scope(index)
    for key in keys:
        try:
            spec = getattr(spec, key)
        except AttributeError:
            attr, index = key.rsplit("_", 1)
            spec = getattr(spec, attr)[int(index)]
    return spec


class LayerSpec(object):
    """Layer specification."""

    def validate(self):
        """Checks that required variables are set to a valid value."""

        def _check(spec, name, value):
            if value is None:
                raise ValueError("Missing value for attribute %s" % name)

            if isinstance(value, np.ndarray):
                # Use float32 as the working floating point type.
                if value.dtype in (np.float16, np.float64):
                    value = value.astype(np.float32)
            elif isinstance(value, float):
                value = np.dtype("float32").type(value)
            elif isinstance(value, bool):
                # Convert bool to an integer type.
                value = np.dtype("int8").type(value)
            elif isinstance(value, str):
                if value != OPTIONAL:
                    value = np.frombuffer(value.encode("utf-8"), dtype=np.int8)

            attr_name = _split_scope(name)[-1]
            setattr(spec, attr_name, value)

        self.visit(_check)

    def variables(self, prefix="", ordered=False):
        """Returns a dict mapping variables name to value. If ordered is True,
        returns an ordered list of (name, value) pairs instead.
        """
        var = {}

        def _register_var(spec, name, value):
            if isinstance(value, str) and value == OPTIONAL:
                return
            var[_join_scope(prefix, name)] = value

        self.visit(_register_var)
        if ordered:
            return list(sorted(var.items(), key=lambda x: x[0]))
        return var

    def _alias_variables(self):
        """Find duplicate variables in spec and create aliases."""
        # When a variable is duplicated, keep the version that comes first in
        # the alphabetical order and alias the others.
        variables = self.variables(ordered=True)
        for name, value in reversed(variables):
            for other_name, other_value in variables:
                if name == other_name:
                    break
                # Because variables can be transformed on load (e.g. transposed),
                # we use an element-wise equality check.
                if (
                    not np.isscalar(value)
                    and value.dtype == other_value.dtype
                    and np.array_equal(value, other_value)
                ):
                    # Replace variable value by the alias name.
                    scope, attr_name = _parent_scope(name)
                    spec = index_spec(self, scope)
                    setattr(spec, attr_name, other_name)
                    break

    def _quantize(self, quantization):
        """Possibly quantizes the variable of the layer."""

        def _quantize(spec, name, value):
            if not isinstance(value, np.ndarray):
                return
            if "weight" in name and quantization != "float16":
                if quantization == "int16":
                    # Represent the value with 10 bits so the multiplication is 20 bits
                    # and 12 bits are left for accumulation.
                    scale = np.dtype(value.dtype).type(
                        2**10 / np.amax(np.absolute(value))
                    )
                    value *= scale
                    value = np.rint(value)
                    value = np.clip(
                        value, np.iinfo(np.int16).min, np.iinfo(np.int16).max
                    )
                    value = value.astype(np.int16)
                elif quantization in ("int8", "int8_float16"):
                    amax = np.amax(np.absolute(value), axis=1)
                    amax[amax == 0] = 127.0
                    scale = 127.0 / amax
                    value *= np.expand_dims(scale, 1)
                    value = np.rint(value)
                    value = value.astype(np.int8)
                setattr(spec, "weight_scale", scale)
                setattr(spec, "weight", value)
            elif quantization in ("float16", "int8_float16"):
                if value.dtype == np.float32:
                    key = _split_scope(name)[-1]
                    setattr(spec, key, value.astype(np.float16))

        self.visit(_quantize)

    def optimize(self, quantization=None):
        """Applies some optimizations on this layer."""
        self._alias_variables()
        if quantization is not None:
            self._quantize(quantization)

    def visit(self, fn):
        """Recursively visits this layer and its children."""
        visit_spec(self, fn)


def _dtype_to_type_id(object_dtype):
    # Order should match the DataType enum in include/ctranslate2/types.h
    dtypes = (np.float32, np.int8, np.int16, np.int32, np.float16)
    try:
        return dtypes.index(object_dtype)
    except ValueError:
        raise ValueError(
            "%s is not in list of supported dtypes: %s"
            % (str(object_dtype), ", ".join(map(str, dtypes)))
        )


class ModelSpec(LayerSpec):
    """The top level layer specification."""

    @property
    def name(self):
        """The name of the model specification."""
        raise NotImplementedError()

    @property
    def revision(self):
        """The model specification revision.

        This value is incremented each time the weights layout of the model is
        changed (e.g. a weight is renamed).
        """
        return 1

    def save(self, output_dir):
        """Saves this model specification."""
        self.serialize(os.path.join(output_dir, "model.bin"))

    def serialize(self, path):
        """Serializes the model variables."""
        variables = []
        aliases = []
        for variable in self.variables(ordered=True):
            if isinstance(variable[1], str):
                aliases.append(variable)
            else:
                variables.append(variable)

        with open(path, "wb") as model:

            def _write_string(string):
                model.write(struct.pack("H", len(string) + 1))
                model.write(string.encode("utf-8"))
                model.write(struct.pack("B", 0))

            model.write(struct.pack("I", CURRENT_BINARY_VERSION))
            _write_string(self.name)
            model.write(struct.pack("I", self.revision))
            model.write(struct.pack("I", len(variables)))
            for name, value in variables:
                _write_string(name)
                model.write(struct.pack("B", len(value.shape)))
                for dim in value.shape:
                    model.write(struct.pack("I", dim))
                model.write(struct.pack("B", _dtype_to_type_id(value.dtype)))
                model.write(struct.pack("I", value.nbytes))
                model.write(value.tobytes())
            model.write(struct.pack("I", len(aliases)))
            for alias, variable_name in aliases:
                _write_string(alias)
                _write_string(variable_name)


def _flatten_vocabularies(vocabularies):
    for name, vocabulary in vocabularies.items():
        if len(vocabulary) == 1:
            yield name, vocabulary[0]
        else:
            for i, vocab in enumerate(vocabulary):
                yield "%s_%d" % (name, i + 1), vocab


class SequenceToSequenceModelSpec(ModelSpec):
    """Base specification for sequence to sequence models."""

    def __init__(self, source_embeddings_specs, target_embeddings_specs):
        """Initializes a sequence to sequence model specification.

        Args:
          source_embeddings_specs: List of source EmbeddingsSpec modules.
          target_embeddings_specs: List of target EmbeddingsSpec modules.
        """
        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.with_source_bos = False
        self.with_source_eos = False
        self.with_target_bos = True
        self.user_decoder_start_tokens = False
        self._embeddings_specs = {
            "source": source_embeddings_specs,
            "target": target_embeddings_specs,
        }
        self._vocabularies = {
            "source": [],
            "target": [],
        }
        self._vmap = None

    def register_source_vocabulary(self, tokens):
        """Registers a source vocabulary of tokens."""
        self._vocabularies["source"].append(tokens)

    def register_target_vocabulary(self, tokens):
        """Registers a target vocabulary of tokens."""
        self._vocabularies["target"].append(tokens)

    def register_vocabulary_mapping(self, path):
        """Registers a vocabulary mapping file."""
        self._vmap = path

    def validate(self):
        # Check that vocabularies are registered and have the correct size.
        for name, embeddings_specs in self._embeddings_specs.items():
            vocabularies = self._vocabularies[name]
            if len(vocabularies) != len(embeddings_specs):
                raise ValueError(
                    "Incorrect number of %s vocabularies: %d registered, but expected %d"
                    % (name, len(vocabularies), len(embeddings_specs))
                )
            for i, (vocabulary, embeddings_spec) in enumerate(
                zip(vocabularies, embeddings_specs)
            ):
                expected_size = embeddings_spec.weight.shape[0]
                if len(vocabulary) != expected_size:
                    raise ValueError(
                        "%s vocabulary %d has size %d but the model expected a vocabulary "
                        "of size %d"
                        % (name.capitalize(), i, len(vocabulary), expected_size)
                    )

        if self._vmap is not None and not os.path.exists(self._vmap):
            raise ValueError("Vocabulary mapping file %s does not exist" % self._vmap)

        # Validate the rest of the model.
        super().validate()

    def save(self, output_dir):
        # Save the vocabularies.
        vocabularies = dict(_flatten_vocabularies(self._vocabularies))
        all_vocabularies = list(vocabularies.values())
        if all(vocabulary == all_vocabularies[0] for vocabulary in all_vocabularies):
            vocabularies = {"shared": all_vocabularies[0]}

        for name, tokens in vocabularies.items():
            path = os.path.join(output_dir, "%s_vocabulary.txt" % name)
            _save_lines(path, tokens)

        if self._vmap is not None:
            shutil.copy(self._vmap, os.path.join(output_dir, "vmap.txt"))

        # Save the rest of the model.
        super().save(output_dir)


def _save_lines(path, lines):
    with open(path, "w", encoding="utf-8", newline="") as f:
        for line in lines:
            f.write(line)
            f.write("\n")
