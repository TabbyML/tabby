"""Specifications declare the expected variables layout of CTranslate2 models
that do not load a computation graph. The model converter should make sure that
each required variable of the specification is set.
"""

import struct
import six
import numpy as np

OPTIONAL = "optional"


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
    for name, value in list(six.iteritems(spec.__dict__)):
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

    @property
    def revision(self):
        return 1

    def validate(self):
        """Checks that required variables are set to a valid value."""
        def _check(spec, name, value):
            if value is None:
                raise ValueError("Missing value for attribute %s" % name)
            # Promote float16 to float32 as it is currently an unsupported type.
            if isinstance(value, np.ndarray) and value.dtype == np.float16:
                attr_name = _split_scope(name)[-1]
                setattr(spec, attr_name, value.astype(np.float32))
        self.visit(_check)
        self._alias_variables()

    def variables(self, prefix="", ordered=False):
        """Returns a dict mapping variables name to value. If ordered is True,
        returns an ordered list of (name, value) pairs instead.
        """
        var = {}
        def _register_var(spec, name, value):
            if isinstance(value, six.string_types) and value == OPTIONAL:
                return
            var[_join_scope(prefix, name)] = value
        self.visit(_register_var)
        if ordered:
            return list(sorted(six.iteritems(var), key=lambda x: x[0]))
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
                if value.dtype == other_value.dtype and np.array_equal(value, other_value):
                    # Replace variable value by the alias name.
                    scope, attr_name = _parent_scope(name)
                    spec = index_spec(self, scope)
                    setattr(spec, attr_name, other_name)
                    break

    def quantize(self, quantization):
        """Possibly quantizes the variable of the layer."""
        def _quantize(spec, name, value):
            if "weight" in name and isinstance(value, np.ndarray):
                if quantization == "int16":
                    # Represent the value with 10 bits so the multiplication is 20 bits
                    # and 12 bits are left for accumulation.
                    scale = np.dtype(value.dtype).type(2**10 / np.amax(np.absolute(value)))
                    value *= scale
                    value = np.clip(value, np.iinfo(np.int16).min, np.iinfo(np.int16).max)
                    value = value.astype(np.int16)
                elif quantization == "int8":
                    scale = 127.0 / np.amax(np.absolute(value), axis=1)
                    value *= np.expand_dims(scale, 1)
                    value = value.astype(np.int8)
                setattr(spec, "weight_scale", scale)
                setattr(spec, "weight", value)
        self.visit(_quantize)

    def visit(self, fn):
        """Recursively visits this layer and its children."""
        visit_spec(self, fn)

    def serialize(self, path):
        """Serializes this specification."""
        variables = []
        aliases = []
        for variable in self.variables(ordered=True):
            if isinstance(variable[1], six.string_types):
                aliases.append(variable)
            else:
                variables.append(variable)

        with open(path, "wb") as model:

            def _write_string(string):
                model.write(struct.pack("H", len(string) + 1))
                model.write(six.b(string))
                model.write(struct.pack('B', 0))

            model.write(struct.pack("I", 3))  # Binary version.
            _write_string(self.__class__.__name__)
            model.write(struct.pack("I", self.revision))
            model.write(struct.pack("I", len(variables)))
            for name, value in variables:
                _write_string(name)
                model.write(struct.pack("B", len(value.shape)))
                for dim in value.shape:
                    model.write(struct.pack("I", dim))
                model.write(struct.pack("B", value.dtype.itemsize))
                model.write(struct.pack("I", value.size))
                model.write(value.tobytes())
            model.write(struct.pack("I", len(aliases)))
            for alias, variable_name in aliases:
                _write_string(alias)
                _write_string(variable_name)
