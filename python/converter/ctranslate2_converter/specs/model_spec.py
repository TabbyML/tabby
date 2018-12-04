"""Specifications declare the expected variables layout of CTranslate2 models
that do not load a computation graph. The model converter should make sure that
each required variable of the specification is set.
"""

import six
import numpy as np

OPTIONAL = "optional"


def _join_scope(scope, name):
    if not scope:
        return name
    return "%s/%s" % (scope, name)

def visit_spec(spec, fn, scope=""):
    """Recursively visits a layer spec."""
    for name, value in six.iteritems(spec.__dict__):
        if isinstance(value, list):
            for i, elem in enumerate(value):
                visit_spec(elem, fn, scope=_join_scope(scope, "%s_%d" % (name, i)))
        elif isinstance(value, LayerSpec):
            visit_spec(value, fn, scope=_join_scope(scope, name))
        else:
            fn(spec, _join_scope(scope, name), value)


class LayerSpec(object):
    """Layer specification."""

    def validate(self):
        """Checks that required variables are set to a value."""
        def _is_defined(spec, name, value):
            if value is None:
                raise ValueError("Missing value for attribute %s" % name)
        self.visit(_is_defined)

    def variables(self, prefix=""):
        """Returns a dict mapping variables name to value."""
        var = {}
        def _register_var(spec, name, value):
            if isinstance(value, six.string_types) and value == OPTIONAL:
                return
            var[_join_scope(prefix, name)] = value
        self.visit(_register_var)
        return var

    def quantize(self, quantization):
        """Possibly quantizes the variable of the layer."""
        def _quantize(spec, name, value):
            if "weight" in name:
                if quantization == "int16":
                    value *= 1000
                    value = value.astype(np.int16)
                    setattr(spec, "weight", value)
        self.visit(_quantize)

    def visit(self, fn):
        """Recursively visits this layer and its children."""
        visit_spec(self, fn)
