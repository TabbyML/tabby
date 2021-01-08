"""Catalog of model specifications."""

from ctranslate2.specs import transformer_spec


class TransformerBase(transformer_spec.TransformerSpec):
    def __init__(self):
        super(TransformerBase, self).__init__(6, 8)


class TransformerBig(transformer_spec.TransformerSpec):
    def __init__(self):
        super(TransformerBig, self).__init__(6, 16)


class TransformerBaseRelative(transformer_spec.TransformerSpec):
    def __init__(self):
        super(TransformerBaseRelative, self).__init__(6, 8, with_relative_position=True)


class TransformerBigRelative(transformer_spec.TransformerSpec):
    def __init__(self):
        super(TransformerBigRelative, self).__init__(6, 16, with_relative_position=True)
