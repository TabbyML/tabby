"""Catalog of model specifications."""

from ctranslate2_converter.specs import transformer_spec


class TransformerBase(transformer_spec.TransformerSpec):
    def __init__(self, fused_projections=False):
        super(TransformerBase, self).__init__(6, fused_projections=fused_projections)

class TransformerBig(transformer_spec.TransformerSpec):
    def __init__(self, fused_projections=False):
        super(TransformerBig, self).__init__(6, fused_projections=fused_projections)
