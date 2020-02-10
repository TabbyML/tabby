from ctranslate2.specs import common_spec
from ctranslate2.specs import model_spec


class MultiHeadAttentionSpec(model_spec.LayerSpec):
    def __init__(self, self_attention=False):
        self.layer_norm = common_spec.LayerNormSpec()
        if self_attention:
            num_projections = 2
        else:
            num_projections = 3
        self.linear = [common_spec.LinearSpec() for _ in range(num_projections)]
        self.relative_position_keys = model_spec.OPTIONAL
        self.relative_position_values = model_spec.OPTIONAL
