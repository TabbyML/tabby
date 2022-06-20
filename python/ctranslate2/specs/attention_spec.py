from ctranslate2.specs import common_spec, model_spec


class MultiHeadAttentionSpec(model_spec.LayerSpec):
    def __init__(self, self_attention=False, relative_position=False):
        self.layer_norm = common_spec.LayerNormSpec()
        self.linear = [
            common_spec.LinearSpec() for _ in range(2 if self_attention else 3)
        ]

        if relative_position:
            self.relative_position_keys = None
            self.relative_position_values = None
        else:
            self.relative_position_keys = model_spec.OPTIONAL
            self.relative_position_values = model_spec.OPTIONAL
