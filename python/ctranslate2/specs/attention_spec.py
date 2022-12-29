from ctranslate2.specs import common_spec, model_spec


class MultiHeadAttentionSpec(model_spec.LayerSpec):
    def __init__(
        self,
        self_attention=False,
        relative_position=False,
        relative_attention_bias=False,
        rms_norm=False,
    ):
        self.queries_scale = model_spec.OPTIONAL

        self.layer_norm = common_spec.LayerNormSpec(rms_norm=rms_norm)
        self.linear = [
            common_spec.LinearSpec() for _ in range(2 if self_attention else 3)
        ]

        if relative_position:
            self.relative_position_keys = None
            self.relative_position_values = None

        if relative_attention_bias:
            self.relative_attention_bias = None
            self.relative_attention_max_distance = None
