def fuse_linear(spec, layers):
    import numpy as np
    spec.weight = np.concatenate([layer.weight for layer in layers])
    spec.bias = np.concatenate([layer.bias for layer in layers])
