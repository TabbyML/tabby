import argparse
import struct

import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--export_dir", required=True,
                    help="Saved model directory.")
parser.add_argument("--output_model", required=True,
                    help="Output model file.")
parser.add_argument("--quantization", default="none", choices=["none", "int16"],
                    help="Weight quantization type.")
args = parser.parse_args()

with tf.Session() as sess:
    meta_graph = tf.saved_model.loader.load(sess, ["serve"], args.export_dir)
    variables = tf.trainable_variables()
    variables_value = sess.run(variables)

    with open(args.output_model, "wb") as model:
        model.write(struct.pack("I", len(variables)))
        for tensor, value in zip(variables, variables_value):
            if "kernel" in tensor.name:
                value = np.transpose(np.squeeze(value))
            if "kernel" in tensor.name or "w_embs" in tensor.name:
                if args.quantization == "int16":
                    value *= 1000
                    value = value.astype(np.int16)
            model.write(struct.pack("H", len(tensor.name) + 1))
            model.write(tf.compat.as_bytes(tensor.name))
            model.write(struct.pack('B', 0))
            model.write(struct.pack("B", len(value.shape)))
            for dim in value.shape:
                model.write(struct.pack("I", dim))
            model.write(struct.pack("B", value.dtype.itemsize))
            model.write(struct.pack("I", value.size))
            model.write(value.tobytes())
