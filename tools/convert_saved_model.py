import argparse
import struct
import shutil
import os
import six

import tensorflow as tf
import numpy as np

def copy_asset(export_dir, filepath, destination):
    shutil.copy(os.path.join(six.b(args.export_dir), b"assets", os.path.basename(filepath)),
                destination)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--export_dir", required=True,
                    help="Saved model directory.")
parser.add_argument("--output_dir", required=True,
                    help="Output model directory.")
parser.add_argument("--quantization", default="none", choices=["none", "int16"],
                    help="Weight quantization type.")
parser.add_argument("--force", action="store_true",
                    help="Force conversion even if the output directory already exists.")
args = parser.parse_args()

if os.path.exists(args.output_dir):
    if not args.force:
        raise RuntimeError(
            "output directory %s already exists, use --force to override" % args.output_dir)
    else:
        shutil.rmtree(args.output_dir)

os.makedirs(args.output_dir)

with tf.Session() as sess:
    meta_graph = tf.saved_model.loader.load(sess, ["serve"], args.export_dir)
    variables = tf.trainable_variables()
    variables_value = sess.run(variables)
    assets = sess.run(tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS))

    copy_asset(args.export_dir, assets[0], os.path.join(args.output_dir, "source_vocabulary.txt"))
    copy_asset(args.export_dir, assets[1], os.path.join(args.output_dir, "target_vocabulary.txt"))

    with open(os.path.join(args.output_dir, "model.bin"), "wb") as model:
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
