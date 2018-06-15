import argparse
import struct
import shutil
import os
import six

import tensorflow as tf
import numpy as np

def _serialize_variables(output_dir, names, variables, quantization):
    with open(os.path.join(output_dir, "model.bin"), "wb") as model:
        model.write(struct.pack("I", len(variables)))
        for name, value in zip(names, variables):
            if "kernel" in name:
                value = np.transpose(np.squeeze(value))
            if "kernel" in name or "w_embs" in name:
                if quantization == "int16":
                    value *= 1000
                    value = value.astype(np.int16)
            model.write(struct.pack("H", len(name) + 1))
            model.write(tf.compat.as_bytes(name))
            model.write(struct.pack('B', 0))
            model.write(struct.pack("B", len(value.shape)))
            for dim in value.shape:
                model.write(struct.pack("I", dim))
            model.write(struct.pack("B", value.dtype.itemsize))
            model.write(struct.pack("I", value.size))
            model.write(value.tobytes())

def _copy_asset(export_dir, filepath, destination):
    shutil.copy(os.path.join(six.b(export_dir), b"assets", os.path.basename(filepath)),
                destination)

def convert_checkpoint(checkpoint_dir, output_dir, src_vocab, tgt_vocab, quantization="int16"):
    os.makedirs(output_dir)
    shutil.copy(src_vocab, os.path.join(output_dir, "source_vocabulary.txt"))
    shutil.copy(tgt_vocab, os.path.join(output_dir, "target_vocabulary.txt"))
    excludes_pattern = ["optim", "words_per_sec", "global_step"]
    reader = tf.train.load_checkpoint(checkpoint_dir)
    variable_map = reader.get_variable_to_shape_map()
    names = variable_map.keys()
    names = list(filter(lambda n: not any(e in n for e in excludes_pattern), names))
    variables = [reader.get_tensor(name) for name in names]
    _serialize_variables(output_dir, names, variables, quantization)

def convert_saved_model(export_dir, output_dir, quantization="int16"):
    os.makedirs(output_dir)
    with tf.Session() as sess:
        meta_graph = tf.saved_model.loader.load(sess, ["serve"], export_dir)
        tensors = tf.trainable_variables()
        variables = sess.run(tensors)
        assets = sess.run(tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS))
    names = [tensor.name for tensor in tensors]
    _copy_asset(export_dir, assets[0], os.path.join(output_dir, "source_vocabulary.txt"))
    _copy_asset(export_dir, assets[1], os.path.join(output_dir, "target_vocabulary.txt"))
    _serialize_variables(output_dir, names, variables, quantization)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_dir", required=True,
                        help="Model directory.")
    parser.add_argument("--output_dir", required=True,
                        help="Output model directory.")
    parser.add_argument("--type", default="saved_model", choices=["saved_model", "checkpoint"],
                        help="Type of TensorFlow model.")
    parser.add_argument("--src_vocab", default=None,
                        help="Source vocabulary file (required if converting a checkpoint).")
    parser.add_argument("--tgt_vocab", default=None,
                        help="Target vocabulary file (required if converting a checkpoint).")
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

    if args.type == "checkpoint":
        if args.src_vocab is None or args.tgt_vocab is None:
            raise ValueError("vocabularies must be defined for checkpoint conversion")
        convert_checkpoint(args.model_dir, args.output_dir,
                           args.src_vocab, args.tgt_vocab,
                           quantization=args.quantization)
    else:
        convert_saved_model(args.model_dir, args.output_dir,
                            quantization=args.quantization)
